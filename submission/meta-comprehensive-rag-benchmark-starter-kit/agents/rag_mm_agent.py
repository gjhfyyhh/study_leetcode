import torch
from typing import Dict, List, Any
from PIL import Image

from .base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
from .structured_data_parser import StructuredDataParser
from .domain_utils import (
    infer_domain_from_query, get_domain_name,
    get_query_category_name, get_dynamism_name
)

import vllm

# 配置常量
BATCH_SIZE = 8
MAX_GENERATION_TOKENS = 75


class RAGMMAgent(BaseAgent):
    """CRAG-MM多模态RAG代理 - 支持所有三个任务"""

    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        """初始化代理"""
        super().__init__(search_pipeline)

        # 保存搜索管道
        self.search_pipeline = search_pipeline

        # 初始化大型语言模型
        self.initialize_models()

        # 初始化数据解析器
        self.parser = StructuredDataParser()

        # 会话历史缓存 - 用于多轮对话(任务3)
        self.session_cache = {}

        print("RAGMMAgent初始化完成")

    def initialize_models(self):
        """初始化vLLM模型"""
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        print(f"正在加载模型 {model_name}...")
        self.llm = vllm.LLM(
            model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
            max_num_seqs=2,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1  # CRAG-MM数据集中每个对话最多有1张图像
            }
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("模型加载完成")

    def get_batch_size(self) -> int:
        """返回批处理大小"""
        return BATCH_SIZE

    def extract_domain_from_turn(self, turn):
        """从turn数据中提取领域"""
        if not turn:
            return "other"

        domain_id = turn.get("domain")
        if domain_id is not None:
            return get_domain_name(domain_id)
        return "other"

    def extract_session_info(self, message_histories):
        """从消息历史中提取会话信息"""
        session_info = []

        for history in message_histories:
            current_info = {"domain": "other", "is_multi_turn": len(history) > 0} # 遍历所有消息对

            for i, message in enumerate(history):
                if i % 2 == 0 and message.get("role") == "user":
                    content = message.get("content", "") # 从用户消息中尝试提取领域信息
                    if isinstance(content, dict) and "domain" in content:
                        domain_id = content.get("domain")
                        current_info["domain"] = get_domain_name(domain_id)

            session_info.append(current_info)

        return session_info

    def prepare_search_queries(self, queries, images, message_histories):
        """准备搜索查询"""
        search_queries = []

        # 提取会话信息
        session_info = self.extract_session_info(message_histories)

        for i, (query, image, info) in enumerate(zip(queries, images, session_info)):
            # 提取领域
            domain = info.get("domain", "other")
            if domain == "other":
                # 如果无法从历史中提取，则从查询文本推断
                domain = infer_domain_from_query(query)

            # 增强查询
            enhanced_query = self._enhance_query(query, domain)
            search_queries.append(enhanced_query)

        return search_queries

    def _enhance_query(self, query, domain):
        """根据领域增强查询"""
        # 简单的查询增强策略
        domain_prefixes = {
            "food": "food dish ",
            "shopping": "product item ",
            "landmark": "landmark building ",
            "nature": "nature landscape recycling bin ",
            "people": "person people ",
            "sports": "sports game ",
            "technology": "technology device battery ",
            "transportation": "vehicle transportation "
        }

        prefix = domain_prefixes.get(domain, "")
        return f"{prefix}{query}"

    def process_image_search_results(self, search_results):
        """处理图像搜索结果"""
        formatted_results = []

        for result_list in search_results:
            if not result_list:  # 跳过空结果
                formatted_results.append("")
                continue

            parsed_result = ""

            # 处理多个搜索结果
            for result in result_list:
                # 从搜索结果中提取实体信息
                entities = result.get("entities", [])

                # 使用解析器处理实体数据
                for entity in entities:
                    entity_name = entity.get("entity_name", "Unknown")
                    entity_attrs = entity.get("entity_attributes", {})

                    # 确保 entity_attrs 不是 None
                    if entity_attrs is None:
                        entity_attrs = {}

                    # 解析结构化数据
                    domain = self._infer_entity_domain(entity_attrs)
                    parsed_entity = self.parser.parse(domain, {"name": entity_name, **entity_attrs})

                    if parsed_entity:
                        parsed_result += parsed_entity + "\n\n"

            formatted_results.append(parsed_result.strip())

        return formatted_results

    def process_web_search_results(self, search_results):
        """处理网页搜索结果"""
        formatted_results = []

        for result_list in search_results:
            if not result_list:  # 跳过空结果
                formatted_results.append("")
                continue

            context = ""

            # 处理多个搜索结果
            for result in result_list:
                if "page_snippet" in result:
                    snippet = result.get("page_snippet", "")
                    page_name = result.get("page_name", "Unknown Source")

                    if snippet:
                        context += f"From {page_name}:\n{snippet}\n\n"

            formatted_results.append(context.strip())

        return formatted_results

    def _infer_entity_domain(self, entity_attrs):
        """从实体属性推断领域"""
        # 基于实体属性简单推断领域
        attrs_str = str(entity_attrs).lower()

        if any(keyword in attrs_str for keyword in ["food", "dish", "meal", "recipe", "ingredient", "cuisine"]):
            return "food"
        elif any(keyword in attrs_str for keyword in ["product", "price", "brand", "buy", "store"]):
            return "shopping"
        elif any(keyword in attrs_str for keyword in ["building", "architect", "address", "location"]):
            return "landmark"
        elif any(keyword in attrs_str for keyword in ["person", "born", "birth"]):
            return "people"
        elif any(keyword in attrs_str for keyword in ["recycle", "battery", "bin"]):
            return "nature"

        # 默认领域
        return "other"

    def prepare_llm_prompts(self, queries, images, message_histories, image_contexts=None, web_contexts=None):
        """准备LLM输入"""
        prompts = []

        for idx, (query, image, history) in enumerate(
                zip(queries, images, message_histories)
        ):
            # 构建系统提示词
            system_prompt = (
                "You are a helpful visual assistant. Answer questions about the image "
                "accurately and concisely based on what you can see and the provided context. "
                "If you don't know the answer, say 'I don't know'."
            )

            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]

            # 添加对话历史(如果有)
            if history:
                # 转换历史格式确保兼容性
                processed_history = self._process_message_history(history)
                messages.extend(processed_history)

            # 添加图像相关上下文(如果有)
            if image_contexts and idx < len(image_contexts) and image_contexts[idx]:
                context_message = {
                    "role": "system",
                    "content": f"Information about what's in the image:\n\n{image_contexts[idx]}"
                }
                messages.append(context_message)

            # 添加网页相关上下文(如果有) - 仅适用于任务2
            if web_contexts and idx < len(web_contexts) and web_contexts[idx]:
                web_context_message = {
                    "role": "system",
                    "content": f"Additional information from the web:\n\n{web_contexts[idx]}"
                }
                messages.append(web_context_message)

            # 添加当前查询和图像
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]
            }
            messages.append(user_message)

            # 应用聊天模板
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            prompts.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })

        return prompts

    def _process_message_history(self, message_history):
        """处理消息历史，确保格式正确"""
        processed_history = []

        for message in message_history:
            # 确保消息包含role和content字段
            if "role" not in message or "content" not in message:
                continue

            # 如果content是字典或列表，处理为字符串
            content = message["content"]
            if isinstance(content, (dict, list)):
                # 尝试提取文本内容
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    content = " ".join(text_parts)
                elif isinstance(content, dict):
                    content = str(content)

            # 添加处理后的消息
            processed_history.append({
                "role": message["role"],
                "content": content
            })

        return processed_history

    def _update_session_cache(self, queries, responses, message_histories):
        """更新会话缓存，用于多轮对话"""
        for query, response, history in zip(queries, responses, message_histories):
            # 为每个会话生成唯一标识符
            if not history:
                continue

            # 尝试从历史中提取会话ID
            session_id = None
            for message in history:
                if "session_id" in message:
                    session_id = message["session_id"]
                    break

            if not session_id:
                continue

            # 更新会话缓存
            if session_id not in self.session_cache:
                self.session_cache[session_id] = []

            # 添加最新一轮对话
            self.session_cache[session_id].append({
                "query": query,
                "response": response
            })

    def batch_generate_response(
            self,
            queries: List[str],
            images: List[Image.Image],
            message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """生成批量回答"""
        if not images or len(images) == 0:
            return ["I don't have an image to analyze."] * len(queries)

        # 1. 准备搜索查询
        search_queries = self.prepare_search_queries(queries, images, message_histories)

        # 2. 执行图像搜索 (任务1、2、3共用)
        image_search_results = []
        for i, (query, image) in enumerate(zip(search_queries, images)):
            try:
                # 使用图像搜索API
                result = self.search_pipeline(image, k=3)
                image_search_results.append(result)
            except Exception as e:
                print(f"图像搜索时出错: {e}")
                image_search_results.append([])

        # 3. 执行网页搜索 (仅任务2)
        web_search_results = []
        try:
            # 检查是否支持web搜索(任务2)
            if hasattr(self.search_pipeline, 'search_web') and self.search_pipeline.web_index is not None:
                for query in search_queries:
                    try:
                        # 使用web搜索API
                        result = self.search_pipeline(query, k=3)
                        web_search_results.append(result)
                    except Exception as e:
                        print(f"网页搜索时出错: {e}")
                        web_search_results.append([])
        except Exception as e:
            print(f"网页搜索初始化出错: {e}")

        # 4. 处理搜索结果
        image_contexts = self.process_image_search_results(image_search_results)
        web_contexts = self.process_web_search_results(web_search_results) if web_search_results else None

        # 5. 准备LLM输入
        prompts = self.prepare_llm_prompts(queries, images, message_histories, image_contexts, web_contexts)

        # 6. 批量生成回答
        outputs = self.llm.generate(
            prompts,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )

        # 7. 提取生成的文本
        responses = [output.outputs[0].text for output in outputs]

        # 8. 更新会话缓存(任务3)
        self._update_session_cache(queries, responses, message_histories)

        return responses