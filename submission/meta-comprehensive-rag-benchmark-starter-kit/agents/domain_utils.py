# 领域映射
DOMAIN_MAP = {
    0: "food",
    1: "shopping",
    2: "landmark",
    3: "math_science",
    4: "art",
    5: "entertainment",
    6: "events",
    7: "nature",
    8: "people",
    9: "sports",
    10: "technology",
    11: "transportation",
    12: "other"
}

# 查询类别映射
QUERY_CATEGORY_MAP = {
    0: "simple",
    1: "simple_w_condition",
    2: "comparison",
    3: "aggregation",
    4: "set",
    5: "false_premise",
    6: "post-processing",
    7: "multi-hop"
}

# 动态性映射
DYNAMISM_MAP = {
    0: "static",
    1: "slow-changing",
    2: "fast-changing",
    3: "real-time"
}

def get_domain_name(domain_id):
    """获取领域名称"""
    return DOMAIN_MAP.get(domain_id, "other")

def get_query_category_name(category_id):
    """获取查询类别名称"""
    return QUERY_CATEGORY_MAP.get(category_id, "simple")

def get_dynamism_name(dynamism_id):
    """获取动态性名称"""
    return DYNAMISM_MAP.get(dynamism_id, "static")

def infer_domain_from_query(query):
    """从查询中推断领域"""
    query_lower = query.lower()
    
    # 定义领域关键词
    domain_keywords = {
        "food": ["food", "dish", "meal", "recipe", "ingredient", "cuisine", "eat", "drink", "cook", "bake"],
        "shopping": ["product", "price", "brand", "buy", "purchase", "cost", "store", "shop", "mall"],
        "landmark": ["landmark", "building", "monument", "museum", "tourist", "location", "place", "visit"],
        "math_science": ["math", "science", "formula", "equation", "calculate", "physics", "chemistry", "biology"],
        "art": ["art", "painting", "sculpture", "museum", "artist", "gallery", "draw", "creative"],
        "entertainment": ["movie", "film", "series", "show", "actor", "actress", "director", "watch"],
        "events": ["event", "conference", "meeting", "festival", "concert", "exhibition", "ceremony"],
        "nature": ["nature", "plant", "animal", "flower", "tree", "landscape", "outdoor", "environment", "bin", "recycle", "battery"],
        "people": ["person", "people", "man", "woman", "child", "family", "group"],
        "sports": ["sport", "game", "team", "player", "match", "competition", "athlete", "race"],
        "technology": ["technology", "device", "computer", "phone", "gadget", "tech", "digital", "battery"],
        "transportation": ["car", "vehicle", "train", "bus", "transportation", "travel", "trip", "journey"],
    }
    
    # 查找匹配的领域
    for domain, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return domain
    
    # 默认领域
    return "other"

def extract_domain_from_message_history(message_history):
    """从消息历史中提取领域信息"""
    if not message_history:
        return None
        
    # 尝试从历史中提取领域ID
    for message in message_history:
        if isinstance(message, dict):
            # 尝试从用户消息的content中提取domain
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    # 处理多模态内容
                    for part in content:
                        if isinstance(part, dict) and part.get("domain") is not None:
                            domain_id = part.get("domain")
                            return get_domain_name(domain_id)
                elif isinstance(content, dict) and content.get("domain") is not None:
                    domain_id = content.get("domain")
                    return get_domain_name(domain_id)
            
            # 尝试从消息本身提取domain
            if "domain" in message:
                domain_id = message.get("domain")
                return get_domain_name(domain_id)
            
    return None

def process_multi_turn_conversation(turn_data):
    """处理多轮对话数据，提取有用信息"""
    result = {
        "is_multi_turn": len(turn_data) > 1,
        "domains": [],
        "query_categories": [],
        "dynamism": []
    }
    
    # 处理每个回合
    for turn in turn_data:
        domain_id = turn.get("domain")
        domain = get_domain_name(domain_id) if domain_id is not None else "other"
        result["domains"].append(domain)
        
        category_id = turn.get("query_category")
        category = get_query_category_name(category_id) if category_id is not None else "simple"
        result["query_categories"].append(category)
        
        dynamism_id = turn.get("dynamism")
        dynamism = get_dynamism_name(dynamism_id) if dynamism_id is not None else "static"
        result["dynamism"].append(dynamism)
    
    return result