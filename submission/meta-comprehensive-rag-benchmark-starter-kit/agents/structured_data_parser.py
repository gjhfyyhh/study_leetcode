class StructuredDataParser:
    """结构化数据解析器，处理不同领域的信息"""

    def __init__(self):
        """初始化解析器"""
        # 初始化领域特定解析函数
        self.parsers = {
            "food": self.parse_food,
            "shopping": self.parse_shopping,
            "landmark": self.parse_landmark,
            "math_science": self.parse_math_science,
            "art": self.parse_art,
            "entertainment": self.parse_entertainment,
            "events": self.parse_events,
            "nature": self.parse_nature,
            "people": self.parse_people,
            "sports": self.parse_sports,
            "technology": self.parse_technology,
            "transportation": self.parse_transportation,
            "other": self.parse_generic
        }

    def parse(self, domain, data):
        """解析结构化数据"""
        parser = self.parsers.get(domain, self.parse_generic)
        return parser(data)

    def parse_food(self, data):
        """解析食物领域的结构化数据"""
        result = "<Food>\n"

        # 提取菜名
        name = data.get("name", data.get("dish_name", "Unknown dish"))
        result += f"Name: {name}\n"

        # 提取食物类型
        food_type = data.get("food_type", data.get("type", ""))
        if food_type:
            result += f"Type: {food_type}\n"

        # 提取配料
        ingredients = data.get("ingredients", [])
        if ingredients:
            if isinstance(ingredients, list):
                result += f"Ingredients: {', '.join(ingredients)}\n"
            else:
                result += f"Ingredients: {ingredients}\n"

        # 提取营养信息
        nutrition = data.get("nutrition", {})
        if nutrition:
            result += "Nutrition: "
            if isinstance(nutrition, dict):
                for key, value in nutrition.items():
                    result += f"{key}: {value}, "
                result = result.rstrip(", ") + "\n"
            else:
                result += f"{nutrition}\n"

        # 提取烹饪方法
        cooking_method = data.get("cooking_method", data.get("preparation", ""))
        if cooking_method:
            result += f"Cooking Method: {cooking_method}\n"

        # 提取菜系
        cuisine = data.get("cuisine", data.get("origin", ""))
        if cuisine:
            result += f"Cuisine: {cuisine}\n"

        result += "</Food>"
        return result

    def parse_shopping(self, data):
        """解析购物领域的结构化数据"""
        result = "<Product>\n"

        # 提取产品名称
        name = data.get("name", data.get("product_name", "Unknown product"))
        result += f"Name: {name}\n"

        # 提取品牌
        brand = data.get("brand", data.get("manufacturer", ""))
        if brand:
            result += f"Brand: {brand}\n"

        # 提取价格
        price = data.get("price", data.get("cost", ""))
        if price:
            result += f"Price: {price}\n"

        # 提取类别
        category = data.get("category", data.get("type", ""))
        if category:
            result += f"Category: {category}\n"

        # 提取规格
        specifications = data.get("specifications", data.get("specs", {}))
        if specifications:
            result += "Specifications:\n"
            if isinstance(specifications, dict):
                for key, value in specifications.items():
                    result += f"  - {key}: {value}\n"
            else:
                result += f"  - {specifications}\n"

        result += "</Product>"
        return result

    def parse_landmark(self, data):
        """解析地标领域的结构化数据"""
        result = "<Landmark>\n"

        # 处理维基百科格式的数据
        # 提取名称
        name = data.get("name", "Unknown landmark")
        if '{{' in name:  # 处理维基百科模板
            name = name.split('<br')[0]  # 移除<br>标签
        result += f"Name: {name}\n"

        # 提取地址
        address = data.get("address", "")
        if address:
            # 清理维基百科格式
            address = address.replace("[[", "").replace("]]", "")
            address = address.split('<br')[0] if '<br' in address else address
            result += f"Address: {address}\n"

        # 提取建筑类型
        building_type = data.get("building_type", "")
        if building_type:
            building_type = building_type.replace("[[", "").replace("]]", "")
            result += f"Type: {building_type}\n"

        # 提取建筑风格
        architectural_style = data.get("architectural_style", "")
        if architectural_style:
            architectural_style = architectural_style.replace("[[", "").replace("]]", "")
            result += f"Style: {architectural_style}\n"

        # 提取楼层数量
        floor_count = data.get("floor_count", "")
        if floor_count:
            result += f"Floors: {floor_count}\n"

        # 提取建筑师
        architect = data.get("architect", "")
        if architect:
            architect = architect.replace("[[", "").replace("]]", "")
            result += f"Architect: {architect}\n"

        # 提取完工日期
        completion_date = data.get("completion_date", "")
        if completion_date:
            result += f"Completed: {completion_date}\n"

        result += "</Landmark>"
        return result

    def parse_nature(self, data):
        """解析自然/环境领域的结构化数据"""
        result = "<Nature>\n"

        # 提取名称
        name = data.get("name", "Unknown")
        result += f"Name: {name}\n"

        # 提取类型
        type_info = data.get("type", "")
        if type_info:
            result += f"Type: {type_info}\n"

        # 提取位置
        location = data.get("location", "")
        if location:
            result += f"Location: {location}\n"

        # 提取规则/政策
        rules = data.get("rules", data.get("policy", ""))
        if rules:
            result += f"Rules: {rules}\n"

        # 提取可回收性
        recyclable = data.get("recyclable", data.get("can_recycle", ""))
        if recyclable:
            result += f"Recyclable: {recyclable}\n"

        # 提取材料
        materials = data.get("materials", "")
        if materials:
            if isinstance(materials, list):
                result += f"Materials: {', '.join(materials)}\n"
            else:
                result += f"Materials: {materials}\n"

        result += "</Nature>"
        return result

    def parse_people(self, data):
        """解析人物领域的结构化数据"""
        result = "<Person>\n"

        # 提取姓名
        name = data.get("name", "Unknown person")
        result += f"Name: {name}\n"

        # 提取生日
        birth_date = data.get("birth_date", data.get("born", ""))
        if birth_date:
            result += f"Born: {birth_date}\n"

        # 提取职业
        occupation = data.get("occupation", "")
        if occupation:
            result += f"Occupation: {occupation}\n"

        # 提取国籍
        nationality = data.get("nationality", "")
        if nationality:
            result += f"Nationality: {nationality}\n"

        # 提取成就
        achievements = data.get("achievements", data.get("known_for", ""))
        if achievements:
            result += f"Known for: {achievements}\n"

        result += "</Person>"
        return result

    def parse_math_science(self, data):
        """解析数学科学领域的结构化数据"""
        return self.parse_generic(data)

    def parse_art(self, data):
        """解析艺术领域的结构化数据"""
        return self.parse_generic(data)

    def parse_entertainment(self, data):
        """解析娱乐领域的结构化数据"""
        return self.parse_generic(data)

    def parse_events(self, data):
        """解析事件领域的结构化数据"""
        return self.parse_generic(data)

    def parse_sports(self, data):
        """解析体育领域的结构化数据"""
        return self.parse_generic(data)

    def parse_technology(self, data):
        """解析技术领域的结构化数据"""
        result = "<Technology>\n"

        # 提取名称
        name = data.get("name", "Unknown device")
        result += f"Name: {name}\n"

        # 提取类型
        type_info = data.get("type", data.get("category", ""))
        if type_info:
            result += f"Type: {type_info}\n"

        # 提取品牌
        brand = data.get("brand", data.get("manufacturer", ""))
        if brand:
            result += f"Brand: {brand}\n"

        # 提取规格
        specifications = data.get("specifications", data.get("specs", {}))
        if specifications:
            result += "Specifications:\n"
            if isinstance(specifications, dict):
                for key, value in specifications.items():
                    result += f"  - {key}: {value}\n"
            else:
                result += f"  - {specifications}\n"

        # 提取废弃处理方法
        disposal = data.get("disposal", data.get("recycling", ""))
        if disposal:
            result += f"Disposal: {disposal}\n"

        # 提取环境影响
        environmental = data.get("environmental_impact", "")
        if environmental:
            result += f"Environmental Impact: {environmental}\n"

        result += "</Technology>"
        return result

    def parse_transportation(self, data):
        """解析交通领域的结构化数据"""
        return self.parse_generic(data)

    def parse_generic(self, data):
        """通用解析方法，适用于任何领域"""
        result = "<Info>\n"

        # 遍历数据中的所有属性
        for key, value in data.items():
            # 跳过特殊字段
            if key in ["image", "image_size", "coordinates", "mapframe_wikidata"]:
                continue

            # 清理维基百科格式
            if isinstance(value, str):
                # 移除维基百科标记
                value = value.replace("[[", "").replace("]]", "")
                value = value.replace("{{", "").replace("}}", "")

                # 跳过空值或网页链接
                if not value or value.startswith("http") or "URL|" in value:
                    continue

            # 格式化复杂值
            if isinstance(value, dict):
                result += f"{key}:\n"
                for k, v in value.items():
                    if isinstance(v, str) and (not v or v.startswith("http")):
                        continue
                    result += f"  - {k}: {v}\n"
            elif isinstance(value, list):
                result += f"{key}: {', '.join(map(str, value))}\n"
            else:
                result += f"{key}: {value}\n"

        result += "</Info>"
        return result