"""
TPCLLM - Natural Language to Symbolic Constraint Translator.

Translates natural language travel queries into executable Python
constraint code using LLM-based translation with reflection.

Author: YZ1553927
License: MIT
"""

import re
import json
from typing import Dict, List, Any
import time

from chinatravel.agent.llms import AbstractLLM
from tpc_agent.converter import SimpleTravelConverter

class TPCLLM(AbstractLLM):
    def __init__(self, model_path=None):
        """初始化TPCLLM - 与TPC Agent对接的版本"""
        super().__init__()
        self.name = "TPC-Algorithm"
        self.travel_converter = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "Qwen3-4B", "Qwen", "Qwen3-4B")
        if model_path:
            try:
                self.travel_converter = SimpleTravelConverter(model_path)
                print("✅ SimpleTravelConverter 初始化成功")
            except Exception as e:
                print(f"❌ SimpleTravelConverter 初始化失败: {e}")
                self.travel_converter = None

        # 定义景点类型集合，用于区分名称和类型
        self.attraction_types = {
            "人文景观", "公园", "博物馆/纪念馆", "历史古迹", "商业街区","大学校园",
            "文化旅游区", "游乐园/体育娱乐", "红色景点", "美术馆/艺术馆", "自然风光"
        }


        self.restaurant_types = {
            "东北菜", "东南亚菜", "亚洲菜", "其他", "其他中餐", "创意菜", "咖啡店",
            "客家菜", "小吃", "川菜", "快餐简餐", "日本料理", "江浙菜", "海鲜",
            "湘菜", "火锅", "烧烤", "粤菜", "素食", "自助餐", "融合菜", "西北菜",
            "西餐", "酒吧/酒馆", "面包甜点", "韩国料理", "鲁菜","新疆菜","茶馆/茶室"
        }

        # 酒店类型
        self.hotel_types = {
            "酒店公寓", "设计师酒店", "行李寄存", "行政酒廊", "茶室", "自营亲子房",
            "美食酒店", "网红泳池", "管家服务", "湖景房", "温泉", "洗衣服务", "洗衣房",
            "泳池", "江河景房", "棋牌室", "桑拿", "桌球室", "机器人服务", "智能客控",
            "日光浴场", "提前入园", "影音房", "山景房", "家庭房", "套房", "多功能厅",
            "四合院", "动人夜景", "免费停车", "充电桩", "儿童乐园", "健身室", "停车场",
            "会议厅", "亲子主题房", "SPA"
        }

        self.city_specialty_mapping = {
            "北京": ["其他中餐", "火锅", "小吃", "烧烤"],
            "成都": ["川菜", "火锅", "小吃"],
            "重庆": ["川菜", "火锅"],
            "广州": ["粤菜", "海鲜", "小吃"],
            "杭州": ["江浙菜", "小吃"],
            "南京": ["江浙菜", "小吃"],
            "上海": ["江浙菜", "小吃", "融合菜"],
            "深圳": ["粤菜", "客家菜", "融合菜"],
            "苏州": ["江浙菜", "小吃"],
            "武汉": ["湘菜", "小吃"]
        }

        # 添加类型映射
        self.type_mapping = {
            "大床房": "单床房",
            "双床房": "双床房",
            "历史景点": "历史古迹",
            "历史文化景点": "历史古迹",
            "自然风景":"自然风光",
            "博物馆": "博物馆/纪念馆",
            "当地美食": "小吃",
            "当地小吃": "小吃",
            "故宫":"故宫博物馆",
            "茶馆":"茶馆/茶室",
            "简餐":"快餐简餐",
            "打电子游戏":"电竞房",
            "健身":"健身室",
            "苏州园林":"园林",
            "高校":"大学校园",
            "当地高校": "大学校园",
            "美术馆":"美术馆/艺术馆",
            "水立方":"水立方嬉水乐园",
            "詹天佑故居":"詹天佑故居纪念馆"
        }


    def nl_to_python_direct(self, natural_language=None, days=None, people_number=None,
                            start_city=None, target_city=None):
        """直接将自然语言转换为Python代码 - 与TPC Agent对接的版本"""

        print("🔍 TPC-Algorithm 开始处理自然语言输入...")

        # 使用 travel_converter 将自然语言转换为 structured_data
        if natural_language and self.travel_converter:
            print("📝 使用 SimpleTravelConverter 转换自然语言...")
            print(f"{natural_language}")
            structured_data_old = self.travel_converter.convert_with_fallback(natural_language)
            print("✅ 自然语言转换完成")

            # 获取美化后的数据
            structured_data = self.travel_converter.get_pretty_data(structured_data_old)
            print(f"🎨 美化数据: {structured_data}")

            if not structured_data:
                structured_data = {
                    "start_city": start_city,
                    "target_city": target_city,
                    "days": days,
                    "people_number": people_number,
                    "城内交通": ["taxi", "metro"],
                    "房型": "大床房",
                    "想去的景点": ["历史景点", "博物馆"],
                    "食物偏好": ["火锅", "本地特色美食"]
                }

            # 打印转换结果用于调试
            self.travel_converter.print_result(structured_data)
        else:
            # 如果没有自然语言或转换器不可用，创建空的 structured_data
            print("⚠️ 无自然语言输入或转换器不可用")
            structured_data = {
                "start_city": start_city,
                "target_city": target_city,
                "days": days,
                "people_number":people_number,
                "往返交通": ["train"],
                "城内交通": ["taxi", "train"],
                "房型": "大床房",
                "想去的景点": ["历史景点", "博物馆"],
                "食物偏好": ["火锅","本地特色美食"]
            }

        basic_info = self._build_basic_info(structured_data, days, people_number, start_city, target_city)
        print(f"📊 使用的基础信息: {basic_info}")

        # 提取约束条件
        constraints_info = self._extract_constraints_from_structured(natural_language , structured_data , target_city)
        print(f"🎯 提取的约束条件: {constraints_info}")

        # 生成Python代码
        hard_logic_py = self._generate_python_code(basic_info, constraints_info)
        print(f"🐍 生成的Python代码块数: {len(hard_logic_py)}")

        # 构建与TPC Agent兼容的返回格式
        result = {
            "start_city": basic_info["start_city"],
            "target_city": basic_info["target_city"],
            "days": basic_info["days"],
            "people_number": basic_info["people_number"],
            "hard_logic_py": hard_logic_py
        }

        return result


    def _build_basic_info(self, structured_data, days=None, people_number=None, start_city=None, target_city=None):
        """构建基础信息，优先级：传入参数 > 结构化数据 > 默认值"""
        basic_info = {}

        # 人数：参数 > 结构化数据 > 默认值
        if people_number is not None:
            basic_info["people_number"] = people_number
        elif "people_count" in structured_data:
            basic_info["people_number"] = structured_data["people_count"]
        else:
            basic_info["people_number"] = 1

        # 出发城市：参数 > 结构化数据 > 默认值
        if start_city is not None:
            basic_info["start_city"] = start_city
        elif "start_city" in structured_data:
            basic_info["start_city"] = structured_data["start_city"]
        else:
            basic_info["start_city"] = "北京"

        # 目标城市：参数 > 结构化数据 > 默认值
        if target_city is not None:
            basic_info["target_city"] = target_city
        elif "target_city" in structured_data:
            basic_info["target_city"] = structured_data["target_city"]
        else:
            basic_info["target_city"] = "上海"

        # 天数：参数 > 结构化数据 > 默认值
        if days is not None:
            basic_info["days"] = days
        elif "travel_days" in structured_data:
            basic_info["days"] = structured_data["travel_days"]
        else:
            basic_info["days"] = 3

        return basic_info

    def _extract_constraints_from_structured(self,natural_language , data , target_city):
        """从结构化数据提取约束条件"""
        constraints = []

        # 提取预算约束
        constraints.extend(self._extract_budget_constraints(data))

        # 提取交通约束
        constraints.extend(self._extract_transport_constraints(natural_language,data))

        # 提取住宿约束
        constraints.extend(self._extract_accommodation_constraints(data))

        # 提取景点约束
        constraints.extend(self._extract_attraction_constraints(data))

        # 提取餐厅约束
        constraints.extend(self._extract_restaurant_constraints(data,target_city))

        # 提取特殊偏好约束
        constraints.extend(self._extract_special_preference_constraints(data))

        print(f"📝 提取到的约束数量: {len(constraints)}")

        return {"logic_type": "and", "constraints": constraints}


    def _extract_budget_constraints(self, data):
        """提取预算约束"""
        constraints = []

        # 总预算
        if "在旅行上的总预算" in data:
            amount = float(data["在旅行上的总预算"].replace("元", ""))
            constraints.append({"type": "budget_total", "amount": amount})

        # 住宿预算
        if "在住宿上的预算" in data:
            amount = float(data["在住宿上的预算"].replace("元", ""))
            constraints.append({"type": "budget_accommodation", "amount": amount})

        # 跨城市交通预算
        if "跨城市交通费的预算" in data:
            amount = float(data["跨城市交通费的预算"].replace("元", ""))
            constraints.append({"type": "budget_inter_transport", "amount": amount})

        # 城市内出行预算
        if "在城市内出行的预算" in data:
            amount = float(data["在城市内出行的预算"].replace("元", ""))
            constraints.append({"type": "budget_inner_transport", "amount": amount})

        # 用餐预算
        if "在用餐用餐预算" in data:
            amount = float(data["在用餐用餐预算"].replace("元", ""))
            constraints.append({"type": "budget_meal", "amount": amount})

        # 游览预算
        if "在游览上的预算" in data:
            amount = float(data["在游览上的预算"].replace("元", ""))
            if amount==0:
                constraints.append({"type": "attraction_free_only"})
            constraints.append({"type": "budget_attraction", "amount": amount})

        return constraints

    def map_intercity_mode(self, modes, natural_language=""):
        """映射城际交通方式，包含自然语言验证"""
        # 定义关键词映射
        train_keywords = ["高铁", "动车", "火车", "列车", "铁路", "站票", "硬座", "软卧", "G", "D", "K", "T", "Z"]
        airplane_keywords = ["飞机", "航班", "航空", "机场", "登机", "经济舱", "头等舱", "飞", "航站楼"]

        # 确保 modes 是列表
        if not isinstance(modes, list):
            modes = [modes]

        mode_str = " ".join(str(mode) for mode in modes)
        natural_language_str = str(natural_language)

        # 检查自然语言中是否包含交通方式关键词
        has_train_in_nl = any(keyword in natural_language_str for keyword in train_keywords)
        has_airplane_in_nl = any(keyword in natural_language_str for keyword in airplane_keywords)

        # 检查交通方式列表中包含的类型
        has_train = False
        has_airplane = False

        for mode in modes:
            mode_lower = str(mode).lower()
            if any(keyword in mode_lower for keyword in train_keywords):
                has_train = True
            elif any(keyword in mode_lower for keyword in airplane_keywords):
                has_airplane = True
            elif mode_lower == "train":
                has_train = True
            elif mode_lower == "airplane":
                has_airplane = True

        # 根据逻辑决定返回结果
        if has_train and not has_airplane:
            # 只有火车，没有飞机
            if natural_language_str and not has_train_in_nl:
                return None  # 自然语言验证失败
            return "train"
        elif has_airplane and not has_train:
            # 只有飞机，没有火车
            if natural_language_str and not has_airplane_in_nl:
                return None  # 自然语言验证失败
            return "airplane"
        elif has_train and has_airplane:
            # 既有火车又有飞机，相当于没有限制
            return None
        else:
            # 没有明确的交通方式
            return None


    def _extract_transport_constraints(self,natural_language, data):
        """提取交通约束"""
        constraints = []


        # 城内交通方式映射表
        inner_city_transport_mapping = {
            "地铁": "metro",
            "轨道交通": "metro",
            "打车": "taxi",
            "出租车": "taxi",
            "的士": "taxi",
            "步行": "walk",
            "走路": "walk",
            "徒步": "walk",
        }


        def map_inner_city_mode(mode):
            if mode in ["walk", "taxi", "metro"]:
                return mode
            return inner_city_transport_mapping.get(mode, mode)

        if "往返交通" in data and data["往返交通"]:
            transport_mode = self.map_intercity_mode(data["往返交通"], natural_language)
            if transport_mode:
                constraints.append({
                    "type": "intercity_transport_wanted",
                    "go_mode": transport_mode,
                    "back_mode": transport_mode
                })


        # if "去程交通" in data and data["去程交通"]:
        #     transport_go = map_intercity_mode(data["去程交通"][0])
        #
        #     if "返程交通" in data and data["返程交通"]:
        #         transport_back = map_intercity_mode(data["返程交通"][0])
        #         constraints.append({
        #             "type": "intercity_transport_wanted",
        #             "go_mode": transport_go,
        #             "back_mode": transport_back
        #         })
        #     else:
        #         constraints.append({
        #             "type": "intercity_transport_wanted",
        #             "go_mode": transport_go,
        #             "back_mode": transport_go
        #         })
        #
        #
        # if "返程交通" in data and data["返程交通"] and "去程交通" not in data:
        #     transport_mode = map_intercity_mode(data["返程交通"][0])
        #     constraints.append({
        #         "type": "intercity_transport_wanted",
        #         "go_mode": transport_mode,
        #         "back_mode": transport_mode
        #     })


        if "城内交通" in data and data["城内交通"]:
            wanted_modes = [map_inner_city_mode(mode) for mode in data["城内交通"]]
            constraints.append({
                "type": "transport_mode_wanted",
                "modes": wanted_modes
            })

        if "避免城内交通" in data and data["避免城内交通"]:
            unwanted_modes = [map_inner_city_mode(mode) for mode in data["避免城内交通"]]
            constraints.append({
                "type": "transport_mode_unwanted",
                "modes": unwanted_modes
            })

        return constraints

    def _extract_accommodation_constraints(self, data):
        """提取住宿约束"""
        constraints = []

        # 酒店偏好@@@@@
        if "酒店偏好" in data and data["酒店偏好"]:
            hotel_types = [self.type_mapping.get(hotel, hotel) for hotel in data["酒店偏好"]]
            if all(hotel in self.hotel_types for hotel in hotel_types):
                constraints.append({
                    "type": "accommodation_type_wanted",
                    "types": hotel_types
                })
            else:
                constraints.append({
                    "type": "accommodation_wanted",
                    "names": hotel_types
                })

        if "酒店类型偏好" in data and data["酒店类型偏好"]:
            hotel_types = [self.type_mapping.get(hotel, hotel) for hotel in data["酒店类型偏好"]]
            if all(hotel in self.hotel_types for hotel in hotel_types):
                constraints.append({
                    "type": "accommodation_type_wanted",
                    "types": hotel_types
                })


        # # 房型
        # if "房型" in data:
        #     room_type = data["房型"]
        #     if room_type == "大床房":
        #         constraints.append({"type": "room_type", "room_type": 1})
        #     elif room_type == "双床房":
        #         constraints.append({"type": "room_type", "room_type": 2})
        #     elif room_type == "标间":
        #         constraints.append({"type": "room_type", "room_type": 2})
        #     elif room_type == "标准房":
        #         constraints.append({"type": "room_type", "room_type": 2})
        #     elif room_type == "标准间":
        #         constraints.append({"type": "room_type", "room_type": 2})
        #     elif room_type == "单床房":
        #         constraints.append({"type": "room_type", "room_type": 1})
        #     elif room_type == "双人房":
        #         constraints.append({"type": "room_type", "room_type": 2})
        #     elif room_type == "标准件双床房":
        #         constraints.append({"type": "room_type", "room_type": 2})

        return constraints

    def _extract_attraction_constraints(self, data):
        """提取景点约束"""
        constraints = []

        if "想去的景点" in data and data["想去的景点"]:
            attractions = [self.type_mapping.get(attr, attr) for attr in data["想去的景点"]]

            # 判断是景点名称还是类型
            if all(attr in self.attraction_types for attr in attractions):
                constraints.append({
                    "type": "attraction_type_wanted",
                    "types": attractions
                })
            else:
                constraints.append({
                    "type": "attraction_wanted",
                    "names": attractions
                })

        if "想去的景点类型" in data and data["想去的景点类型"]:
            attractions = [self.type_mapping.get(attr, attr) for attr in data["想去的景点类型"]]

            # 判断是景点名称还是类型
            if all(attr in self.attraction_types for attr in attractions):
                constraints.append({
                    "type": "attraction_type_wanted",
                    "types": attractions
                })


        return constraints

    def _process_restaurant_name(self, restaurant_name: str) -> str:
        """处理餐厅名称，提取餐厅名和地点信息"""
        import re

        # 模式1: "地点附近的餐厅名"
        pattern1 = r'(.+?)附近的(.+)'
        match1 = re.match(pattern1, restaurant_name)
        if match1:
            location = match1.group(1).strip()
            restaurant = match1.group(2).strip()
            return f"{restaurant}({location}店)"

        # 模式2: "餐厅名在/位于地点"
        pattern2 = r'(.+?)(在|位于)(.+)'
        match2 = re.match(pattern2, restaurant_name)
        if match2:
            restaurant = match2.group(1).strip()
            location = match2.group(3).strip()
            return f"{restaurant}({location}店)"

        # 模式3: "地点的餐厅名"
        pattern3 = r'(.+?)的(.+)'
        match3 = re.match(pattern3, restaurant_name)
        if match3:
            location = match3.group(1).strip()
            restaurant = match3.group(2).strip()
            return f"{restaurant}({location}店)"

        # 模式4: 包含具体商圈或地标
        common_locations = ['新街口', '夫子庙', '鼓楼', '玄武门', '中山陵', '总统府',
                            '老门东', '湖南路', '珠江路', '上海路', '日坛公园']

        for location in common_locations:
            if location in restaurant_name:
                # 移除地点词，保留餐厅名
                restaurant_main = restaurant_name.replace(location, '').strip()
                # 清理多余的连接词
                restaurant_main = re.sub(r'[的附近周边]', '', restaurant_main).strip()
                if restaurant_main:
                    return f"{restaurant_main}({location}店)"

        return restaurant_name

    def _extract_restaurant_constraints(self, data,target_city):
        """提取餐厅约束"""
        constraints = []
        restaurant_types = []
        restaurant_type_unwanted = []
        if "餐厅偏好" in data and data["餐厅偏好"]:
            processed_restaurants = []
            for rest in data["餐厅偏好"]:
                # 处理包含地点的餐厅名称
                processed_name = self._process_restaurant_name(rest)
                processed_restaurants.append(processed_name)

            # 使用映射
            restaurants = [self.type_mapping.get(rest, rest) for rest in processed_restaurants]
            constraints.append({
                "type": "restaurant_name_wanted",
                "names": restaurants
            })


        if "食物偏好" in data and data["食物偏好"]:
            # 获取目标城市target_city才会成功
            target_city = target_city
            for preference in data["食物偏好"]:
                if preference in self.restaurant_types:
                    restaurant_types.append(preference)
                    #只有在数据传入
                elif target_city and any(keyword in preference for keyword in ["美食", "特色", "当地", "本地"]):
                    city_types = self.city_specialty_mapping.get(target_city, [])
                    valid_types = [t for t in city_types if t in self.restaurant_types]
                    restaurant_types.extend(valid_types)

                else:
                    mapped_type = self.type_mapping.get(preference, preference)
                    if mapped_type in self.restaurant_types:
                        restaurant_types.append(mapped_type)


        if "饮食限制" in data and data["饮食限制"]:
            for restriction in data["饮食限制"]:
                if restriction in ["吃辣", "微辣"]:
                    if "川菜" not in restaurant_types:
                        restaurant_types.append("川菜")

                elif restriction == "不吃辣":
                    if restaurant_types:
                        spicy_cuisines = ["川菜", "湘菜"]
                        restaurant_types = [t for t in restaurant_types if t not in spicy_cuisines]
                    restaurant_type_unwanted.extend(["川菜", "湘菜"])  # 或者分开append

        if "菜系偏好" in data and data["菜系偏好"]:
            for cuisine in data["菜系偏好"]:

                if cuisine in self.restaurant_types:
                    restaurant_types.append(cuisine)
                else:
                    mapped_cuisine = self.type_mapping.get(cuisine, cuisine)
                    if mapped_cuisine in self.restaurant_types:
                        restaurant_types.append(mapped_cuisine)


        restaurant_types = list(set(restaurant_types))

        if restaurant_types:
            constraints.append({
                "type": "restaurant_type_wanted",
                "types": restaurant_types
            })
        if restaurant_type_unwanted:
            constraints.append({
                "type": "restaurant_type_unwanted",
                "types": restaurant_type_unwanted
            })

        return constraints

    def _extract_special_preference_constraints(self, data):
        """提取特殊偏好约束"""
        constraints = []

        # 检查是否有特殊偏好字段
        if "特殊偏好" in data and data["特殊偏好"]:
            for preference in data["特殊偏好"]:
                constraint = self._parse_special_preference(preference)
                if constraint:
                    constraints.append(constraint)

        return constraints

    def _parse_special_preference(self, preference_text):
        """解析单个特殊偏好"""
        preference_text = preference_text.lower().strip()

        # 免费景点偏好
        if any(keyword in preference_text for keyword in ["免费景点", "免费游览", "不花钱", "免门票","不要钱","免费"]):
            return {"type": "attraction_free_only"}


        return None


    def _extract_names(self, text: str, prefix: str) -> List[str]:
        """从文本中提取名称列表"""
        clean_text = text.replace(prefix, "").strip().replace("：", "").replace(":", "")
        clean_text = re.sub(r'\d+\.\s*', '', clean_text)

        # 处理各种分隔符
        names = []
        if "和" in clean_text:
            names = [name.strip() for name in clean_text.split("和")]
        elif "与" in clean_text:
            names = [name.strip() for name in clean_text.split("与")]
        elif "、" in clean_text:
            names = [name.strip() for name in clean_text.split("、")]
        elif "或" in clean_text:
            names = [name.strip() for name in clean_text.split("或")]
        else:
            names = [clean_text]

        return [name for name in names if name]

    def _generate_python_code(self, basic_info: Dict, constraints_info: Dict) -> List[str]:
        """生成Python代码 - 与TPC Agent兼容的版本"""
        code_blocks = []
        logic_type = constraints_info["logic_type"]
        constraints = constraints_info["constraints"]

        print(f"🔧 生成代码，逻辑类型: {logic_type}, 约束数量: {len(constraints)}")

        # 处理OR逻辑的情况
        if logic_type == "or" and constraints:
            result_list_code = "result_list=[]\n"

            for constraint in constraints:
                constraint_code = self._generate_single_constraint_code(constraint, basic_info)
                if constraint_code:
                    result_list_code += constraint_code + "\nresult_list.append(result)\n"

            result_list_code += "result=False\nfor r in result_list:\n  result=result or r"
            code_blocks.append(result_list_code)

        # 处理AND逻辑的情况
        else:
            for constraint in constraints:
                constraint_code = self._generate_single_constraint_code(constraint, basic_info)
                if constraint_code:
                    code_blocks.append(constraint_code)
            code_blocks.extend(self._generate_basic_validation_code(basic_info))
        # 添加基础验证代码（天数、人数、票务等）
        # code_blocks.extend(self._generate_basic_validation_code(basic_info))

        return code_blocks

    def _generate_single_constraint_code(self, constraint: Dict, basic_info: Dict) -> str:
        """生成单个约束的Python代码 - 修复版本"""
        constraint_type = constraint["type"]

        print(f"🔧 生成约束代码: {constraint_type}")

        # 1. 景点约束 - 修复：正确区分名称和类型
        if constraint_type == "attraction_unwanted":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"attraction_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\n"
                f"result=not({names_set}&attraction_name_set)"
            )

        elif constraint_type == "attraction_wanted":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"attraction_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\n"
                f"result=({names_set}<=attraction_name_set)"
            )

        elif constraint_type == "attraction_free_only":
            return (
                f"attraction_cost=0\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='attraction': attraction_cost+=activity_cost(activity)\n"
                f"result=attraction_cost<=0"
            )

        elif constraint_type == "attraction_type_unwanted":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"attraction_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='attraction': attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
                f"result=not({types_set}&attraction_type_set)"
            )
        #@@@@
        elif constraint_type == "attraction_type_wanted":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"attraction_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='attraction': attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
                f"result=({types_set}<=attraction_type_set)"
            )

        # 2. 餐厅约束
        elif constraint_type == "restaurant_name_unwanted":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"restaurant_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: restaurant_name_set.add(activity_position(activity))\n"
                f"result=not({names_set}&restaurant_name_set)"
            )

        elif constraint_type == "restaurant_name_wanted":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"restaurant_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: restaurant_name_set.add(activity_position(activity))\n"
                f"result=({names_set}<=restaurant_name_set)"
            )

        elif constraint_type == "restaurant_wanted_one":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"restaurant_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: restaurant_name_set.add(activity_position(activity))\n"
                f"result=({names_set}&restaurant_name_set)"
            )

        elif constraint_type == "restaurant_type_unwanted":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"restaurant_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\n"
                f"result=not({types_set}&restaurant_type_set)"
            )

        elif constraint_type == "restaurant_type_wanted":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"restaurant_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\n"
                f"result=({types_set}<=restaurant_type_set)"
            )

        elif constraint_type == "restaurant_type_wanted_one":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"restaurant_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\n"
                f"result=({types_set}&restaurant_type_set)"
            )

        # 3. 住宿约束
        elif constraint_type == "accommodation_unwanted":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"accommodation_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_name_set.add(activity_position(activity))\n"
                f"result=not({names_set}&accommodation_name_set)"
            )

        elif constraint_type == "accommodation_wanted":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"accommodation_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_name_set.add(activity_position(activity))\n"
                f"result=({names_set}<=accommodation_name_set)"
            )

        elif constraint_type == "accommodation_wanted_one":
            names_set = "{" + ", ".join([f'"{name}"' for name in constraint["names"]]) + "}"
            return (
                f"accommodation_name_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_name_set.add(activity_position(activity))\n"
                f"result=({names_set}&accommodation_name_set)"
            )

        elif constraint_type == "accommodation_type_unwanted":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"accommodation_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_type_set.add(accommodation_type(activity, target_city(plan)))\n"
                f"result=not({types_set}&accommodation_type_set)"
            )

        elif constraint_type == "accommodation_type_wanted":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"accommodation_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_type_set.add(accommodation_type(activity, target_city(plan)))\n"
                f"result=({types_set}<=accommodation_type_set)"
            )

        elif constraint_type == "accommodation_type_wanted_one":
            types_set = "{" + ", ".join([f'"{t}"' for t in constraint["types"]]) + "}"
            return (
                f"accommodation_type_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_type_set.add(accommodation_type(activity, target_city(plan)))\n"
                f"result=({types_set}&accommodation_type_set)"
            )

        elif constraint_type == "room_type":
            room_code = constraint["room_type"]
            return (
                f"result=True\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation' and room_type(activity)!={room_code}: result=False"
            )

        elif constraint_type == "accommodation_location":
            poi = constraint["poi"]
            distance = constraint["distance"]
            return (
                f"result=False\n"
                f"accommodation_position=''\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_position=activity_position(activity)\n"
                f"result=(poi_distance(target_city(plan), '{poi}', accommodation_position)<={distance})"
            )

        # 4. 预算约束
        elif constraint_type == "budget_meal":
            amount = constraint["amount"]
            return (
                f"restaurant_cost=0\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']: restaurant_cost+=activity_cost(activity)\n"
                f"result=restaurant_cost<={amount}"
            )

        elif constraint_type == "budget_accommodation":
            amount = constraint["amount"]
            return (
                f"accommodation_cost=0\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='accommodation': accommodation_cost+=activity_cost(activity)\n"
                f"result=accommodation_cost<={amount}"
            )

        elif constraint_type == "budget_inner_transport":
            amount = constraint["amount"]
            return (
                f"inner_city_transportation_cost=0\n"
                f"for activity in allactivities(plan):\n"
                f"  inner_city_transportation_cost+=innercity_transport_cost(activity_transports(activity))\n"
                f"result=(inner_city_transportation_cost<={amount})"
            )

        elif constraint_type == "budget_inter_transport":
            amount = constraint["amount"]
            return (
                f"inter_city_transportation_cost=0\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity) in ['airplane','train']: inter_city_transportation_cost+=activity_cost(activity)\n"
                f"result=inter_city_transportation_cost<={amount}"
            )

        elif constraint_type == "budget_attraction":
            amount = constraint["amount"]
            return (
                f"attraction_cost=0\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='attraction': attraction_cost+=activity_cost(activity)\n"
                f"result=attraction_cost<={amount}"
            )

        elif constraint_type == "budget_total":
            amount = constraint["amount"]
            return (
                f"total_cost=0\n"
                f"for activity in allactivities(plan):\n"
                f"  total_cost+=activity_cost(activity)\n"
                f"  total_cost+=innercity_transport_cost(activity_transports(activity))\n"
                f"result=(total_cost<={amount})"
            )

        # 5. 时间约束 - 修复版本
        elif constraint_type == "time_window":
            location = constraint["location"]
            start_time = constraint["start_time"]
            end_time = constraint["end_time"]
            return (
                f"result=False\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_position(activity)=='{location}':\n"
                f"    if activity_start_time(activity)<='{start_time}' and activity_end_time(activity)>='{end_time}':\n"
                f"      result=True"
            )

        elif constraint_type == "arrive_no_later":
            location = constraint["location"]
            time = constraint["time"]
            return (
                f"result=False\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_position(activity)=='{location}':\n"
                f"    if activity_start_time(activity)<='{time}':\n"
                f"      result=True"
            )

        elif constraint_type == "leave_no_earlier":
            location = constraint["location"]
            time = constraint["time"]
            return (
                f"result=False\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_position(activity)=='{location}':\n"
                f"    if activity_end_time(activity)>='{time}':\n"
                f"      result=True"
            )

        elif constraint_type == "min_duration":
            location = constraint["location"]
            minutes = constraint["minutes"]
            return (
                f"result=False\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_position(activity)=='{location}':\n"
                f"    if activity_time(activity)>={minutes}:\n"
                f"      result=True"
            )

        elif constraint_type == "visit_sequence":
            location_a = constraint["location_a"]
            location_b = constraint["location_b"]
            return (
                f"result=False\n"
                f"idx_activity0=0\n"
                f"idx_activity1=0\n"
                f"i=0\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_position(activity)=='{location_a}':\n"
                f"    idx_activity0=i\n"
                f"  if activity_position(activity)=='{location_b}':\n"
                f"    idx_activity1=i\n"
                f"  i+=1\n"
                f"if idx_activity0<idx_activity1:\n"
                f"  result=True"
            )

        # 6. 出行约束
        elif constraint_type == "transport_mode_wanted":
            modes_set = "{" + ", ".join([f'"{mode}"' for mode in constraint["modes"]]) + "}"
            return (
                f"innercity_transport_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_transports(activity)!=[]: innercity_transport_set.add(innercity_transport_type(activity_transports(activity)))\n"
                f"result=({modes_set}<=innercity_transport_set)"
            )

        elif constraint_type == "transport_mode_unwanted":
            modes_set = "{" + ", ".join([f'"{mode}"' for mode in constraint["modes"]]) + "}"
            return (
                f"inner_city_transportation_set=set()\n"
                f"for activity in allactivities(plan):\n"
                f"  if activity_type(activity)=='transportation': inner_city_transportation_set.add(activity_position(activity))\n"
                f"result=not({modes_set}&inner_city_transportation_set)"
            )


        elif constraint_type == "intercity_transport_unwanted":
            go_mode = constraint["go_mode"]
            back_mode = constraint["back_mode"]
            return (
                f"result=False\n"
                f"intercity_transport_go=''\n"
                f"intercity_transport_back=''\n"
                f"if allactivities(plan)[0]['type'] != \"{go_mode}\" and intercity_transport_origin(allactivities(plan)[0])==start_city(plan) and allactivities(plan)[-1]['type'] != \"{back_mode}\" and intercity_transport_origin(allactivities(plan)[-1])==target_city(plan):\n"
                f"  result=True"
            )

        elif constraint_type == "intercity_transport_wanted":
            go_mode = constraint["go_mode"]
            back_mode = constraint["back_mode"]
            return (
                f"result=False\n"
                f"intercity_transport_go=''\n"
                f"intercity_transport_back=''\n"
                f"if allactivities(plan)[0]['type'] == \"{go_mode}\" and intercity_transport_origin(allactivities(plan)[0])==start_city(plan) and allactivities(plan)[-1]['type'] == \"{back_mode}\" and intercity_transport_origin(allactivities(plan)[-1])==target_city(plan):\n"
                f"  result=True"
            )

        elif constraint_type == "taxi_over_distance":
            distance = constraint["distance"]
            return (
                f"result=True\n"
                f"for activity in allactivities(plan):\n"
                f"  if innercity_transport_type(activity_transports(activity)) != 'taxi' and innercity_transport_distance(activity_transports(activity)) > {distance}:\n"
                f"    result=False\n"
                f"    break"
            )

        print(f"❌ 未知约束类型: {constraint_type}")
        return ""

    def _generate_basic_validation_code(self, basic_info: Dict) -> List[str]:
        """生成基础验证代码 - 与TPC Agent兼容"""
        people_number = basic_info["people_number"]
        days = basic_info["days"]

        code_blocks = [
            f"result=(day_count(plan)=={days})",
            f"result=(people_count(plan)=={people_number})",
            f"result=True\n"
            f"for activity in allactivities(plan):\n"
            f"  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!={people_number}: result=False\n"
            f"  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!={people_number}: result=False",
            f"result=True\n"
            f"for activity in allactivities(plan):\n"
            f"  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False"
        ]

        return code_blocks

    def _get_response(self, messages, one_line=True, json_mode=False):
        """实现AbstractLLM的抽象方法 - 简化版本"""
        # 这里可以简单实现或者抛出异常，因为TPC Agent主要使用nl_to_python_direct
        if json_mode:
            return '{"hard_logic_py": []}'
        else:
            return "TPC-Algorithm response"


def test_natural_language_input():
    """测试自然语言输入功能"""


    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Qwen3-4B", "Qwen", "Qwen3-4B")
    llm = TPCLLM(model_path=model_path)


    # 初始化 TPCLLM，传入模型路径
    # model_path = "/home/yz/chinatravel/local_llm/Qwen3-4B/Qwen/Qwen3-4B"
    # llm = TPCLLM(model_path=model_path)

    # 测试用例：你提供的JSON格式数据
    print(f"\n{'=' * 60}")
    print("测试用例: JSON格式自然语言输入")
    print(f"{'=' * 60}")

    input_data ={
        "uid": "20250424183230548475",
        "start_city": "广州",
        "target_city": "成都",
        "people_number": 3,
        "days": 4,
        "nature_language": "我们一家三口广州坐高铁去四川成都玩四天，希望推荐几个著名好吃的 地方打卡，每顿饭人均不超过100元，酒店的话，首选干净整洁交通便利的如家，汉庭酒店即可，每晚房间价格预算400以内，带个小孩子，要大床房，市内交通不坐地铁，总预算五千."
    }
    print("输入数据:")
    print(json.dumps(input_data, ensure_ascii=False, indent=2))

    # 提取自然语言和基础信息
    natural_language = input_data["nature_language"]
    start_city = input_data["start_city"]
    target_city = input_data["target_city"]
    people_number = input_data["people_number"]
    days = input_data["days"]

    print(f"\n📝 提取的自然语言: {natural_language}")
    print(f"📍 出发城市: {start_city}")
    print(f"🎯 目标城市: {target_city}")
    print(f"👥 人数: {people_number}")
    print(f"📅 天数: {days}")

    # 使用自然语言调用 nl_to_python_direct
    result = llm.nl_to_python_direct(
        natural_language=natural_language,
        days=days,
        people_number=people_number,
        start_city=start_city,
        target_city=target_city
    )

    print(f"\n📋 生成的完整结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"\n🐍 生成的Python代码 ({len(result['hard_logic_py'])} 个代码块):")
    for j, code in enumerate(result["hard_logic_py"], 1):
        print(f"\n{j}. {code}")

if __name__ == "__main__":
    test_natural_language_input()