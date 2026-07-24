"""
SimpleTravelConverter - Natural Language to Structured JSON Parser.

Parses natural language travel queries into structured JSON format
using a local Qwen language model.

Author: YZ1553927
License: MIT
"""

import json
import re
import time
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleTravelConverter:
    def __init__(self, model_path: str):
        """
        简化版旅行需求转换器 - 只提取核心信息

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._check_environment()
        self._load_model_optimized()

        # 简化输出格式 - 只保留核心字段
        self.output_template = {
            # 基础信息
            "start_city": "",
            "target_city": "",
            "travel_days": "",
            "people_count": "",

            # 预算信息
            "budget_total": "",

            # 交通偏好
            "intercity_transport_go": [],  # train/airplane
            "intercity_transport_back": [],  # train/airplane
            "transport_mode_unwanted": [],  # walk/taxi/metro

            # 住宿偏好
            "accommodation_wanted": [],
            "room_type": "",  # 单床房/双床房

            # 景点偏好
            "attraction_wanted": [],

            # 餐饮偏好
            "restaurant_wanted": [],
            "food_preference": [],
            "dietary_restrictions": [],
            "cuisiine_types": [],

            # 其他重要需求
            "special_requirements": []
        }

    def _check_environment(self):
        """检查运行环境"""
        print("=== 环境诊断 ===")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
        else:
            print("警告: 在CPU上运行，速度会较慢")
        print("================")

    def _load_model_optimized(self):
        """优化加载模型和tokenizer"""
        try:
            print("正在优化加载模型...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='left'
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            self.model.eval()
            print("模型优化加载完成!")

        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def _create_simple_prompt(self, user_input: str) -> str:
        """创建简化提示词 - 只要求核心信息"""
        prompt = f"""请从用户的旅行需求中提取信息，只输出用户明确提到的约束字段：

用户输入："{user_input}"

请只提取以下核心信息，未提及的字段保持为空：
{{
  "请输出以下JSON格式，只包含提及的信息，未提及的字段保持为空：
{{
  请只提取以下核心信息，未提及的字段保持为空：
{{
  "start_city": "出发城市",
  "target_city": "目标城市", 
  "travel_days": "旅行天数",
  "people_count": "人数",
  "budget_total": "总预算金额"（人均多少乘以总人数的金额）,
  "intercity_transport": ["城市来回的交通方式(train/airplane)"],
  "transport_mode_unwanted": ["不希望的市内交通方式(walk/taxi/metro)"],
  "transport_mode_wanted": ["希望的市内交通方式(walk/taxi/metro)"],
  "budget_accommodation":住宿预算金额（每晚预算乘以天数等于的金额）,
  "budget_meal":用餐预算金额,
  "budget_attraction":游玩景点预算金额,
  "budget_inter_transport":跨城市交通预算金额，
  "budget_inner_transport":在城市内出行的预算金额，
  "accommodation_wanted": ["希望住的酒店名称"],
  "accommodation_type_wanted": ["希望住的酒店类型，如停车场,健身室,家庭房,免费停车,山景房等"],
  "room_type": "房型(单床房/双床房/大床房/标间/双人房)",
  "attraction_wanted": ["希望游览的景点名称"],
  "attraction_type_wanted": ["想去游玩的地方类型，如历史古迹，公园，自然风光，博物馆/纪念馆等"],
  "restaurant_wanted": ["想吃的餐厅名称，如餐厅名称，什么地点附近的餐厅，如浣花北路附近的国土六扇门"],
  "food_preferences": ["食物偏好，如:当地特色美食,小吃,火锅,海鲜,烧烤等"],
  "dietary_restrictions": ["饮食限制，如:吃辣,微辣,不吃辣,素食,不吃猪肉,过敏食物等"],
  "cuisine_types": ["喜欢的菜系，如:川菜,粤菜,湘菜,江浙菜,徽菜,融合菜等"],
  "special_requirements": ["其他特殊要求,比如说了想去免费不要钱的景点"]
}}
}}

餐饮信息提取指南：
- 如果用户说"想吃当地小吃"，则"food_preferences"中加入"当地小吃"
- 如果用户说"可以吃辣"，则"food_preferences"中加入"吃辣"  
- 如果用户说“吃不太辣”，则“food_preference"中加入"微辣"
- 如果用户说"不吃辣"，则"dietary_restrictions"中加入"不吃辣"
- 如果用户说"喜欢川菜"，则"cuisine_types"中加入"川菜"
- 如果用户提到具体餐厅，如"全聚德"，则"restaurant_wanted"中加入"全聚德"

示例1:
输入: "我们想去成都玩，想吃当地小吃和火锅，可以吃辣"
输出: {{
  "food_preferences": ["当地小吃", "火锅", "吃辣"],
  "cuisine_types": ["川菜"]
}}

示例2:
输入: "我们不吃辣，想尝试上海的本帮菜"
输出: {{
  "dietary_restrictions": ["不吃辣"],
  "cuisine_types": ["本帮菜"]
}}


要求：
1. 只输出JSON，不要其他内容，只输出JSON，不要其他内容，只输出JSON，不要其他内容
2. 有train/airplane就提取出来，没有的话就不要提取
3. 数组字段使用字符串数组
4. 数字字段只填数字
5. 住宿预算是每晚预算乘以旅游的天数，住宿预算是每晚预算乘以旅游的天数，住宿预算是每晚预算乘以旅游的天数
6. 没有的话就不要提取，只提取明确提及的信息,只提取明确提及的信息，只提取明确提及的信息
7. 没有提到往返交通跟城内交通就不要提取往返交通跟城内交通，提到了再提取，不要多提取
8. 如果只有总预算的话，不需要你帮他分配各个领域的预算，只有总预算的话只需要提取总预算就行了，有什么预算就提取什么预算，没提到的预算不要提取
9. 如果只有总预算的话，不需要你帮他分配各个领域的预算，只有总预算的话只需要提取总预算就行了，有什么预算就提取什么预算，没提到的预算不要提取
10. 如果只有总预算的话，不需要你帮他分配各个领域的预算，只有总预算的话只需要提取总预算就行了，有什么预算就提取什么预算，没提到的预算不要提取
现在请输出JSON："""
        return prompt

    def _extract_json_fast(self, response: str) -> Dict[str, Any]:
        """快速提取JSON"""
        try:
            # 找到第一个{
            start_idx = response.find('{')
            if start_idx == -1:
                return self.output_template.copy()

            # 找到第一个完整的JSON对象
            stack = []
            end_idx = -1

            for i in range(start_idx, len(response)):
                char = response[i]
                if char == '{':
                    stack.append('{')
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack:
                            end_idx = i
                            break

            if end_idx == -1:
                return self.output_template.copy()

            json_str = response[start_idx:end_idx + 1]

            # 清理格式问题
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)

            result = json.loads(json_str)

            # 确保所有字段都存在
            for key in self.output_template:
                if key not in result:
                    result[key] = self.output_template[key]

            return result

        except json.JSONDecodeError:
            print("JSON解析失败，返回空模板")
            return self.output_template.copy()

    def convert_fast(self, user_input: str) -> Dict[str, Any]:
        """
        快速转换版本 - 专注于核心信息

        Args:
            user_input: 用户输入的自然语言

        Returns:
            结构化的旅行需求字典
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未正确加载")

        total_start = time.time()
        prompt = self._create_simple_prompt(user_input)

        try:
            # 编码输入 - 使用更短的长度
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1400,  # 减少输入长度
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # 极速生成参数 - 大幅减少生成token
            generate_start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1300,  # 大幅减少！
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            generate_time = time.time() - generate_start
            print(f"生成时间: {generate_time:.2f}秒")

            # 解码响应
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()

            print(f"生成内容预览: {generated_text[:100]}...")

            # 提取JSON
            result = self._extract_json_fast(generated_text)

            total_time = time.time() - total_start
            print(f"总耗时: {total_time:.2f}秒")

            return result

        except Exception as e:
            print(f"转换过程中出错: {e}")
            return self.output_template.copy()

    def convert_with_fallback(self, user_input: str, max_retries: int = 1) -> Dict[str, Any]:
        """
        带重试机制的转换方法

        Args:
            user_input: 用户输入
            max_retries: 最大重试次数

        Returns:
            结构化的旅行需求
        """
        for attempt in range(max_retries):
            try:
                print(f"第{attempt + 1}次尝试...")
                result = self.convert_fast(user_input)

                if self._has_valid_content(result):
                    return result
                else:
                    print(f"第{attempt + 1}次尝试结果为空")
            except Exception as e:
                print(f"第{attempt + 1}次尝试失败: {e}")

        print("返回模板")
        return self.output_template.copy()

    def _has_valid_content(self, result: Dict[str, Any]) -> bool:
        """验证结果是否有实际内容"""
        if not isinstance(result, dict):
            return False

        # 检查是否有非空字段
        for key, value in result.items():
            if value and value != self.output_template[key]:
                return True

        return False

    def save_result(self, result: Dict[str, Any], filepath: str):
        """保存结果到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def print_result(self, result: Dict[str, Any]):
        """美化打印结果"""
        print("=" * 50)
        print("旅行需求分析结果:")
        print("=" * 50)

        has_content = False

        # 基础信息
        basic_info = []
        for field in ['start_city', 'target_city', 'travel_days', 'people_count']:
            if result.get(field):
                basic_info.append(f"  \"{field}\": \"{result[field]}\"")

        if basic_info:
            has_content = True
            for line in basic_info:
                print(line)

        # 预算信息
        if result.get('budget_total'):
            has_content = True
            print(f"  \"在旅行上的总预算\": \"{result['budget_total']}元\"")

        if result.get('budget_accommodation'):
            has_content = True
            print(f"  \"在住宿上的预算\": \"{result['budget_accommodation']}元\"")

        if result.get('budget_inter_transport'):
            has_content = True
            print(f"  \"跨城市交通的预算\": \"{result['budget_inter_transport']}元\"")

        if result.get('budget_inner_transport'):
            has_content = True
            print(f"  \"在城市内出行的预算\": \"{result['budget_inner_transport']}元\"")

        if result.get('budget_meal'):
            has_content = True
            print(f"  \"在用餐上预算\": \"{result['budget_meal']}元\"")

        if result.get('budget_attraction'):
            has_content = True
            print(f"  \"在游览上的预算\": \"{result['budget_attraction']}元\"")

        # 交通偏好
        transport_info = []

        # 首先检查是否有统一的往返交通
        if result.get('intercity_transport'):
            transport_info.append(f"  \"往返交通\": {result['intercity_transport']}")
        # else:
        #     # 如果没有统一往返交通，再分别检查去程和返程
        #     has_go = result.get('intercity_transport_go')
        #     has_back = result.get('intercity_transport_back')
        #
        #     if has_go and has_back:
        #         go_transport = result['intercity_transport_go']
        #         back_transport = result['intercity_transport_back']
        #
        #         if go_transport == back_transport:
        #             transport_info.append(f"  \"往返交通\": {go_transport}")
        #         else:
        #             transport_info.append(f"  \"去程交通\": {go_transport}")
        #             transport_info.append(f"  \"返程交通\": {back_transport}")
        #     elif has_go and not has_back:
        #         transport_info.append(f"  \"去程交通\": {result['intercity_transport_go']}")
        #     elif has_back and not has_go:
        #         transport_info.append(f"  \"返程交通\": {result['intercity_transport_back']}")


        if result.get('transport_mode_wanted'):
            transport_info.append(f"  \"城内交通\": {result['transport_mode_wanted']}")
        if result.get('transport_mode_unwanted'):
            transport_info.append(f"  \"避免的城内交通\": {result['transport_mode_unwanted']}")

        if transport_info:
            has_content = True
            for line in transport_info:
                print(line)

        # 住宿偏好
        accommodation_info = []
        if result.get('accommodation_wanted'):
            accommodation_info.append(f"  \"酒店偏好\": {result['accommodation_wanted']}")
        if result.get('accommodation_type_wanted'):
            accommodation_info.append(f"  \"酒店类型偏好\": {result['accommodation_type_wanted']}")
        if result.get('room_type'):
            accommodation_info.append(f"  \"房型\": \"{result['room_type']}\"")

        if accommodation_info:
            has_content = True
            for line in accommodation_info:
                print(line)

        # 景点偏好
        if result.get('attraction_wanted'):
            has_content = True
            print(f"  \"想去的景点\": {result['attraction_wanted']}")

        if result.get('attraction_type_wanted'):
            has_content = True
            print(f"  \"想去的景点类型\": {result['attraction_type_wanted']}")

        # 餐厅偏好
        food_info = []
        if result.get('restaurant_wanted'):
            food_info.append(f"  \"餐厅偏好\": {result['restaurant_wanted']}")
        if result.get('food_preferences'):
            food_info.append(f"  \"食物偏好\": {result['food_preferences']}")
        if result.get('dietary_restrictions'):
            food_info.append(f"  \"饮食限制\": {result['dietary_restrictions']}")
        if result.get('cuisine_types'):
            food_info.append(f"  \"菜系偏好\": {result['cuisine_types']}")

        if food_info:
            has_content = True
            for line in food_info:
                print(line)

        # 特殊要求
        if result.get('special_requirements'):
            has_content = True
            print(f"\n【特殊偏好】")
            for req in result['special_requirements']:
                print(f"  - {req}")

        if not has_content:
            print("\n未提取到有效信息")

    def get_pretty_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """获取美化后的数据结构"""
        pretty_data = {}

        # 基础信息
        for field in ['start_city', 'target_city', 'travel_days', 'people_count']:
            if result.get(field):
                pretty_data[field] = result[field]

        # 预算信息 - 使用中文键名
        if result.get('budget_total'):
            pretty_data["在旅行上的总预算"] = f"{result['budget_total']}元"

        if result.get('budget_accommodation'):
            pretty_data["在住宿上的预算"] = f"{result['budget_accommodation']}元"

        if result.get('budget_inter_transport'):
            pretty_data["跨城市交通的预算"] = f"{result['budget_inter_transport']}元"

        if result.get('budget_inner_transport'):
            pretty_data["在城市内出行的预算"] = f"{result['budget_inner_transport']}元"

        if result.get('budget_meal'):
            pretty_data["在用餐上预算"] = f"{result['budget_meal']}元"

        if result.get('budget_attraction'):
            pretty_data["在游览上的预算"] = f"{result['budget_attraction']}元"

        # 交通偏好
        if result.get('intercity_transport'):
            pretty_data["往返交通"] = result['intercity_transport']

        if result.get('transport_mode_wanted'):
            pretty_data["城内交通"] = result['transport_mode_wanted']

        if result.get('transport_mode_unwanted'):
            pretty_data["避免的城内交通"] = result['transport_mode_unwanted']

        # 住宿偏好
        if result.get('room_type'):
            pretty_data["房型"] = result['room_type']

        if result.get('accommodation_wanted'):
            pretty_data["酒店偏好"] = result['accommodation_wanted']

        if result.get('accommodation_type_wanted'):
            pretty_data["酒店类型偏好"] = result['accommodation_type_wanted']

        # 景点偏好
        if result.get('attraction_wanted'):
            pretty_data["想去的景点"] = result['attraction_wanted']

        if result.get('attraction_type_wanted'):
            pretty_data["想去的景点类型"] = result['attraction_type_wanted']

        # 餐厅偏好
        if result.get('food_preferences'):
            pretty_data["食物偏好"] = result['food_preferences']

        if result.get('dietary_restrictions'):
            pretty_data["饮食限制"] = result['dietary_restrictions']

        if result.get('cuisine_types'):
            pretty_data["菜系偏好"] = result['cuisine_types']

        if result.get('restaurant_wanted'):
            pretty_data["餐厅偏好"] = result['restaurant_wanted']

        # 特殊要求
        if result.get('special_requirements'):
            pretty_data["特殊偏好"] = result['special_requirements']

        return pretty_data

# 使用示例
if __name__ == "__main__":
    # 初始化转换器
    converter = SimpleTravelConverter("/home/yz/chinatravel/local_llm/Qwen3-4B/Qwen/Qwen3-4B")

    # 测试输入
    test_inputs = [

        "我们5人从成都出发，前往上海，计划旅行5天。预算10000元。期望入住双床房（标间），选择高铁G1974次列车，07：17从上海虹桥出发，18：28到达成都东"

    ]

    # 执行转换
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{'=' * 60}")
        print(f"测试用例 {i}:")
        print(f"输入: {test_input}")
        print(f"{'=' * 60}")

        result = converter.convert_with_fallback(test_input)
        converter.print_result(result)

