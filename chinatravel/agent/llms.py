# 开发自己的本地大型语言模型（LLM），
# 您需要继承 chinatravel/agent/llms.py 的 AbstractLLM 类，
# 并在 llms.py 中添加相应的本地 LLM 推理代码。
# 加完之后向init_agent函数添加代码：
# 打开 chinatravel/agent/load_model.py 文件，
# 并在 init_llm 函数中添加对新 llm 的支持。

from abc import ABC, abstractmethod
from openai import OpenAI
from json_repair import repair_json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import tiktoken
import torch
import re
import sys
import os


def chat_template(messages):
    """
    将 messages 列表转成符合 Chat 模板格式的字符串
    用于 tiktoken.encode 计算 token 数。
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|{role}|>\n{content}\n"
    formatted += "<|assistant|>\n"  # 留空表示用户希望 assistant 继续回复
    return formatted

def merge_repeated_role(messages):
    ptr = len(messages) - 1
    last_role = ""
    while ptr >= 0:
        cur_role = messages[ptr]["role"]
        if cur_role == last_role:
            messages[ptr]["content"] += "\n" + messages[ptr + 1]["content"]
            del messages[ptr + 1]
        last_role = cur_role
        ptr -= 1
    return messages


class AbstractLLM(ABC):
    class ModeError(Exception):
        pass

    def __init__(self):
        self.input_token_count = 0
        self.output_token_count = 0
        self.input_token_maxx = 0
        pass

    def __call__(self, messages, one_line=True, json_mode=False):
        if one_line and json_mode:
            raise self.ModeError(
                "one_line and json_mode cannot be True at the same time"
            )
        return self._get_response(messages, one_line, json_mode)

    @abstractmethod
    def _get_response(self, messages, one_line, json_mode):
        pass


class Deepseek(AbstractLLM):
    def __init__(self):
        super().__init__()
        self.llm = OpenAI(
            base_url="https://api.deepseek.com",
        )
        self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", "deepseek_v3_tokenizer"
        )
        self.name = "DeepSeek-V3"

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

    def _send_request(self, messages, kwargs):

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = self.tokenizer(text)["input_ids"]

        self.input_token_count += len(input_tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

        res_str = (
            self.llm.chat.completions.create(messages=messages, **kwargs)
            .choices[0]
            .message.content
        )
        output_tokens = self.tokenizer(res_str)["input_ids"]
        self.output_token_count += len(output_tokens)

        res_str = res_str.strip()
        return res_str

    def _get_response(self, messages, one_line, json_mode):
        kwargs = {
            "model": "deepseek-chat",
            "max_tokens": 4096,
            "temperature": 0,
            "top_p": 0.00000001,
        }
        if one_line:
            kwargs["stop"] = ["\n"]
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = repair_json(res_str, ensure_ascii=False)
        except Exception as e:
            print(e)
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class GLM4Plus(AbstractLLM):
    def __init__(self):
        super().__init__()
        self.llm = OpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        self.name = "GLM4Plus"

    def _send_request(self, messages, kwargs):
        res_str = (
            self.llm.chat.completions.create(messages=messages, **kwargs)
            .choices[0]
            .message.content
        )
        res_str = res_str.strip()
        return res_str

    def _get_response(self, messages, one_line, json_mode):
        kwargs = {
            "model": "glm-4-plus",
            "max_tokens": 4095,
            "temperature": 0,
            "top_p": 0.01,
        }
        if one_line:
            kwargs["stop"] = ["<STOP>"]
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = repair_json(res_str, ensure_ascii=False)
        except Exception as e:
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class GPT4o(AbstractLLM):
    def __init__(self):
        super().__init__()
        self.llm = OpenAI()
        self.name = "GPT4o"
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _send_request(self, messages, kwargs):

        # print(messages)
        tokens = self.tokenizer.encode(chat_template(messages))
        self.input_token_count += len(tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(tokens))

        # print(tokens)
        # print(self.input_token_count)
        # exit(0)

        res_str = (
            self.llm.chat.completions.create(messages=messages, **kwargs)
            .choices[0]
            .message.content
        )

        tokens = self.tokenizer.encode(res_str)
        self.output_token_count += len(tokens)

        res_str = res_str.strip()
        return res_str

    def _get_response(self, messages, one_line, json_mode):
        kwargs = {
            "model": "chatgpt-4o-latest",
            "max_tokens": 4095,
            "temperature": 0,
            "top_p": 0.01,
        }
        if one_line:
            kwargs["stop"] = ["\n"]
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = repair_json(res_str, ensure_ascii=False)
        except Exception as e:
            print(e)
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class Qwen(AbstractLLM):
    def __init__(self, model_name, max_model_len=None):
        super().__init__()
        self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", model_name
        )

        print(f"Loading {model_name} with ultimate solution...")

        # 终极方案：手动修复配置并直接加载
        import json
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # 1. 读取并修复配置文件
        config_path = os.path.join(self.path, "config.json")
        print(f"Reading config from: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 强制设置正确的模型类型
        config_data["model_type"] = "qwen2"
        config_data["auto_map"] = {
            "AutoModelForCausalLM": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM"
        }

        # 保存修复后的配置
        fixed_config_path = os.path.join(self.path, "config_fixed.json")
        with open(fixed_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        print("Config fixed and saved")

        # 2. 直接加载 tokenizer，不依赖配置检测
        print("Loading tokenizer directly...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            trust_remote_code=True,
            local_files_only=True
        )

        # 3. 直接加载模型，强制信任远程代码
        print("Loading model directly...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # 确保 tokenizer 设置正确
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 设置生成参数
        self.max_model_len = max_model_len or 32768
        self.name = model_name

        if "Qwen3" in model_name:
            self.generation_config = {
                "max_new_tokens": 4096,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
        else:
            self.generation_config = {
                "max_new_tokens": 4096,
                "temperature": 0,
                "top_p": 0.001,
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id
            }

        print(f"Model {model_name} loaded successfully!")

    def _get_response(self, messages, one_line, json_mode):
        try:
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            input_tokens = inputs["input_ids"][0]

            self.input_token_count += len(input_tokens)
            self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

            if len(input_tokens) >= self.max_model_len:
                return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )

            generated_tokens = outputs[0][len(input_tokens):]
            res_str = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            self.output_token_count += len(generated_tokens)

            if "Qwen3" in self.name:
                try:
                    m = re.match(r"<think>\n(.+)</think>\n\n", res_str, flags=re.DOTALL)
                    if m:
                        thinking_content = m.group(1).strip()
                        res_str = res_str[len(m.group(0)):].strip()
                except Exception:
                    pass

            if json_mode:
                res_str = repair_json(res_str, ensure_ascii=False)
            elif one_line:
                res_str = res_str.split("\n")[0]

            return res_str

        except Exception as e:
            print(f"Qwen inference error: {e}")
            import traceback
            traceback.print_exc()
            return '{"error": "Request failed, please try again."}'



class Mistral(AbstractLLM):
    def __init__(self, max_model_len=None):
        super().__init__()
        self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", "Mistral-7B-Instruct-v0.3",
        )

        # 使用 transformers 替代 vLLM
        print("Loading Mistral with transformers...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.name = "Mistral-7B-Instruct-v0.3"
        self.max_model_len = max_model_len or 32768

    def _get_response(self, messages, one_line, json_mode):
        messages = merge_repeated_role(messages)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        input_tokens = inputs["input_ids"][0]

        self.input_token_count += len(input_tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

        if len(input_tokens) >= self.max_model_len:
            return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0,
                top_p=0.001,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][len(input_tokens):]
        res_str = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        self.output_token_count += len(generated_tokens)

        if json_mode:
            res_str = repair_json(res_str, ensure_ascii=False)
        elif one_line:
            res_str = res_str.split("\n")[0]

        return res_str


class Llama(AbstractLLM):
    def __init__(self, model_name):
        super().__init__()

        Llama_supported = ["Llama3-3B", "Llama3-8B"]
        if model_name not in Llama_supported:
            raise ValueError(f"Unsupported model name: {model_name}. Supported models: {Llama_supported}")

        if model_name == "Llama3-3B":
            self.path = os.path.join(
                project_root_path, "chinatravel", "local_llm", "Llama-3.2-3B-Instruct"
            )
        elif model_name == "Llama3-8B":
            self.path = os.path.join(
                project_root_path, "chinatravel", "local_llm", "Meta-Llama-3.1-8B-Instruct"
            )

        # 使用 transformers 替代 vLLM
        print(f"Loading {model_name} with transformers...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.name = model_name

    def _get_response(self, messages, one_line, json_mode):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        input_tokens = inputs["input_ids"][0]

        self.input_token_count += len(input_tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

        if len(input_tokens) >= 131072:
            return '{"error": "Input prompt is longer than 131072 tokens."}'

        try:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0,
                    top_p=0.001,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][len(input_tokens):]
            res_str = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            self.output_token_count += len(generated_tokens)

            if json_mode:
                res_str = repair_json(res_str, ensure_ascii=False)
            elif one_line:
                res_str = res_str.split("\n")[0]
        except Exception as e:
            print(f"Llama inference error: {e}")
            res_str = '{"error": "Request failed, please try again."}'

        return res_str


class EmptyLLM(AbstractLLM):
    def __init__(self):
        super().__init__()
        self.name = "EmptyLLM"

    def _get_response(self, messages, one_line, json_mode):
        return "Empty LLM response"


if __name__ == "__main__":
    # model = Mistral()
    model = GPT4o()
    print(model([{"role": "user", "content": "hello!"}], one_line=False))

