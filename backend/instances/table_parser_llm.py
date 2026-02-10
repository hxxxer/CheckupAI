import json
import tomllib
from typing import Any, Union
from vllm import LLM, SamplingParams
from backend.config import settings


class TableParserLLM:
    def __init__(self, model_path: str, prompt_path: str):
        """
        初始化LLM处理器
        :param model_path: 模型路径
        :param prompt_path: Prompt配置文件路径
        """
        self.model_path = model_path
        self.prompt_path = prompt_path
        self._llm = None
        self._tokenizer = None
        self._prompt = None

    def _load_model(self) -> Any:
        """加载LLM模型"""
        if self._llm is None:
            self._llm = LLM(
                model=self.model_path,
                dtype="float16",
                quantization="awq",
                # gpu_memory_utilization=0.8,
                max_model_len=16384,
                enforce_eager=False,
                trust_remote_code=True,
            )
        return self._llm

    def _load_tokenizer(self) -> Any:
        """加载Tokenizer"""
        if self._tokenizer is None:
            llm = self._load_model()
            self._tokenizer = llm.get_tokenizer()
        return self._tokenizer

    def _build_prompt(self) -> str:
        """构建Prompt"""
        if self._prompt is None:
            with open(self.prompt_path, "rb") as f:
                raw_prompt = tomllib.load(f)

            prompt = raw_prompt.get("prompt", {}).get("role", "") + "\n\n"

            if 'rules' in raw_prompt:
                prompt += "规则：\n"
                for key, rule in raw_prompt["rules"].items():
                    prompt += f"{key}：{rule}\n"
                prompt += "\n"

            if 'output' in raw_prompt:
                prompt += "输出格式：\n"
                prompt += raw_prompt.get("title", "") + "\n"
                prompt += raw_prompt.get("discription", "") + "\n\n"

            if 'examples' in raw_prompt:
                prompt += "示例：\n"
                for key, example in raw_prompt["examples"].items():
                    prompt += f"例子{key} - {example.get('title', '')}\n"
                    prompt += "输入：\n"
                    prompt += example.get("input", "") + "\n"
                    prompt += "输出：\n"
                    prompt += example.get("output", "") + "\n\n"

            prompt += raw_prompt.get("task", {}).get("instruction", "")
            self._prompt = prompt

        return self._prompt

    def generate(self, table_md: str) -> Union[dict, list, None]:
        """
        使用LLM生成结果并解析为JSON
        :param table_md: Markdown格式的表格数据
        :return: 解析后的JSON结果
        """
        # 加载模型和Tokenizer
        tokenizer = self._load_tokenizer()

        # 构建Prompt
        prompt = self._build_prompt()

        # 构造消息
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": table_md}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # 推理参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            max_tokens=8192,
            stop=["<|im_end|>", "<|endoftext|>"]  # 设置停止词
        )

        # 执行推理
        llm = self._load_model()
        outputs = llm.generate([text], sampling_params)
        content = outputs[0].outputs[0].text

        # print("提取结果:", content)

        # 安全解析JSON
        return self.safe_json_parse(content)

    @staticmethod
    def safe_json_parse(text: str) -> Union[dict, list, None]:
        """
        安全地解析JSON，包含多种清理策略
        """
        if not text:
            return None

        # 清理文本
        cleaned_text = text.strip()

        # 移除常见的前缀/后缀
        prefixes_to_remove = ['```json', '```', 'json']
        suffixes_to_remove = ['```']

        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break

        for suffix in suffixes_to_remove:
            if cleaned_text.endswith(suffix):
                cleaned_text = cleaned_text[:-len(suffix)].strip()
                break

        # 尝试直接解析
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # 尝试提取数组部分
        array_matches = re.findall(r'\[[\s\S]*?\]', cleaned_text)
        if array_matches:
            try:
                return json.loads(array_matches[-1])  # 使用最后一个匹配的数组
            except json.JSONDecodeError:
                pass

        # 尝试提取对象部分
        object_matches = re.findall(r'\{[\s\S]*?\}', cleaned_text)
        if object_matches:
            try:
                return json.loads(object_matches[-1])
            except json.JSONDecodeError:
                pass

        print(f"无法解析JSON: {text[:100]}...")
        return None


table_parser = TableParserLLM(
    model_path=settings.llm_table_model_path,
    prompt_path=settings.llm_table_prompt
)
