import json
import os
import re
import tomllib
from typing import Any, Union

from openai import OpenAI


class TableParserLLM:
    def __init__(
        self,
        prompt_path,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-4B"
    ):
        """
        初始化客户端
        
        Args:
            base_url: vLLM API地址
            api_key: API密钥（vLLM默认为EMPTY）
            model: 模型名称
        """
        self.prompt_path = prompt_path
        self._prompt = None

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        
        # 采样参数配置
        self.sampling_params = {
            "temperature": 1.0,          # 温度
            "top_p": 0.95,               # 核采样
            "presence_penalty": 2.0,     # 存在惩罚
            "max_tokens": 32768,         # 最大生成长度
            "extra_body": {
                "top_k": 40,             # Top-K采样
            }
        }


    def _build_prompt(self) -> str:
        """构建Prompt"""
        if self._prompt is None:
            try:
                with open(self.prompt_path, "rb") as f:
                    raw_prompt = tomllib.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"提取表格Prompt文件缺失: {self.prompt_path}")
            except tomllib.TOMLDecodeError as e:
                raise ValueError(f"TOML文件格式不合法: {self.prompt_path}: {e}")

            prompt = raw_prompt.get("prompt", {}).get("role", "") + "\n\n"

            if 'rules' in raw_prompt:
                prompt += "规则：\n"
                for key, rule in raw_prompt["rules"].items():
                    prompt += f"{key}：{rule}\n"
                prompt += "\n"

            if 'output' in raw_prompt:
                prompt += "输出格式：\n"
                prompt += raw_prompt.get("title", "") + "\n"
                prompt += raw_prompt.get("description", "") + "\n\n"

            if 'examples' in raw_prompt:
                prompt += "以下为若干个示例，涵盖绝大部分特殊情况处理方式：\n"
                for key, example in raw_prompt["examples"].items():
                    prompt += f"例子{key} - {example.get('title', '')}\n"
                    prompt += "输入：\n"
                    prompt += example.get("input", "") + "\n"
                    prompt += "输出：\n"
                    prompt += example.get("output", "") + "\n\n"

            prompt += raw_prompt.get("task", {}).get("instruction", "")
            prompt += "\n\n"
            self._prompt = prompt

        return self._prompt

    def parse(self, table_data: dict) -> Union[list, None]:
        """
        使用 LLM 解析表格数据并解析为 JSON
        :param table_data: table_html_to_md 返回的结构化数据，包含：
            - table_count: 表格数量
            - tables: 表格列表，每个表格包含：
              - matrix: 二维矩阵
              - segments: 分段信息
              - is_double_column: 是否双栏
              - markdown: Markdown 字符串
              - html: HTML 字符串
              - context: 第一个表格前的上下文文本（可选）
        :return: 解析后的JSON结果
        """
        if not table_data or not table_data.get("tables"):
            return []

        # 构建Prompt
        prompt = self._build_prompt()

        results = []

        for table in table_data["tables"]:
            context = table.get("context", "")
            context = f"【表格上方原始文本】:\n{context}\n\n" if context else ""

            md_content = table["markdown"]
            html_content = table["html"]

            user_content = (
                f"{context}"
                f"【Markdown】:\n{md_content}\n\n"
                # f"【HTML源码】:\n{html_content}"
            )

            # 构造消息
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ]
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self.sampling_params
                )
                
            except Exception as e:
                raise RuntimeError(f"API调用失败: {str(e)}")

            content = response.choices[0].message.content
            parsed_result = self.safe_json_parse(content)

            # 安全解析JSON
            results.append(parsed_result)

        # print("提取结果:", content)

        return results

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


# 获取环境变量
openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

_table_parser = None
def get_table_parser(prompt_path: str = None) -> TableParserLLM:
    global _table_parser
    if _table_parser is None:
        _table_parser = TableParserLLM(prompt_path=prompt_path,
                                       openai_base_url=openai_base_url,
                                       openai_api_key=openai_api_key,
                                       model="Qwen3.5-4B")
    return _table_parser

table_parser = get_table_parser()

