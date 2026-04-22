import enum
import os

from backend.config import settings
from .base_llm import BaseLLM

class TableType(str, enum.Enum):
    personal = "personal"
    measured = "measured"
    summary = "summary"
    other = "other"


class TableParseRouter(BaseLLM):
    def __init__(
        self,
        prompt_path,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-4B",
        enable_thinking: bool = False
    ):
        """
        初始化客户端

        Args:
            base_url: vLLM API地址
            api_key: API密钥（vLLM默认为EMPTY）
            model: 模型名称
        """
        super().__init__(
            prompt_path=prompt_path,
            base_url=base_url,
            api_key=api_key,
            model=model,
            enable_thinking=enable_thinking
        )

    def router(self, table_data: dict) -> str:
        '''
        使用 LLM 分类表格类型
        :param table_data: table_html_to_md 返回的结构化数据中的表格，包含：
            - matrix: 二维矩阵
            - segments: 分段信息
            - is_double_column: 是否双栏
            - markdown: Markdown 字符串
            - html: HTML 字符串
        :return: 分类后的类型，用字符串表示，包含：
            - personal: 个人信息类
            - measured: 检验数值类
            - summary: 小结建议类
            - other: 其它类，一般是健康小知识或医师信息等
        '''
        if not table_data:
            return TableType.other

        prompt = self._load_prompt()

        md_content = table_data["markdown"]
        html_content = table_data["html"]

        user_content = (
            f"【Markdown】:\n{md_content}\n\n"
            # f"【HTML源码】:\n{html_content}"
        )

        # 调用 LLM
        times = 3
        while (times > 0):
            content = self._call_llm(prompt, user_content)

            # 解析 JSON
            parsed_result = self._parse_json_response(content)

            answer_val = parsed_result.get("answer")
            if answer_val and answer_val in [t.value for t in TableType]:
                break

            times -= 1

        # 尝试将字符串转换为枚举对象
        if parsed_result and parsed_result.get("answer"):
            try:
                return TableType(parsed_result["answer"])
            except ValueError:
                # 如果 LLM 抽风返回了不在定义里的值，捕获异常并兜底
                return TableType.other
        
        return TableType.other


# 获取环境变量
base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

_table_parse_router = None


def get_table_parse_router() -> TableParseRouter:
    global _table_parse_router
    if _table_parse_router is None:
        _table_parse_router = TableParseRouter(prompt_path=settings.llm_table_parse_router_prompt,
                                               base_url=base_url,
                                               api_key=api_key,
                                               model="Qwen3.5-4B")
    return _table_parse_router


table_parse_router = get_table_parse_router()
