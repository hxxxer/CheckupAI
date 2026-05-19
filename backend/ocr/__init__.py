"""
OCR 模块
使用 PaddleOCR 处理体检报告

所有重依赖模块（schema / runner / checkup_parser / parser）均懒加载，
避免测试环境缺少依赖时导入崩溃，同时解决与 backend.llm 的循环引用。
"""

# 轻依赖，直接导入（不触发 backend.llm）
from .utils import table_html_clean, table_html_to_md

# schema 符号名集合（用于懒加载路由）
_SCHEMA_NAMES = frozenset({
    "OCRResult", "Page", "PersonalInfo", "PositiveFinding",
    "TextAnalysis", "TextRegion", "Table", "TableItem",
    "Image", "RawBlock", "RawPageOutput", "RawFileOutput",
})


def __getattr__(name):
    # schema 懒加载（避免与 backend.llm 循环引用）
    if name in _SCHEMA_NAMES:
        from . import schema
        for nm in _SCHEMA_NAMES:
            if hasattr(schema, nm):
                globals()[nm] = getattr(schema, nm)
        return getattr(schema, name)

    # 重依赖懒加载
    if name == "PaddleOCRRunner":
        from .runner import PaddleOCRRunner
        return PaddleOCRRunner
    if name == "parse_checkup":
        from .checkup_parser import parse_checkup
        return parse_checkup
    if name == "UniversalParser":
        from .parser import UniversalParser
        return UniversalParser

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PaddleOCRRunner",
    "parse_checkup",
    "UniversalParser",
    "table_html_clean",
    "table_html_to_md",
    "OCRResult",
    "Page",
    "Table",
    "TableItem",
    "TextAnalysis",
    "TextRegion",
    "PersonalInfo",
    "PositiveFinding",
    "Image",
    "RawBlock",
    "RawPageOutput",
    "RawFileOutput",
]
