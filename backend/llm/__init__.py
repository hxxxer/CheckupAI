"""
LLM 模块
懒加载避免测试环境缺少 openai 等依赖时导入即崩溃
"""


def __getattr__(name):
    if name == "table_parser":
        from .table_parser import table_parser
        return table_parser
    if name == "text_analyzer":
        from .text_analyzer import text_analyzer
        return text_analyzer
    if name == "table_parse_router":
        from .table_parse_router import table_parse_router
        return table_parse_router
    if name == "TableType":
        from .table_parse_router import TableType
        return TableType
    if name == "query_rewriter":
        from .query_rewriter import query_rewriter
        return query_rewriter
    if name == "QueryRewriter":
        from .query_rewriter import QueryRewriter
        return QueryRewriter
    if name == "chat_llm":
        from .chat_llm import chat_llm
        return chat_llm
    if name == "ChatLLM":
        from .chat_llm import ChatLLM
        return ChatLLM
    if name == "safe_json_parse":
        from .utils import safe_json_parse
        return safe_json_parse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")