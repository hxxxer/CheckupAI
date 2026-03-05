"""
OCR 模块
使用 PaddleOCR 处理体检报告

注意：PaddleOCRRunner 使用懒加载，只在首次访问时导入。
这样可以避免在测试等场景中因缺少 OCR 环境而报错。
"""

# 轻依赖，直接导入
from .utils import table_html_clean, table_html_to_md
from .schema import RawOCRResult


def __getattr__(name):
    """
    懒加载：只在访问时才导入重依赖模块

    这样可以避免在测试等场景中因缺少 OCR 环境（PaddlePaddle、特定 Python 路径等）而报错。
    只有在真正使用 PaddleOCRRunner 时才会触发导入和初始化。
    """
    if name == 'PaddleOCRRunner':
        from .ocr_runner import PaddleOCRRunner
        return PaddleOCRRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 用于 IDE 类型提示和__all__导出
__all__ = ['PaddleOCRRunner', 'table_html_clean', 'table_html_to_md', 'RawOCRResult']
