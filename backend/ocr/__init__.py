"""
OCR 模块
使用 PaddleOCR 处理体检报告
"""
from .ocr_runner import PaddleOCRRunner
from .utils import table_html_clean, table_html_to_md
