# file:d:\Python\CheckupAI\backend\ocr\schema.py
"""
schema.py — OCR 模型输出的中间结构定义
Python 3.12+, 纯 dataclass，无第三方依赖
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# 核心单元：带 bbox 的文本块
# ---------------------------------------------------------------------------


@dataclass
class TextRegion:
    """OCR 识别出的单个文本区域，是 RawOCRResult 的最小单元。

    bbox 坐标系：左上角为原点，单位为像素，格式 (x0, y0, x1, y1)。
    """

    index: int  # block 序号
    label: str
    text: str  # OCR 识别出的原始文本，保留原文不做任何清洗
    bbox: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) 像素坐标
    confidence: float | None = None  # 0.0~1.0，引擎置信度，不支持则为 None
    block_index: int | None = None  # 在当前页中的块序号（从上到下，0-based）


# ---------------------------------------------------------------------------
# 结构化表格（能识别表格结构的引擎才填）
# ---------------------------------------------------------------------------


@dataclass
class TableItem:
    """表格中的单条检验项目。"""

    # --- 必填：原始字符串，保留原文，方便拼接展示 ---
    item: str  # 项目名称，如"血红蛋白"
    result: str  # 结果值，如"135"

    # --- 原始字符串（可选）---
    unit: str = ""  # 单位，如"g/L"
    abnormal: str = ""  # 原始异常标志字符，如"↑"、"H"
    reference_range: str = ""  # 原始参考值字符串，如"115-150"


@dataclass
class Table:
    """一张检验 / 检查表格。"""

    index: int  # block 序号
    title: str  # 表格标题，如"血常规"
    items: list[TableItem] = field(default_factory=list)


@dataclass
class Image:
    """页面中提取的图片。"""

    index: int  # block 序号
    image_path: str  # 图片保存路径（相对或绝对）
    bbox: tuple[int, int, int, int] | None = None


# ---------------------------------------------------------------------------
# 页面级原始结果
# ---------------------------------------------------------------------------


@dataclass
class RawPage:
    """单页的 OCR 原始输出。"""

    page_index: int  # 0-based
    image_width: int  # 页面图像宽度（像素）
    image_height: int  # 页面图像高度（像素）

    # 所有引擎都应填充：带 bbox 的文本块，兜底数据源
    regions: list[TextRegion] = field(default_factory=list)

    # 结构化表格
    tables: list[Table] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)

    # --- 可选：引擎级别的原始输出（调试用，不参与解析）---
    # 保留原始 JSON 字符串，方便排查 OCR 输出和中间结构之间的转换问题
    raw_output_json: str | None = None


# ---------------------------------------------------------------------------
# 顶层：单份文件的完整 OCR 原始结果
# ---------------------------------------------------------------------------


@dataclass
class RawOCRResult:
    """一份原始文件经 OCRRunner 处理后的完整中间结构。

    所有 Runner 对外输出统一为此类型。

    典型用法::

        # 在 PaddleOCRRunner 内部
        result = RawOCRResult(
            source_path="/data/reports/2024_checkup.pdf",
            ocr_engine="paddleocr",
            ocr_model_version="PP-OCRv4",
            scanned_at=datetime.now(),
            total_pages=3,
            pages=[RawPage(...), RawPage(...), RawPage(...)],
        )
        # 交给 parser.py
        report = parse(result)
    """

    # --- 文件元数据 ---
    source_path: str  # 原始文件路径（绝对路径）
    file_format: str  # "pdf" / "jpg" / "png" 等，小写
    total_pages: int  # 文件总页数

    # --- OCR 引擎元数据 ---
    ocr_engine: str  # 引擎标识，如 "paddleocr" / "tesseract" / "azure"

    # --- 时间 ---
    scanned_at: datetime  # 本次 OCR 扫描时间

    # --- 页面内容 ---
    pages: list[RawPage] = field(default_factory=list)