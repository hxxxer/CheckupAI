"""
raw_ocr_result.py — OCR 模型输出的中间结构
Python 3.12+, 纯 dataclass，无第三方依赖

职责：
    各 OCRRunner（PaddleOCR / Tesseract / Azure 等）消化掉各自的输出差异，
    统一转换成 RawOCRResult 后交给 parser.py 处理。

设计原则：
    - 同时支持两种粒度，Runner 按自身能力选择填充深度：
        · TextRegion：带 bbox 的文本块，所有引擎都能输出，作为兜底
        · RawTable：结构化行列表格，能识别表格结构的引擎（如 Azure）才填
    - 只保留 OCR 层能直接输出的信息，语义理解留给 parser.py
    - parser.py 优先消费 raw_tables，无结构化表格时回退到 regions 走 LLM 解析
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# 核心单元：带 bbox 的文本块
# ---------------------------------------------------------------------------

@dataclass
class TextRegion:
    """
    OCR 识别出的单个文本区域，是 RawOCRResult 的最小单元。

    bbox 坐标系：左上角为原点，单位为像素，格式 (x0, y0, x1, y1)。
    """
    index: int # block序号
    text: str    # OCR 识别出的原始文本，保留原文不做任何清洗
    bbox: Optional[list[int, int, int, int]] = None  # (x0, y0, x1, y1) 像素坐标
    confidence: Optional[float] = None  # 0.0~1.0，引擎置信度，不支持则为 None
    block_index: Optional[int] = None  # 在当前页中的块序号（从上到下，0-based）


# ---------------------------------------------------------------------------
# 结构化表格（能识别表格结构的引擎才填）
# ---------------------------------------------------------------------------

@dataclass
class TableItem:
    """表格中的单条检验项目。"""

    # --- 必填：原始字符串，保留原文，方便拼接展示 ---
    name:            str  # 项目名称，如"血红蛋白"
    result:          str  # 结果值，如"135"

    # --- 原始字符串（可选）---
    unit:            str = ""  # 单位，如"g/L"
    flag:            str = ""  # 原始异常标志字符，如"↑"、"H"
    reference_range: str = ""  # 原始参考值字符串，如"115-150"


@dataclass
class Table:
    """一张检验 / 检查表格。"""
    index: int # block序号
    title: str  # 表格标题，如"血常规"
    items: list[TableItem] = field(default_factory=list)


@dataclass
class Image:
    """页面中提取的图片。"""
    index: int # block序号
    image_path: str  # 图片保存路径（相对或绝对）
    bbox: Optional[list[int, int, int, int]] = None


# ---------------------------------------------------------------------------
# 页面级原始结果
# ---------------------------------------------------------------------------


@dataclass
class RawPage:
    """单页的 OCR 原始输出。"""

    page_number:  int  # 1-based
    image_width:  int  # 页面图像宽度（像素）
    image_height: int  # 页面图像高度（像素）

    # 所有引擎都应填充：带 bbox 的文本块，兜底数据源
    regions: list[TextRegion] = field(default_factory=list)

    # 结构化表格
    tables: list[Table] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)

    # --- 可选：引擎级别的原始输出（调试用，不参与解析）---
    # 保留原始 JSON 字符串，方便排查 OCR 输出和中间结构之间的转换问题
    raw_output_json: Optional[str] = None


# ---------------------------------------------------------------------------
# 顶层：单份文件的完整 OCR 原始结果
# ---------------------------------------------------------------------------

@dataclass
class RawOCRResult:
    """
    一份原始文件经 OCRRunner 处理后的完整中间结构。
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
    source_path:  str       # 原始文件路径（绝对路径）
    file_format:  str       # "pdf" / "jpg" / "png" 等，小写
    total_pages:  int       # 文件总页数

    # --- OCR 引擎元数据 ---
    ocr_engine:        str  # 引擎标识，如 "paddleocr" / "tesseract" / "azure"

    # --- 时间 ---
    scanned_at: datetime    # 本次 OCR 扫描时间

    # --- 页面内容 ---
    pages: list[RawPage] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    # def all_regions(self) -> list[TextRegion]:
    #     """返回所有页面的所有文本区域（跨页展平）。"""
    #     return [r for page in self.pages for r in page.regions]

    # def all_raw_tables(self) -> list[tuple[int, RawTable]]:
    #     """
    #     返回所有页面的结构化表格，格式为 (page_number, table)。
    #     为空说明所有 Runner 均不支持表格识别，parser.py 应回退到 regions。
    #     """
    #     return [
    #         (page.page_number, table)
    #         for page in self.pages
    #         for table in page.raw_tables
    #     ]

    # def get_page(self, page_number: int) -> Optional[RawPage]:
    #     """按页码（1-based）获取单页，不存在返回 None。"""
    #     for page in self.pages:
    #         if page.page_number == page_number:
    #             return page
    #     return None

    # def low_confidence_regions(
    #     self, threshold: float = 0.7
    # ) -> list[tuple[int, TextRegion]]:
    #     """
    #     返回置信度低于阈值的区域，格式为 (page_number, region)。
    #     用于入库前的质量检查或人工复核标记。
    #     """
    #     return [
    #         (page.page_number, region)
    #         for page in self.pages
    #         for region in page.regions
    #         if region.confidence is not None and region.confidence < threshold
    #     ]
