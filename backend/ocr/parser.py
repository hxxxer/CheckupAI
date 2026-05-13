"""
parser.py - OCR结果解析器（引擎无关）
负责将 RawFileOutput 转换为 OCRResult
"""

import re
from datetime import datetime
from typing import Any, Dict, List

from backend.llm import table_parser, table_parse_router, TableType
from backend.ocr import (
    Image,
    OCRResult,
    Page,
    RawBlock,
    RawFileOutput,
    RawPageOutput,
    Table,
    TableItem,
    TextRegion,
)
from backend.ocr import table_html_clean, table_html_to_md


class UniversalParser:
    """通用OCR结果解析器，将引擎无关的原始输出转换为标准OCRResult"""

    def parse(self, raw_outputs: list[RawFileOutput]) -> list[OCRResult]:
        """
        将原始输出转换为标准OCRResult

        Args:
            raw_outputs: 原始文件输出列表

        Returns:
            OCRResult 列表
        """
        results = []
        for raw_output in raw_outputs:
            result = self._parse_file(raw_output)
            results.append(result)
        return results

    def _parse_file(self, raw_output: RawFileOutput) -> OCRResult:
        """
        解析单个文件

        Args:
            raw_output: 单个文件的原始输出

        Returns:
            OCRResult 对象
        """
        pages = []
        for raw_page in raw_output.pages:
            page = self._parse_single_page(raw_page)
            pages.append(page)

        file_format = self._extract_file_format(raw_output.input_path)

        return OCRResult(
            source_path=raw_output.input_path,
            file_format=file_format,
            total_pages=len(pages),
            ocr_engine=raw_output.ocr_engine,
            scanned_at=datetime.now(),
            pages=pages,
        )

    def _parse_single_page(self, raw_page: RawPageOutput) -> Page:
        """
        解析单个页面

        Args:
            raw_page: 单页的原始输出

        Returns:
            Page 对象
        """
        text_regions = []
        tables = []
        images = []
        context_before_table = []
        first_table_found = False
        context_text = None

        for block in raw_page.blocks:
            if block.label in [
                "text",
                "content",
                "doc_title",
                "figure_title",
                "paragraph_title",
            ]:
                text_regions.append(
                    TextRegion(
                        index=block.block_id,
                        label=block.label,
                        text=block.content,
                        bbox=block.bbox,
                        confidence=block.confidence,
                        block_index=len(text_regions),
                    )
                )
                if not first_table_found:
                    context_before_table.append(block)
            elif block.label == "table":
                first_table_found = True
                table_html = block.content
                
                # 检查表格内是否有内嵌图片（兼容不同OCR引擎）
                embedded_images = self._extract_embedded_images_from_table(
                    table_html, raw_page.image_paths
                )
                if embedded_images:
                    for img_path in embedded_images:
                        images.append(
                            Image(
                                index=block.block_id,
                                image_path=img_path,
                                bbox=None,
                            )
                        )

                table_html = table_html_clean(table_html)
                tables_md = table_html_to_md(table_html)

                if tables_md:
                    # 只给第一个表格添加上下文（每个页面独立处理）
                    if len(tables) == 0 and context_text is None:
                        # 过滤 context 内容
                        context_text = self._filter_context_text(
                            context_before_table)
                        if context_text:
                            tables_md["tables"][0]["context"] = context_text

                    for table_md in tables_md["tables"]:
                        table_types = table_parse_router.router(table_md)
                        if TableType.measured in table_types:
                            llm_result = table_parser.parse(table_md)
                            # 转换为 Table 对象
                            if llm_result:
                                table_obj = Table(
                                    index=block.block_id,
                                    title=llm_result.get('title', '') if isinstance(
                                        llm_result, dict) else '',
                                    items=self._build_table_items(
                                        llm_result) if isinstance(llm_result, dict) else [],
                                    raw_md=table_md["markdown"],
                                    types=table_types
                                )
                        else:
                            table_obj = Table(
                                index=block.block_id,
                                raw_md=table_md["markdown"],
                                types=table_types
                            )
                        tables.append(table_obj)
            elif block.label == "image":
                img_path = self._match_image_path(
                    raw_page.image_paths, block.bbox)
                if img_path:
                    images.append(
                        Image(
                            index=block.block_id,
                            image_path=img_path,
                            bbox=block.bbox,
                        )
                    )

        return Page(
            page_index=raw_page.page_index,
            image_width=raw_page.width,
            image_height=raw_page.height,
            regions=text_regions,
            tables=tables,
            images=images,
        )

    @staticmethod
    def _build_table_items(table_data: dict) -> List[TableItem]:
        """
        从 LLM 解析结果构建 TableItem 列表

        Args:
            table_data: LLM 解析后的表格数据

        Returns:
            TableItem 列表
        """
        items = []
        table_list = table_data.get('table', [])
        if not isinstance(table_list, list):
            return items

        for item_data in table_list:
            if not isinstance(item_data, dict):
                continue
            items.append(TableItem(
                item=item_data.get('item', ''),
                result=item_data.get('result', ''),
                unit=item_data.get('unit', ''),
                abnormal=item_data.get('abnormal', ''),
                reference_range=item_data.get('reference_range', '')
            ))

        return items

    @staticmethod
    def _filter_context_text(blocks) -> str:
        """
        过滤掉无关的个人信息，保留可能作为表格标题的内容

        排除：姓名、性别、年龄、门诊号、No.、体检号等
        保留：短文本、医疗检查关键词（血常规、肝功能等）

        Args:
            blocks: text blocks 列表 (list[RawBlock])

        Returns:
            过滤后的文本字符串
        """
        # 排除模式（正则）
        exclude_patterns = [
            r'姓名 [：:\s]',
            r'性别 [：:\s]',
            r'年龄 [：:\s]',
            r'门诊号 [：:\s]',
            r'住院号 [：:\s]',
            r'体检号 [：:\s]',
            r'No[.:]\s*\w+',
            r'编号 [：:\s]',
            r'\bID[：:\s]',
            r'报告日期',
            r'体检日期',
            r'打印日期',
            r'就诊卡',
        ]

        # 医疗检查关键词
        keywords = [
            '血常规', '尿常规', '便常规',
            '肝功能', '肾功能', '血脂', '血糖',
            '心电图', '胸片', 'CT', 'B 超', '彩超',
            '肿瘤标志物', '凝血', '生化', '肝炎', '乙肝',
            '甲功', '糖化', '离子', '心肌酶',
            '免疫', '激素', '贫血', '检验', '检查',
        ]

        filtered_lines = []
        for block in blocks:
            content = getattr(block, 'content', '').strip()
            if not content:
                continue

            # 排除过长的文本
            if len(content) > 50:
                continue

            # 检查是否包含排除模式
            if any(re.search(p, content, re.IGNORECASE) for p in exclude_patterns):
                continue

            # 保留：包含关键词 或 短文本（<20 字符）
            if any(kw in content for kw in keywords) or len(content) < 20:
                filtered_lines.append(content)

        return "\n".join(filtered_lines)

    @staticmethod
    def _match_image_path(image_paths: List[str], bbox: List[int]) -> str:
        """
        根据 bbox 匹配图片路径

        图片命名格式：img_in_image_box_x0_y0_x1_y1.jpg

        Args:
            image_paths: 图片路径列表
            bbox: block 的 bbox 列表 [x0, y0, x1, y1]

        Returns:
            匹配的图片路径
        """
        import os
        if not bbox or not image_paths:
            return ""

        bbox_text = "_".join(str(x) for x in bbox)
        for img_path in image_paths:
            img_name = os.path.basename(img_path)

            name_without_ext = os.path.splitext(img_name)[0]

            if name_without_ext.endswith(bbox_text):
                return img_path

        return ""
    
    @staticmethod
    def _extract_embedded_images_from_table(table_content: str, image_paths: List[str]) -> List[Dict[str, str]]:
        """
        从表格内容中提取内嵌图片的路径（兼容不同OCR引擎）
        
        通过暴力匹配 image_paths 中每个文件的 basename 来查找表格中引用的图片

        Args:
            table_content: 表格的 HTML 或文本内容
            image_paths: 页面图片路径列表

        Returns:
            匹配到的图片路径列表
        """
        import os
        
        if not table_content or not image_paths:
            return []
        
        embedded_images = []
        
        # 遍历所有图片路径，检查其文件名是否出现在表格内容中
        for img_path in image_paths:
            img_filename = os.path.basename(img_path)
            
            # 检查文件名是否在表格内容中出现
            if img_filename in table_content:
                # 尝试找到文件名的引用位置（可能是完整路径或只是文件名）
                embedded_images.append(img_path)
        
        return embedded_images

    @staticmethod
    def _extract_file_format(file_path: str) -> str:
        """
        从文件路径提取文件格式

        Args:
            file_path: 文件路径

        Returns:
            文件格式（小写，不含点）
        """
        import os

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext or "jpg"
