"""
OCR Runner Module
封装 PaddleOCR 的 subprocess 调用
"""

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.llm import table_parser
from backend.ocr import (Image, OCRResult, Page, Table, TableItem,
                         TextRegion, table_html_clean, table_html_to_md)


class PaddleOCRRunner:
    """PaddleOCR 运行器，通过 subprocess 调用独立的 OCR 脚本"""

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        gpu_id: Optional[int] = None,
    ):
        """
        初始化 PaddleOCR 运行器

        Args:
            use_gpu: 是否使用 GPU，默认从 settings 读取
            gpu_id: GPU 设备 ID，默认从 settings 读取
        """
        self.use_gpu = use_gpu if use_gpu is not None else settings.ocr_use_gpu
        self.gpu_id = gpu_id if gpu_id is not None else settings.ocr_gpu_id
        self._script_path = os.path.join(
            os.path.dirname(__file__), "paddle_runner.py"
        )

    def run(self, file_path: str, output_dir: Optional[str] = None) -> str:
        """
        执行 OCR 处理

        Args:
            file_path: 输入图片/pdf路径，或者包含多个图片/pdf的文件夹路径
            output_dir: 输出 JSON 目录，默认生成带时间戳的目录

        Returns:
            输出 JSON 目录路径
        """
        # 确定输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(settings.project_root / f"data/sensitive/ocr_output/{timestamp}/")

        os.makedirs(output_dir, exist_ok=True)

        # 构建命令
        cmd = [
            settings.ocr_python,
            self._script_path,
            "--image", str(file_path),
            "--output", str(output_dir),
        ]

        if self.use_gpu:
            cmd.extend(["--gpu", "--gpu-id", str(self.gpu_id)])

        # 执行 subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # 检查执行结果
        # if result.returncode != 0:
        #     raise RuntimeError(f"OCR 执行失败：{result.stderr.strip()}")

        return output_dir

    @staticmethod
    def load_result(output_dir: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        加载 OCR 输出结果

        Args:
            output_dir: 输出目录

        Returns:
            (按文件分组的 OCR 数据，图片路径列表)
            每个文件包含：
            - input_path: 原始文件路径
            - pages: 该文件的页面列表
        """
        files_data = []  # List[Dict]: 按文件分组的数据

        # 遍历所有以文件名命名的子目录
        for file_dir in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_dir)
            if os.path.isfile(file_path):
                continue

            file_pages = []  # 该文件的所有页面

            # 遍历所有页面索引目录
            for page_dir in os.listdir(file_path):
                page_path = os.path.join(file_path, page_dir)
                if os.path.isfile(page_path):
                    continue

                page_jsons = []
                page_images = []

                # 收集该页面目录下的所有图片路径
                for img_file in os.listdir(page_path):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        page_images.append(os.path.join(page_path, img_file))

                for filename in os.listdir(page_path):
                    if not filename.endswith('.json'):
                        continue

                    json_file = os.path.join(page_path, filename)

                    if os.path.isdir(json_file):
                        continue

                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            page_data = json.load(f)
                            page_jsons.append(page_data)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"⚠️ 加载 JSON 文件失败 {json_file}: {str(e)}")
                        continue
                
                if len(page_jsons) == 0:
                    raise RuntimeError(f"文件 {page_path} 中没有有效的 JSON 文件")
                elif len(page_jsons) > 1:
                    print(f"⚠️ 文件 {page_path} 中有多个 JSON 文件，将只使用第一个")

                file_pages.append({
                    'page_json': page_jsons[0],
                    'page_images': page_images
                })

            if file_pages:
                # 按 file 分组
                files_data.append({
                    'input_path': file_pages[0].get('input_path', file_path),
                    'pages': file_pages
                })

        return files_data


    def parse_result(
        self,
        ocr_output: List[Dict[str, Any]]
    ) -> List[OCRResult]:
        """
        解析 OCR 输出结果，转换为 schema.py 定义的结构

        Args:
            ocr_output: OCR 输出数据列表（已按文件分组）
            image_paths: 图片路径列表

        Returns:
            RawOCRResult 列表（每个文件一个结果）
        """
        results = []

        for file_data in ocr_output:
            pages = file_data.get('pages', [])
            input_path = file_data.get('input_path', '')

            raw_pages = []
            for page_data in pages:
                image_paths = page_data.get('page_images', [])
                page_json = page_data.get('page_json', {})

                raw_page = self._parse_single_page(page_json, image_paths)
                raw_pages.append(raw_page)

            file_format = os.path.splitext(input_path)[1].lower().lstrip('.')
            result = OCRResult(
                source_path=input_path,
                file_format=file_format or 'jpg',
                total_pages=len(raw_pages),
                ocr_engine='paddleocr',
                scanned_at=datetime.now(),
                pages=raw_pages
            )
            results.append(result)

        return results

    def _parse_single_page(
        self,
        page_data: Dict[str, Any],
        image_paths: List[str]
    ) -> Page:
        """
        解析单个页面

        Args:
            page_data: 单个页面的 JSON 数据
            image_paths: 图片路径列表

        Returns:
            RawPage 对象
        """
        blocks = page_data.get("parsing_res_list", [])
        page_index = page_data.get("page_index", 0) or 0
        page_count = page_data.get("page_count", 1) or 1

        # 分类 blocks
        text_regions = []
        tables = []
        images = []
        context_before_table = []
        first_table_found = False
        context_text = None

        for block in blocks:
            label = block.get("block_label")
            bbox = block.get("block_bbox")
            block_id = block.get("block_id", len(text_regions) + len(tables) + len(images))
            block_text = block.get("block_content", "")

            if label in ["text", "content", "doc_title", "figure_title", "paragraph_title"]:
                text_regions.append(TextRegion(
                    index=block_id,
                    label=label,
                    text=block_text,
                    bbox=tuple(bbox) if bbox else None,
                    block_index=len(text_regions)
                ))
                if not first_table_found:
                    context_before_table.append(block)

            elif label == "table":
                first_table_found = True
                # 处理表格
                table_html = block_text
                table_html = table_html_clean(table_html)
                table_md = table_html_to_md(table_html)

                if table_md:
                    # 只给第一个表格添加上下文（每个页面独立处理）
                    if len(tables) == 0 and context_text is None:
                        # 过滤 context 内容
                        context_text = self._filter_context_text(context_before_table)
                        if context_text:
                            table_md['tables'][0]['context'] = context_text

                    llm_result = table_parser.parse(table_md)

                    # 转换为 Table 对象
                    if llm_result and len(llm_result) > 0:
                        for table in llm_result:
                            table_obj = Table(
                                index=block_id,
                                title=table.get('title', '') if isinstance(table, dict) else '',
                                items=self._build_table_items(table) if isinstance(table, dict) else []
                            )
                            tables.append(table_obj)

            elif label == "image":
                # 根据 bbox 匹配图片路径
                img_path = self._match_image_path(image_paths, bbox)
                images.append(Image(
                    index=block_id,
                    image_path=img_path,
                    bbox=tuple(bbox) if bbox else None
                ))

        # 构建 RawPage
        return Page(
            page_index=page_index,
            image_width=page_data.get("width", 0),
            image_height=page_data.get("height", 0),
            regions=text_regions,
            tables=tables,
            images=images
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
    def _filter_context_text(blocks: List[Dict[str, Any]]) -> str:
        """
        过滤掉无关的个人信息，保留可能作为表格标题的内容

        排除：姓名、性别、年龄、门诊号、No.、体检号等
        保留：短文本、医疗检查关键词（血常规、肝功能等）

        Args:
            blocks: text blocks 列表

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
            content = block.get("block_content", "").strip()
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
        if not bbox or not image_paths:
            return ""

        for img_path in image_paths:
            img_name = os.path.basename(img_path)

            # 提取 bbox 数字：img_in_image_box_77_15_776_877.jpg -> [77, 15, 776, 877]
            name_without_ext = os.path.splitext(img_name)[0]
            parts = name_without_ext.split('_')

            if len(parts) >= 6 and name_without_ext.startswith("img_in_image_box_"):
                img_bbox = list(map(int, parts[-4:]))
                if img_bbox == bbox:
                    return img_path

        # 如果没有精确匹配，返回第一个
        return image_paths[0] if image_paths else ""
