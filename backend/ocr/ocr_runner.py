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
from backend.ocr import table_html_clean, table_html_to_md
from backend.instances import table_parser


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
            table_parser: 表格解析器实例（依赖注入），用于 LLM 表格解析
        """
        self.use_gpu = use_gpu if use_gpu is not None else settings.ocr_use_gpu
        self.gpu_id = gpu_id if gpu_id is not None else settings.ocr_gpu_id
        self._script_path = os.path.join(
            os.path.dirname(__file__), "paddle_runner.py"
        )

    def run(self, image_path: str, output_dir: Optional[str] = None) -> str:
        """
        执行 OCR 处理

        Args:
            image_path: 输入图片路径
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
            "--image", str(image_path),
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
    def load_result(output_dir: str) -> List[Dict[str, Any]]:
        """
        加载 OCR 输出结果

        Args:
            output_dir: 输出目录

        Returns:
            OCR 输出数据（已加载的 JSON）
        """
        results = []
        
        # 遍历所有以文件名命名的子目录
        for file_dir in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_dir)
            if not os.path.isdir(file_path):
                continue
            
            # 遍历所有页面索引目录
            for page_dir in os.listdir(file_path):
                page_path = os.path.join(file_path, page_dir)
                if not os.path.isdir(page_path):
                    continue
                
                for filename in os.listdir(page_path):
                    if not filename.endswith('.json'):
                        continue
                    
                    json_file = os.path.join(page_path, filename)
                    
                    if os.path.isdir(json_file):
                        continue
                    
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            results.append(json.load(f))
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"⚠️ 加载 JSON 文件失败 {json_file}: {str(e)}")
                        continue
        
        return results


    def parse_result(self, ocr_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        解析 OCR 输出结果，提取表格和文本

        Args:
            ocr_output: OCR 输出数据（已加载的 JSON）

        Returns:
            包含 tables、full_text 和 stats 的字典
        """
        if not ocr_output:
            raise ValueError("OCR 输出为空")

        # 单图处理
        page_data = ocr_output[0].get("res", ocr_output[0])
        blocks = page_data.get("parsing_res_list", [])

        text_blocks = []
        table_blocks = []
        image_blocks = []
        context_before_table = []  # 第一个 table 之前的 text blocks
        first_table_found = False

        for block in blocks:
            label = block.get("block_label")
            if label == "text":
                text_blocks.append(block)
                # 在遇到第一个 table 之前，累积 text 到 context
                if not first_table_found:
                    context_before_table.append(block)
            elif label == "table":
                first_table_found = True
                table_blocks.append(block)
            elif label == "image":
                image_blocks.append(block)

        # 过滤 context 内容，提取可能有用的表格标题信息
        context_text = self._filter_context_text(context_before_table)

        # 处理表格
        tables_data = []
        if table_parser:
            table_parser.wake_up()
            for idx, table_block in enumerate(table_blocks):
                try:
                    table_html = table_block.get("block_content")
                    table_html = table_html_clean(table_html)
                    table_md = table_html_to_md(table_html)
                    if table_md:
                        table_data = table_parser.parse(table_md)
                        # 只给第一个表格添加上下文
                        if context_text and idx == 0:
                            table_data['tables'][0]['context'] = context_text
                        tables_data.append()
                except Exception as e:
                    print(f"⚠️ 表格块 {idx} 解析失败：{str(e)}")
                    raise
            table_parser.sleep()

        full_text = "\n".join([b["block_content"] for b in text_blocks])

        return {
            "tables": tables_data,
            "full_text": full_text,
            "stats": {
                "table_blocks": len(table_blocks),
                "parsed_tables": len(tables_data),
                "text_blocks": len(text_blocks),
            }
        }

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
