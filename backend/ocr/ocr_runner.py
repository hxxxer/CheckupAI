"""
OCR Runner Module
封装 PaddleOCR 的 subprocess 调用
"""

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.ocr import table_html_clean, table_html_to_md


class PaddleOCRRunner:
    """PaddleOCR 运行器，通过 subprocess 调用独立的 OCR 脚本"""

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        gpu_id: Optional[int] = None,
        table_parser: Optional[Any] = None,
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
        self.table_parser = table_parser
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
        for block in blocks:
            label = block.get("block_label")
            if label == "text":
                text_blocks.append(block)
            elif label == "table":
                table_blocks.append(block)
            elif label == "image":
                image_blocks.append(block)

        # 处理表格
        tables_data = []
        if self.table_parser:
            self.table_parser.wake_up()
            for idx, table_block in enumerate(table_blocks):
                try:
                    table_html = table_block.get("block_content")
                    table_html = table_html_clean(table_html)
                    table_md = table_html_to_md(table_html)
                    if table_md:
                        tables_data.append(self.table_parser.parse(table_md))
                except Exception as e:
                    print(f"⚠️ 表格块 {idx} 解析失败：{str(e)}")
                    raise
            self.table_parser.sleep()

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
