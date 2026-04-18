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
from backend.ocr import RawBlock, RawFileOutput, RawPageOutput


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

    def load_result(self, output_dir: str) -> List[RawFileOutput]:
        """
        加载 OCR 输出结果为统一的中间层结构

        Args:
            output_dir: 输出目录

        Returns:
            RawFileOutput 列表（每个文件一个结果）
        """
        raw_outputs = []

        # 遍历所有以文件名命名的子目录
        for file_dir in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_dir)
            if os.path.isfile(file_path):
                continue

            raw_pages = []

            # 遍历所有页面索引目录
            for page_dir in os.listdir(file_path):
                page_path = os.path.join(file_path, page_dir)
                if os.path.isfile(page_path):
                    continue

                page_images = []
                page_json = None

                # 收集该页面目录下的所有图片路径
                for img_file in os.listdir(page_path):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        page_images.append(os.path.join(page_path, img_file))

                # 加载 JSON 文件
                for filename in os.listdir(page_path):
                    if not filename.endswith('.json'):
                        continue

                    json_file = os.path.join(page_path, filename)

                    if os.path.isdir(json_file):
                        continue

                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            page_json = json.load(f)
                            break
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"⚠️ 加载 JSON 文件失败 {json_file}: {str(e)}")
                        continue

                if page_json is None:
                    raise RuntimeError(f"文件 {page_path} 中没有有效的 JSON 文件")

                # 转换为 RawPageOutput
                raw_page = self._convert_to_raw_page(page_json, page_images)
                raw_pages.append(raw_page)

            if raw_pages:
                input_path = raw_pages[0].raw_json.get('input_path', file_path)
                raw_output = RawFileOutput(
                    input_path=input_path,
                    pages=raw_pages,
                    ocr_engine="paddleocr"
                )
                raw_outputs.append(raw_output)

        return raw_outputs

    @staticmethod
    def _convert_to_raw_page(
        page_json: Dict[str, Any], image_paths: List[str]
    ) -> RawPageOutput:
        """
        将单个页面的 JSON 数据转换为 RawPageOutput

        Args:
            page_json: 单个页面的 JSON 数据
            image_paths: 图片路径列表

        Returns:
            RawPageOutput 对象
        """
        blocks = page_json.get("parsing_res_list", [])
        page_index = page_json.get("page_index", 0) or 0

        raw_blocks = []
        for block in blocks:
            label = block.get("block_label", "")
            bbox = block.get("block_bbox")
            block_id = block.get("block_id", len(raw_blocks))
            content = block.get("block_content", "")
            confidence = block.get("confidence")

            raw_block = RawBlock(
                block_id=block_id,
                label=label,
                content=content,
                bbox=tuple(bbox) if bbox else None,
                confidence=confidence,
                # metadata={k: v for k, v in block.items() if k not in ["block_label", "block_bbox", "block_id", "block_content", "confidence"]}
            )
            raw_blocks.append(raw_block)

        return RawPageOutput(
            page_index=page_index,
            width=page_json.get("width", 0),
            height=page_json.get("height", 0),
            blocks=raw_blocks,
            image_paths=image_paths,
            raw_json=page_json
        )
