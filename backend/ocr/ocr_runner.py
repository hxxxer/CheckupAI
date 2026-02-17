"""
OCR Runner Module
封装 PaddleOCR 的 subprocess 调用
"""

import os
import subprocess
from datetime import datetime
from typing import Optional
from backend.config import settings


class PaddleOCRRunner:
    """PaddleOCR 运行器，通过 subprocess 调用独立的 OCR 脚本"""
    
    def __init__(self, use_gpu: Optional[bool] = None, gpu_id: Optional[int] = None):
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
