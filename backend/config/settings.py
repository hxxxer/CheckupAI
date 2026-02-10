import os
import sys
from pathlib import Path
from typing import Optional
import tomllib


class Settings:
    """配置管理器：TOML + 路径验证"""

    def __init__(self, config_path: Optional[str] = None):
        # 1. 确定配置文件路径
        if config_path:
            self.config_file = Path(config_path)
        else:
            # 优先级：项目config目录 > 当前目录
            candidates = [
                Path("backend/config/settings.toml"),
                Path("config/settings.toml"),
                Path("settings.toml")
            ]
            self.config_file = next(
                (p for p in candidates if p.exists()), None)

        if not self.config_file or not self.config_file.exists():
            raise FileNotFoundError(
                "❌ 配置文件未找到！请：\n"
                "1. 复制 backend/config/settings.toml.example → settings.toml\n"
                "2. 填写实际conda环境路径"
            )

        # 2. 加载TOML
        with open(self.config_file, "rb") as f:
            raw_config = tomllib.load(f)

        # 3. 解析为属性 + 路径标准化
        self._parse_config(raw_config)
        self._validate_paths()

    def _parse_config(self, config: dict):
        # 环境路径
        env = config.get("env", {})
        self.ocr_python: str = env.get("ocr_python", "")
        self.llm_python: str = env.get("llm_python", sys.executable)  # 默认当前环境

        # 路径处理（转为绝对路径）
        paths = config.get("paths", {})
        project_root = Path(paths.get("project_root", ".")).resolve()
        self.project_root: Path = project_root
        self.ocr_script: str = str(
            (project_root / paths.get("ocr_script", "")).resolve())
        self.temp_dir: Path = (
            project_root / paths.get("temp_dir", "data/temp")).resolve()
        self.log_dir: Path = (
            project_root / paths.get("log_dir", "logs")).resolve()

        # 业务参数
        self.ocr_use_gpu: bool = config.get("ocr", {}).get("use_gpu", True)
        self.ocr_gpu_id: int = config.get("ocr", {}).get("gpu_id", 0)

        llm_config = config.get("llm", {})
        self.llm_table_prompt: str = str(
            (project_root / llm_config.get("table_prompt", "")).resolve())
        self.llm_table_model_path: str = llm_config.get("model_path", "")

    def _validate_paths(self):
        """关键路径存在性校验"""
        if not self.ocr_python:
            raise ValueError("❌ ocr_python 未配置！请检查 settings.toml")

        if not Path(self.ocr_python).exists():
            raise FileNotFoundError(f"❌ OCR环境Python不存在: {self.ocr_python}")

        if not Path(self.ocr_script).exists():
            raise FileNotFoundError(f"❌ OCR脚本不存在: {self.ocr_script}")

        # 自动创建必要目录
        # self.temp_dir.mkdir(parents=True, exist_ok=True)
        # self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_ocr_command(self, image_path: str, output_path: str) -> list:
        """生成跨环境调用命令（llm.py中直接使用）"""
        return [
            self.ocr_python,
            str(self.ocr_script),
            "--image", str(image_path),
            "--output", str(output_path),
            "--gpu" if self.use_gpu else "--cpu",
            f"--gpu-id={self.gpu_id}" if self.use_gpu else ""
        ]


# 全局单例（按需初始化）
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(config_path)
    return _settings


# 便捷导入：from backend.config import settings
settings = get_settings()
