"""
体检报告 OCR 解析模块
封装对 PaddleOCRRunner 的调用，提供高层业务接口
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.llm import text_analyzer
from backend.ocr import PaddleOCRRunner


def parse_checkup(input_path: str, runner: Optional[PaddleOCRRunner] = None):
    """
    执行体检报告 OCR 解析

    Args:
        input_path: 输入图片路径
        runner: PaddleOCRRunner 运行实例，默认创建新实例

    Returns:
        包含 tables、full_text 和 stats 的字典
    """
    if runner is None:
        runner = PaddleOCRRunner()
    output_path = runner.run(input_path)

    dates = runner.load_result(output_path)

    structured_data = runner.parse_result(dates)

    text_results = text_analyzer.analyze(structured_data)


def run_ocr(input_path: str, runner: Optional[PaddleOCRRunner] = None) -> str:
    """
    执行 OCR 处理

    Args:
        input_path: 输入图片路径
        runner: PaddleOCRRunner 实例，默认创建新实例

    Returns:
        输出 JSON 目录路径
    """
    if runner is None:
        runner = PaddleOCRRunner()
    output_json_path = runner.run(input_path)
    return output_json_path


# 测试
if __name__ == "__main__":
    runner = PaddleOCRRunner()

    input_path = settings.project_root / "tests/test_ocr/cam2/4.jpg"
    output_path = runner.run(input_path)

    dates = runner.load_result(output_path)

    # 核心：解析结构化数据
    try:
        structured_data = runner.parse_result(dates)

        print("="*50)
        print(f"📊 共解析 {len(structured_data)} 个文件")
        print("="*50)

        print(f"✅ 结构化数据：{structured_data}")

        # # 表格数据示例（供后续标准化/画像生成）
        # if structured_data["tables"]:
        #     print("\n【表格数据示例】")
        #     sample_row = structured_data["tables"][0][0]
        #     print(f"首行：{sample_row}")

        # # 文本段落示例（供 NER/LLM 处理）
        # print("\n【文本段落】")
        # print(f"\n{structured_data['full_text']}\n")
    except Exception as e:
        print(f"❌ 解析失败：{str(e)}")
        import traceback
        traceback.print_exc()
