"""
体检报告 OCR 解析模块
封装对 PaddleOCRRunner 的调用，提供高层业务接口
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.ocr import PaddleOCRRunner


def parse_ocr_result(runner: PaddleOCRRunner, ocr_output: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    解析 OCR 输出结果（委托给 PaddleOCRRunner 处理）

    Args:
        runner: PaddleOCRRunner 实例（包含 table_parser）
        ocr_output: OCR 输出数据（已加载的 JSON）

    Returns:
        包含 tables、full_text 和 stats 的字典
    """
    return runner.parse_result(ocr_output)


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
    from backend.instances import table_parser

    # 创建 runner 并注入 table_parser
    runner = PaddleOCRRunner(table_parser=table_parser)

    output_json_path = run_ocr(
        settings.project_root / "tests/test_ocr/cam2/4.jpg",
        runner=runner
    )

    dates = []
    for filename in os.listdir(output_json_path):
        if filename.endswith('.json'):
            file_path = os.path.join(output_json_path, filename)
            dates.append(json.load(open(file_path)))
            print(f"成功读取：{filename}")

    # 核心：解析结构化数据
    try:
        structured_data = runner.parse_result(dates)

        print("="*50)
        print(f"📊 共解析 {structured_data['stats']['parsed_tables']} 个表格")
        print("="*50)

        # 表格数据示例（供后续标准化/画像生成）
        if structured_data["tables"]:
            print("\n【表格数据示例】")
            sample_row = structured_data["tables"][0][0]
            print(f"首行：{sample_row}")

        # 文本段落示例（供 NER/LLM 处理）
        print("\n【文本段落】")
        print(f"\n{structured_data['full_text']}\n")
    except Exception as e:
        print(f"❌ 解析失败：{str(e)}")
        import traceback
        traceback.print_exc()
