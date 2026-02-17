import json
import os
from datetime import datetime
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Union
from backend.config import settings
from backend.instances import table_parser
from backend.ocr import PaddleOCRRunner


def parse_ocr_result(ocr_output: list) -> Dict[str, Any]:
    if not ocr_output:
        raise ValueError("OCR输出为空")

    # 单图处理
    page_data = ocr_output[0].get("res", ocr_output[0])
    blocks = page_data.get("parsing_res_list", [])

    text_blocks = []
    table_blocks = []
    for block in blocks:
        if block.get("block_label") == "text":
            text_blocks.append(block)
        elif block.get("block_label") == "table":
            table_blocks.append(block)

    table_parser.wake_up()
    tables_data = []
    for idx, table_block in enumerate(table_blocks):
        try:
            table_html = table_block.get("block_content")
            table_html = table_html_clean(table_html)
            table_md = table_html_to_md(table_html)
            tables_data.append(table_parser.parse(table_md))
        except Exception as e:
            print(f"⚠️ 表格块 {idx} 解析失败: {str(e)}")
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


def table_html_clean(table_html: str) -> str:
    """
    清洗HTML中的转义字符，将其转换为正常字符
    """
    if not table_html:
        return table_html

    # 定义转义字符映射表
    escape_map = {
        r'\\uparrow ': '↑ ',      # 上箭头
        r'\\downarrow ': '↓ ',    # 下箭头
        r'\\times ': ' × ',        # 乘号
        r'\\mu ': 'μ',           # 删除\m
    }

    # 逐个替换转义字符
    cleaned_html = table_html
    for escape_seq, normal_char in escape_map.items():
        cleaned_html = cleaned_html.replace(escape_seq, normal_char)

    return cleaned_html


def table_html_to_md(table_html: str) -> str:
    soup = BeautifulSoup(table_html, 'lxml')
    table = soup.find('table')

    if not table or not table.find('tr'):
        return None

    md_lines = []
    header_row = table.find('tr')
    raw_headers = [th.get_text(strip=True)
                   for th in header_row.find_all(['td', 'th'])]

    # 标准化表头别名（关键！统一后续判断基准）
    header_map = {
        '项目': '项目名称', '检验项目': '项目名称', '指标': '项目名称', '检查项目': '项目名称',
        '结果': '检查结果', '测定值': '检查结果', '实测值': '检查结果',
        '参考范围': '参考值', '正常值': '参考值', '参考区间': '参考值',
        '单位': '单位', '计量单位': '单位'
    }
    headers = [header_map.get(h, h) for h in raw_headers]

    rows = []
    for tr in table.find_all('tr')[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        # row_dict = dict(zip(headers, cells))
        rows.append(cells)

    md_lines.append('| ' + ' | '.join(headers) + ' |')
    md_lines.append('| ' + ' | '.join([' --- ' for _ in headers]) + ' |')
    for row in rows:
        md_lines.append('| ' + ' | '.join(row) + ' |')

    md = '\n'.join(md_lines)

    return md


def run_ocr(input_path):
    """
    执行 OCR 处理
    
    Args:
        input_path: 输入图片路径
        
    Returns:
        输出 JSON 目录路径
    """
    runner = PaddleOCRRunner()
    output_json_path = runner.run(input_path)
    return output_json_path


# 测试
if __name__ == "__main__":
    output_json_path = run_ocr(
        settings.project_root / "tests/test_ocr/cam2/4.jpg")

    dates = []
    for filename in os.listdir(output_json_path):
        if filename.endswith('.json'):
            file_path = os.path.join(output_json_path, filename)

            # with open(file_path, 'r', encoding='utf-8') as f:
            #     data = json.load(f)
            #     json_data_list.append(data)
            dates.append(json.load(open(file_path)))
            print(f"成功读取: {filename}")
    # 核心：解析结构化数据
    try:
        structured_data = parse_ocr_result(dates)

        print("="*50)
        print(f"📊 共解析 {structured_data['stats']['parsed_tables']} 个表格")
        print("="*50)

        # 表格数据示例（供后续标准化/画像生成）
        if structured_data["tables"]:
            print("\n【表格数据示例】")
            sample_row = structured_data["tables"][0][0]
            print(f"首行: {sample_row}")

        # 文本段落示例（供NER/LLM处理）
        print("\n【文本段落】")
        print(f"\n{structured_data['full_text']}\n")
    except Exception as e:
        print(f"❌ 解析失败: {str(e)}")
        import traceback
        traceback.print_exc()
