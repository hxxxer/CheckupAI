import gc
import json
import re
import os
import subprocess
from datetime import datetime
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Union
# from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from backend.settings import settings


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

    tables_data = []
    for idx, table_block in enumerate(table_blocks):
        try:
            table_html = table_block.get("block_content")
            table_html = table_html_clean(table_html)
            table_md = table_html_to_md(table_html)
            tables_data.append(table_md_to_json(table_md))
        except Exception as e:
            print(f"⚠️ 表格块 {idx} 解析失败: {str(e)}")
            raise

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
        r'\\times ': '× ',        # 乘号
        r'\\mu ': 'μ ',           # 删除\m
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


def table_md_to_json(table_md: str) -> Union[dict, list, None]:
    model_path = "/root/autodl-tmp/models/Qwen/Qwen3-8B-AWQ"

    llm = LLM(
        model=model_path,
        dtype="float16",
        quantization="awq",
        # gpu_memory_utilization=0.8,
        max_model_len=16384,
        enforce_eager=False,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    prompt = """
# Role
你是一个专业的医疗数据结构化助手。你的任务是从体检报告的 OCR 识别结果（Markdown 格式）中提取关键指标，并将其转换为标准的 JSON 格式。
# Rules
1. **严格输出 JSON**：只返回 JSON 代码块，不要包含任何解释性文字、Markdown 标记（如 ```json）等内容。
2. **噪音过滤**：请忽略所有非检查数据的信息，包括但不限于：
   - 报告标题、医院名称、体检号、条形码。
   - 页眉、页码、打印时间。
   - 审核医生、检验者、报告日期、送检日期等。
3. **语义推断**：
   - 如果提取到的文本中包含箭头符号（↑ ↓ + -），将其提取到 `is_abnormal` 字段中。
   - 如果数据没有单位，`unit` 填 null。
# Output Schema
请输出一个包含以下结构的 JSON 列表：
[
  {
    "item_name": "检查项目名称 (如: 白细胞)",
    "result": "检查结果 (字符串，保留原始符号)",
    "unit": "单位 (如: g/L，若无则为 null)",
    "reference_range": "参考范围 (如: 3.5-9.5，若无则为null)",
    "is_abnormal": "异常标记 (如有箭头则保留箭头字符，如 '↑'，否则为 null)"
  }
]
# Examples
## Example 1 (标准单栏表格，包括大部分正常情况)
### Input Markdown:
| 项目名称 | 检查结果 | 单位 | 参考范围 |
|---|---|---|---|
| 红细胞 | ↑ 6 | 10^12/L | 4.0-5.5 |
| 血红蛋白 | 135 | g/L | 120-160 |
| 透明度 | 透明 |  |  |
### Output JSON:
[
  {"item_name": "红细胞", "result": "6", "unit": "10^12/L", "reference_range": "4.0-5.5", "is_abnormal": "↑"},
  {"item_name": "血红蛋白", "result": "135", "unit": "g/L", "reference_range": "120-160", "is_abnormal": null},
  {"item_name": "透明度", "result": "透明", "unit": null, "reference_range": null, "is_abnormal": null}
]
## Example 2 (包含噪音的脏数据)
### Input Markdown:
| 项目名称 | 检查结果 | 单位 | 参考范围 |
|---|---|---|---|
| 蛋白质 | ↑ 45 | U/L | 0-40 |
| 白细胞 | 28 | U/L | 0-40 |
| 审核医生：张三 |  | 报告日期：2023-10-22 14:00:00 |  |
| 第 1 页 / 共 2 页 |  |  |  |
### Output JSON:
[
  {"item_name": "蛋白质", "result": "45", "unit": "U/L", "reference_range": "0-40", "is_abnormal": "↑"},
  {"item_name": "白细胞", "result": "28", "unit": "U/L", "reference_range": "0-40", "is_abnormal": null}
]
## Example 3 (双栏表格)
### Input Markdown:
| 项目名称 | 检查结果 | 单位 | 参考值 | 项目名称 | 检查结果 | 单位 | 参考值 |
|  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |
| 尿胆原 | 阴性 |  | 阴性 | 维生素C | 1.2 | mmol/L | 0.7-2.0 |
| 葡萄糖 | 阴性 |  | 阴性 | 酸碱度 | ↓ 4.2 |  | 4.5-8.0 |

### Output JSON:
[
  {"item_name": "尿胆原", "result": "阴性", "unit": null, "reference_range": "阴性", "is_abnormal": null},
  {"item_name": "维生素C", "result": "1.2", "unit": "mmol/L", "reference_range": "0.7-2.0", "is_abnormal": null},
  {"item_name": "葡萄糖", "result": "阴性", "unit": null, "reference_range": "阴性", "is_abnormal": null},
  {"item_name": "酸碱度", "result": "4.2", "unit": null, "reference_range": "4.5-8.0", "is_abnormal": "↓"}
]
# Real Task
请根据上述规则，处理以下真实的体检报告 Markdown 内容：
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": table_md}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        max_tokens=8192,
        stop=["<|im_end|>", "<|endoftext|>"]  # 设置停止词
    )

    outputs = llm.generate([text], sampling_params)
    content = outputs[0].outputs[0].text

    print("提取结果:", content)
    parsed_result = safe_json_parse(content)

    if parsed_result is None:
        return []
    # elif isinstance(parsed_result, list):
    #     return {"items": parsed_result}
    else:
        return parsed_result


def safe_json_parse(text: str) -> Union[dict, list, None]:
    """
    安全地解析JSON，包含多种清理策略
    """
    if not text:
        return None

    # 清理文本
    cleaned_text = text.strip()

    # 移除常见的前缀/后缀
    prefixes_to_remove = ['```json', '```', 'json']
    suffixes_to_remove = ['```']

    for prefix in prefixes_to_remove:
        if cleaned_text.startswith(prefix):
            cleaned_text = cleaned_text[len(prefix):].strip()
            break

    for suffix in suffixes_to_remove:
        if cleaned_text.endswith(suffix):
            cleaned_text = cleaned_text[:-len(suffix)].strip()
            break

    # 尝试直接解析
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # 尝试提取数组部分
    array_matches = re.findall(r'\[[\s\S]*?\]', cleaned_text)
    if array_matches:
        try:
            return json.loads(array_matches[-1])  # 使用最后一个匹配的数组
        except json.JSONDecodeError:
            pass

    # 尝试提取对象部分
    object_matches = re.findall(r'\{[\s\S]*?\}', cleaned_text)
    if object_matches:
        try:
            return json.loads(object_matches[-1])
        except json.JSONDecodeError:
            pass

    print(f"无法解析JSON: {text[:100]}...")
    return None


def run_ocr(input_path):
    # 1. 主程序自己决定输出路径（比如带时间戳）
    timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    output_json_path = settings.project_root / f"data/sensitive/ocr_output/{timestamp}/"

    # 2. 调用子进程
    result = subprocess.run(
        [settings.ocr_python, settings.ocr_script, input_path, output_json_path],
        capture_output=True,
        text=True
    )

    # 3. 检查是否成功
    # if result.returncode != 0:
    #     raise RuntimeError(f"OCR failed: {result.stderr.strip()}")

    # 4. 返回 JSON 路径（主程序完全掌控）
    return output_json_path


if __name__ == "__main__":
    output_json_path = run_ocr(settings.project_root / "tests/test_ocr/cam2/4.jpg")

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
