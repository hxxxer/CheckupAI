import json
import os
from dataclasses import asdict
from datetime import datetime

from backend.config import settings
from backend.ocr import parse_checkup

test_files_folder = os.path.join(settings.project_root, "tests/test_ocr/test_files/")
test_output_folder = os.path.join(settings.project_root, "tests/test_ocr/test_output/", datetime.now().strftime("%Y%m%d_%H%M%S"))

if not os.path.isdir(test_files_folder):
    raise FileNotFoundError(f"测试文件夹不存在：{test_files_folder}")

# valid_extensions = {'.png', '.jpg', '.jpeg', '.pdf'}
# files = [f for f in os.listdir(test_files_folder) if os.path.isfile(os.path.join(test_files_folder, f))]
# valid_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

# if not valid_files:
#     raise ValueError(f"测试文件夹下未找到图片或 PDF 文件：{test_files_folder}")

structured_data = parse_checkup(test_files_folder)

for i in len(structured_data):
    try:
        dict_data = structured_data[i].asdict()
        with open(os.path.join(test_output_folder, f"{i}"), "w", encoding="utf-8") as f:
            json.dump(dict_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error processing file {i}: {e}")
