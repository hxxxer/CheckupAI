import json
import os
from dataclasses import asdict
from datetime import datetime

from backend.config import settings
from backend.ocr import parse_checkup


class DateTimeEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，支持 datetime 对象序列化"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


test_files_folder = os.path.join(settings.project_root, "tests/test_ocr/test_files/")
test_output_folder = os.path.join(settings.project_root, "tests/test_ocr/test_output/", datetime.now().strftime("%Y%m%d_%H%M%S"))

os.makedirs(test_output_folder, exist_ok=True)

if not os.path.isdir(test_files_folder):
    raise FileNotFoundError(f"测试文件夹不存在：{test_files_folder}")

try:
    structured_data = parse_checkup(test_files_folder)
    
    for i in range(len(structured_data)):
        try:
            dict_data = asdict(structured_data[i])
            output_file = os.path.join(test_output_folder, f"{i}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dict_data, f, ensure_ascii=False, indent=4, cls=DateTimeEncoder)
            print(f"成功保存文件：{output_file}")
        except Exception as e:
            print(f"Error processing file {i}: {e}")
            
except Exception as e:
    print(f"解析过程出错：{e}")
    raise