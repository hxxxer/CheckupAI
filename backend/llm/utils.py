import json
import re
from typing import Union


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
