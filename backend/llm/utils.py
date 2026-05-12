import json
import os
import re
from typing import Dict, List, Union


class MedicalTermNormalizer:
    """医学名词统一化器，加载 medical_terms_map.json 并提供标准化方法"""

    def __init__(self, map_file_path: str | None = None):
        self.medical_terms_map = self._load_map(map_file_path)

    def _load_map(self, map_file_path: str | None = None) -> Dict[str, str]:
        if map_file_path is None:
            map_file_path = os.path.join(os.path.dirname(__file__), "medical_terms_map.json")
        if os.path.exists(map_file_path):
            try:
                with open(map_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告：加载医学名词统一表失败 {e}")
                return {}
        return {}

    def normalize_item(self, raw_item: str | None) -> str:
        """将原始 item 名称转换为标准名称，若无映射则保留原值"""
        if raw_item is None or not isinstance(raw_item, str):
            return raw_item if raw_item is not None else ""
        cleaned = raw_item.strip()
        if not cleaned:
            return raw_item
        return self.medical_terms_map.get(cleaned, raw_item)

    def normalize_list(self, items: List[str]) -> List[str]:
        """对字符串列表逐一标准化并去重"""
        normalized = []
        for item in items:
            norm = self.normalize_item(item)
            if norm and norm.strip():
                normalized.append(norm.strip())
        return list(dict.fromkeys(normalized))  # 保序去重


# 模块级单例：供 table_parser、rag 等共用同一张映射表
medical_term_normalizer = MedicalTermNormalizer()


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
