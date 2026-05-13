"""
测试 MedicalTermNormalizer — 医学名词统一化
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_normalize_item_with_map():
    """测试从 medical_terms_map.json 加载并标准化"""
    from backend.llm.utils import MedicalTermNormalizer

    normalizer = MedicalTermNormalizer()

    # 映射表中的标准对照
    assert normalizer.normalize_item("血糖") == "葡萄糖"
    assert normalizer.normalize_item("空腹血糖") == "空腹葡萄糖"
    assert normalizer.normalize_item("血压") == "血压测量值"
    assert normalizer.normalize_item("收缩压") == "收缩压"  # 自身为标准名
    assert normalizer.normalize_item("白细胞") == "白细胞计数"
    assert normalizer.normalize_item("ALT") == "丙氨酸氨基转移酶"


def test_normalize_item_no_map():
    """测试未在映射表中的词保持原样"""
    from backend.llm.utils import MedicalTermNormalizer

    normalizer = MedicalTermNormalizer()

    assert normalizer.normalize_item("支原体抗体") == "支原体抗体"
    assert normalizer.normalize_item("癌胚抗原") == "癌胚抗原"


def test_normalize_item_edge_cases():
    """边界情况：None、空字符串、空白"""
    from backend.llm.utils import MedicalTermNormalizer

    normalizer = MedicalTermNormalizer()

    assert normalizer.normalize_item(None) == ""
    assert normalizer.normalize_item("") == ""
    assert normalizer.normalize_item("   ") == "   "  # 纯空白保留原值


def test_normalize_item_strip():
    """前后空格应被 trim"""
    from backend.llm.utils import MedicalTermNormalizer

    normalizer = MedicalTermNormalizer()

    assert normalizer.normalize_item(" 血糖 ") == "葡萄糖"
    assert normalizer.normalize_item("  ALT  ") == "丙氨酸氨基转移酶"


def test_normalize_list():
    """批量标准化 + 去重"""
    from backend.llm.utils import MedicalTermNormalizer

    normalizer = MedicalTermNormalizer()

    result = normalizer.normalize_list(["血糖", "空腹血糖", "白细胞", "血糖", "血压"])
    # 去重后按首次出现顺序
    assert result == ["葡萄糖", "空腹葡萄糖", "白细胞计数", "血压测量值"]


def test_normalize_list_handles_empty_strings():
    """空字符串应被过滤"""
    from backend.llm.utils import MedicalTermNormalizer

    normalizer = MedicalTermNormalizer()

    result = normalizer.normalize_list(["血糖", "", "  ", "白细胞"])
    assert result == ["葡萄糖", "白细胞计数"]


def test_normalize_list_empty():
    """空列表返回空列表"""
    from backend.llm.utils import MedicalTermNormalizer

    normalizer = MedicalTermNormalizer()
    assert normalizer.normalize_list([]) == []


def test_custom_map_path():
    """使用自定义映射表"""
    import tempfile
    from backend.llm.utils import MedicalTermNormalizer

    custom_map = {"A": "Alpha", "B": "Beta"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(custom_map, f)
        tmp_path = f.name

    try:
        normalizer = MedicalTermNormalizer(map_file_path=tmp_path)
        assert normalizer.normalize_item("A") == "Alpha"
        assert normalizer.normalize_item("B") == "Beta"
        assert normalizer.normalize_item("C") == "C"  # 无映射保留
    finally:
        os.unlink(tmp_path)


def test_module_singleton():
    """测试模块级单例 medical_term_normalizer"""
    from backend.llm.utils import medical_term_normalizer

    assert medical_term_normalizer is not None
    assert len(medical_term_normalizer.medical_terms_map) > 0
    # 验证一致性：两次调用同一实例
    result1 = medical_term_normalizer.normalize_item("血糖")
    result2 = medical_term_normalizer.normalize_item("血糖")
    assert result1 == result2 == "葡萄糖"


def test_map_loaded():
    """验证 map 实际加载了数据"""
    from backend.llm.utils import medical_term_normalizer

    assert len(medical_term_normalizer.medical_terms_map) >= 20, (
        f"映射表应包含至少20个映射，实际 {len(medical_term_normalizer.medical_terms_map)}"
    )
