"""
测试 table_html_to_md 函数的重构
覆盖：双栏、HTML 实体、连体表格、colspan、footer 行等特殊情况

使用懒加载导入，避免 OCR 环境依赖问题
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用懒加载导入：只导入轻量级函数，不触发 OCR 环境检查
from backend.ocr.utils import table_html_to_md


def test_double_column_with_split():
    """测试双栏布局且包含连体表格拆分"""
    html = '''
    <table>
    <tr>
        <td>项目名称</td><td>检查结果</td><td>参考值</td><td>单位</td>
        <td>项目名称</td><td>检查结果</td><td>参考值</td><td>单位</td>
    </tr>
    <tr>
        <td>甲胎蛋白</td><td>阴性（-）</td><td>&lt;9.0</td><td></td>
        <td>癌胚抗原</td><td>阴性（-）</td><td>&lt;3.0</td><td></td>
    </tr>
    <tr>
        <td>检验者：崔英玉</td><td></td><td>审核日期：2011-05-23 11:37</td><td></td>
        <td></td><td></td><td>审核者：姚家奎</td><td></td>
    </tr>
    <tr>
        <td colspan="8">肝功能 + 空腹血糖 + 肾功能 + 血脂</td>
    </tr>
    <tr>
        <td>项目名称</td><td>检查结果</td><td>参考值</td><td>单位</td>
        <td>项目名称</td><td>检查结果</td><td>参考值</td><td>单位</td>
    </tr>
    <tr>
        <td>白蛋白</td><td>46</td><td>35-55 g/L</td><td></td>
        <td>总蛋白</td><td>96</td><td>↑ 60-80 g/L</td><td></td>
    </tr>
    <tr>
        <td>低密度脂蛋白胆固醇</td><td>2.36</td><td>2.07-3.36 m</td><td></td>
        <td></td><td></td><td></td><td></td>
    </tr>
    <tr>
        <td>检验者：朱永昌</td><td></td><td>审核日期：2011-05-23 10:47</td><td></td>
        <td></td><td></td><td>审核者：姜润涵</td><td></td>
    </tr>
    </table>
    '''

    result = table_html_to_md(html)

    print("\n=== 测试：双栏布局 + 连体表格拆分 ===")
    
    assert result is not None, "解析失败"
    assert 'table_count' in result
    assert 'tables' in result

    print(f"表格数量：{result['table_count']}")
    assert result['table_count'] == 2, f"期望拆分 2 个表格，实际 {result['table_count']} 个"

    table1 = result['tables'][0]
    print(f"表格 1 双栏：{table1['is_double_column']}")
    assert table1['is_double_column'] is True

    table2 = result['tables'][1]
    print(f"表格 2 双栏：{table2['is_double_column']}")

    segments2 = table2['segments']
    print(f"表格 2 分段：{segments2}")
    title_seg = next((s for s in segments2 if s['type'] == 'title'), None)
    
    if title_seg:
        print(f"表格 2 title: {title_seg['text']}")
        assert '肝功能' in title_seg['text']

    matrix1 = table1['matrix']
    has_less_than = any('<9.0' in cell or '<3.0' in cell for row in matrix1 for cell in row)
    print(f"HTML 实体解码 (<): {has_less_than}")
    assert has_less_than, "HTML 实体 &lt; 应该被解码为 <"

    md2 = table2['markdown']
    print(f"Markdown 包含 title 加粗：{'**肝功能' in md2}")
    assert '**肝功能' in md2 or '肝功能**' in md2

    print("[PASS]\n")
    return True


def test_simple_table():
    """测试简单单栏表格"""
    html = '''
    <table>
    <tr>
        <th>项目名称</th><th>检查结果</th><th>参考值</th><th>单位</th>
    </tr>
    <tr>
        <td>白细胞</td><td>5.2</td><td>3.5-9.5</td><td>10^9/L</td>
    </tr>
    <tr>
        <td>红细胞</td><td>4.8</td><td>4.0-5.5</td><td>10^12/L</td>
    </tr>
    </table>
    '''

    result = table_html_to_md(html)

    print("\n=== 测试：简单单栏表格 ===")
    assert result is not None
    assert result['table_count'] == 1

    table = result['tables'][0]
    print(f"双栏：{table['is_double_column']}")
    assert table['is_double_column'] is False
    print(f"矩阵行数：{len(table['matrix'])}")
    assert len(table['matrix']) == 3

    print("[PASS]\n")
    return True


def test_rowspan_colspan():
    """测试 rowspan 和 colspan 处理"""
    html = '''
    <table>
    <tr>
        <th rowspan="2">项目</th>
        <th colspan="2">结果</th>
    </tr>
    <tr>
        <th>测定值</th><th>参考值</th>
    </tr>
    <tr>
        <td>血糖</td><td>5.6</td><td>3.9-6.1</td>
    </tr>
    </table>
    '''

    result = table_html_to_md(html)

    print("\n=== 测试：rowspan 和 colspan ===")
    assert result is not None

    table = result['tables'][0]
    matrix = table['matrix']

    print(f"矩阵：{matrix}")

    # 验证 rowspan: "项目" 应该占据两行第一列（两行都有值）
    assert matrix[0][0] == '项目'
    assert matrix[1][0] == '项目'

    # 验证 colspan: "结果" 只在左上角 (0,1)，(0,2) 为空（方案 A：不重复填充）
    assert matrix[0][1] == '结果'
    assert matrix[0][2] == ''

    print("[PASS]\n")
    return True


def test_footer_row():
    """测试 footer 行识别"""
    html = '''
    <table>
    <tr>
        <td>项目名称</td><td>检查结果</td><td>参考值</td>
    </tr>
    <tr>
        <td>血糖</td><td>5.6</td><td>3.9-6.1</td>
    </tr>
    <tr>
        <td>检验者：张三</td><td></td><td>审核日期：2024-01-01</td>
    </tr>
    </table>
    '''

    result = table_html_to_md(html)

    print("\n=== 测试：footer 行识别 ===")
    assert result is not None
    
    table = result['tables'][0]
    segments = table['segments']
    print(f"分段：{segments}")

    footer_seg = next((s for s in segments if s['type'] == 'footer'), None)
    print(f"footer 段：{footer_seg}")
    assert footer_seg is not None, "应该识别出 footer 行"

    print("[PASS]\n")
    return True


def test_html_entities():
    """测试 HTML 实体解码"""
    html = '''
    <table>
    <tr>
        <td>项目</td><td>结果</td>
    </tr>
    <tr>
        <td>白细胞</td><td>&lt;5.0</td>
    </tr>
    <tr>
        <td>红细胞</td><td>&gt;4.0</td>
    </tr>
    </table>
    '''

    result = table_html_to_md(html)

    print("\n=== 测试：HTML 实体解码 ===")
    assert result is not None
    
    table = result['tables'][0]
    matrix = table['matrix']
    print(f"矩阵：{matrix}")

    has_less_than = any('<5.0' in row[1] for row in matrix if len(row) > 1)
    has_greater_than = any('>4.0' in row[1] for row in matrix if len(row) > 1)

    print(f"<5.0: {has_less_than}, >4.0: {has_greater_than}")
    assert has_less_than, "应该包含解码后的 <5.0"
    assert has_greater_than, "应该包含解码后的 >4.0"

    print("[PASS]\n")
    return True


def test_empty_input():
    """测试空输入"""
    print("\n=== 测试：空输入 ===")
    assert table_html_to_md('') is None
    assert table_html_to_md(None) is None
    print("[PASS]\n")
    return True


def test_invalid_html():
    """测试无效 HTML"""
    print("\n=== 测试：无效 HTML ===")
    result = table_html_to_md('<div>不是表格</div>')
    print(f"结果：{result}")
    assert result is None
    print("[PASS]\n")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("table_html_to_md 重构测试")
    print("=" * 60)

    tests = [
        ("双栏布局 + 连体表格拆分", test_double_column_with_split),
        ("简单单栏表格", test_simple_table),
        ("rowspan/colspan", test_rowspan_colspan),
        ("footer 行识别", test_footer_row),
        ("HTML 实体解码", test_html_entities),
        ("空输入", test_empty_input),
        ("无效 HTML", test_invalid_html),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {e}\n")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {e}\n")
            failed += 1

    print("=" * 60)
    print(f"测试结果：{passed} 通过，{failed} 失败")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
