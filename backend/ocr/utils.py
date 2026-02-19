"""
表格处理工具模块
提供 HTML 表格清洗和 Markdown 转换的通用工具函数
可被多个 OCR 模型复用
"""

from typing import Optional
from bs4 import BeautifulSoup


def table_html_clean(table_html: str) -> str:
    """
    清洗 HTML 中的转义字符，将其转换为正常字符

    Args:
        table_html: 原始 HTML 字符串

    Returns:
        清洗后的 HTML 字符串
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


def table_html_to_md(table_html: str) -> Optional[str]:
    """
    将 HTML 表格转换为 Markdown 格式

    Args:
        table_html: HTML 表格字符串

    Returns:
        Markdown 格式表格，如果解析失败返回 None
    """
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
        rows.append(cells)

    md_lines.append('| ' + ' | '.join(headers) + ' |')
    md_lines.append('| ' + ' | '.join([' --- ' for _ in headers]) + ' |')
    for row in rows:
        md_lines.append('| ' + ' | '.join(row) + ' |')

    md = '\n'.join(md_lines)

    return md
