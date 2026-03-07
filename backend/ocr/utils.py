"""
表格处理工具模块
提供 HTML 表格清洗和 Markdown 转换的通用工具函数
可被多个 OCR 模型复用
"""

import html
from typing import Optional, Dict, List, Any, Tuple
from bs4 import BeautifulSoup, Tag


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
        r'\uparrow ': '↑ ',      # 上箭头
        r'\downarrow ': '↓ ',    # 下箭头
        r'\times ': ' × ',        # 乘号
        r'\mu ': 'μ',           # 删除\m
    }

    # 逐个替换转义字符
    cleaned_html = table_html
    for escape_seq, normal_char in escape_map.items():
        cleaned_html = cleaned_html.replace(escape_seq, normal_char)

    return cleaned_html


def _build_matrix(table: Tag) -> List[List[str]]:
    """
    解析 HTML 表格，构建二维矩阵，处理 rowspan 和 colspan

    Args:
        table: BeautifulSoup 的 table 标签

    Returns:
        二维矩阵，每个元素是单元格文本
    """
    rows = table.find_all('tr')
    if not rows:
        return []

    # 先确定矩阵的最大行列数
    max_cols = 0
    for tr in rows:
        col_count = 0
        for cell in tr.find_all(['td', 'th']):
            colspan = int(cell.get('colspan', 1))
            col_count += colspan
        max_cols = max(max_cols, col_count)

    # 初始化矩阵（用空字符串填充）
    matrix: List[List[str]] = [['' for _ in range(max_cols)] for _ in range(len(rows))]

    # 记录每行每列是否被占用（处理 rowspan）
    occupied: List[List[bool]] = [[False for _ in range(max_cols)] for _ in range(len(rows))]

    for row_idx, tr in enumerate(rows):
        col_idx = 0
        for cell in tr.find_all(['td', 'th']):
            # 跳过被 rowspan 占用的位置
            while col_idx < max_cols and occupied[row_idx][col_idx]:
                col_idx += 1

            if col_idx >= max_cols:
                break

            # 获取单元格属性
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            text = html.unescape(cell.get_text(strip=True))

            # 方案 A：rowspan 向下重复填充，colspan 不向右填充
            # rowspan 需要填充：因为每行都需要这个值
            # colspan 不填充：因为内容只在左侧显示
            for r in range(row_idx, min(row_idx + rowspan, len(rows))):
                matrix[r][col_idx] = text

            # 标记所有被占据的位置为 occupied（包括自身）
            for r in range(row_idx, min(row_idx + rowspan, len(rows))):
                for c in range(col_idx, min(col_idx + colspan, max_cols)):
                    occupied[r][c] = True

            col_idx += colspan

    return matrix


def _is_header_row(row: List[str]) -> bool:
    """
    判断某行是否为 header 行（通过关键词匹配）

    Args:
        row: 行数据列表

    Returns:
        是否为 header 行
    """
    header_keywords = ['项目名称', '检验项目', '检查项目', '指标', '项目',
                       '检查结果', '测定值', '实测值', '结果',
                       '参考值', '参考范围', '正常值', '参考区间',
                       '单位', '计量单位']

    row_text = ''.join(row).lower()
    # 至少匹配 2 个关键词才认为是 header
    match_count = sum(1 for kw in header_keywords if kw in row_text)
    return match_count >= 2


def _is_footer_row(row: List[str]) -> bool:
    """
    判断某行是否为 footer 行（检验者、审核者等）
    使用两部分组合匹配：(动作/角色词) + (后缀词)

    Args:
        row: 行数据列表

    Returns:
        是否为 footer 行
    """
    # 第一部分：动作/角色词
    action_keywords = ['检验', '检查', '审核', '报告', '核对', '医师', '医生']
    
    # 第二部分：后缀词
    suffix_keywords = ['者', '医生', '日期', '时间']
    
    row_text = ''.join(row)
    
    # 检查是否同时包含两部分关键词（组合匹配）
    has_action = any(kw in row_text for kw in action_keywords)
    has_suffix = any(kw in row_text for kw in suffix_keywords)
    
    return has_action and has_suffix


def _is_colspan_separator_row(row: List[str], matrix_row_idx: int, matrix: List[List[str]]) -> bool:
    """
    判断某行是否为 colspan 分隔行（title 行）

    Args:
        row: 行数据列表
        matrix_row_idx: 在矩阵中的行索引
        matrix: 完整矩阵

    Returns:
        是否为分隔行

    判断依据（优先级从高到低）：
    1. 结构特征：只有一个非空单元格（colspan 展开后）
    2. 上下文特征：下一行是 header 行（强特征）
    3. 内容特征：包含医疗检查类别关键词
    """
    non_empty = [cell for cell in row if cell.strip()]

    # 结构特征：应该只有一个非空单元格（colspan 展开后）
    if len(non_empty) != 1:
        return False

    separator_text = non_empty[0]

    # 上下文特征：检查下一行是否为 header 行（强特征）
    next_row_idx = matrix_row_idx + 1
    if next_row_idx < len(matrix):
        next_row = matrix[next_row_idx]
        if _is_header_row(next_row):
            # 下一行是 header，当前行很可能是 title
            # 排除明显不是 title 的情况（如 footer 行）
            if not _is_footer_row(row):
                return True

    # 内容特征：包含医疗检查类别关键词（辅助判断）
    title_keywords = ['血常规', '尿常规', '肝功能', '肾功能', '血糖', '血脂',
                      '心电图', '胸片', 'CT', 'B 超', '肿瘤标志物',
                      '激素', '免疫', '凝血', '生化', '肝炎', '乙肝',
                      '甲功', '糖化', '离子', '心肌酶', '贫血']
    return any(kw in separator_text for kw in title_keywords)


def _detect_segments(matrix: List[List[str]]) -> List[Dict[str, Any]]:
    """
    使用状态机检测表格的分段信息

    Args:
        matrix: 二维矩阵

    Returns:
        分段信息列表
    """
    if not matrix:
        return []

    segments: List[Dict[str, Any]] = []
    row_idx = 0

    # 状态：INIT -> DETECT_TITLE -> DETECT_HEADER -> IN_DATA
    state = 'INIT'

    while row_idx < len(matrix):
        row = matrix[row_idx]

        if state == 'INIT':
            # 检查是否是 title 行（colspan 分隔行）
            if _is_colspan_separator_row(row, row_idx, matrix):
                # title 文本只取第一个单元格（避免 colspan 重复）
                segments.append({
                    'type': 'title',
                    'row_index': row_idx,
                    'text': row[0].strip() if row else ''
                })
                state = 'DETECT_TITLE'
                row_idx += 1
            else:
                state = 'DETECT_HEADER'

        elif state == 'DETECT_TITLE':
            # title 行之后应该是 header
            if _is_header_row(row):
                segments.append({'type': 'header', 'row_index': row_idx})
                state = 'IN_DATA'
                row_idx += 1
            else:
                # 如果 title 后不是 header，可能是数据行
                state = 'IN_DATA'

        elif state == 'DETECT_HEADER':
            if _is_header_row(row):
                segments.append({'type': 'header', 'row_index': row_idx})
                state = 'IN_DATA'
                row_idx += 1
            elif _is_footer_row(row):
                # 特殊情况：没有 header 直接 footer
                segments.append({'type': 'footer', 'row_index': row_idx})
                row_idx += 1
            else:
                # 默认当作数据行
                segments.append({'type': 'data', 'row_index': row_idx})
                row_idx += 1

        elif state == 'IN_DATA':
            # 检查是否遇到新的分隔行（连体表格拆分点）
            if _is_colspan_separator_row(row, row_idx, matrix):
                # title 文本只取第一个单元格（避免 colspan 重复）
                segments.append({
                    'type': 'title',
                    'row_index': row_idx,
                    'text': row[0].strip() if row else '',
                    'split_table': True  # 标记需要拆分
                })
                row_idx += 1
            elif _is_header_row(row):
                # 遇到新的 header，可能是新表格
                segments.append({
                    'type': 'header',
                    'row_index': row_idx,
                    'new_table': True  # 标记可能是新表格
                })
                row_idx += 1
            elif _is_footer_row(row):
                segments.append({'type': 'footer', 'row_index': row_idx})
                row_idx += 1
            else:
                segments.append({'type': 'data', 'row_index': row_idx})
                row_idx += 1

    return segments


def _split_tables_by_segments(matrix: List[List[str]], segments: List[Dict[str, Any]]) -> List[Tuple[List[List[str]], List[Dict[str, Any]]]]:
    """
    根据分段信息拆分连体表格

    Args:
        matrix: 完整矩阵
        segments: 分段信息

    Returns:
        列表，每个元素是 (子矩阵，子分段)
    """
    if not matrix:
        return []

    # 找到所有拆分点
    split_points = []
    for i, seg in enumerate(segments):
        if seg.get('split_table') or seg.get('new_table'):
            split_points.append((i, seg['row_index']))

    if not split_points:
        # 没有拆分点，返回完整表格
        return [(matrix, segments)]

    tables = []
    start_row = 0

    for seg_idx, split_row in split_points:
        # 找到 split_table 类型的 segment，它应该是新表格的 title
        seg = segments[seg_idx]
        
        # 如果是 split_table，表示这行是 colspan 分隔行，应该作为新表格的 title
        if seg.get('split_table'):
            # 当前表格不包含 title 行
            end_row = split_row
            if end_row > start_row:
                sub_matrix = matrix[start_row:end_row]
                sub_segments = [
                    {**s, 'row_index': s['row_index'] - start_row}
                    for s in segments if start_row <= s['row_index'] < end_row
                ]
                tables.append((sub_matrix, sub_segments))
            
            # 新表格从 title 行开始
            start_row = split_row
        else:
            # new_table: 只是新的 header，不需要拆分
            pass

    # 处理最后一个表格（包含 title 行）
    if start_row < len(matrix):
        sub_matrix = matrix[start_row:]
        sub_segments = [
            {**s, 'row_index': s['row_index'] - start_row}
            for s in segments if s['row_index'] >= start_row
        ]
        # 将第一个 segment 标记为 title（如果它是 split_table）
        if sub_segments and sub_segments[0].get('split_table'):
            sub_segments[0]['type'] = 'title'
            if 'split_table' in sub_segments[0]:
                del sub_segments[0]['split_table']
        tables.append((sub_matrix, sub_segments))

    return tables


def _is_double_column(matrix: List[List[str]]) -> bool:
    """
    判断表格是否为双栏布局

    Args:
        matrix: 二维矩阵

    Returns:
        是否为双栏布局
    """
    if not matrix:
        return False

    # 双栏特征：列数 >= 6（通常左右各 3-4 列）
    # 且中间列与两侧列有明显的"重复 header"模式
    first_row = matrix[0] if matrix else []
    if len(first_row) < 6:
        return False

    # 检查是否有重复的 header 模式
    # 例如：[项目名称，检查结果，参考值，单位，项目名称，检查结果，参考值，单位]
    mid = len(first_row) // 2
    left_part = first_row[:mid]
    right_part = first_row[mid:]

    # 简化判断：如果左右两侧有相同的关键词
    left_text = ' '.join(left_part)
    right_text = ' '.join(right_part)

    common_keywords = ['项目名称', '结果', '参考值', '单位']
    left_matches = sum(1 for kw in common_keywords if kw in left_text)
    right_matches = sum(1 for kw in common_keywords if kw in right_text)

    return left_matches >= 2 and right_matches >= 2


def _matrix_to_html(matrix: List[List[str]], segments: List[Dict[str, Any]]) -> str:
    """
    将矩阵和分段信息转换为 HTML 表格字符串（不包含 footer 行）

    Args:
        matrix: 二维矩阵
        segments: 分段信息

    Returns:
        HTML 表格字符串
    """
    if not matrix:
        return ''

    html_lines = ['<table>']

    for row_idx, row in enumerate(matrix):
        seg = next((s for s in segments if s['row_index'] == row_idx), None)

        if seg and seg['type'] == 'title':
            # title 行：使用 colspan 跨越所有列
            title_text = seg.get('text', row[0].strip() if row else '')
            # 转义 HTML 特殊字符
            title_text = title_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_lines.append(f'<tr><td colspan="{len(row)}">{title_text}</td></tr>')

        elif seg and seg['type'] == 'header':
            # header 行：使用 th 标签
            cells = []
            for cell in row:
                # 转义 HTML 特殊字符
                cell_escaped = cell.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                cells.append(f'<th>{cell_escaped}</th>')
            html_lines.append(f'<tr>{"".join(cells)}</tr>')

        elif seg and seg['type'] == 'data':
            # 数据行：使用 td 标签（跳过 footer）
            cells = []
            for cell in row:
                # 转义 HTML 特殊字符
                cell_escaped = cell.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                cells.append(f'<td>{cell_escaped}</td>')
            html_lines.append(f'<tr>{"".join(cells)}</tr>')

        # footer 行：跳过不输出

    html_lines.append('</table>')
    return ''.join(html_lines)


def _matrix_to_markdown(matrix: List[List[str]], segments: List[Dict[str, Any]]) -> str:
    """
    将矩阵和分段信息转换为 Markdown 格式（不包含 footer 行）

    Args:
        matrix: 二维矩阵
        segments: 分段信息

    Returns:
        Markdown 字符串
    """
    if not matrix:
        return ''

    md_lines = []
    header_found = False
    header_row_idx = -1

    # 先找到 header 行
    for seg in segments:
        if seg['type'] == 'header':
            header_row_idx = seg['row_index']
            header_found = True
            break

    for row_idx, row in enumerate(matrix):
        seg = next((s for s in segments if s['row_index'] == row_idx), None)

        if seg and seg['type'] == 'title':
            # 使用 segment 中存储的 text，而不是 join 整行（避免 colspan 重复）
            title_text = seg.get('text', row[0].strip() if row else '')
            md_lines.append(f"**{title_text}**")
            md_lines.append('')  # 空行分隔

        elif seg and seg['type'] == 'header':
            # Markdown 表头
            md_lines.append('| ' + ' | '.join(row) + ' |')
            md_lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')

        elif seg and seg['type'] == 'data':
            # 数据行（跳过 footer）
            if header_found:
                md_lines.append('| ' + ' | '.join(row) + ' |')
            elif row_idx == 0 and not header_found:
                # 如果没有 header，第一行当作 header
                md_lines.append('| ' + ' | '.join(row) + ' |')
                md_lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')
                header_found = True

    return '\n'.join(md_lines)


def table_html_to_md(table_html: str) -> Optional[Dict[str, Any]]:
    """
    将 HTML 表格转换为结构化数据和 Markdown/HTML 格式

    Args:
        table_html: HTML 表格字符串

    Returns:
        字典，包含：
        - table_count: 表格数量
        - tables: 表格列表，每个表格包含：
          - matrix: 二维矩阵
          - segments: 分段信息
          - is_double_column: 是否双栏
          - markdown: Markdown 字符串（不含 footer 行）
          - html: HTML 表格字符串（不含 footer 行）

        如果解析失败返回 None
    """
    if not table_html:
        return None

    soup = BeautifulSoup(table_html, 'lxml')
    table = soup.find('table')

    if not table or not table.find('tr'):
        return None

    # 步骤 1: 构建矩阵（处理 rowspan/colspan）
    matrix = _build_matrix(table)
    if not matrix:
        return None

    # 步骤 2: 检测分段
    segments = _detect_segments(matrix)

    # 步骤 3: 拆分连体表格
    sub_tables = _split_tables_by_segments(matrix, segments)

    # 步骤 4: 构建返回结果
    tables_result = []
    for sub_matrix, sub_segments in sub_tables:
        tables_result.append({
            'matrix': sub_matrix,
            'segments': sub_segments,
            'is_double_column': _is_double_column(sub_matrix),
            'markdown': _matrix_to_markdown(sub_matrix, sub_segments),
            'html': _matrix_to_html(sub_matrix, sub_segments)
        })

    return {
        'table_count': len(tables_result),
        'tables': tables_result
    }
