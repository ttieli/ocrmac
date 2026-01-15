"""表格恢复模块 - 基于 bbox 坐标重建表格结构"""

import sys
if sys.version_info < (3, 9):
    from typing import List, Dict, Tuple, Optional
else:
    from typing import Optional
    List, Dict, Tuple = list, dict, tuple


class TableCell:
    """表格单元格"""

    def __init__(self, text: str, row: int, col: int, bbox: List[float]):
        self.text = text
        self.row = row
        self.col = col
        self.bbox = bbox  # [x, y, w, h]
        self.rowspan = 1
        self.colspan = 1

    def __repr__(self):
        return f"Cell({self.text[:20]}, r{self.row}c{self.col})"


class Table:
    """表格结构"""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.cells: List[List[Optional[TableCell]]] = [
            [None] * cols for _ in range(rows)
        ]

    def set_cell(self, row: int, col: int, cell: TableCell):
        """设置单元格"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.cells[row][col] = cell

    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """获取单元格"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.cells[row][col]
        return None

    def to_markdown(self) -> str:
        """转换为 Markdown 表格"""
        if self.rows == 0 or self.cols == 0:
            return ""

        lines = []

        # 表头（第一行）
        header = []
        for col in range(self.cols):
            cell = self.get_cell(0, col)
            text = cell.text if cell else ""
            header.append(text)
        lines.append("| " + " | ".join(header) + " |")

        # 分隔线
        lines.append("| " + " | ".join(["---"] * self.cols) + " |")

        # 数据行
        for row in range(1, self.rows):
            row_data = []
            for col in range(self.cols):
                cell = self.get_cell(row, col)
                text = cell.text if cell else ""
                row_data.append(text)
            lines.append("| " + " | ".join(row_data) + " |")

        return "\n".join(lines)

    def __repr__(self):
        return f"Table({self.rows}x{self.cols})"


class TableRowDetector:
    """表格行检测器 - 基于 y 坐标聚类"""

    def __init__(self, y_tolerance=0.015):
        """
        初始化行检测器

        Args:
            y_tolerance: y 坐标容忍度（同一行内允许的 y 差异）
        """
        self.y_tolerance = y_tolerance

    def detect_rows(self, bboxes_with_text: List[Tuple]) -> List[Dict]:
        """
        检测表格行

        Args:
            bboxes_with_text: [(text, confidence, [x, y, w, h]), ...]

        Returns:
            行列表: [{'y_avg': 0.5, 'items': [...]}, ...]
        """
        if not bboxes_with_text:
            return []

        # 按 y 坐标排序（从上到下 = y_center 从大到小，Vision 坐标系）
        sorted_items = sorted(bboxes_with_text, key=lambda item: self._get_y_center(item[2]), reverse=True)

        rows = []
        for item in sorted_items:
            text, conf, bbox = item
            y_center = self._get_y_center(bbox)

            # 查找是否属于已有行
            found_row = False
            for row in rows:
                if abs(y_center - row['y_avg']) <= self.y_tolerance:
                    row['items'].append(item)
                    # 更新行的平均 y
                    row['y_avg'] = sum(self._get_y_center(it[2]) for it in row['items']) / len(row['items'])
                    found_row = True
                    break

            if not found_row:
                # 创建新行
                rows.append({
                    'y_avg': y_center,
                    'items': [item],
                })

        # 按 y 坐标排序行（从上到下 = y_avg 从大到小）
        rows.sort(key=lambda r: r['y_avg'], reverse=True)

        # 在每行内按 x 坐标排序
        for row in rows:
            row['items'].sort(key=lambda item: item[2][0])  # 按 x 排序

        return rows

    def _get_y_center(self, bbox):
        """获取 bbox 中心 y 坐标（Vision 坐标系）"""
        x, y, w, h = bbox
        return y + h / 2  # Vision 坐标系


class TableColumnDetector:
    """表格列检测器 - 基于 x 坐标对齐"""

    def __init__(self, x_tolerance=0.025, alignment_ratio=0.6):
        """
        初始化列检测器

        Args:
            x_tolerance: x 坐标容忍度
            alignment_ratio: 列对齐率阈值（至少多少比例的行对齐才算一列）
        """
        self.x_tolerance = x_tolerance
        self.alignment_ratio = alignment_ratio

    def detect_columns(self, rows: List[Dict]) -> List[float]:
        """
        检测表格列边界

        Args:
            rows: 行列表（来自 TableRowDetector）

        Returns:
            列边界的 x 坐标列表（排序）
        """
        if not rows:
            return []

        # 收集所有 x 坐标（左边界）
        all_x_positions = []
        for row in rows:
            for item in row['items']:
                bbox = item[2]
                all_x_positions.append(bbox[0])  # x

        # 聚类 x 坐标
        column_boundaries = self._cluster_positions(all_x_positions, self.x_tolerance)

        # 验证对齐率
        valid_columns = []
        for x_boundary in column_boundaries:
            aligned_count = sum(
                1 for row in rows
                if any(abs(item[2][0] - x_boundary) <= self.x_tolerance for item in row['items'])
            )
            alignment_rate = aligned_count / len(rows)

            if alignment_rate >= self.alignment_ratio:
                valid_columns.append(x_boundary)

        return sorted(valid_columns)

    def _cluster_positions(self, positions: List[float], tolerance: float) -> List[float]:
        """
        聚类坐标点

        Args:
            positions: 坐标列表
            tolerance: 容忍度

        Returns:
            聚类中心列表
        """
        if not positions:
            return []

        sorted_positions = sorted(positions)
        clusters = []
        current_cluster = [sorted_positions[0]]

        for pos in sorted_positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                # 保存当前簇的中心
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [pos]

        # 添加最后一个簇
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters


class TableDetector:
    """表格检测器 - 综合检测和重建表格"""

    def __init__(
        self,
        y_tolerance=0.015,
        x_tolerance=0.025,
        min_rows=2,
        min_cols=2,
    ):
        """
        初始化表格检测器

        Args:
            y_tolerance: 行检测 y 容忍度
            x_tolerance: 列检测 x 容忍度
            min_rows: 最少行数
            min_cols: 最少列数
        """
        self.row_detector = TableRowDetector(y_tolerance)
        self.col_detector = TableColumnDetector(x_tolerance)
        self.min_rows = min_rows
        self.min_cols = min_cols

    def detect(self, ocr_results: List[Tuple]) -> Optional[Table]:
        """
        检测表格

        Args:
            ocr_results: OCR 结果列表

        Returns:
            Table 对象，如果未检测到表格则返回 None
        """
        if not ocr_results:
            return None

        # 1. 检测行
        rows = self.row_detector.detect_rows(ocr_results)

        if len(rows) < self.min_rows:
            return None

        # 2. 检测列
        column_boundaries = self.col_detector.detect_columns(rows)

        if len(column_boundaries) < self.min_cols:
            return None

        # 3. 构建表格结构
        table = Table(rows=len(rows), cols=len(column_boundaries))

        for row_idx, row in enumerate(rows):
            for item in row['items']:
                text, conf, bbox = item
                x = bbox[0]

                # 找到所属列
                col_idx = self._find_column_index(x, column_boundaries)

                if col_idx is not None:
                    cell = TableCell(text, row_idx, col_idx, bbox)
                    table.set_cell(row_idx, col_idx, cell)

        return table

    def detect_all(self, ocr_results: List[Tuple]) -> List[Table]:
        """
        检测所有表格（支持多表格）

        Args:
            ocr_results: OCR 结果列表

        Returns:
            Table 列表
        """
        # 简化版：假设整个区域是一个表格
        # 未来可扩展：空间分割、多表格检测
        table = self.detect(ocr_results)
        return [table] if table else []

    def _find_column_index(self, x: float, column_boundaries: List[float]) -> Optional[int]:
        """
        找到 x 坐标对应的列索引

        Args:
            x: x 坐标
            column_boundaries: 列边界列表

        Returns:
            列索引
        """
        for idx, boundary in enumerate(column_boundaries):
            if abs(x - boundary) <= self.col_detector.x_tolerance:
                return idx

        # 如果没有精确匹配，找最接近的
        if column_boundaries:
            distances = [abs(x - boundary) for boundary in column_boundaries]
            return distances.index(min(distances))

        return None


class TableFormatter:
    """表格格式化工具"""

    @staticmethod
    def to_markdown(table: Table) -> str:
        """转换为 Markdown 表格"""
        return table.to_markdown()

    @staticmethod
    def to_html(table: Table) -> str:
        """转换为 HTML 表格"""
        lines = ['<table>']

        for row in range(table.rows):
            lines.append('  <tr>')
            for col in range(table.cols):
                cell = table.get_cell(row, col)
                text = cell.text if cell else ""
                tag = 'th' if row == 0 else 'td'
                lines.append(f'    <{tag}>{text}</{tag}>')
            lines.append('  </tr>')

        lines.append('</table>')
        return '\n'.join(lines)

    @staticmethod
    def to_json(table: Table) -> dict:
        """转换为 JSON 结构"""
        data = {
            'rows': table.rows,
            'cols': table.cols,
            'cells': []
        }

        for row in range(table.rows):
            row_data = []
            for col in range(table.cols):
                cell = table.get_cell(row, col)
                row_data.append({
                    'text': cell.text if cell else "",
                    'bbox': cell.bbox if cell else None,
                })
            data['cells'].append(row_data)

        return data
