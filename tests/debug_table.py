#!/usr/bin/env python
"""调试表格检测"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocrmac.table_recovery import TableDetector

ocr_results = [
    # 表头
    ("姓名", 1.0, [0.1, 0.9, 0.1, 0.05]),
    ("年龄", 1.0, [0.3, 0.9, 0.1, 0.05]),
    # 第一行数据
    ("张三", 1.0, [0.1, 0.7, 0.1, 0.05]),
    ("25", 1.0, [0.3, 0.7, 0.1, 0.05]),
    # 第二行数据
    ("李四", 1.0, [0.1, 0.5, 0.1, 0.05]),
    ("30", 1.0, [0.3, 0.5, 0.1, 0.05]),
]

detector = TableDetector()
table = detector.detect(ocr_results)

if table:
    print(f"表格检测成功: {table.rows} 行 x {table.cols} 列")
    print("\n单元格内容:")
    for r in range(table.rows):
        for c in range(table.cols):
            cell = table.get_cell(r, c)
            if cell:
                print(f"  [{r},{c}]: '{cell.text}'")
            else:
                print(f"  [{r},{c}]: None")
else:
    print("未检测到表格")
