#!/usr/bin/env python
"""调试段落检测"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocrmac.layout_analyzer import ParagraphDetector

# 测试数据
ocr_results = [
    ("第一段第一行", 1.0, [0.1, 0.87, 0.8, 0.03]),  # y=0.87, h=0.03 -> top=0.90
    ("第一段第二行", 1.0, [0.1, 0.83, 0.8, 0.03]),  # y=0.83, h=0.03 -> top=0.86
    ("第二段第一行", 1.0, [0.1, 0.67, 0.8, 0.03]),  # y=0.67, h=0.03 -> top=0.70
]

detector = ParagraphDetector(line_spacing_threshold=1.5)

# 按 y 排序
sorted_lines = sorted(ocr_results, key=lambda r: detector._get_y_top(r[2]))
print("排序后的行:")
for i, line in enumerate(sorted_lines):
    text, _, bbox = line
    y_top = detector._get_y_top(bbox)
    y_bottom = detector._get_y_bottom(bbox)
    print(f"  {i}: '{text}' y_top={y_top:.3f}, y_bottom={y_bottom:.3f}, h={bbox[3]:.3f}")

# 计算平均行高
avg_height = detector._calculate_avg_height(sorted_lines)
print(f"\n平均行高: {avg_height:.3f}")
print(f"分段阈值: {avg_height * detector.line_spacing_threshold:.3f}")

# 计算间距
print("\n行间距:")
for i in range(1, len(sorted_lines)):
    prev_y_bottom = detector._get_y_bottom(sorted_lines[i-1][2])
    curr_y_top = detector._get_y_top(sorted_lines[i][2])
    spacing = curr_y_top - prev_y_bottom
    should_split = detector._should_split_paragraph(spacing, avg_height)
    print(f"  行{i-1}→行{i}: spacing={spacing:.3f}, should_split={should_split}")

# 执行检测
paragraphs = detector.detect_paragraphs(ocr_results)
print(f"\n检测结果: {len(paragraphs)} 个段落")
for i, para in enumerate(paragraphs):
    print(f"段落 {i+1}: {len(para['lines'])} 行 - {para['text']}")
