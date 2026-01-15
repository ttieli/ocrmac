#!/usr/bin/env python
"""调试标题检测"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocrmac.layout_analyzer import HeadingDetector

# 模拟段落（包含标题）
paragraphs = [
    # 标题（字体大）
    {
        'lines': [("这是标题", 1.0, [0.1, 0.9, 0.8, 0.06])],  # h=0.06（大字体）
        'text': "这是标题",
    },
    # 普通段落
    {
        'lines': [("这是正文", 1.0, [0.1, 0.8, 0.8, 0.03])],  # h=0.03（普通字体）
        'text': "这是正文",
    },
]

avg_height = 0.03  # 平均行高
detector = HeadingDetector(size_threshold=1.3)

print(f"平均行高: {avg_height}")
print(f"标题阈值: {avg_height * detector.size_threshold}")
print(f"\n段落1 高度: {paragraphs[0]['lines'][0][2][3]}")
print(f"段落2 高度: {paragraphs[1]['lines'][0][2][3]}")

result = detector.detect_headings(paragraphs, avg_height)

print(f"\n检测结果:")
for i, para in enumerate(result):
    print(f"段落 {i+1}:")
    print(f"  is_heading: {para.get('is_heading')}")
    print(f"  heading_level: {para.get('heading_level')}")
    print(f"  高度比率: {paragraphs[i]['lines'][0][2][3] / avg_height:.2f}")
