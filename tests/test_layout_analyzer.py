"""测试布局分析功能"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocrmac.layout_analyzer import (
    ParagraphDetector,
    HeadingDetector,
    ListDetector,
    LayoutAnalyzer,
)


def test_paragraph_detection():
    """测试段落检测"""
    # 模拟 OCR 结果（3 行，分成 2 段）
    # Vision 坐标系: y 是底部坐标, y+h 是顶部坐标, 从上往下 y 递减
    ocr_results = [
        # 第一段（2 行，行间距小）
        ("第一段第一行", 1.0, [0.1, 0.87, 0.8, 0.03]),  # y=0.87, h=0.03 -> top=0.90
        ("第一段第二行", 1.0, [0.1, 0.83, 0.8, 0.03]),  # y=0.83, h=0.03 -> top=0.86, spacing=0.01

        # 第二段（1 行，与第一段间距大）
        ("第二段第一行", 1.0, [0.1, 0.67, 0.8, 0.03]),  # y=0.67, h=0.03 -> top=0.70, spacing=0.13
    ]

    detector = ParagraphDetector(line_spacing_threshold=1.5)
    paragraphs = detector.detect_paragraphs(ocr_results)

    assert len(paragraphs) == 2, f"应检测到 2 个段落，实际 {len(paragraphs)} 个"
    assert len(paragraphs[0]['lines']) == 2, "第一段应有 2 行"
    assert len(paragraphs[1]['lines']) == 1, "第二段应有 1 行"

    print("✅ 段落检测测试通过")


def test_heading_detection():
    """测试标题检测"""
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
    result = detector.detect_headings(paragraphs, avg_height)

    assert result[0]['is_heading'] is True, "第一段应被识别为标题"
    assert result[0]['heading_level'] == 1, "应为一级标题"
    assert result[1]['is_heading'] is False, "第二段不应是标题"

    print("✅ 标题检测测试通过")


def test_list_detection():
    """测试列表检测"""
    paragraphs = [
        {'text': '• 列表项 1'},
        {'text': '1. 有序列表项'},
        {'text': '普通段落'},
    ]

    detector = ListDetector()
    result = detector.detect_lists(paragraphs)

    assert result[0]['is_list'] is True
    assert result[0]['list_type'] == 'unordered'

    assert result[1]['is_list'] is True
    assert result[1]['list_type'] == 'ordered'

    assert result[2]['is_list'] is False

    print("✅ 列表检测测试通过")


def test_layout_analyzer():
    """测试综合布局分析"""
    ocr_results = [
        # 标题（大字体）
        ("文档标题", 1.0, [0.1, 0.95, 0.8, 0.06]),

        # 第一段
        ("第一段第一行", 1.0, [0.1, 0.85, 0.8, 0.03]),
        ("第一段第二行", 1.0, [0.1, 0.81, 0.8, 0.03]),

        # 列表
        ("• 列表项 1", 1.0, [0.1, 0.7, 0.8, 0.03]),
        ("• 列表项 2", 1.0, [0.1, 0.66, 0.8, 0.03]),

        # 第二段
        ("第二段内容", 1.0, [0.1, 0.5, 0.8, 0.03]),
    ]

    analyzer = LayoutAnalyzer(
        line_spacing_threshold=1.5,
        heading_size_threshold=1.3,
    )
    result = analyzer.analyze(ocr_results)

    paragraphs = result['paragraphs']

    # 验证段落数量
    assert len(paragraphs) >= 3, f"至少应有 3 个段落，实际 {len(paragraphs)} 个"

    # 验证标题
    headings = [p for p in paragraphs if p.get('is_heading')]
    assert len(headings) >= 1, "至少应检测到 1 个标题"

    # 验证列表
    lists = [p for p in paragraphs if p.get('is_list')]
    assert len(lists) >= 1, "至少应检测到列表项"

    print("✅ 综合布局分析测试通过")


def test_to_markdown():
    """测试 Markdown 输出"""
    ocr_results = [
        ("一级标题", 1.0, [0.1, 0.95, 0.8, 0.08]),
        ("普通段落文本", 1.0, [0.1, 0.8, 0.8, 0.03]),
        ("• 列表项", 1.0, [0.1, 0.6, 0.8, 0.03]),
    ]

    analyzer = LayoutAnalyzer()
    result = analyzer.analyze(ocr_results)
    markdown = analyzer.to_markdown(result)

    assert "# 一级标题" in markdown or "## 一级标题" in markdown, "应包含标题"
    assert "普通段落文本" in markdown, "应包含段落文本"
    assert "• 列表项" in markdown, "应包含列表"

    print("✅ Markdown 输出测试通过")
    print("\n生成的 Markdown:")
    print("=" * 60)
    print(markdown)
    print("=" * 60)


if __name__ == '__main__':
    print("🧪 运行布局分析测试...\n")

    try:
        test_paragraph_detection()
        test_heading_detection()
        test_list_detection()
        test_layout_analyzer()
        test_to_markdown()

        print("\n✅ 所有测试通过！")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
