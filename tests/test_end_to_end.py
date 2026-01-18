#!/usr/bin/env python3
"""
端到端测试 - 真实 OCR + 表格恢复 + 文章分段
需要在 macOS 上运行（依赖 Vision/LiveText）
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ocrmac import OCR
    from ocrmac.table_recovery import TableDetector
    from ocrmac.layout_analyzer import LayoutAnalyzer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先安装: pip install -e .")
    sys.exit(1)


def test_simple_text():
    """测试1: 简单文本识别"""
    print("\n" + "="*60)
    print("测试 1: 简单文本识别")
    print("="*60)

    # 检查测试图片
    test_image = Path(__file__).parent / "test_images" / "simple_text.png"
    if not test_image.exists():
        print(f"⚠️  测试图片不存在: {test_image}")
        print("   请准备一张包含简单文本的图片")
        return False

    try:
        # OCR 识别
        print(f"📸 正在识别: {test_image.name}")
        ocr = OCR(str(test_image), framework='livetext', unit='line')
        results = ocr.recognize()

        # 输出结果
        print(f"✅ 识别完成，共 {len(results)} 行文本")
        print("\n识别的文本:")
        print("-" * 60)
        for i, (text, conf, bbox) in enumerate(results, 1):
            print(f"{i}. {text} (置信度: {conf:.2f})")
        print("-" * 60)

        return len(results) > 0

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_table_detection():
    """测试2: 表格识别和恢复"""
    print("\n" + "="*60)
    print("测试 2: 表格识别和恢复")
    print("="*60)

    test_image = Path(__file__).parent / "test_images" / "table.png"
    if not test_image.exists():
        print(f"⚠️  测试图片不存在: {test_image}")
        print("   请准备一张包含表格的图片（推荐: 2x3 或 3x3 的简单表格）")
        return False

    try:
        # OCR 识别
        print(f"📸 正在识别: {test_image.name}")
        ocr = OCR(str(test_image), framework='livetext', unit='line')
        results = ocr.recognize()

        print(f"✅ OCR 完成，共识别 {len(results)} 个文本块")

        # 显示识别的文本和坐标
        print("\n识别的文本块:")
        print("-" * 60)
        for i, (text, conf, bbox) in enumerate(results, 1):
            x, y, w, h = bbox
            print(f"{i}. '{text}' @ [{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}]")
        print("-" * 60)

        # 表格检测
        print("\n🔍 正在检测表格结构...")
        detector = TableDetector(
            y_tolerance=0.015,
            x_tolerance=0.025,
            min_rows=2,
            min_cols=2,
        )
        table = detector.detect(results)

        if table:
            print(f"✅ 检测到表格: {table.rows} 行 x {table.cols} 列")
            print("\n生成的 Markdown 表格:")
            print("-" * 60)
            print(table.to_markdown())
            print("-" * 60)
            return True
        else:
            print("⚠️  未检测到表格结构")
            print("   可能原因:")
            print("   1. 表格单元格未对齐")
            print("   2. 行列数不足（< 2）")
            print("   3. 需要调整容忍度参数")

            # 尝试放宽参数
            print("\n🔄 尝试放宽检测参数...")
            detector2 = TableDetector(
                y_tolerance=0.03,
                x_tolerance=0.05,
                min_rows=2,
                min_cols=2,
            )
            table2 = detector2.detect(results)
            if table2:
                print(f"✅ 放宽参数后检测到表格: {table2.rows} 行 x {table2.cols} 列")
                print(table2.to_markdown())
                return True

            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_paragraph_detection():
    """测试3: 多段落文档分析"""
    print("\n" + "="*60)
    print("测试 3: 多段落文档分析")
    print("="*60)

    test_image = Path(__file__).parent / "test_images" / "multi_paragraph.png"
    if not test_image.exists():
        print(f"⚠️  测试图片不存在: {test_image}")
        print("   请准备一张包含多段落的文档图片（至少 3 段）")
        return False

    try:
        # OCR 识别
        print(f"📸 正在识别: {test_image.name}")
        ocr = OCR(str(test_image), framework='livetext', unit='line')
        results = ocr.recognize()

        print(f"✅ OCR 完成，共识别 {len(results)} 行")

        # 布局分析
        print("\n🔍 正在分析文档结构...")
        analyzer = LayoutAnalyzer(
            line_spacing_threshold=1.5,
            heading_size_threshold=1.3,
        )
        layout = analyzer.analyze(results)

        paragraphs = layout['paragraphs']
        print(f"✅ 检测到 {len(paragraphs)} 个段落")

        # 统计
        headings = [p for p in paragraphs if p.get('is_heading')]
        lists = [p for p in paragraphs if p.get('is_list')]

        print(f"   - 标题: {len(headings)} 个")
        print(f"   - 列表: {len(lists)} 个")
        print(f"   - 普通段落: {len(paragraphs) - len(headings) - len(lists)} 个")

        # 输出 Markdown
        markdown = analyzer.to_markdown(layout)
        print("\n生成的 Markdown:")
        print("-" * 60)
        print(markdown)
        print("-" * 60)

        return len(paragraphs) > 0

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heading_and_list():
    """测试4: 标题和列表识别"""
    print("\n" + "="*60)
    print("测试 4: 标题和列表识别")
    print("="*60)

    test_image = Path(__file__).parent / "test_images" / "heading_list.png"
    if not test_image.exists():
        print(f"⚠️  测试图片不存在: {test_image}")
        print("   请准备一张包含标题和列表的图片")
        return False

    try:
        # OCR 识别
        print(f"📸 正在识别: {test_image.name}")
        ocr = OCR(str(test_image), framework='livetext', unit='line')
        results = ocr.recognize()

        print(f"✅ OCR 完成，共识别 {len(results)} 行")

        # 布局分析
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(results)

        paragraphs = layout['paragraphs']

        # 显示识别结果
        print("\n识别的结构:")
        print("-" * 60)
        for i, para in enumerate(paragraphs, 1):
            text = para['text'][:50] + '...' if len(para['text']) > 50 else para['text']

            if para.get('is_heading'):
                level = para.get('heading_level', 1)
                print(f"{i}. [标题 H{level}] {text}")
            elif para.get('is_list'):
                list_type = para.get('list_type', 'unknown')
                print(f"{i}. [列表-{list_type}] {text}")
            else:
                print(f"{i}. [段落] {text}")
        print("-" * 60)

        return len(paragraphs) > 0

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_content():
    """测试5: 混合内容（表格 + 段落）"""
    print("\n" + "="*60)
    print("测试 5: 混合内容（表格 + 段落）")
    print("="*60)

    test_image = Path(__file__).parent / "test_images" / "mixed_content.png"
    if not test_image.exists():
        print(f"⚠️  测试图片不存在: {test_image}")
        print("   请准备一张包含表格和文本段落的图片")
        return False

    try:
        # OCR 识别
        print(f"📸 正在识别: {test_image.name}")
        ocr = OCR(str(test_image), framework='livetext', unit='line')
        results = ocr.recognize()

        print(f"✅ OCR 完成，共识别 {len(results)} 个文本块")

        # 尝试表格检测
        print("\n🔍 尝试检测表格...")
        detector = TableDetector()
        table = detector.detect(results)

        if table:
            print(f"✅ 检测到表格: {table.rows} 行 x {table.cols} 列")
            print(table.to_markdown())
        else:
            print("⚠️  未检测到明显的表格结构")

        # 段落分析
        print("\n🔍 分析文档结构...")
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(results)

        paragraphs = layout['paragraphs']
        print(f"✅ 检测到 {len(paragraphs)} 个文本区域")

        # 输出结果
        markdown = analyzer.to_markdown(layout)
        print("\n完整的 Markdown 输出:")
        print("-" * 60)
        print(markdown)
        print("-" * 60)

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 OCR 升级功能端到端测试")
    print("="*60)
    print("\n⚠️  注意: 需要在 macOS 上运行（依赖 Vision/LiveText）\n")

    # 检查是否在 macOS 上
    if sys.platform != 'darwin':
        print("❌ 当前系统不是 macOS")
        print(f"   检测到: {sys.platform}")
        print("\n   这些测试需要 Apple Vision/LiveText 框架，只能在 macOS 上运行。")
        print("   请在 macOS 设备上运行此测试脚本。")
        sys.exit(1)

    # 检查测试图片目录
    test_images_dir = Path(__file__).parent / "test_images"
    if not test_images_dir.exists():
        print(f"📁 创建测试图片目录: {test_images_dir}")
        test_images_dir.mkdir(parents=True, exist_ok=True)
        print("\n⚠️  请将测试图片放入以下目录:")
        print(f"   {test_images_dir}")
        print("\n需要的测试图片:")
        print("   1. simple_text.png - 简单文本")
        print("   2. table.png - 表格（2-3行x2-3列）")
        print("   3. multi_paragraph.png - 多段落文档")
        print("   4. heading_list.png - 标题和列表")
        print("   5. mixed_content.png - 混合内容（可选）")
        print("\n准备好后再次运行此脚本。")
        sys.exit(0)

    # 运行测试
    results = {}

    tests = [
        ("简单文本识别", test_simple_text),
        ("表格识别和恢复", test_table_detection),
        ("多段落文档分析", test_paragraph_detection),
        ("标题和列表识别", test_heading_and_list),
        ("混合内容", test_mixed_content),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 发生异常: {e}")
            results[name] = False

    # 总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}  {name}")

    print("-" * 60)
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
