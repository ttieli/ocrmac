#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强型 OCR 演示 - 展示表格恢复和段落检测功能

使用方法:
    python examples/enhanced_ocr_demo.py image.png
    python examples/enhanced_ocr_demo.py table.png --enable-table
    python examples/enhanced_ocr_demo.py document.png --detect-paragraphs
"""

import sys
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocrmac import OCR
from ocrmac.layout_analyzer import LayoutAnalyzer
from ocrmac.table_recovery import TableDetector, TableFormatter


def main():
    parser = argparse.ArgumentParser(description='增强型 OCR - 支持表格恢复和段落检测')
    parser.add_argument('image', type=str, help='图片路径')
    parser.add_argument('--enable-table', action='store_true', help='启用表格检测')
    parser.add_argument('--detect-paragraphs', action='store_true', help='启用段落检测')
    parser.add_argument('-l', '--language', type=str, default='zh-Hans', help='语言偏好')
    parser.add_argument('--framework', type=str, default='livetext', choices=['vision', 'livetext'])
    parser.add_argument('-o', '--output', type=str, help='输出文件路径')

    args = parser.parse_args()

    print(f"📷 处理图片: {args.image}")
    print(f"🔧 OCR 框架: {args.framework}")
    print(f"🌐 语言: {args.language}")

    # 执行 OCR
    print("\n⏳ 执行 OCR...")
    try:
        # 使用 line-level 输出（适合段落检测）
        unit = 'line' if args.framework == 'livetext' else 'token'
        ocr = OCR(
            args.image,
            framework=args.framework,
            language_preference=[args.language] if args.language else None,
            detail=True,
            unit=unit,
        )
        results = ocr.recognize()
        print(f"✅ OCR 完成，识别了 {len(results)} 个文本块")

    except Exception as e:
        print(f"❌ OCR 失败: {e}")
        return 1

    output_lines = []

    # 表格检测
    if args.enable_table:
        print("\n📊 检测表格...")
        try:
            detector = TableDetector(
                y_tolerance=0.015,
                x_tolerance=0.025,
                min_rows=2,
                min_cols=2,
            )
            tables = detector.detect_all(results)

            if tables:
                print(f"✅ 检测到 {len(tables)} 个表格")
                for idx, table in enumerate(tables):
                    print(f"   表格 {idx + 1}: {table.rows} 行 x {table.cols} 列")
                    output_lines.append(f"\n## 表格 {idx + 1}\n")
                    output_lines.append(table.to_markdown())
                    output_lines.append("\n")
            else:
                print("⚠️  未检测到表格")
                # 回退到普通文本
                output_lines.append("\n".join(r[0] for r in results))

        except Exception as e:
            print(f"❌ 表格检测失败: {e}")
            import traceback
            traceback.print_exc()

    # 段落检测
    elif args.detect_paragraphs:
        print("\n📝 检测段落...")
        try:
            analyzer = LayoutAnalyzer(
                line_spacing_threshold=1.5,
                heading_size_threshold=1.3,
            )
            layout = analyzer.analyze(results)

            paragraphs = layout['paragraphs']
            print(f"✅ 检测到 {len(paragraphs)} 个段落")

            # 统计
            headings = sum(1 for p in paragraphs if p.get('is_heading'))
            lists = sum(1 for p in paragraphs if p.get('is_list'))
            print(f"   - 标题: {headings}")
            print(f"   - 列表: {lists}")
            print(f"   - 普通段落: {len(paragraphs) - headings - lists}")

            # 转换为 Markdown
            markdown_output = analyzer.to_markdown(layout)
            output_lines.append(markdown_output)

        except Exception as e:
            print(f"❌ 段落检测失败: {e}")
            import traceback
            traceback.print_exc()

    # 默认：简单文本输出
    else:
        print("\n📄 提取文本（无结构分析）")
        text_output = "\n".join(r[0] for r in results)
        output_lines.append(text_output)

    # 输出结果
    final_output = "\n".join(output_lines)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_output, encoding='utf-8')
        print(f"\n💾 结果已保存到: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("📋 OCR 结果:")
        print("=" * 60)
        print(final_output)
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
