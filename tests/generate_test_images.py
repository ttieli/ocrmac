#!/usr/bin/env python3
"""
生成测试图片

如果你没有准备测试图片，可以运行此脚本生成示例图片。
这些图片可以用于快速验证功能，但建议使用真实文档进行最终测试。
"""

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("❌ 需要安装 Pillow: pip install Pillow")
    sys.exit(1)


def get_font(size):
    """获取字体"""
    # 尝试使用系统字体
    font_paths = [
        # macOS 中文字体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        # macOS 英文字体
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux 字体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue

    # 降级使用默认字体
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except:
        return ImageFont.load_default()


def generate_simple_text(output_path):
    """生成简单文本图片"""
    print(f"📝 生成: {output_path.name}")

    # 创建白色背景
    img = Image.new('RGB', (800, 400), 'white')
    draw = ImageDraw.Draw(img)

    # 设置字体
    font = get_font(36)

    # 绘制文本
    texts = [
        "这是第一行文本",
        "这是第二行文本",
        "这是第三行文本",
        "This is the fourth line",
    ]

    y = 50
    for text in texts:
        draw.text((50, y), text, fill='black', font=font)
        y += 80

    # 保存
    img.save(output_path)
    print(f"   ✅ 已保存")


def generate_table(output_path):
    """生成表格图片"""
    print(f"📊 生成: {output_path.name}")

    # 创建白色背景
    img = Image.new('RGB', (600, 400), 'white')
    draw = ImageDraw.Draw(img)

    # 字体
    font = get_font(32)

    # 表格数据
    table_data = [
        ["姓名", "年龄", "城市"],
        ["张三", "25", "北京"],
        ["李四", "30", "上海"],
    ]

    # 绘制表格
    cell_width = 180
    cell_height = 80
    start_x = 40
    start_y = 60

    for row_idx, row in enumerate(table_data):
        for col_idx, cell in enumerate(row):
            x = start_x + col_idx * cell_width
            y = start_y + row_idx * cell_height

            # 绘制单元格边框
            draw.rectangle(
                [x, y, x + cell_width, y + cell_height],
                outline='black',
                width=2
            )

            # 绘制文本（居中）
            bbox = draw.textbbox((0, 0), cell, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x + (cell_width - text_width) // 2
            text_y = y + (cell_height - text_height) // 2

            draw.text((text_x, text_y), cell, fill='black', font=font)

    img.save(output_path)
    print(f"   ✅ 已保存")


def generate_multi_paragraph(output_path):
    """生成多段落文档图片"""
    print(f"📄 生成: {output_path.name}")

    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)

    font = get_font(28)

    paragraphs = [
        "第一段的第一行内容。\n第一段的第二行内容。",
        "第二段的第一行内容。\n第二段的第二行内容。\n第二段的第三行内容。",
        "第三段的内容。",
    ]

    y = 50
    for para in paragraphs:
        lines = para.split('\n')
        for line in lines:
            draw.text((50, y), line, fill='black', font=font)
            y += 45
        y += 40  # 段落间距

    img.save(output_path)
    print(f"   ✅ 已保存")


def generate_heading_list(output_path):
    """生成标题和列表图片"""
    print(f"📋 生成: {output_path.name}")

    img = Image.new('RGB', (700, 500), 'white')
    draw = ImageDraw.Draw(img)

    # 标题字体（大）
    heading_font = get_font(48)
    # 普通字体
    normal_font = get_font(32)

    # 标题
    draw.text((50, 50), "购物清单", fill='black', font=heading_font)

    # 列表
    list_items = [
        "• 苹果",
        "• 香蕉",
        "• 橙子",
        "• 葡萄",
    ]

    y = 150
    for item in list_items:
        draw.text((70, y), item, fill='black', font=normal_font)
        y += 70

    img.save(output_path)
    print(f"   ✅ 已保存")


def generate_mixed_content(output_path):
    """生成混合内容图片"""
    print(f"🎨 生成: {output_path.name}")

    img = Image.new('RGB', (800, 700), 'white')
    draw = ImageDraw.Draw(img)

    heading_font = get_font(42)
    normal_font = get_font(28)

    # 标题
    draw.text((50, 40), "月度报告", fill='black', font=heading_font)

    # 段落
    draw.text((50, 130), "以下是本月销售数据：", fill='black', font=normal_font)

    # 小表格
    table_data = [
        ["姓名", "销售额"],
        ["张三", "10000"],
        ["李四", "12000"],
    ]

    cell_width = 200
    cell_height = 70
    start_x = 50
    start_y = 220

    for row_idx, row in enumerate(table_data):
        for col_idx, cell in enumerate(row):
            x = start_x + col_idx * cell_width
            y = start_y + row_idx * cell_height

            draw.rectangle(
                [x, y, x + cell_width, y + cell_height],
                outline='black',
                width=2
            )

            bbox = draw.textbbox((0, 0), cell, font=normal_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x + (cell_width - text_width) // 2
            text_y = y + (cell_height - text_height) // 2

            draw.text((text_x, text_y), cell, fill='black', font=normal_font)

    # 结尾段落
    draw.text((50, 520), "整体表现良好，继续保持。", fill='black', font=normal_font)

    img.save(output_path)
    print(f"   ✅ 已保存")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎨 生成测试图片")
    print("="*60)
    print("\n⚠️  注意: 这些是程序生成的示例图片")
    print("   建议使用真实文档进行最终测试。\n")

    # 创建目录
    output_dir = Path(__file__).parent / "test_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 输出目录: {output_dir}\n")

    # 生成图片
    generators = [
        ("simple_text.png", generate_simple_text),
        ("table.png", generate_table),
        ("multi_paragraph.png", generate_multi_paragraph),
        ("heading_list.png", generate_heading_list),
        ("mixed_content.png", generate_mixed_content),
    ]

    for filename, generator in generators:
        output_path = output_dir / filename
        try:
            generator(output_path)
        except Exception as e:
            print(f"   ❌ 生成失败: {e}")

    print("\n" + "="*60)
    print("✅ 测试图片生成完成")
    print("="*60)
    print(f"\n图片位置: {output_dir}")
    print("\n现在可以运行: python3 test_end_to_end.py")


if __name__ == '__main__':
    main()
