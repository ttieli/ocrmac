"""测试表格恢复功能"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocrmac.table_recovery import (
    TableRowDetector,
    TableColumnDetector,
    TableDetector,
    Table,
    TableCell,
)


def test_table_row_detection():
    """测试表格行检测"""
    # 模拟 OCR 结果（2 行，每行 2 个单元格）
    ocr_results = [
        # 第一行
        ("A1", 1.0, [0.1, 0.8, 0.1, 0.05]),  # x=0.1, y=0.8
        ("A2", 1.0, [0.3, 0.8, 0.1, 0.05]),  # x=0.3, y=0.8
        # 第二行
        ("B1", 1.0, [0.1, 0.6, 0.1, 0.05]),  # x=0.1, y=0.6
        ("B2", 1.0, [0.3, 0.6, 0.1, 0.05]),  # x=0.3, y=0.6
    ]

    detector = TableRowDetector(y_tolerance=0.02)
    rows = detector.detect_rows(ocr_results)

    assert len(rows) == 2, f"应检测到 2 行，实际 {len(rows)} 行"
    assert len(rows[0]['items']) == 2, "第一行应有 2 个单元格"
    assert len(rows[1]['items']) == 2, "第二行应有 2 个单元格"
    print("✅ 行检测测试通过")


def test_table_column_detection():
    """测试表格列检测"""
    # 模拟已检测的行
    rows = [
        {
            'y_avg': 0.825,
            'items': [
                ("A1", 1.0, [0.1, 0.8, 0.1, 0.05]),
                ("A2", 1.0, [0.3, 0.8, 0.1, 0.05]),
            ]
        },
        {
            'y_avg': 0.625,
            'items': [
                ("B1", 1.0, [0.1, 0.6, 0.1, 0.05]),
                ("B2", 1.0, [0.3, 0.6, 0.1, 0.05]),
            ]
        },
    ]

    detector = TableColumnDetector(x_tolerance=0.03, alignment_ratio=0.5)
    columns = detector.detect_columns(rows)

    assert len(columns) == 2, f"应检测到 2 列，实际 {len(columns)} 列"
    assert abs(columns[0] - 0.1) < 0.01, "第一列应在 x=0.1"
    assert abs(columns[1] - 0.3) < 0.01, "第二列应在 x=0.3"
    print("✅ 列检测测试通过")


def test_table_full_detection():
    """测试完整表格检测"""
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

    detector = TableDetector(
        y_tolerance=0.02,
        x_tolerance=0.03,
        min_rows=2,
        min_cols=2,
    )
    table = detector.detect(ocr_results)

    assert table is not None, "应检测到表格"
    assert table.rows == 3, f"应有 3 行，实际 {table.rows} 行"
    assert table.cols == 2, f"应有 2 列，实际 {table.cols} 列"

    # 检查单元格内容
    assert table.get_cell(0, 0).text == "姓名"
    assert table.get_cell(0, 1).text == "年龄"
    assert table.get_cell(1, 0).text == "张三"
    assert table.get_cell(1, 1).text == "25"

    print("✅ 完整表格检测测试通过")


def test_table_to_markdown():
    """测试 Markdown 输出"""
    table = Table(rows=2, cols=2)
    table.set_cell(0, 0, TableCell("A1", 0, 0, [0.1, 0.8, 0.1, 0.05]))
    table.set_cell(0, 1, TableCell("A2", 0, 1, [0.3, 0.8, 0.1, 0.05]))
    table.set_cell(1, 0, TableCell("B1", 1, 0, [0.1, 0.6, 0.1, 0.05]))
    table.set_cell(1, 1, TableCell("B2", 1, 1, [0.3, 0.6, 0.1, 0.05]))

    markdown = table.to_markdown()

    assert "| A1 | A2 |" in markdown
    assert "| --- | --- |" in markdown
    assert "| B1 | B2 |" in markdown

    print("✅ Markdown 输出测试通过")
    print("\n生成的 Markdown 表格:")
    print(markdown)


def test_empty_table():
    """测试空表格"""
    detector = TableDetector()
    table = detector.detect([])

    assert table is None, "空输入应返回 None"
    print("✅ 空表格测试通过")


def test_insufficient_rows():
    """测试行数不足的情况"""
    ocr_results = [
        ("A1", 1.0, [0.1, 0.8, 0.1, 0.05]),
        ("A2", 1.0, [0.3, 0.8, 0.1, 0.05]),
    ]

    detector = TableDetector(min_rows=2)
    table = detector.detect(ocr_results)

    assert table is None, "只有 1 行不应检测为表格"
    print("✅ 行数不足测试通过")


if __name__ == '__main__':
    print("🧪 运行表格恢复测试...\n")

    try:
        test_table_row_detection()
        test_table_column_detection()
        test_table_full_detection()
        test_table_to_markdown()
        test_empty_table()
        test_insufficient_rows()

        print("\n✅ 所有测试通过！")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
