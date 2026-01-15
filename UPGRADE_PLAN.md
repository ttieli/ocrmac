# OCR 升级方案：文章分段 + 表格恢复

## 📋 项目背景

基于现有的 ocrmac 项目（支持图片、PDF、DOCX OCR），新增智能文档结构识别能力，包括：
- **文章分段**：识别段落、标题、列表等文档结构
- **表格恢复**：从 bbox 坐标重建完整表格
- **格式保留**：输出保持原文档的层次结构

## 🎯 升级目标

### 1. 保持现有应用方式
- ✅ 保留现有 CLI 接口和 Python API
- ✅ 支持现有的输入类型（图片、PDF、DOCX、URL）
- ✅ 支持现有的输出格式（Markdown、Text、JSON）
- ✅ 增强 Markdown 输出，支持表格和结构化格式

### 2. 新增核心功能
- 📝 **段落识别**：基于 y 坐标和行间距聚类分段
- 📊 **表格恢复**：
  - 行列检测（y 轴聚类 + x 轴对齐）
  - 单元格合并检测
  - Markdown 表格输出
  - 可选 DOCX 表格输出
- 🔤 **格式识别**：
  - 标题检测（字号、粗体）
  - 列表检测（缩进、符号）
  - 代码块检测（等宽字体）

## 🏗️ 技术架构

### 现有模块
```
ocrmac/
├── ocrmac.py          # OCR 核心（Vision/LiveText）
├── formatter.py       # 输出格式化
├── cli.py             # CLI 命令行
└── utils.py           # 工具函数
```

### 新增模块
```
ocrmac/
├── layout_analyzer.py    # 文档布局分析（新增）
│   ├── ParagraphDetector      # 段落检测
│   ├── TableDetector          # 表格检测
│   ├── HeadingDetector        # 标题检测
│   └── ListDetector           # 列表检测
│
├── table_recovery.py     # 表格重建（新增）
│   ├── TableRowDetector       # 行检测（y 轴聚类）
│   ├── TableColumnDetector    # 列检测（x 轴对齐）
│   ├── CellMergeDetector      # 合并单元格检测
│   └── TableFormatter         # 表格格式化
│
└── enhanced_formatter.py # 增强格式化（新增）
    ├── StructuredMarkdownFormatter  # 结构化 Markdown
    └── DocxTableFormatter           # Word 表格输出
```

## 🔬 核心算法设计

### 1. 段落识别算法

```python
class ParagraphDetector:
    """基于 bbox 的段落识别"""

    def __init__(self, line_spacing_threshold=1.5):
        """
        参数:
            line_spacing_threshold: 行间距阈值（相对于平均行高的倍数）
        """
        self.line_spacing_threshold = line_spacing_threshold

    def detect_paragraphs(self, ocr_results):
        """
        段落检测流程:
        1. 按 y 坐标排序所有行
        2. 计算相邻行的间距
        3. 间距超过阈值则分段
        4. 返回段落列表
        """
        # 实现细节见代码
        pass
```

**核心逻辑**：
- 使用 LiveText 的 `unit='line'` 获取行级结果
- 按 y 坐标排序，计算行高中位数
- 相邻行间距 > 1.5倍平均行高 → 新段落
- 支持左对齐检测（段首缩进识别）

### 2. 表格恢复算法

#### 2.1 行检测（Y轴聚类）
```python
class TableRowDetector:
    """基于 y 坐标的行检测"""

    def detect_rows(self, bboxes, y_tolerance=0.02):
        """
        算法:
        1. 提取所有 bbox 的 y_center
        2. 使用 DBSCAN/简单聚类（y 差异 < tolerance）
        3. 每个簇 = 一行
        4. 按 y 坐标排序返回
        """
        rows = []
        for bbox in sorted(bboxes, key=lambda b: b['y']):
            # 找到 y 坐标接近的行
            if not rows or abs(bbox['y'] - rows[-1]['y_avg']) > y_tolerance:
                rows.append({'items': [bbox], 'y_avg': bbox['y']})
            else:
                rows[-1]['items'].append(bbox)
                # 更新行的平均 y
                rows[-1]['y_avg'] = sum(b['y'] for b in rows[-1]['items']) / len(rows[-1]['items'])
        return rows
```

#### 2.2 列检测（X轴对齐）
```python
class TableColumnDetector:
    """基于 x 坐标对齐的列检测"""

    def detect_columns(self, rows, x_tolerance=0.03):
        """
        算法:
        1. 收集所有单元格的 x_left 坐标
        2. 聚类找到列边界（多行统计）
        3. 返回列分割点列表
        """
        # 统计所有 x 坐标
        x_positions = []
        for row in rows:
            for item in row['items']:
                x_positions.append(item['x'])

        # 聚类得到列边界
        columns = self._cluster_x_positions(x_positions, x_tolerance)
        return sorted(columns)
```

#### 2.3 合并单元格检测
```python
class CellMergeDetector:
    """检测跨行/跨列的合并单元格"""

    def detect_merged_cells(self, table_grid):
        """
        启发式规则:
        1. 单元格宽度 > 1.5倍平均列宽 → 可能跨列
        2. 单元格高度 > 1.5倍平均行高 → 可能跨行
        3. 空白单元格 + 邻近大单元格 → 合并
        """
        pass
```

### 3. 格式识别算法

```python
class HeadingDetector:
    """标题检测"""

    def is_heading(self, line_bbox, avg_height):
        """
        启发式:
        1. 字体大小 > 1.3倍平均 → 可能是标题
        2. 行首独立（前后空行）
        3. 文本长度 < 100 字符
        """
        if line_bbox['height'] > avg_height * 1.3:
            return True
        return False

class ListDetector:
    """列表检测"""

    def is_list_item(self, line_text):
        """
        检测:
        1. 以 •/-/*/数字. 开头
        2. 缩进一致
        """
        import re
        patterns = [r'^\s*[-•*]\s+', r'^\s*\d+\.\s+']
        return any(re.match(p, line_text) for p in patterns)
```

## 📊 输出格式增强

### Markdown 增强示例

**输入**：表格图片
**现有输出**：
```
单元格1 单元格2
单元格3 单元格4
```

**升级后输出**：
```markdown
| 列1    | 列2    |
|--------|--------|
| 单元格1 | 单元格2 |
| 单元格3 | 单元格4 |
```

### DOCX 表格输出（可选）

```python
from docx import Document
from docx.shared import Inches

class DocxTableFormatter:
    """输出 Word 表格"""

    def write_table_to_docx(self, table_data, output_path):
        doc = Document()
        table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))

        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                table.rows[i].cells[j].text = cell

        doc.save(output_path)
```

## 🔧 CLI 接口增强

### 新增参数

```bash
# 启用表格恢复
ocrmac image.png --enable-table-recovery

# 指定输出格式包含表格
ocrmac image.png -f markdown --table-format markdown

# 输出到 Word（保留表格）
ocrmac image.png -o output.docx --preserve-tables

# 调整表格检测灵敏度
ocrmac image.png --table-sensitivity high  # low/medium/high

# 启用段落检测
ocrmac image.png --detect-paragraphs --paragraph-spacing 1.5
```

### 保持向后兼容

```bash
# 现有用法完全不变
ocrmac image.png                    # ✅ 照常工作
ocrmac image.png -o result.md       # ✅ 照常工作
ocrmac image.png -f json --details  # ✅ 照常工作
```

## 🎬 实现路线图

### Phase 1: 基础布局分析（优先级：高）
**时间估算**：2-3天

- [ ] 实现 `layout_analyzer.py`
  - [ ] ParagraphDetector（段落检测）
  - [ ] 基于 LiveText line-level 输出
  - [ ] 行间距分析
- [ ] 单元测试
- [ ] CLI 集成：`--detect-paragraphs`

**验证标准**：
- 识别包含多段的文档，正确分段
- Markdown 输出段落间有空行

### Phase 2: 表格恢复核心（优先级：高）
**时间估算**：3-4天

- [ ] 实现 `table_recovery.py`
  - [ ] TableRowDetector（行检测）
  - [ ] TableColumnDetector（列检测）
  - [ ] TableFormatter（Markdown 表格）
- [ ] 调优参数（y_tolerance, x_tolerance）
- [ ] 单元测试（含边界 case）
- [ ] CLI 集成：`--enable-table-recovery`

**验证标准**：
- 简单 2x2 表格 100% 识别
- 5x5 复杂表格 >80% 准确率
- 输出标准 Markdown 表格

### Phase 3: 高级特性（优先级：中）
**时间估算**：2-3天

- [ ] CellMergeDetector（合并单元格）
- [ ] HeadingDetector（标题识别）
- [ ] ListDetector（列表识别）
- [ ] 集成到 Markdown 输出

**验证标准**：
- 识别标题并用 `#` 标记
- 识别列表并用 `-` 格式化

### Phase 4: DOCX 输出（优先级：低）
**时间估算**：2天

- [ ] DocxTableFormatter
- [ ] 依赖 python-docx
- [ ] CLI 支持：`-o output.docx`

**验证标准**：
- 表格正确写入 Word
- 保留单元格合并

### Phase 5: 性能优化与文档（优先级：中）
**时间估算**：1-2天

- [ ] 性能基准测试
- [ ] 大图优化（>4K 分辨率）
- [ ] README 更新
- [ ] 示例图库

## 🧪 测试策略

### 测试数据集
```
tests/fixtures/
├── simple_table_2x2.png          # 简单表格
├── complex_table_5x5.png         # 复杂表格
├── merged_cells_table.png        # 合并单元格
├── multi_paragraph_doc.png       # 多段落文档
├── heading_and_list.png          # 标题+列表
└── mixed_content.png             # 混合内容
```

### 准确率指标
- **段落检测**：F1 > 0.90
- **表格行列检测**：精确率 > 0.85
- **端到端表格恢复**：
  - 简单表格（3x3）：> 95%
  - 复杂表格（>5x5）：> 80%

### 测试用例
```python
def test_paragraph_detection():
    """测试段落检测"""
    ocr = OCR('tests/fixtures/multi_paragraph_doc.png')
    results = ocr.recognize()
    detector = ParagraphDetector()
    paragraphs = detector.detect_paragraphs(results)
    assert len(paragraphs) >= 3  # 至少检测到3个段落

def test_table_recovery_simple():
    """测试简单表格恢复"""
    ocr = OCR('tests/fixtures/simple_table_2x2.png')
    results = ocr.recognize()
    table = TableDetector().detect(results)
    assert table.rows == 2
    assert table.cols == 2
```

## 📚 使用示例

### Python API

```python
from ocrmac import OCR
from ocrmac.layout_analyzer import ParagraphDetector
from ocrmac.table_recovery import TableDetector

# 基础 OCR
ocr = OCR('document.png', framework='livetext', unit='line')
results = ocr.recognize()

# 段落检测
detector = ParagraphDetector()
paragraphs = detector.detect_paragraphs(results)
for i, para in enumerate(paragraphs):
    print(f"段落 {i+1}: {para}")

# 表格检测
table_detector = TableDetector()
tables = table_detector.detect(results)
for table in tables:
    print(table.to_markdown())  # 输出 Markdown 表格
```

### CLI 使用

```bash
# 文章分段 + 表格恢复
ocrmac document.png \
    --detect-paragraphs \
    --enable-table-recovery \
    -o output.md

# 高级选项
ocrmac document.png \
    --framework livetext \
    -l zh-Hans \
    --table-sensitivity high \
    --paragraph-spacing 1.8 \
    -o output.md
```

## 🚀 快速开始（开发者）

### 1. 安装依赖
```bash
# 基础依赖（已有）
pip install -r requirements.txt

# 新增依赖
pip install numpy>=1.20.0      # 数值计算
pip install scikit-learn>=1.0  # 聚类算法（可选）
pip install python-docx>=0.8.11  # Word 输出（可选）
```

### 2. 开发新功能
```bash
# 创建新模块
touch ocrmac/layout_analyzer.py
touch ocrmac/table_recovery.py

# 运行测试
pytest tests/test_table_recovery.py -v

# 本地测试 CLI
python -m ocrmac.cli test.png --enable-table-recovery
```

### 3. 验证效果
```bash
# 准备测试图片
cp sample_table.png tests/fixtures/

# 运行端到端测试
ocrmac tests/fixtures/sample_table.png \
    --enable-table-recovery \
    -o /tmp/result.md

# 检查输出
cat /tmp/result.md
```

## 🔍 技术细节

### 为什么选择 LiveText 的 line-level 输出？

**现状分析**：
- `framework='vision'`：返回词/短语级 bbox，需手动合并为行
- `framework='livetext', unit='token'`：返回字符级（CJK）或词级
- `framework='livetext', unit='line'`：✅ **直接返回行级结果**

**优势**：
1. 行已经预分组，无需 y 轴聚类
2. 行内文本已排序，顺序正确
3. 简化段落检测（只需判断行间距）

### 坐标系统说明

Vision/LiveText 使用归一化坐标（0-1）：
```python
# bbox 格式：[x, y, w, h]
x = bbox[0]  # 左边距（占图片宽度的比例）
y = bbox[1]  # 底边距（占图片高度的比例，注意是底部！）
w = bbox[2]  # 宽度比例
h = bbox[3]  # 高度比例

# 转换为绝对坐标（像素）
x_px = x * image.width
y_px = (1 - y - h) * image.height  # y 坐标需翻转
```

### 表格检测阈值调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `y_tolerance` | 0.01-0.02 | 同一行内 y 坐标差异容忍度 |
| `x_tolerance` | 0.02-0.03 | 列对齐容忍度 |
| `min_rows` | 2 | 最少行数（少于此不视为表格） |
| `min_cols` | 2 | 最少列数 |
| `alignment_ratio` | 0.7 | 列对齐率（70% 以上单元格对齐） |

## ⚠️ 已知限制与改进方向

### 当前限制
1. **复杂表格**：嵌套表格、不规则表格识别率较低
2. **表格检测**：需要明显的网格线或对齐，手写表格效果差
3. **多列布局**：报纸式分栏布局可能误判
4. **图文混排**：图片区域需手动排除

### 未来改进
1. 引入 CV 预处理：线条检测辅助表格识别
2. 深度学习模型：表格结构检测（如 TableNet）
3. 布局分析模型：区分文本区、表格区、图片区
4. OCR 质量评估：低置信度区域二次识别

## 📖 参考资料

- [Apple Vision Framework](https://developer.apple.com/documentation/vision)
- [Apple LiveText](https://developer.apple.com/documentation/visionkit)
- [表格检测论文: TableNet](https://arxiv.org/abs/2001.01469)
- [文档布局分析: LayoutParser](https://layout-parser.github.io/)

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：
1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/table-recovery`
3. 提交改动：`git commit -m 'Add table recovery'`
4. 推送分支：`git push origin feature/table-recovery`
5. 提交 Pull Request

## 📄 许可证

MIT License - 详见 LICENSE 文件

---

**方案版本**：v1.0
**更新日期**：2026-01-15
**作者**：Claude AI (规划) + ttieli (实现)
