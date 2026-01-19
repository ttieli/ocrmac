# 🚀 快速开始 - OCR 升级功能

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/ttieli/ocrmac.git
cd ocrmac

# 安装依赖
pip install -r requirements.txt

# 安装到系统（可选）
pip install -e .
```

## 🎯 核心功能演示

### 1️⃣ 基础 OCR（原有功能）

```bash
# 识别图片
ocrmac image.png

# 保存为 Markdown
ocrmac image.png -o result.md

# JSON 格式（含 bbox）
ocrmac image.png -f json --details
```

### 2️⃣ 表格恢复（新功能）

```bash
# 使用 Python API
python examples/enhanced_ocr_demo.py table.png --enable-table
```

**示例代码**：
```python
from ocrmac import OCR
from ocrmac.table_recovery import TableDetector

# OCR 识别
ocr = OCR('table.png', framework='livetext', unit='line')
results = ocr.recognize()

# 检测表格
detector = TableDetector()
table = detector.detect(results)

# 输出 Markdown 表格
if table:
    print(table.to_markdown())
```

**输出示例**：
```markdown
| 姓名 | 年龄 | 城市 |
| --- | --- | --- |
| 张三 | 25 | 北京 |
| 李四 | 30 | 上海 |
```

### 3️⃣ 文章分段（新功能）

```bash
# 使用 Python API
python examples/enhanced_ocr_demo.py document.png --detect-paragraphs
```

**示例代码**：
```python
from ocrmac import OCR
from ocrmac.layout_analyzer import LayoutAnalyzer

# OCR 识别
ocr = OCR('document.png', framework='livetext', unit='line')
results = ocr.recognize()

# 布局分析
analyzer = LayoutAnalyzer()
layout = analyzer.analyze(results)

# 输出结构化 Markdown
markdown = analyzer.to_markdown(layout)
print(markdown)
```

**输出示例**：
```markdown
# 文档标题

这是第一段的内容。Lorem ipsum dolor sit amet.

这是第二段的内容。Consectetur adipiscing elit.

• 列表项 1
• 列表项 2
• 列表项 3

这是第三段。
```

## 🧪 运行测试

```bash
# 测试表格恢复
python tests/test_table_recovery.py

# 测试布局分析
python tests/test_layout_analyzer.py
```

**预期输出**：
```
🧪 运行表格恢复测试...

✅ 行检测测试通过
✅ 列检测测试通过
✅ 完整表格检测测试通过
✅ Markdown 输出测试通过
✅ 空表格测试通过
✅ 行数不足测试通过

✅ 所有测试通过！
```

## 📊 参数调优

### 表格检测参数

```python
detector = TableDetector(
    y_tolerance=0.015,      # 行检测容忍度（默认：0.015）
    x_tolerance=0.025,      # 列检测容忍度（默认：0.025）
    min_rows=2,             # 最少行数（默认：2）
    min_cols=2,             # 最少列数（默认：2）
)
```

**调优建议**：
- **密集表格**（单元格间距小）：减小 `y_tolerance` 和 `x_tolerance`（如 0.01）
- **稀疏表格**（单元格间距大）：增大容忍度（如 0.03-0.05）
- **复杂表格**：降低 `alignment_ratio`（在 `TableColumnDetector` 中）

### 段落检测参数

```python
analyzer = LayoutAnalyzer(
    line_spacing_threshold=1.5,    # 段落间距阈值（默认：1.5）
    heading_size_threshold=1.3,    # 标题字体阈值（默认：1.3）
)
```

**调优建议**：
- **段落间距大的文档**：增大 `line_spacing_threshold`（如 2.0）
- **段落紧凑的文档**：减小阈值（如 1.2）
- **标题不明显**：减小 `heading_size_threshold`（如 1.2）

## 🎨 使用场景

### 场景 1: 扫描文档数字化

```bash
# 处理扫描的论文 PDF
ocrmac paper.pdf -l en-US -o paper.md
python examples/enhanced_ocr_demo.py paper_page1.png --detect-paragraphs -o output.md
```

### 场景 2: 发票/表格数据提取

```python
from ocrmac import OCR
from ocrmac.table_recovery import TableDetector, TableFormatter
import json

# OCR 发票
ocr = OCR('invoice.png', framework='livetext', unit='line')
results = ocr.recognize()

# 提取表格
detector = TableDetector()
table = detector.detect(results)

# 转换为 JSON
table_json = TableFormatter.to_json(table)
with open('invoice_data.json', 'w') as f:
    json.dump(table_json, f, ensure_ascii=False, indent=2)
```

### 场景 3: 微信/截图文字提取

```bash
# 批量处理截图
python examples/enhanced_ocr_demo.py ./screenshots/*.png \
    --detect-paragraphs \
    -l zh-Hans \
    -o ./results/
```

## 🔍 故障排查

### 问题 1: 表格未被检测到

**可能原因**：
- 表格行列数不足（< 2）
- 单元格未对齐（手写表格）
- 容忍度设置过严格

**解决方案**：
```python
# 放宽检测条件
detector = TableDetector(
    y_tolerance=0.03,   # 增大
    x_tolerance=0.04,   # 增大
    min_rows=2,
    min_cols=2,
)

# 或降低对齐率要求
from ocrmac.table_recovery import TableColumnDetector
col_detector = TableColumnDetector(alignment_ratio=0.5)  # 从 0.6 降到 0.5
```

### 问题 2: 段落分段错误

**可能原因**：
- 行间距不规则
- 阈值设置不当

**解决方案**：
```python
# 手动调整阈值
detector = ParagraphDetector(line_spacing_threshold=2.0)  # 增大阈值

# 或降低阈值（分段更细）
detector = ParagraphDetector(line_spacing_threshold=1.2)
```

### 问题 3: OCR 识别率低

**解决方案**：
```bash
# 1. 尝试不同框架
ocrmac image.png --framework vision    # 或 livetext

# 2. 设置语言偏好
ocrmac image.png -l zh-Hans            # 简体中文
ocrmac image.png -l en-US              # 英文

# 3. 使用 accurate 模式
ocrmac image.png --level accurate --framework vision
```

## 📖 更多示例

查看 `examples/` 目录获取更多示例：
- `enhanced_ocr_demo.py` - 完整演示脚本
- （待添加）`batch_process.py` - 批量处理
- （待添加）`export_to_docx.py` - 导出为 Word

## 🤝 贡献

发现 Bug 或有改进建议？欢迎提交 Issue 或 Pull Request！

## 📚 下一步

- 阅读完整的 [升级方案文档](./UPGRADE_PLAN.md)
- 查看 [API 参考文档](./docs/)（待补充）
- 尝试自己的图片和表格！

---

**Happy OCR-ing! 📸✨**
