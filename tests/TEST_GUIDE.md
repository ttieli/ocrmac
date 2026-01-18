# 端到端测试指南

## 📋 测试说明

本测试包用于验证 OCR 升级功能（文章分段 + 表格恢复）在**真实 macOS 环境**下的完整功能。

---

## ⚙️ 测试环境要求

### 必需条件
- ✅ **操作系统**: macOS 10.15+ (推荐 macOS 13 Sonoma 或更高版本)
- ✅ **Python**: 3.9+
- ✅ **依赖**: 已安装 ocrmac 及其依赖

### 检查方式
```bash
# 检查系统版本
sw_vers

# 检查 Python 版本
python3 --version

# 检查是否安装 ocrmac
python3 -c "from ocrmac import OCR; print('✅ ocrmac 已安装')"
```

---

## 📦 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/ttieli/ocrmac.git
cd ocrmac
git checkout claude/ocr-table-recovery-upgrade-aPKL7
```

### 2. 安装依赖
```bash
pip install -e .
```

### 3. 验证安装
```bash
python3 -c "
from ocrmac import OCR
from ocrmac.table_recovery import TableDetector
from ocrmac.layout_analyzer import LayoutAnalyzer
print('✅ 所有模块导入成功')
"
```

---

## 🖼️ 准备测试图片

### 测试图片目录
```
tests/test_images/
├── simple_text.png         # 测试 1: 简单文本
├── table.png               # 测试 2: 表格
├── multi_paragraph.png     # 测试 3: 多段落
├── heading_list.png        # 测试 4: 标题和列表
└── mixed_content.png       # 测试 5: 混合内容（可选）
```

### 创建测试图片目录
```bash
cd tests
mkdir -p test_images
```

---

## 🎨 测试图片准备指南

### 测试 1: simple_text.png - 简单文本

**内容要求**:
- 3-5 行纯文本
- 清晰可读
- 中文或英文均可

**示例内容**:
```
这是第一行文本
这是第二行文本
这是第三行文本
```

**制作方式**:
1. 打开任意文本编辑器（TextEdit/Pages/Word）
2. 输入上述内容，字号设置为 24pt
3. 截图保存为 `simple_text.png`

**预期结果**:
- 识别出 3-5 行文本
- 每行文本准确无误

---

### 测试 2: table.png - 表格

**内容要求**:
- 简单的 2x2 或 3x3 表格
- 单元格对齐清晰
- 文本简短（1-3 个字/单词）

**示例表格**:

| 姓名 | 年龄 | 城市 |
|------|------|------|
| 张三 | 25   | 北京 |
| 李四 | 30   | 上海 |

**制作方式**:

**方法 1: 使用 Excel/Numbers**
1. 打开 Excel 或 Numbers
2. 创建上述表格
3. 调整单元格大小，确保内容清晰
4. 截图保存为 `table.png`

**方法 2: 使用 Word/Pages**
1. 插入表格（3列 x 3行）
2. 填入内容
3. 设置表格边框（所有边框）
4. 截图保存

**方法 3: 手绘表格**
1. 打开 Preview 或其他绘图工具
2. 画一个规整的表格
3. 填入文字
4. 保存

**重要提示**:
- ✅ 表格线条清晰
- ✅ 单元格对齐
- ✅ 文字居中或左对齐一致
- ❌ 避免斜线、合并单元格（简单测试）

**预期结果**:
```markdown
| 姓名 | 年龄 | 城市 |
| --- | --- | --- |
| 张三 | 25 | 北京 |
| 李四 | 30 | 上海 |
```

---

### 测试 3: multi_paragraph.png - 多段落文档

**内容要求**:
- 至少 3 个段落
- 段落间有明显空行
- 每段 2-3 行

**示例内容**:
```
第一段的第一行内容。
第一段的第二行内容。

第二段的第一行内容。
第二段的第二行内容。

第三段的第一行内容。
第三段的第二行内容。
```

**制作方式**:
1. 打开 TextEdit 或 Word
2. 输入上述内容，注意段落间空行
3. 字号 18-24pt，行间距 1.5 倍
4. 截图保存为 `multi_paragraph.png`

**预期结果**:
- 识别出 3 个独立段落
- 段落内文本连续
- Markdown 输出段落间有空行

---

### 测试 4: heading_list.png - 标题和列表

**内容要求**:
- 1 个大标题（字号较大）
- 1 个列表（3-4 项）

**示例内容**:
```
购物清单          ← 大字号（32pt）

• 苹果            ← 普通字号（18pt）
• 香蕉
• 橙子
```

**制作方式**:
1. 打开 TextEdit
2. 输入标题，设置为**粗体 + 32pt**
3. 空一行
4. 输入列表项，使用 • 符号，18pt
5. 截图保存为 `heading_list.png`

**预期结果**:
```markdown
# 购物清单

• 苹果
• 香蕉
• 橙子
```

---

### 测试 5: mixed_content.png - 混合内容（可选）

**内容要求**:
- 标题 + 段落 + 表格

**示例内容**:
```
月度报告                    ← 标题 (28pt)

以下是本月销售数据：        ← 段落 (18pt)

姓名  | 销售额              ← 表格
张三  | 10000
李四  | 12000
```

**制作方式**:
1. 综合使用 Word 或 Pages
2. 标题使用大字号
3. 插入一个小段落
4. 插入一个 2x3 表格
5. 截图保存

**预期结果**:
- 识别标题
- 识别段落
- 识别表格结构

---

## 🚀 运行测试

### 1. 完整测试（推荐）
```bash
cd tests
python3 test_end_to_end.py
```

**预期输出**:
```
🚀 OCR 升级功能端到端测试
============================================================

============================================================
测试 1: 简单文本识别
============================================================
📸 正在识别: simple_text.png
✅ 识别完成，共 3 行文本

识别的文本:
------------------------------------------------------------
1. 这是第一行文本 (置信度: 1.00)
2. 这是第二行文本 (置信度: 1.00)
3. 这是第三行文本 (置信度: 1.00)
------------------------------------------------------------

... (其他测试)

============================================================
📊 测试结果总结
============================================================
✅ 通过  简单文本识别
✅ 通过  表格识别和恢复
✅ 通过  多段落文档分析
✅ 通过  标题和列表识别
✅ 通过  混合内容
------------------------------------------------------------
总计: 5/5 通过

🎉 所有测试通过！
```

### 2. 单独测试某个功能

**测试简单文本**:
```bash
python3 -c "
from ocrmac import OCR
ocr = OCR('test_images/simple_text.png', framework='livetext')
results = ocr.recognize()
for text, conf, bbox in results:
    print(text)
"
```

**测试表格检测**:
```bash
python3 -c "
from ocrmac import OCR
from ocrmac.table_recovery import TableDetector

ocr = OCR('test_images/table.png', framework='livetext', unit='line')
results = ocr.recognize()

detector = TableDetector()
table = detector.detect(results)

if table:
    print(table.to_markdown())
else:
    print('未检测到表格')
"
```

**测试段落分析**:
```bash
python3 -c "
from ocrmac import OCR
from ocrmac.layout_analyzer import LayoutAnalyzer

ocr = OCR('test_images/multi_paragraph.png', framework='livetext', unit='line')
results = ocr.recognize()

analyzer = LayoutAnalyzer()
layout = analyzer.analyze(results)

print(analyzer.to_markdown(layout))
"
```

---

## 🔧 参数调优（如果测试失败）

### 表格检测失败
如果表格未被识别，尝试调整参数：

```python
detector = TableDetector(
    y_tolerance=0.03,    # 增大行容忍度（默认 0.015）
    x_tolerance=0.05,    # 增大列容忍度（默认 0.025）
    min_rows=2,
    min_cols=2,
)
```

### 段落分段过多/过少
```python
analyzer = LayoutAnalyzer(
    line_spacing_threshold=2.0,    # 增大阈值 → 分段更少
    # 或
    line_spacing_threshold=1.2,    # 减小阈值 → 分段更多
)
```

### 标题未识别
```python
analyzer = LayoutAnalyzer(
    heading_size_threshold=1.2,    # 降低阈值 → 更容易识别为标题
)
```

---

## 📊 预期测试结果

### ✅ 成功标准

| 测试项 | 成功标准 |
|--------|----------|
| 简单文本识别 | 识别准确率 > 95% |
| 表格识别 | 能正确检测行列数，输出标准 Markdown 表格 |
| 多段落分析 | 正确分段（误差 ±1 段可接受） |
| 标题和列表 | 能识别标题和列表项 |
| 混合内容 | 综合识别成功 |

### ⚠️ 常见问题

**问题 1: 表格未检测到**
- **原因**: 单元格对齐不清晰
- **解决**: 使用更规整的表格，或放宽 `y_tolerance` 和 `x_tolerance`

**问题 2: 段落分段不准确**
- **原因**: 行间距不够明显
- **解决**: 调整 `line_spacing_threshold` 参数

**问题 3: OCR 识别率低**
- **原因**: 图片模糊、字体过小
- **解决**: 使用清晰的截图，字号至少 18pt

**问题 4: 中文识别不准**
- **原因**: 未指定语言偏好
- **解决**: `OCR(image, language_preference=['zh-Hans'])`

---

## 📝 测试报告模板

测试完成后，请填写以下报告：

```markdown
## 测试环境
- macOS 版本: __________
- Python 版本: __________
- ocrmac 版本/分支: __________

## 测试结果
- [ ] 测试 1: 简单文本识别 - ✅ 通过 / ❌ 失败
- [ ] 测试 2: 表格识别 - ✅ 通过 / ❌ 失败
- [ ] 测试 3: 多段落分析 - ✅ 通过 / ❌ 失败
- [ ] 测试 4: 标题和列表 - ✅ 通过 / ❌ 失败
- [ ] 测试 5: 混合内容 - ✅ 通过 / ❌ 失败

## 遇到的问题
(如有问题请描述)

## 建议
(可选)
```

---

## 🤝 反馈

如果测试过程中遇到任何问题，请：

1. 检查测试图片是否符合要求
2. 查看 [参数调优](#-参数调优如果测试失败) 部分
3. 提交 Issue 附带：
   - 测试图片
   - 完整错误信息
   - 系统环境信息

---

## ✅ 检查清单

测试前请确认：

- [ ] 在 macOS 系统上运行
- [ ] Python 3.9+ 已安装
- [ ] ocrmac 及依赖已安装
- [ ] 测试图片已准备（至少前 4 张）
- [ ] 测试图片符合质量要求（清晰、字号合适）

---

**祝测试顺利！🎉**

如有疑问，请查看 `QUICKSTART.md` 或 `UPGRADE_PLAN.md`。
