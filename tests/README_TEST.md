# 测试指南

## 快速开始

### 1️⃣ 环境检查
```bash
# 必须在 macOS 上运行
sw_vers

# 检查 Python
python3 --version  # 需要 3.9+

# 检查依赖
python3 -c "from ocrmac import OCR; print('✅')"
```

### 2️⃣ 准备测试图片

**选项 A: 自动生成（快速测试）**
```bash
cd tests
python3 generate_test_images.py
```

**选项 B: 使用真实图片（推荐）**

将以下图片放入 `tests/test_images/` 目录：

| 文件名 | 内容 | 要求 |
|--------|------|------|
| `simple_text.png` | 3-5 行纯文本 | 清晰可读 |
| `table.png` | 2x2 或 3x3 表格 | 单元格对齐 |
| `multi_paragraph.png` | 3+ 段落文档 | 段落间有空行 |
| `heading_list.png` | 标题 + 列表 | 标题字号大 |
| `mixed_content.png` | 混合内容 | 可选 |

**图片质量建议**:
- ✅ 分辨率: 800x600 以上
- ✅ 字号: 18pt+
- ✅ 清晰度: 截图或扫描 300dpi+
- ❌ 避免: 模糊、倾斜、光线不均

### 3️⃣ 运行测试
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
...

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

---

## 测试内容

### ✅ 测试 1: 简单文本识别
- **测试目标**: 验证基础 OCR 功能
- **输入**: 多行纯文本图片
- **预期**: 准确识别所有文本

### ✅ 测试 2: 表格识别和恢复
- **测试目标**: 验证表格检测和 Markdown 生成
- **输入**: 带表格的图片
- **预期**:
  ```markdown
  | 姓名 | 年龄 | 城市 |
  | --- | --- | --- |
  | 张三 | 25 | 北京 |
  | 李四 | 30 | 上海 |
  ```

### ✅ 测试 3: 多段落文档分析
- **测试目标**: 验证段落检测
- **输入**: 多段落文档
- **预期**: 正确识别段落边界

### ✅ 测试 4: 标题和列表识别
- **测试目标**: 验证文档结构识别
- **输入**: 包含标题和列表的图片
- **预期**: 识别标题级别和列表项

### ✅ 测试 5: 混合内容
- **测试目标**: 综合测试
- **输入**: 包含多种元素的文档
- **预期**: 综合识别成功

---

## 常见问题

### ❓ 提示 "当前系统不是 macOS"
**原因**: 此测试依赖 Apple Vision/LiveText，只能在 macOS 上运行

**解决**: 在 Mac 设备上运行测试

---

### ❓ 表格未检测到
**可能原因**:
1. 单元格对齐不清晰
2. 表格行列数不足（< 2x2）
3. OCR 识别遗漏部分单元格

**解决方法**:

**1. 检查 OCR 结果**
```bash
python3 -c "
from ocrmac import OCR
ocr = OCR('test_images/table.png', framework='livetext', unit='line')
results = ocr.recognize()
for text, conf, bbox in results:
    print(f'{text} @ {bbox}')
"
```

**2. 放宽检测参数**
修改 `test_end_to_end.py` 中的参数：
```python
detector = TableDetector(
    y_tolerance=0.03,   # 从 0.015 增加到 0.03
    x_tolerance=0.05,   # 从 0.025 增加到 0.05
)
```

**3. 使用更规整的表格**
- 使用 Excel/Numbers 制作
- 确保网格线清晰
- 单元格大小一致

---

### ❓ 段落分段不准确
**调整阈值**:
```python
analyzer = LayoutAnalyzer(
    line_spacing_threshold=2.0,    # 增大 → 分段更少
    # 或
    line_spacing_threshold=1.2,    # 减小 → 分段更多
)
```

---

### ❓ 中文识别不准确
**指定语言**:
```python
ocr = OCR(
    image_path,
    framework='livetext',
    language_preference=['zh-Hans']  # 简体中文
)
```

---

## 单独测试某个功能

### 只测试表格检测
```bash
python3 -c "
from ocrmac import OCR
from ocrmac.table_recovery import TableDetector

ocr = OCR('test_images/table.png', framework='livetext', unit='line')
results = ocr.recognize()

detector = TableDetector()
table = detector.detect(results)

if table:
    print(f'检测到: {table.rows}行 x {table.cols}列')
    print(table.to_markdown())
else:
    print('未检测到表格')
"
```

### 只测试段落分析
```bash
python3 -c "
from ocrmac import OCR
from ocrmac.layout_analyzer import LayoutAnalyzer

ocr = OCR('test_images/multi_paragraph.png', framework='livetext', unit='line')
results = ocr.recognize()

analyzer = LayoutAnalyzer()
layout = analyzer.analyze(results)

print(f'检测到 {len(layout[\"paragraphs\"])} 个段落')
print(analyzer.to_markdown(layout))
"
```

---

## 测试报告模板

```markdown
## 测试环境
- macOS: _________
- Python: _________
- 测试日期: _________

## 测试结果
- [ ] 简单文本识别: ✅ 通过 / ❌ 失败
- [ ] 表格识别: ✅ 通过 / ❌ 失败
- [ ] 多段落分析: ✅ 通过 / ❌ 失败
- [ ] 标题和列表: ✅ 通过 / ❌ 失败
- [ ] 混合内容: ✅ 通过 / ❌ 失败

## 问题和建议
(如有)
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_end_to_end.py` | 主测试脚本（运行这个） |
| `generate_test_images.py` | 生成示例测试图片 |
| `TEST_GUIDE.md` | 详细测试指南 |
| `README_TEST.md` | 快速入门（本文档） |
| `test_images/` | 测试图片目录 |

---

## 下一步

测试通过后：

1. ✅ 确认所有功能正常
2. 📝 填写测试报告
3. 🔀 合并 PR 到 main 分支
4. 🎉 开始使用新功能！

---

**详细文档**:
- 完整测试指南: `TEST_GUIDE.md`
- 使用示例: `../QUICKSTART.md`
- 技术方案: `../UPGRADE_PLAN.md`

**遇到问题?** 查看 [常见问题](#常见问题) 或提交 Issue
