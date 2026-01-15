[![Downloads](https://static.pepy.tech/badge/ocrmac)](https://pepy.tech/project/ocrmac)
# ocrmac
A small Python wrapper to extract text from images on a Mac system. Uses the vision framework from Apple. Simply pass a path to an image or a `PIL` image directly and get lists of texts, their confidence, and bounding box.

This only works on macOS systems with newer macOS versions (10.15+).

## Installation

Install via pip:
```bash
pip install ocrmac
```

Or install from GitHub (with CLI support):
```bash
pipx install git+https://github.com/ttieli/ocrmac.git
```

## CLI Usage

After installation, you can use the `ocrmac` command directly in terminal:

```bash
# Basic usage - OCR an image
ocrmac image.png

# Save output to markdown file
ocrmac image.png -o result.md

# OCR a PDF file (all pages)
ocrmac document.pdf -o result.md

# OCR from URL
ocrmac https://example.com/image.png

# Batch process a directory
ocrmac ./images/ -o ./results/

# Use different output formats
ocrmac image.png -f text          # Plain text
ocrmac image.png -f json          # JSON with coordinates
ocrmac image.png -f json --details # JSON with bounding boxes

# Set language preference
ocrmac image.png -l zh-Hans       # Chinese (Simplified)
ocrmac image.png -l en-US         # English

# Choose OCR framework
ocrmac image.png --framework vision    # Apple Vision
ocrmac image.png --framework livetext  # Apple LiveText (default, macOS Sonoma+)
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path (prints to stdout if not specified) |
| `-f, --format` | Output format: `markdown` (default), `text`, `json` |
| `-l, --language` | Language preference (e.g., `zh-Hans`, `en-US`) |
| `--framework` | OCR framework: `vision` or `livetext` (default) |
| `--level` | Recognition level: `accurate` (default) or `fast` |
| `-r, --recursive` | Process directories recursively |
| `--no-metadata` | Exclude metadata from markdown output |
| `--details` | Include bounding box details in JSON output |

### Supported Input Types

- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
- **PDF**: All pages converted to images and OCR'd
- **DOCX**: Text extracted + embedded images OCR'd
- **URL**: Remote images, PDFs, or DOCX files

## Python API

### Basic Usage

```python
from ocrmac import ocrmac
annotations = ocrmac.OCR('test.png').recognize()
print(annotations)
```

Output (Text, Confidence, BoundingBox):

```
[("GitHub: Let's build from here - X", 0.5, [0.16, 0.91, 0.17, 0.01]),
('github.com', 0.5, [0.174, 0.87, 0.06, 0.01]),
('Qi &0 O M #O', 0.30, [0.65, 0.87, 0.23, 0.02]),
[...]
('P&G U TELUS', 0.5, [0.64, 0.16, 0.22, 0.03])]
```
(BoundingBox precision capped for readability reasons)

### Create Annotated Images

```python
from ocrmac import ocrmac
ocrmac.OCR('test.png').annotate_PIL()
```

![Plot](https://github.com/straussmaximilian/ocrmac/blob/main/output.png?raw=true)

## Functionality

- You can pass the path to an image or a PIL image as an object
- You can use as a class (`ocrmac.OCR`) or function `ocrmac.text_from_image`)
- You can pass several arguments:
    - `recognition_level`: `fast` or `accurate`
    - `language_preference`: A list with languages for post-processing, e.g. `['en-US', 'zh-Hans', 'de-DE']`.
- You can get an annotated output either as PIL image (`annotate_PIL`) or matplotlib figure (`annotate_matplotlib`)
- You can either use the `vision` or the `livetext` framework as backend.

#### Example: Select Language Preference

You can set a language preference like so:

```python
ocrmac.OCR('test.png',language_preference=['en-US'])
```

What abbreviation should you use for your language of choice? [Here](https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry) is an overview of language codes, e.g.: `Chinese (Simplified)` -> `zh-Hans`, `English` -> `en-US` ..

If you set a wrong language you will see an error message showing the languages available. Note that the `recognition_level` will affect the languages available (fast has fewer)

See also this [Example Notebook](https://github.com/straussmaximilian/ocrmac/blob/main/ExampleNotebook.ipynb) for implementation details.


## Speed

Timings for the  above recognize-statement:
MacBook Pro (Apple M3 Max):
- `accurate`: 207 ms ± 1.49 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
- `fast`: 131 ms ± 702 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
- `livetext`: 174 ms ± 4.12 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


## About LiveText
Since MacOS Sonoma, `LiveText` is now supported, which is stronger than the `VisionKit` OCR. You can try this feature by:
```python
# Use the OCR class
from ocrmac import ocrmac
annotations = ocrmac.OCR('test.png', framework="livetext").recognize()
print(annotations)

# Or use the helper directly
annotations = ocrmac.livetext_from_image('test.png')
```
Notice, when using this feature, the `recognition_level` and `confidence_threshold` are not available. The `confidence` output will always be 1. Additionally, LiveText supports an optional `unit` parameter for flat output: use `unit='line'` to return full-line items (instead of token-level).

## Technical Background & Motivation
If you want to do Optical character recognition (OCR) with Python, widely used tools are [`pytesseract`](https://github.com/madmaze/pytesseract) or [`EasyOCR`](https://github.com/JaidedAI/EasyOCR). For me, tesseract never did give great results. EasyOCR did, but it is slow on CPU. While there is GPU acceleration with CUDA, this does not work for Mac. *(Update from 9/2023: Apparently EasyOCR now has mps support for Mac.)*
In any case, as a Mac user you might notice that you can, with newer versions, directly copy and paste from images. The built-in OCR functionality is quite good. The underlying functionality for this is [`VNRecognizeTextRequest`](https://developer.apple.com/documentation/vision/vnrecognizetextrequest) from Apple's Vision Framework. Unfortunately it is in Swift; luckily, a wrapper for this exists. [`pyobjc-framework-Vision`](https://github.com/ronaldoussoren/pyobjc). `ocrmac` utilizes this wrapper and provides an easy interface to use this for OCR.

I found the following resources very helpful when implementing this:
- [Gist from RheTbull](https://gist.github.com/RhetTbull/1c34fc07c95733642cffcd1ac587fc4c)
- [Apple Documentation](https://developer.apple.com/documentation/vision/recognizing_text_in_images/)
- [Using Pythonista with VNRecognizeTextRequest](https://forum.omz-software.com/topic/6016/recognize-text-from-picture)

I also did a small writeup about OCR on mac in this blogpost on [medium.com](https://betterprogramming.pub/a-practical-guide-to-extract-text-from-images-ocr-in-python-d8c9c30ae74b).

## Contributing

If you have a feature request or a bug report, please post it either as an idea in the discussions or as an issue on the GitHub issue tracker.  If you want to contribute, put a PR for it. Thanks!

If you like the project, consider starring it!

---

# ocrmac 中文文档

一个轻量级的 Python 工具，用于在 Mac 系统上从图片中提取文字。使用 Apple 的 Vision 框架，只需传入图片路径或 PIL 图像对象，即可获取文字内容、置信度和边界框信息。

仅支持 macOS 10.15+ 系统。

## 安装

通过 pip 安装：
```bash
pip install ocrmac
```

或从 GitHub 安装（支持命令行工具）：
```bash
pipx install git+https://github.com/ttieli/ocrmac.git
```

## 命令行使用

安装后，可直接在终端使用 `ocrmac` 命令：

```bash
# 基本用法 - 识别图片
ocrmac image.png

# 保存为 Markdown 文件
ocrmac image.png -o result.md

# 识别 PDF 文件（所有页面）
ocrmac document.pdf -o result.md

# 识别网络图片
ocrmac https://example.com/image.png

# 批量处理目录
ocrmac ./images/ -o ./results/

# 不同输出格式
ocrmac image.png -f text          # 纯文本
ocrmac image.png -f json          # JSON 格式
ocrmac image.png -f json --details # JSON 含坐标信息

# 设置语言偏好
ocrmac image.png -l zh-Hans       # 简体中文
ocrmac image.png -l en-US         # 英文

# 选择 OCR 框架
ocrmac image.png --framework vision    # Apple Vision
ocrmac image.png --framework livetext  # Apple LiveText（默认，需 macOS Sonoma+）
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `-o, --output` | 输出文件路径（不指定则输出到终端） |
| `-f, --format` | 输出格式：`markdown`（默认）、`text`、`json` |
| `-l, --language` | 语言偏好（如 `zh-Hans`、`en-US`） |
| `--framework` | OCR 框架：`vision` 或 `livetext`（默认） |
| `--level` | 识别级别：`accurate`（默认）或 `fast` |
| `-r, --recursive` | 递归处理目录 |
| `--no-metadata` | 不在 Markdown 中包含元数据 |
| `--details` | 在 JSON 中包含边界框详情 |

### 支持的输入类型

- **图片**：PNG、JPG、JPEG、GIF、BMP、TIFF、WebP
- **PDF**：所有页面转为图片后进行 OCR
- **DOCX**：提取文本 + 对嵌入图片进行 OCR
- **URL**：远程图片、PDF 或 DOCX 文件

## Python API 使用

### 基本用法

```python
from ocrmac import ocrmac
annotations = ocrmac.OCR('test.png').recognize()
print(annotations)
```

输出格式（文字, 置信度, 边界框）：

```
[("GitHub: Let's build from here - X", 0.5, [0.16, 0.91, 0.17, 0.01]),
('github.com', 0.5, [0.174, 0.87, 0.06, 0.01]),
...]
```

### 创建标注图片

```python
from ocrmac import ocrmac
ocrmac.OCR('test.png').annotate_PIL()
```

## 功能特性

- 支持传入图片路径或 PIL 图像对象
- 可使用类 (`ocrmac.OCR`) 或函数 (`ocrmac.text_from_image`)
- 支持多种参数：
    - `recognition_level`：`fast` 或 `accurate`
    - `language_preference`：语言偏好列表，如 `['zh-Hans', 'en-US']`
- 可生成标注图片（PIL 或 matplotlib）
- 支持 `vision` 和 `livetext` 两种后端框架

### 设置语言偏好

```python
ocrmac.OCR('test.png', language_preference=['zh-Hans'])
```

语言代码参考：`简体中文` -> `zh-Hans`，`英文` -> `en-US`

## 关于 LiveText

macOS Sonoma 起支持 LiveText，识别效果优于 VisionKit：

```python
from ocrmac import ocrmac
annotations = ocrmac.OCR('test.png', framework="livetext").recognize()
```

注意：使用 LiveText 时，`recognition_level` 和 `confidence_threshold` 不可用，置信度始终为 1。

## 贡献

欢迎提交 Issue 或 Pull Request！
