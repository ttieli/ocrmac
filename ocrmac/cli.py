"""Console script for ocrmac."""

import os
import sys
import concurrent.futures
from pathlib import Path

import click

from .ocrmac import OCR
from .formatter import OCRResult, format_result
from .utils import (
    is_url,
    get_input_type,
    get_url_file_type,
    download_file,
    download_image,
    pdf_to_images,
    pdf_from_bytes_to_images,
    extract_docx_content,
    extract_docx_from_bytes,
    list_files_in_directory,
    PDF_AVAILABLE,
    DOCX_AVAILABLE,
)
from .processing import AdaptiveOCR, SmartOCR


def ocr_image(image, framework='livetext', recognition_level='accurate', language=None):
    """Perform OCR on an image and return text."""
    try:
        # Use line-level output for LiveText to get better text structure
        unit = 'line' if framework == 'livetext' else 'token'
        ocr = OCR(
            image,
            framework=framework,
            recognition_level=recognition_level,
            language_preference=[language] if language else None,
            detail=True,
            unit=unit,
        )
        results = ocr.recognize()

        # Extract text, joining by lines
        text_lines = [r[0] for r in results]
        text = "\n".join(text_lines)

        return text, results
    except Exception as e:
        return f"[OCR Error: {e}]", []


MAX_HEIGHT = 10000

def process_single_image(image, source, framework, level, language, adaptive=True, binarize=False, aggressive=False, split_regions=False):
    """
    Process a single image and return OCRResult.

    Args:
        image: PIL Image
        source: Source path/URL
        framework: OCR framework ('vision' or 'livetext')
        level: Recognition level ('accurate' or 'fast')
        language: Language preference
        adaptive: Use adaptive processing (default True)
        binarize: Enable adaptive binarization
        aggressive: Enable aggressive preprocessing
        split_regions: Enable automatic region detection
    """
    result = OCRResult(source=source)
    width, height = image.size

    if adaptive:
        # 使用自适应 OCR 处理
        return process_single_image_adaptive(image, source, framework, language, binarize=binarize, aggressive=aggressive, split_regions=split_regions)

    # 旧版处理逻辑（作为备选）
    if height > MAX_HEIGHT:
        click.echo(f"  Image too tall ({height}px), slicing into chunks...", err=True)
        overlap = 500  # pixels overlap
        current_y = 0
        page_num = 1

        while current_y < height:
            bottom = min(current_y + MAX_HEIGHT, height)
            click.echo(f"    Processing slice {page_num} ({current_y}-{bottom})...", err=True)

            crop = image.crop((0, current_y, width, bottom))

            try:
                text, details = ocr_image(crop, framework, level, language)
                result.add_page(page_num, text, details)
            except Exception as e:
                click.echo(f"    Warning: Failed to process slice {page_num}: {e}", err=True)

            if bottom == height:
                break

            current_y += MAX_HEIGHT - overlap
            page_num += 1
    else:
        text, details = ocr_image(image, framework, level, language)
        result.add_page(1, text, details)

    return result


def process_single_image_adaptive(image, source, framework, language, binarize=False, aggressive=False, split_regions=False):
    """
    使用自适应 OCR 处理图片

    自动检测图片特征并选择最优处理策略：
    - 高质量数字截图 → 智能切片处理
    - 低质量物理照片 → 预处理增强 + OCR
    - 超长图片 → 智能切片 + 坐标合并
    - 多文档图片 → 区域分割 + 分别 OCR

    Args:
        image: PIL Image
        source: Source path/URL
        framework: OCR framework
        language: Language preference
        binarize: Enable adaptive binarization (better for low-contrast)
        aggressive: Enable aggressive preprocessing
        split_regions: Enable automatic region detection
    """
    result = OCRResult(source=source)
    width, height = image.size

    try:
        # 创建自适应 OCR 处理器
        adaptive = AdaptiveOCR(
            framework=framework,
            language=language,
            enable_table_detection=True,
            enable_preprocessing=True,
            enable_binarization=binarize,
            aggressive_preprocessing=aggressive,
            enable_region_detection=split_regions,
            verbose=False,
        )

        # 执行识别
        ocr_output = adaptive.recognize(image)

        # 输出处理信息
        info = ocr_output.processing_info
        profile = ocr_output.profile

        click.echo(f"  Image: {width}x{height}, Source: {info['source']}", err=True)

        if info.get('region_count', 0) > 1:
            click.echo(f"  Detected {info['region_count']} regions", err=True)

        if info['preprocessing_applied']:
            click.echo(f"  Applied preprocessing (contrast: {info['contrast_level']})", err=True)

        if info['needs_slicing']:
            click.echo(f"  Sliced into {info['slice_count']} parts", err=True)

        # 转换为 OCRResult 格式
        # 将 MergedResult 转换为旧格式 (text, confidence, bbox)
        details = [
            (r.text, r.confidence, r.bbox)
            for r in ocr_output.results
        ]

        result.add_page(1, ocr_output.text, details)

        # 如果检测到表格，添加表格信息
        if ocr_output.tables:
            click.echo(f"  Detected {len(ocr_output.tables)} table(s)", err=True)
            # 表格以 Markdown 格式附加到文本
            for i, table in enumerate(ocr_output.tables):
                table_md = table.to_markdown()
                if table_md:
                    result.tables = getattr(result, 'tables', [])
                    result.tables.append(table_md)

        return result

    except Exception as e:
        click.echo(f"  Warning: Adaptive OCR failed, falling back to basic mode: {e}", err=True)
        # 回退到基本模式
        text, details = ocr_image(image, framework, 'accurate', language)
        result.add_page(1, text, details)
        return result


def process_single_image_smart(image, source, framework, language, verbose=False):
    """
    使用智能 OCR 处理图片（自动选择最佳策略）

    Args:
        image: PIL Image
        source: Source path/URL
        framework: OCR framework
        language: Language preference
        verbose: Show detailed processing info
    """
    result = OCRResult(source=source)
    width, height = image.size

    try:
        # 创建智能 OCR 处理器
        smart = SmartOCR(
            framework=framework,
            language=language,
            max_retries=2,
            quality_threshold=0.6,
            verbose=verbose,
        )

        # 执行智能处理
        output = smart.process(image)

        # 输出处理信息
        click.echo(f"  Image: {width}x{height}", err=True)
        click.echo(f"  Strategy: {output['strategy']} (score: {output['score']:.2f})", err=True)
        click.echo(f"  Attempts: {output['attempts']}", err=True)

        if output.get('profile'):
            profile = output['profile']
            if profile.is_long_screenshot:
                click.echo(f"  Detected: Long screenshot", err=True)
            if profile.has_multiple_regions:
                click.echo(f"  Detected: Multiple regions", err=True)

        # 转换为 OCRResult 格式
        details = output.get('details', [])
        if details:
            details_formatted = [
                (r.text, r.confidence, r.bbox)
                for r in details
            ]
        else:
            details_formatted = []

        result.add_page(1, output['text'], details_formatted)
        return result

    except Exception as e:
        click.echo(f"  Warning: Smart OCR failed, falling back to adaptive mode: {e}", err=True)
        # 回退到自适应模式
        return process_single_image_adaptive(image, source, framework, language)


def process_pdf(pdf_path_or_bytes, source, framework, level, language, is_bytes=False):
    """Process a PDF file and return OCRResult."""
    if not PDF_AVAILABLE:
        raise click.ClickException("PyMuPDF is not installed. Install with: pip install PyMuPDF")

    result = OCRResult(source=source)

    if is_bytes:
        pages = pdf_from_bytes_to_images(pdf_path_or_bytes)
    else:
        pages = pdf_to_images(pdf_path_or_bytes)

    for page_num, img in pages:
        click.echo(f"  Processing page {page_num}...", err=True)
        text, details = ocr_image(img, framework, level, language)
        result.add_page(page_num, text, details)

    return result


def process_docx(docx_path_or_bytes, source, framework, level, language, is_bytes=False):
    """Process a DOCX file and return OCRResult."""
    if not DOCX_AVAILABLE:
        raise click.ClickException("python-docx is not installed. Install with: pip install python-docx")

    if is_bytes:
        docx_text, images = extract_docx_from_bytes(docx_path_or_bytes)
    else:
        docx_text, images = extract_docx_content(docx_path_or_bytes)

    result = OCRResult(source=source, docx_text=docx_text if docx_text else None)

    # OCR embedded images
    for idx, img in images:
        click.echo(f"  Processing embedded image {idx}...", err=True)
        text, details = ocr_image(img, framework, level, language)
        result.add_page(idx, text, details)

    return result


def process_url(url, framework, level, language, adaptive=True, binarize=False, aggressive=False, split_regions=False, smart=False, verbose=False):
    """Process a URL and return OCRResult."""
    file_type = get_url_file_type(url)
    click.echo(f"Downloading from {url}...", err=True)

    content = download_file(url)

    if file_type == 'pdf':
        return process_pdf(content, url, framework, level, language, is_bytes=True)
    elif file_type == 'docx':
        return process_docx(content, url, framework, level, language, is_bytes=True)
    else:
        # Treat as image
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(content))
        if smart:
            return process_single_image_smart(img, url, framework, language, verbose=verbose)
        return process_single_image(img, url, framework, level, language, adaptive=adaptive, binarize=binarize, aggressive=aggressive, split_regions=split_regions)


def process_input(input_path, framework, level, language, recursive=False, adaptive=True, binarize=False, aggressive=False, split_regions=False, smart=False, verbose=False):
    """Process input and return list of OCRResult."""
    input_type = get_input_type(input_path)
    results = []

    if input_type == 'url':
        results.append(process_url(input_path, framework, level, language, adaptive=adaptive, binarize=binarize, aggressive=aggressive, split_regions=split_regions, smart=smart, verbose=verbose))

    elif input_type == 'pdf':
        click.echo(f"Processing PDF: {input_path}", err=True)
        results.append(process_pdf(input_path, input_path, framework, level, language))

    elif input_type == 'docx':
        click.echo(f"Processing DOCX: {input_path}", err=True)
        results.append(process_docx(input_path, input_path, framework, level, language))

    elif input_type == 'image':
        click.echo(f"Processing image: {input_path}", err=True)
        from PIL import Image
        img = Image.open(input_path)
        if smart:
            results.append(process_single_image_smart(img, input_path, framework, language, verbose=verbose))
        else:
            results.append(process_single_image(img, input_path, framework, level, language, adaptive=adaptive, binarize=binarize, aggressive=aggressive, split_regions=split_regions))

    elif input_type == 'directory':
        files = list_files_in_directory(input_path, recursive=recursive)
        if not files:
            raise click.ClickException(f"No supported files found in {input_path}")

        click.echo(f"Found {len(files)} files in {input_path}", err=True)
        for file_path in files:
            sub_results = process_input(file_path, framework, level, language, adaptive=adaptive, binarize=binarize, aggressive=aggressive, split_regions=split_regions, smart=smart, verbose=verbose)
            results.extend(sub_results)

    else:
        raise click.ClickException(f"Unsupported input type: {input_path}")

    return results


def process_batch(inputs, framework, level, language, recursive, stdout_mode,
                   output_path, output_format, no_metadata, details, layout,
                   adaptive, binarize, aggressive, split_regions, smart, verbose,
                   workers):
    """并行处理多个输入，返回按输入顺序排列的 (path, results, error) 列表"""
    import objc

    max_workers = workers or min(len(inputs), 8)

    def process_one(index, path):
        """Worker: 处理单个输入"""
        try:
            with objc.autorelease_pool():
                results = process_input(
                    path, framework, level, language, recursive,
                    adaptive=adaptive, binarize=binarize, aggressive=aggressive,
                    split_regions=split_regions, smart=smart, verbose=verbose
                )
                return (index, path, results, None)
        except Exception as e:
            return (index, path, None, str(e))

    click.echo(f"Processing {len(inputs)} inputs with {max_workers} workers...", err=True)

    results_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, i, p): i
            for i, p in enumerate(inputs)
        }
        for future in concurrent.futures.as_completed(futures):
            idx, path, results, error = future.result()
            results_map[idx] = (path, results, error)

    # 按输入顺序返回
    return [results_map[i] for i in range(len(inputs))]


def show_usage_guide():
    """显示友好的使用指南"""
    guide = """
ocrmac - macOS 原生 OCR 工具 (v1.3.0)

基本用法:
  ocrmac <图片/PDF/URL>              # OCR 并自动保存为 Markdown
  ocrmac image.png                   # 保存到 image_macocr_TIMESTAMP.md
  ocrmac image.png --stdout          # 输出到终端
  ocrmac image.png -o result.md      # 保存到指定文件

支持的输入类型:
  • 图片文件: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
  • PDF 文件: 自动识别所有页面
  • DOCX 文件: 提取文本 + OCR 嵌入图片
  • 网络 URL: 自动下载并处理
  • 目录: 批量处理目录中的文件

网络图片:
  ocrmac "https://example.com/image.png"
  ocrmac "https://example.com/doc.pdf" -o result.md

批量处理:
  ocrmac ./images/                   # 处理目录中所有文件
  ocrmac ./images/ -o ./results/     # 输出到指定目录
  ocrmac ./images/ -r                # 递归处理子目录

并行处理（多输入）:
  ocrmac a.png b.png c.png --stdout  # 多文件并行 OCR
  ocrmac "url1" "url2" --stdout      # 多 URL 并行下载+OCR
  cat urls.txt | ocrmac --batch -p   # 从 stdin 读取输入
  ocrmac a.png b.png -w 4            # 指定 4 个并行线程

输出格式:
  ocrmac image.png -f markdown       # Markdown 格式（默认）
  ocrmac image.png -f text           # 纯文本
  ocrmac image.png -f json           # JSON 格式
  ocrmac image.png -f json --details # JSON 含坐标信息

语言设置:
  ocrmac image.png -l zh-Hans        # 简体中文优先
  ocrmac image.png -l en-US          # 英文优先
  ocrmac image.png -l ja             # 日文

OCR 框架:
  ocrmac image.png --framework livetext   # LiveText（默认，macOS Sonoma+）
  ocrmac image.png --framework vision     # Apple Vision

高级选项:
  --verbose, -v          # 显示详细处理信息
  --binarize             # 启用二值化（低对比度图片）
  --aggressive           # 强力预处理
  --split-regions auto   # 自动区域分割（auto/on/off）
  --layout               # 启用布局分析
  --no-smart             # 禁用智能模式，使用旧版处理

智能处理（默认启用）:
  • 自动检测图片类型，选择最佳策略
  • 长截图自动切片，避免切断文字
  • 自动去重、合并断词、格式化标题

别名命令:
  macocr                 # 与 ocrmac 完全相同

查看完整帮助:
  ocrmac --help
"""
    click.echo(guide)


@click.command()
@click.argument('input_path', nargs=-1)
@click.option('-o', '--output', 'output_path', type=str, default=None,
              help='Output file path. If not specified, prints to stdout.')
@click.option('-f', '--format', 'output_format', type=click.Choice(['markdown', 'text', 'json']),
              default='markdown', help='Output format (default: markdown)')
@click.option('-l', '--language', type=str, default=None,
              help='Language preference (e.g., zh-Hans, en-US)')
@click.option('--framework', type=click.Choice(['vision', 'livetext']),
              default='livetext', help='OCR framework (default: livetext)')
@click.option('--level', type=click.Choice(['accurate', 'fast']),
              default='accurate', help='Recognition level (default: accurate, vision only)')
@click.option('-r', '--recursive', is_flag=True, default=False,
              help='Process directories recursively')
@click.option('-p', '--stdout', is_flag=True, default=False,
              help='Print output to stdout instead of saving to file')
@click.option('--no-metadata', is_flag=True, default=False,
              help='Exclude metadata from markdown output')
@click.option('--details', is_flag=True, default=False,
              help='Include bounding box details in JSON output')
@click.option('--no-adaptive', is_flag=True, default=False,
              help='Disable adaptive processing (use legacy mode)')
@click.option('--binarize', is_flag=True, default=False,
              help='Enable adaptive binarization (better for low-contrast images)')
@click.option('--aggressive', is_flag=True, default=False,
              help='Enable aggressive preprocessing (stronger enhancement)')
@click.option('--split-regions', type=click.Choice(['auto', 'on', 'off']), default='auto',
              help='Region detection mode: auto (smart), on (force), off (disable)')
@click.option('--layout', is_flag=True, default=False,
              help='Use layout analysis for better paragraph/heading detection')
@click.option('--no-smart', is_flag=True, default=False,
              help='Disable smart auto-detection mode (use legacy adaptive mode)')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Show detailed processing information')
@click.option('--batch', '-b', is_flag=True, default=False,
              help='Read input paths from stdin (one per line)')
@click.option('--workers', '-w', type=int, default=None,
              help='Number of parallel workers (default: min(inputs, 8))')
def main(input_path, output_path, output_format, language, framework, level, recursive, stdout, no_metadata, details, no_adaptive, binarize, aggressive, split_regions, layout, no_smart, verbose, batch, workers):
    """
    OCR tool for macOS - Extract text from images, PDFs, and DOCX files.

    INPUT_PATH can be one or more of:

    \b
      - Local image file (png, jpg, etc.)
      - Local PDF file
      - Local DOCX file
      - URL to image/PDF/DOCX
      - Directory containing files

    Examples:

    \b
      ocrmac image.png                        # OCR and save to image_macocr_TIMESTAMP.md (Default)
      ocrmac image.png --stdout               # OCR and print to terminal
      ocrmac image.png -o result.md           # Save as specific file
      ocrmac document.pdf -o result.md        # OCR all PDF pages
      ocrmac ./images/ -o ./results/          # Batch process directory
      ocrmac a.png b.png c.png --stdout       # Parallel OCR multiple files
      cat urls.txt | ocrmac --batch --stdout  # Read inputs from stdin
    """
    # 收集所有输入
    inputs = list(input_path)  # nargs=-1 返回 tuple

    if batch:
        if inputs:
            raise click.UsageError("--batch cannot be used with positional arguments")
        inputs = [line.strip() for line in sys.stdin if line.strip()]

    # 无输入时显示使用指南
    if not inputs:
        show_usage_guide()
        return 0

    from datetime import datetime
    adaptive = not no_adaptive  # 默认启用自适应处理
    smart = not no_smart  # 默认启用智能模式

    # === 单输入：走原有逻辑，完全向后兼容 ===
    if len(inputs) == 1:
        try:
            results = process_input(
                inputs[0], framework, level, language, recursive,
                adaptive=adaptive, binarize=binarize, aggressive=aggressive,
                split_regions=split_regions, smart=smart, verbose=verbose
            )

            if not results:
                click.echo("No results.", err=True)
                return 1

            # Format output
            output_parts = []
            for result in results:
                formatted = format_result(
                    result,
                    format_type=output_format,
                    include_metadata=not no_metadata,
                    include_details=details,
                    use_layout=layout,
                )
                output_parts.append(formatted)

            output_text = "\n\n---\n\n".join(output_parts)

            if output_path:
                output_path = Path(output_path)

                if output_path.is_dir() or (len(results) > 1 and not output_path.suffix):
                    output_path.mkdir(parents=True, exist_ok=True)
                    for result in results:
                        source_name = Path(result.source).stem
                        ext = '.md' if output_format == 'markdown' else ('.txt' if output_format == 'text' else '.json')
                        file_path = output_path / f"{source_name}{ext}"

                        formatted = format_result(
                            result,
                            format_type=output_format,
                            include_metadata=not no_metadata,
                            include_details=details,
                            use_layout=layout,
                        )
                        file_path.write_text(formatted, encoding='utf-8')
                        click.echo(f"Saved: {file_path}", err=True)
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(output_text, encoding='utf-8')
                    click.echo(f"Saved: {output_path}", err=True)

            elif stdout:
                click.echo(output_text)

            else:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                ext = '.md' if output_format == 'markdown' else ('.txt' if output_format == 'text' else '.json')

                for i, result in enumerate(results):
                    source_path = Path(result.source)

                    if not source_path.exists() and "://" in result.source:
                         stem = source_path.name.split('?')[0]
                         if not stem:
                             stem = "url_result"
                         output_dir = Path.cwd()
                    else:
                        stem = source_path.stem
                        output_dir = source_path.parent

                    file_name = f"{stem}_macocr_{timestamp}{ext}"
                    file_path = output_dir / file_name

                    formatted_single = format_result(
                        result,
                        format_type=output_format,
                        include_metadata=not no_metadata,
                        include_details=details,
                        use_layout=layout,
                    )

                    file_path.write_text(formatted_single, encoding='utf-8')
                    click.echo(f"Saved: {file_path}", err=True)

            return 0

        except Exception as e:
            raise click.ClickException(str(e))

    # === 多输入：并行处理 ===
    try:
        batch_results = process_batch(
            inputs, framework, level, language, recursive, stdout,
            output_path, output_format, no_metadata, details, layout,
            adaptive, binarize, aggressive, split_regions, smart, verbose,
            workers
        )

        total = len(batch_results)
        has_error = False

        if stdout:
            # --stdout: 分隔符格式输出
            for i, (path, results, error) in enumerate(batch_results):
                label = Path(path).name if not is_url(path) else path
                if error:
                    click.echo(f"--- [{i+1}/{total}] {label} [ERROR] ---")
                    click.echo(error)
                    has_error = True
                else:
                    click.echo(f"--- [{i+1}/{total}] {label} ---")
                    for result in results:
                        formatted = format_result(
                            result,
                            format_type=output_format,
                            include_metadata=not no_metadata,
                            include_details=details,
                            use_layout=layout,
                        )
                        click.echo(formatted)
                if i < total - 1:
                    click.echo("")  # blank line between entries

        elif output_path:
            # -o: 输出到目录，每个独立文件
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = '.md' if output_format == 'markdown' else ('.txt' if output_format == 'text' else '.json')

            for i, (path, results, error) in enumerate(batch_results):
                if error:
                    click.echo(f"[ERROR] {path}: {error}", err=True)
                    has_error = True
                    continue
                for result in results:
                    source_name = Path(result.source).stem
                    file_path = out_dir / f"{source_name}{ext}"
                    formatted = format_result(
                        result,
                        format_type=output_format,
                        include_metadata=not no_metadata,
                        include_details=details,
                        use_layout=layout,
                    )
                    file_path.write_text(formatted, encoding='utf-8')
                    click.echo(f"Saved: {file_path}", err=True)

        else:
            # 默认: auto-save 每个独立文件
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            ext = '.md' if output_format == 'markdown' else ('.txt' if output_format == 'text' else '.json')

            for i, (path, results, error) in enumerate(batch_results):
                if error:
                    click.echo(f"[ERROR] {path}: {error}", err=True)
                    has_error = True
                    continue
                for result in results:
                    source_path = Path(result.source)

                    if not source_path.exists() and "://" in result.source:
                        stem = source_path.name.split('?')[0]
                        if not stem:
                            stem = "url_result"
                        output_dir = Path.cwd()
                    else:
                        stem = source_path.stem
                        output_dir = source_path.parent

                    file_name = f"{stem}_macocr_{timestamp}{ext}"
                    file_path = output_dir / file_name

                    formatted_single = format_result(
                        result,
                        format_type=output_format,
                        include_metadata=not no_metadata,
                        include_details=details,
                        use_layout=layout,
                    )
                    file_path.write_text(formatted_single, encoding='utf-8')
                    click.echo(f"Saved: {file_path}", err=True)

        return 1 if has_error else 0

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    sys.exit(main())
