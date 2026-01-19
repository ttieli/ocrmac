"""Console script for ocrmac."""

import os
import sys
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


def process_single_image(image, source, framework, level, language):
    """Process a single image and return OCRResult."""
    result = OCRResult(source=source)
    text, details = ocr_image(image, framework, level, language)
    result.add_page(1, text, details)
    return result


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


def process_url(url, framework, level, language):
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
        return process_single_image(img, url, framework, level, language)


def process_input(input_path, framework, level, language, recursive=False):
    """Process input and return list of OCRResult."""
    input_type = get_input_type(input_path)
    results = []

    if input_type == 'url':
        results.append(process_url(input_path, framework, level, language))

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
        results.append(process_single_image(img, input_path, framework, level, language))

    elif input_type == 'directory':
        files = list_files_in_directory(input_path, recursive=recursive)
        if not files:
            raise click.ClickException(f"No supported files found in {input_path}")

        click.echo(f"Found {len(files)} files in {input_path}", err=True)
        for file_path in files:
            sub_results = process_input(file_path, framework, level, language)
            results.extend(sub_results)

    else:
        raise click.ClickException(f"Unsupported input type: {input_path}")

    return results


@click.command()
@click.argument('input_path', type=str)
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
def main(input_path, output_path, output_format, language, framework, level, recursive, stdout, no_metadata, details):
    """
    OCR tool for macOS - Extract text from images, PDFs, and DOCX files.

    INPUT_PATH can be:

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
    """
    from datetime import datetime
    try:
        results = process_input(input_path, framework, level, language, recursive)

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
            )
            output_parts.append(formatted)

        output_text = "\n\n---\n\n".join(output_parts)

        # Logic:
        # 1. If -o/--output is provided -> Save to that specific path/dir
        # 2. If --stdout is provided -> Print to terminal
        # 3. Default -> Auto-save to {stem}_macocr_{timestamp}.{ext}

        if output_path:
            output_path = Path(output_path)

            # If output is a directory, create files for each result
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
                    )
                    file_path.write_text(formatted, encoding='utf-8')
                    click.echo(f"Saved: {file_path}", err=True)
            else:
                # Single output file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output_text, encoding='utf-8')
                click.echo(f"Saved: {output_path}", err=True)

        elif stdout:
            # Print to stdout
            click.echo(output_text)

        else:
            # Default: Auto-save mode
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            ext = '.md' if output_format == 'markdown' else ('.txt' if output_format == 'text' else '.json')

            for i, result in enumerate(results):
                source_path = Path(result.source)
                
                # Handle URLs or unknown sources
                if not source_path.exists() and "://" in result.source:
                     # For URLs, save to current directory with sanitized name
                     stem = source_path.name.split('?')[0] # Remove query params
                     if not stem:
                         stem = "url_result"
                     output_dir = Path.cwd()
                else:
                    # For local files, save to same directory
                    stem = source_path.stem
                    output_dir = source_path.parent

                file_name = f"{stem}_macocr_{timestamp}{ext}"
                file_path = output_dir / file_name

                # If processing multiple files (batch), we need specific content for each
                # Re-format just for this result to ensure correct content separation
                formatted_single = format_result(
                    result,
                    format_type=output_format,
                    include_metadata=not no_metadata,
                    include_details=details,
                )
                
                file_path.write_text(formatted_single, encoding='utf-8')
                click.echo(f"Saved: {file_path}", err=True)

        return 0

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    sys.exit(main())
