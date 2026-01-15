"""Output formatters for OCR results."""

import json
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional


class OCRResult:
    """Container for OCR results."""

    def __init__(
        self,
        source: str,
        pages: Optional[List[Dict[str, Any]]] = None,
        docx_text: Optional[str] = None,
    ):
        """
        Initialize OCR result.

        Args:
            source: Source file path or URL
            pages: List of page results, each containing:
                - page_num: Page number
                - text: Recognized text
                - details: Optional list of (text, confidence, bbox)
            docx_text: Extracted text from DOCX (if applicable)
        """
        self.source = source
        self.pages = pages or []
        self.docx_text = docx_text
        self.timestamp = datetime.now()

    def add_page(
        self,
        page_num: int,
        text: str,
        details: Optional[List[Tuple[str, float, List[float]]]] = None,
    ):
        """Add a page result."""
        self.pages.append({
            'page_num': page_num,
            'text': text,
            'details': details,
        })

    @property
    def total_pages(self) -> int:
        """Get total number of pages."""
        return len(self.pages)

    @property
    def all_text(self) -> str:
        """Get all recognized text combined."""
        parts = []
        if self.docx_text:
            parts.append(self.docx_text)
        for page in self.pages:
            if page.get('text'):
                parts.append(page['text'])
        return "\n\n".join(parts)


def format_markdown(result: OCRResult, include_metadata: bool = True) -> str:
    """
    Format OCR result as Markdown.

    Args:
        result: OCR result object
        include_metadata: Whether to include source and timestamp

    Returns:
        Formatted Markdown string
    """
    lines = []

    if include_metadata:
        lines.append(f"**Source**: {result.source}")
        if result.total_pages > 0:
            lines.append(f"**Pages**: {result.total_pages}")
        lines.append(f"**Time**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # DOCX extracted text
    if result.docx_text:
        lines.append("## Document Text")
        lines.append("")
        lines.append(result.docx_text)
        lines.append("")

    # OCR results by page
    if result.total_pages == 1 and not result.docx_text:
        # Single image/page - no page header needed
        page = result.pages[0]
        if page.get('text'):
            lines.append(page['text'])
    else:
        for page in result.pages:
            page_num = page.get('page_num', 1)
            if result.docx_text:
                lines.append(f"## Image {page_num}")
            else:
                lines.append(f"## Page {page_num}")
            lines.append("")
            if page.get('text'):
                lines.append(page['text'])
            else:
                lines.append("*(No text recognized)*")
            lines.append("")

    return "\n".join(lines)


def format_text(result: OCRResult) -> str:
    """
    Format OCR result as plain text.

    Args:
        result: OCR result object

    Returns:
        Plain text string
    """
    return result.all_text


def format_json(result: OCRResult, include_details: bool = False) -> str:
    """
    Format OCR result as JSON.

    Args:
        result: OCR result object
        include_details: Whether to include bounding box details

    Returns:
        JSON string
    """
    data = {
        'source': result.source,
        'timestamp': result.timestamp.isoformat(),
        'total_pages': result.total_pages,
    }

    if result.docx_text:
        data['docx_text'] = result.docx_text

    pages_data = []
    for page in result.pages:
        page_data = {
            'page_num': page.get('page_num', 1),
            'text': page.get('text', ''),
        }
        if include_details and page.get('details'):
            page_data['details'] = [
                {
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox,
                }
                for text, conf, bbox in page['details']
            ]
        pages_data.append(page_data)

    data['pages'] = pages_data

    return json.dumps(data, ensure_ascii=False, indent=2)


def format_result(
    result: OCRResult,
    format_type: str = 'markdown',
    include_metadata: bool = True,
    include_details: bool = False,
) -> str:
    """
    Format OCR result according to specified format.

    Args:
        result: OCR result object
        format_type: 'markdown', 'text', or 'json'
        include_metadata: Include metadata in output (markdown only)
        include_details: Include bbox details (json only)

    Returns:
        Formatted string
    """
    if format_type == 'markdown':
        return format_markdown(result, include_metadata=include_metadata)
    elif format_type == 'text':
        return format_text(result)
    elif format_type == 'json':
        return format_json(result, include_details=include_details)
    else:
        raise ValueError(f"Unknown format type: {format_type}")
