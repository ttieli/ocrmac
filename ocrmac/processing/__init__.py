"""OCR 处理管道模块"""

from .smart import SmartOCR, smart_ocr
from .adaptive import AdaptiveOCR
from .slicer import SmartSlicer
from ..types import SliceInfo

__all__ = [
    'SmartOCR',
    'smart_ocr',
    'AdaptiveOCR',
    'SmartSlicer',
    'SliceInfo',
]
