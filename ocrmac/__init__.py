"""
ocrmac - macOS 原生 OCR 工具

模块结构:
- ocrmac.ocrmac: 核心 OCR 功能 (Apple Vision/LiveText)
- ocrmac.processing: 处理管道 (SmartOCR, AdaptiveOCR)
- ocrmac.analysis: 分析模块 (图像分析、布局分析、区域检测)
- ocrmac.preprocessing: 图像预处理
- ocrmac.postprocessing: 文本后处理
"""

__author__ = """Maximilian Strauss"""
__email__ = "straussmaximilian+ocrmac@gmail.com"
__version__ = "1.3.0"

# Core OCR
from .ocrmac import OCR

# Processing Pipeline (新模块)
from .processing import SmartOCR, smart_ocr, AdaptiveOCR, SmartSlicer

# Analysis (新模块)
from .analysis import (
    ImageAnalyzer, ImageProfile,
    LayoutAnalyzer, ParagraphDetector, HeadingDetector, ListDetector,
    RegionDetector,
    TableDetector,
)
# 坐标合并从旧位置导入（避免循环依赖）
from .coordinate_merger import CoordinateMerger, MergedResult

# Preprocessing (新模块)
from .preprocessing import ImagePreprocessor

# Postprocessing (新模块)
from .postprocessing import TextCleaner

# 向后兼容导入 (从旧模块位置)
from .adaptive_ocr import AdaptiveOCR as _AdaptiveOCR, OCROutput
from .image_analyzer import ImageSource, ContentType
from .preprocessor import AdaptivePreprocessor, ManualPreprocessor
from .types import SliceInfo
from .coordinate_merger import TextMerger
from .region_detector import DetectedRegion, split_document_regions
from .table_recovery import Table
from .layout_analyzer import format_with_layout

__all__ = [
    # Core
    'OCR',

    # Processing Pipeline
    'SmartOCR',
    'smart_ocr',
    'AdaptiveOCR',
    'OCROutput',
    'SmartSlicer',
    'SliceInfo',

    # Analysis
    'ImageAnalyzer',
    'ImageProfile',
    'ImageSource',
    'ContentType',
    'LayoutAnalyzer',
    'ParagraphDetector',
    'HeadingDetector',
    'ListDetector',
    'RegionDetector',
    'DetectedRegion',
    'split_document_regions',
    'TableDetector',
    'Table',
    'CoordinateMerger',
    'MergedResult',
    'TextMerger',
    'format_with_layout',

    # Preprocessing
    'ImagePreprocessor',
    'AdaptivePreprocessor',
    'ManualPreprocessor',

    # Postprocessing
    'TextCleaner',
]
