"""Top-level package for ocrmac."""

__author__ = """Maximilian Strauss"""
__email__ = "straussmaximilian+ocrmac@gmail.com"
__version__ = "1.2.0"

from .ocrmac import OCR

# Adaptive OCR modules
from .adaptive_ocr import AdaptiveOCR, adaptive_ocr, adaptive_ocr_text, OCROutput
from .image_analyzer import ImageAnalyzer, ImageProfile, ImageSource, ContentType
from .preprocessor import AdaptivePreprocessor, ManualPreprocessor
from .smart_slicer import SmartSlicer, SliceInfo
from .coordinate_merger import CoordinateMerger, MergedResult, TextMerger
from .table_recovery import TableDetector, Table
from .layout_analyzer import LayoutAnalyzer
from .region_detector import RegionDetector, DetectedRegion, split_document_regions

__all__ = [
    # Core
    'OCR',
    # Adaptive OCR
    'AdaptiveOCR',
    'adaptive_ocr',
    'adaptive_ocr_text',
    'OCROutput',
    # Image Analysis
    'ImageAnalyzer',
    'ImageProfile',
    'ImageSource',
    'ContentType',
    # Preprocessing
    'AdaptivePreprocessor',
    'ManualPreprocessor',
    # Slicing
    'SmartSlicer',
    'SliceInfo',
    # Coordinate Merging
    'CoordinateMerger',
    'MergedResult',
    'TextMerger',
    # Table & Layout
    'TableDetector',
    'Table',
    'LayoutAnalyzer',
    # Region Detection
    'RegionDetector',
    'DetectedRegion',
    'split_document_regions',
]
