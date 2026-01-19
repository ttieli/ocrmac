"""自适应 OCR 处理器 - 统一入口"""

import sys
from dataclasses import dataclass, field
from PIL import Image

if sys.version_info < (3, 9):
    from typing import List, Optional, Dict, Any
else:
    from typing import Optional, Any
    List, Dict = list, dict

from .image_analyzer import ImageAnalyzer, ImageProfile, ImageSource, ContentType
from .preprocessor import AdaptivePreprocessor
from .smart_slicer import SmartSlicer, SliceInfo
from .coordinate_merger import CoordinateMerger, MergedResult, TextMerger
from .ocrmac import OCR
from .table_recovery import TableDetector, Table


@dataclass
class OCROutput:
    """OCR 输出结果"""
    text: str                           # 完整文本
    results: List[MergedResult]         # 详细结果（含坐标）
    profile: ImageProfile               # 图片特征分析
    tables: List[Table]                 # 检测到的表格
    processing_info: Dict[str, Any]     # 处理信息


class AdaptiveOCR:
    """
    自适应 OCR 处理器

    自动检测图片特征，选择最优处理策略：
    - 高质量数字截图 → 直接切片处理
    - 低质量物理照片 → 预处理 + OCR
    - 超长图片 → 智能切片 + 坐标合并

    特点：
    1. 全本地处理，使用 macOS 原生 OCR 能力（Vision/LiveText）
    2. 自适应各种场景（长截图、手机照片、扫描件等）
    3. 文字识别优先，同时支持表格检测
    """

    def __init__(
        self,
        framework: str = 'livetext',
        language: Optional[str] = None,
        enable_table_detection: bool = True,
        enable_preprocessing: bool = True,
        verbose: bool = False,
    ):
        """
        初始化自适应 OCR 处理器

        Args:
            framework: OCR 框架 ('vision' 或 'livetext')
            language: 语言偏好 (如 'zh-Hans', 'en-US')
            enable_table_detection: 是否启用表格检测
            enable_preprocessing: 是否启用自适应预处理
            verbose: 是否输出详细信息
        """
        self.framework = framework
        self.language = language
        self.enable_table_detection = enable_table_detection
        self.enable_preprocessing = enable_preprocessing
        self.verbose = verbose

        # 初始化各模块
        self.analyzer = ImageAnalyzer()
        self.preprocessor = AdaptivePreprocessor()
        self.slicer = SmartSlicer()
        self.table_detector = TableDetector()

    def recognize(self, image: Image.Image) -> OCROutput:
        """
        自适应 OCR 识别

        Args:
            image: PIL Image 对象

        Returns:
            OCROutput 对象，包含文本、坐标、表格等信息
        """
        width, height = image.size

        # Step 1: 分析图片特征
        profile = self.analyzer.analyze(image)

        processing_info = {
            'source': profile.source.value,
            'content_type': profile.content_type.value,
            'contrast_level': round(profile.contrast_level, 2),
            'noise_level': round(profile.noise_level, 2),
            'needs_preprocessing': profile.needs_preprocessing,
            'needs_slicing': profile.needs_slicing,
            'slice_count': 0,
            'preprocessing_applied': False,
        }

        if self.verbose:
            print(f"[AdaptiveOCR] Image: {width}x{height}")
            print(f"[AdaptiveOCR] Source: {profile.source.value}")
            print(f"[AdaptiveOCR] Contrast: {profile.contrast_level:.2f}")
            print(f"[AdaptiveOCR] Needs preprocessing: {profile.needs_preprocessing}")
            print(f"[AdaptiveOCR] Needs slicing: {profile.needs_slicing}")

        # Step 2: 预处理（如果需要且启用）
        if self.enable_preprocessing and profile.needs_preprocessing:
            processed_image = self.preprocessor.process(image, profile)
            processing_info['preprocessing_applied'] = True
            if self.verbose:
                print("[AdaptiveOCR] Preprocessing applied")
        else:
            processed_image = image

        # Step 3: OCR 处理
        if profile.needs_slicing:
            results = self._process_with_slicing(processed_image, profile)
            processing_info['slice_count'] = self.slicer.get_slice_count(image, profile)
            if self.verbose:
                print(f"[AdaptiveOCR] Sliced into {processing_info['slice_count']} parts")
        else:
            results = self._process_single(processed_image)

        # Step 4: 合并文本
        text = TextMerger.merge_to_text(results, preserve_lines=True)

        # Step 5: 表格检测（如果启用且检测到表格特征）
        tables = []
        if self.enable_table_detection and (
            profile.content_type == ContentType.TABLE or profile.has_tables
        ):
            tables = self._detect_tables(results)
            if self.verbose:
                print(f"[AdaptiveOCR] Detected {len(tables)} tables")

        return OCROutput(
            text=text,
            results=results,
            profile=profile,
            tables=tables,
            processing_info=processing_info,
        )

    def recognize_text_only(self, image: Image.Image) -> str:
        """
        仅识别文本（不返回坐标和表格）

        Args:
            image: PIL Image 对象

        Returns:
            识别的文本
        """
        output = self.recognize(image)
        return output.text

    def _process_single(self, image: Image.Image) -> List[MergedResult]:
        """处理单张图片（无需切片）"""
        ocr = OCR(
            image,
            framework=self.framework,
            language_preference=[self.language] if self.language else None,
            detail=True,
            unit='line',
        )
        raw_results = ocr.recognize()

        return [
            MergedResult(
                text=text,
                confidence=conf,
                bbox=bbox if isinstance(bbox, list) else list(bbox),
                slice_index=0,
                original_bbox=bbox if isinstance(bbox, list) else list(bbox),
            )
            for text, conf, bbox in raw_results
        ]

    def _process_with_slicing(
        self, image: Image.Image, profile: ImageProfile
    ) -> List[MergedResult]:
        """切片处理"""
        width, height = image.size
        merger = CoordinateMerger(height, width)

        slice_results = []

        for slice_info in self.slicer.slice(image, profile):
            if self.verbose:
                print(f"[AdaptiveOCR] Processing slice {slice_info.index + 1} "
                      f"({slice_info.y_start}-{slice_info.y_end})")

            # 对每个切片进行 OCR
            try:
                ocr = OCR(
                    slice_info.image,
                    framework=self.framework,
                    language_preference=[self.language] if self.language else None,
                    detail=True,
                    unit='line',
                )
                raw_results = ocr.recognize()
                slice_results.append((slice_info, raw_results))
            except Exception as e:
                if self.verbose:
                    print(f"[AdaptiveOCR] Warning: Failed to process slice {slice_info.index}: {e}")
                continue

        # 合并结果
        return merger.merge_results(slice_results)

    def _detect_tables(self, results: List[MergedResult]) -> List[Table]:
        """
        检测表格

        Args:
            results: OCR 结果列表

        Returns:
            检测到的表格列表
        """
        if not results:
            return []

        # 转换为 TableDetector 需要的格式
        ocr_results = [
            (r.text, r.confidence, r.bbox)
            for r in results
        ]

        # 检测表格
        tables = self.table_detector.detect_all(ocr_results)

        return tables


def adaptive_ocr(
    image_path_or_pil: str | Image.Image,
    framework: str = 'livetext',
    language: Optional[str] = None,
    enable_table_detection: bool = True,
) -> OCROutput:
    """
    便捷函数：自适应 OCR 识别

    Args:
        image_path_or_pil: 图片路径或 PIL Image 对象
        framework: OCR 框架
        language: 语言偏好
        enable_table_detection: 是否启用表格检测

    Returns:
        OCROutput 对象
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil)
    else:
        image = image_path_or_pil

    ocr = AdaptiveOCR(
        framework=framework,
        language=language,
        enable_table_detection=enable_table_detection,
    )

    return ocr.recognize(image)


def adaptive_ocr_text(
    image_path_or_pil: str | Image.Image,
    framework: str = 'livetext',
    language: Optional[str] = None,
) -> str:
    """
    便捷函数：自适应 OCR 识别（仅返回文本）

    Args:
        image_path_or_pil: 图片路径或 PIL Image 对象
        framework: OCR 框架
        language: 语言偏好

    Returns:
        识别的文本
    """
    output = adaptive_ocr(
        image_path_or_pil,
        framework=framework,
        language=language,
        enable_table_detection=False,
    )
    return output.text
