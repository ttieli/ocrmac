"""自适应 OCR 处理器 - 统一入口"""

import sys
from dataclasses import dataclass, field
from PIL import Image

if sys.version_info < (3, 9):
    from typing import List, Optional, Dict, Any, Tuple
else:
    from typing import Optional, Any
    List, Dict, Tuple = list, dict, tuple

from ..analysis.image import ImageAnalyzer, ImageProfile, ImageSource, ContentType
from ..preprocessing import ImagePreprocessor as AdaptivePreprocessor
from .slicer import SmartSlicer, SliceInfo
from ..analysis.coordinates import CoordinateMerger, MergedResult, TextMerger
from ..ocrmac import OCR
from ..analysis.table import TableDetector, Table
from ..analysis.region import RegionDetector, DetectedRegion


@dataclass
class RegionOCRResult:
    """单个区域的 OCR 结果"""
    region_index: int                   # 区域索引
    region_bbox: Tuple[int, int, int, int]  # 区域边界 (x, y, w, h)
    text: str                           # 区域文本
    results: List[MergedResult]         # 详细结果（含坐标）
    tables: List[Table]                 # 区域内的表格
    dominant_color: Optional[Tuple[int, int, int]] = None  # 区域主要颜色


@dataclass
class OCROutput:
    """OCR 输出结果"""
    text: str                           # 完整文本
    results: List[MergedResult]         # 详细结果（含坐标）
    profile: ImageProfile               # 图片特征分析
    tables: List[Table]                 # 检测到的表格
    processing_info: Dict[str, Any]     # 处理信息
    regions: Optional[List[RegionOCRResult]] = None  # 区域识别结果（如果启用）


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
        enable_binarization: bool = False,
        aggressive_preprocessing: bool = False,
        enable_region_detection: str = 'auto',
        region_method: str = 'auto',
        verbose: bool = False,
    ):
        """
        初始化自适应 OCR 处理器

        Args:
            framework: OCR 框架 ('vision' 或 'livetext')
            language: 语言偏好 (如 'zh-Hans', 'en-US')
            enable_table_detection: 是否启用表格检测
            enable_preprocessing: 是否启用自适应预处理
            enable_binarization: 是否启用自适应二值化（对低对比度图片有效）
            aggressive_preprocessing: 是否使用激进预处理模式
            enable_region_detection: 区域检测模式
                - 'auto': 智能判断（推荐）
                - True/'on': 强制启用
                - False/'off': 强制禁用
            region_method: 区域检测方法 ('auto', 'color', 'contour', 'combined')
            verbose: 是否输出详细信息
        """
        self.framework = framework
        self.language = language
        self.enable_table_detection = enable_table_detection
        self.enable_preprocessing = enable_preprocessing
        self.enable_binarization = enable_binarization
        self.aggressive_preprocessing = aggressive_preprocessing
        # 标准化区域检测参数
        if enable_region_detection in (True, 'on', 'true', 'True'):
            self.region_detection_mode = 'on'
        elif enable_region_detection in (False, 'off', 'false', 'False'):
            self.region_detection_mode = 'off'
        else:
            self.region_detection_mode = 'auto'
        self.region_method = region_method
        self.verbose = verbose

        # 初始化各模块
        self.analyzer = ImageAnalyzer()
        self.preprocessor = AdaptivePreprocessor(
            enable_binarization=enable_binarization,
            aggressive_mode=aggressive_preprocessing,
        )
        self.slicer = SmartSlicer()
        self.table_detector = TableDetector()
        self.region_detector = RegionDetector(method=region_method)

    def recognize(self, image: Image.Image) -> OCROutput:
        """
        自适应 OCR 识别

        Args:
            image: PIL Image 对象

        Returns:
            OCROutput 对象，包含文本、坐标、表格等信息
        """
        width, height = image.size

        # Step 1: 分析图片特征（提前分析，用于智能决策）
        profile = self.analyzer.analyze(image)

        # Step 0: 智能判断是否启用区域检测
        use_region_detection = self._should_use_region_detection(profile)

        if use_region_detection:
            return self._recognize_with_regions(image, profile)

        processing_info = {
            'source': profile.source.value,
            'content_type': profile.content_type.value,
            'contrast_level': round(profile.contrast_level, 2),
            'noise_level': round(profile.noise_level, 2),
            'needs_preprocessing': profile.needs_preprocessing,
            'needs_slicing': profile.needs_slicing,
            'slice_count': 0,
            'preprocessing_applied': False,
            'region_count': 0,
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
            regions=None,
        )

    def _should_use_region_detection(self, profile: ImageProfile) -> bool:
        """
        智能判断是否应该启用区域检测

        Args:
            profile: 图片特征分析结果

        Returns:
            是否启用区域检测
        """
        # 强制模式
        if self.region_detection_mode == 'on':
            # 即使强制启用，长截图也不应该用区域检测
            if profile.is_long_screenshot:
                if self.verbose:
                    print("[AdaptiveOCR] Region detection skipped: long screenshot detected")
                return False
            return True

        if self.region_detection_mode == 'off':
            return False

        # auto 模式：根据图片特征智能判断
        if profile.is_long_screenshot:
            if self.verbose:
                print("[AdaptiveOCR] Auto: long screenshot, using slicing mode")
            return False

        if profile.recommended_region_detection:
            if self.verbose:
                print("[AdaptiveOCR] Auto: multiple regions detected, using region detection")
            return True

        return False

    def _recognize_with_regions(self, image: Image.Image, profile: Optional[ImageProfile] = None) -> OCROutput:
        """
        基于区域检测的 OCR 识别

        1. 检测图片中的独立区域
        2. 对每个区域单独进行 OCR
        3. 合并结果
        """
        width, height = image.size

        # 检测区域
        detected_regions = self.region_detector.detect(image)

        if self.verbose:
            print(f"[AdaptiveOCR] Image: {width}x{height}")
            print(f"[AdaptiveOCR] Detected {len(detected_regions)} regions")

        # 使用传入的 profile 或重新分析
        if profile is None:
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
            'region_count': len(detected_regions),
        }

        # 对每个区域单独 OCR
        region_results = []
        all_results = []
        all_tables = []
        all_texts = []

        for region in detected_regions:
            if self.verbose:
                print(f"[AdaptiveOCR] Processing region {region.index + 1}: "
                      f"bbox={region.bbox}, color={region.dominant_color}")

            # 分析区域特征
            region_profile = self.analyzer.analyze(region.image)

            # 预处理区域
            if self.enable_preprocessing and region_profile.needs_preprocessing:
                processed_region = self.preprocessor.process(region.image, region_profile)
                processing_info['preprocessing_applied'] = True
            else:
                processed_region = region.image

            # OCR 区域
            if region_profile.needs_slicing:
                results = self._process_with_slicing(processed_region, region_profile)
            else:
                results = self._process_single(processed_region)

            # 调整坐标到原图空间
            x_offset, y_offset = region.bbox[0], region.bbox[1]
            adjusted_results = []
            for r in results:
                adjusted_bbox = [
                    r.bbox[0] + x_offset,
                    r.bbox[1] + y_offset,
                    r.bbox[2],
                    r.bbox[3],
                ] if len(r.bbox) == 4 else r.bbox
                adjusted_results.append(MergedResult(
                    text=r.text,
                    confidence=r.confidence,
                    bbox=adjusted_bbox,
                    slice_index=r.slice_index,
                    original_bbox=r.original_bbox,
                ))

            all_results.extend(adjusted_results)

            # 合并区域文本
            region_text = TextMerger.merge_to_text(results, preserve_lines=True)
            all_texts.append(f"--- 区域 {region.index + 1} ---\n{region_text}")

            # 表格检测
            region_tables = []
            if self.enable_table_detection:
                region_tables = self._detect_tables(results)
                all_tables.extend(region_tables)

            # 保存区域结果
            region_results.append(RegionOCRResult(
                region_index=region.index,
                region_bbox=region.bbox,
                text=region_text,
                results=results,
                tables=region_tables,
                dominant_color=region.dominant_color,
            ))

        # 合并所有文本
        text = "\n\n".join(all_texts)

        if self.verbose:
            print(f"[AdaptiveOCR] Total tables detected: {len(all_tables)}")

        return OCROutput(
            text=text,
            results=all_results,
            profile=profile,
            tables=all_tables,
            processing_info=processing_info,
            regions=region_results,
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

        results = []
        for text, conf, bbox in raw_results:
            bbox_list = bbox if isinstance(bbox, list) else list(bbox)

            # 转换 Vision 坐标系 (y=0 在底部) 到像素坐标系 (y=0 在顶部)
            # Vision: y 是文本框底部距离图片底部的比例
            # 转换: y_from_top = 1 - y - h
            x, y, w, h = bbox_list
            converted_bbox = [x, 1 - y - h, w, h]

            results.append(MergedResult(
                text=text,
                confidence=conf,
                bbox=converted_bbox,
                slice_index=0,
                original_bbox=bbox_list,
            ))

        return results

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
