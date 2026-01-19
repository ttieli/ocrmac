"""智能 OCR 模式 - 自动检测最佳处理策略"""

import sys
from dataclasses import dataclass
from PIL import Image

if sys.version_info < (3, 9):
    from typing import List, Optional, Dict, Any, Tuple
else:
    from typing import Optional, Any
    List, Dict, Tuple = list, dict, tuple

from .image_analyzer import ImageAnalyzer, ImageProfile
from .adaptive_ocr import AdaptiveOCR, OCROutput
from .layout_analyzer import LayoutAnalyzer, TextCleaner


@dataclass
class OCRStrategy:
    """OCR 处理策略"""
    name: str
    use_regions: bool = False
    use_layout: bool = False
    use_preprocessing: bool = True
    description: str = ""


@dataclass
class QualityMetrics:
    """OCR 质量指标"""
    char_count: int              # 识别字符数
    line_count: int              # 行数
    avg_confidence: float        # 平均置信度
    duplicate_ratio: float       # 重复比例
    coherence_score: float       # 文本连贯性评分
    structure_score: float       # 结构评分（段落、标题检测）


class SmartOCR:
    """
    智能 OCR 处理器

    自动检测图片类型，选择最佳处理策略：
    1. 分析图片特征（单次快速分析）
    2. 根据特征选择初始策略
    3. 执行 OCR 并评估质量
    4. 如果质量不佳，尝试备选策略
    5. 返回最佳结果
    """

    # 策略定义
    STRATEGY_ARTICLE = OCRStrategy(
        name="article",
        use_regions=False,
        use_layout=True,
        description="长文章模式：切片处理 + 布局分析"
    )

    STRATEGY_MULTI_DOC = OCRStrategy(
        name="multi_doc",
        use_regions=True,
        use_layout=True,
        description="多文档模式：区域检测 + 布局分析"
    )

    STRATEGY_SIMPLE = OCRStrategy(
        name="simple",
        use_regions=False,
        use_layout=False,
        description="简单模式：直接 OCR"
    )

    def __init__(
        self,
        framework: str = 'livetext',
        language: Optional[str] = None,
        max_retries: int = 2,
        quality_threshold: float = 0.6,
        verbose: bool = False,
    ):
        """
        初始化智能 OCR

        Args:
            framework: OCR 框架
            language: 语言偏好
            max_retries: 最大重试次数（不同策略）
            quality_threshold: 质量阈值，低于此值尝试其他策略
            verbose: 详细输出
        """
        self.framework = framework
        self.language = language
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self.verbose = verbose

        self.analyzer = ImageAnalyzer()
        self.layout_analyzer = LayoutAnalyzer()
        self.text_cleaner = TextCleaner()

    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        智能处理图片

        Args:
            image: PIL Image 对象

        Returns:
            处理结果字典，包含：
            - text: 最终文本
            - strategy: 使用的策略
            - quality: 质量指标
            - details: OCR 详情
            - alternatives: 尝试过的其他策略结果（如果有）
        """
        width, height = image.size

        # 1. 分析图片特征
        profile = self.analyzer.analyze(image)

        if self.verbose:
            print(f"[SmartOCR] Image: {width}x{height}")
            print(f"[SmartOCR] Aspect ratio: {height/width:.2f}")
            print(f"[SmartOCR] Is long screenshot: {profile.is_long_screenshot}")
            print(f"[SmartOCR] Has multiple regions: {profile.has_multiple_regions}")

        # 2. 选择初始策略
        strategies = self._select_strategies(profile)

        if self.verbose:
            print(f"[SmartOCR] Selected strategies: {[s.name for s in strategies]}")

        # 3. 尝试策略，评估质量
        results = []
        best_result = None
        best_quality = 0.0

        for i, strategy in enumerate(strategies[:self.max_retries + 1]):
            if self.verbose:
                print(f"[SmartOCR] Trying strategy {i+1}/{len(strategies)}: {strategy.name}")

            # 执行 OCR
            ocr_output = self._execute_strategy(image, profile, strategy)

            # 评估质量
            quality = self._evaluate_quality(ocr_output, strategy)
            quality_score = self._calculate_quality_score(quality)

            if self.verbose:
                print(f"[SmartOCR] Quality score: {quality_score:.2f} "
                      f"(chars: {quality.char_count}, lines: {quality.line_count})")

            result = {
                'strategy': strategy,
                'output': ocr_output,
                'quality': quality,
                'score': quality_score,
            }
            results.append(result)

            # 更新最佳结果
            if quality_score > best_quality:
                best_quality = quality_score
                best_result = result

            # 如果质量足够好，停止尝试
            if quality_score >= self.quality_threshold:
                if self.verbose:
                    print(f"[SmartOCR] Quality sufficient, stopping")
                break

        # 4. 应用布局分析（如果策略要求）
        final_text = best_result['output'].text
        if best_result['strategy'].use_layout and best_result['output'].results:
            final_text = self._apply_layout(best_result['output'].results)

        # 5. 文本清理
        final_text = self.text_cleaner.clean_text(final_text)

        return {
            'text': final_text,
            'strategy': best_result['strategy'].name,
            'quality': best_result['quality'],
            'score': best_quality,
            'details': best_result['output'].results,
            'profile': profile,
            'attempts': len(results),
            'alternatives': results[1:] if len(results) > 1 else [],
        }

    def _select_strategies(self, profile: ImageProfile) -> List[OCRStrategy]:
        """
        根据图片特征选择策略优先级

        Args:
            profile: 图片特征

        Returns:
            策略列表（按优先级排序）
        """
        strategies = []

        # 长截图 → 优先文章模式
        if profile.is_long_screenshot:
            strategies.append(self.STRATEGY_ARTICLE)
            strategies.append(self.STRATEGY_SIMPLE)

        # 多区域 → 优先多文档模式
        elif profile.has_multiple_regions or profile.recommended_region_detection:
            strategies.append(self.STRATEGY_MULTI_DOC)
            strategies.append(self.STRATEGY_ARTICLE)
            strategies.append(self.STRATEGY_SIMPLE)

        # 普通图片 → 先尝试简单模式，再尝试布局分析
        else:
            # 如果图片较大，优先布局分析
            if profile.height > 2000 or profile.width > 2000:
                strategies.append(self.STRATEGY_ARTICLE)
                strategies.append(self.STRATEGY_SIMPLE)
            else:
                strategies.append(self.STRATEGY_SIMPLE)
                strategies.append(self.STRATEGY_ARTICLE)

        return strategies

    def _execute_strategy(
        self,
        image: Image.Image,
        profile: ImageProfile,
        strategy: OCRStrategy
    ) -> OCROutput:
        """执行指定策略的 OCR"""
        ocr = AdaptiveOCR(
            framework=self.framework,
            language=self.language,
            enable_region_detection='on' if strategy.use_regions else 'off',
            enable_preprocessing=strategy.use_preprocessing,
            verbose=self.verbose,
        )

        return ocr.recognize(image)

    def _evaluate_quality(self, output: OCROutput, strategy: OCRStrategy) -> QualityMetrics:
        """
        评估 OCR 结果质量

        Args:
            output: OCR 输出
            strategy: 使用的策略

        Returns:
            质量指标
        """
        text = output.text
        results = output.results

        # 基础指标
        char_count = len(text.replace('\n', '').replace(' ', ''))
        lines = [l for l in text.split('\n') if l.strip()]
        line_count = len(lines)

        # 平均置信度
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
        else:
            avg_confidence = 0.0

        # 重复检测
        duplicate_ratio = self._calculate_duplicate_ratio(lines)

        # 连贯性评分（基于句子完整性）
        coherence_score = self._calculate_coherence(lines)

        # 结构评分（基于段落和标题）
        structure_score = self._calculate_structure_score(text, results)

        return QualityMetrics(
            char_count=char_count,
            line_count=line_count,
            avg_confidence=avg_confidence,
            duplicate_ratio=duplicate_ratio,
            coherence_score=coherence_score,
            structure_score=structure_score,
        )

    def _calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """
        计算综合质量评分 (0-1)

        权重：
        - 字符数量：30%（越多越好，但有上限）
        - 置信度：20%
        - 重复率：20%（越低越好）
        - 连贯性：15%
        - 结构性：15%
        """
        # 字符数量评分（1000字以上满分）
        char_score = min(metrics.char_count / 1000, 1.0)

        # 置信度评分
        conf_score = metrics.avg_confidence

        # 重复率评分（反向）
        dup_score = 1.0 - metrics.duplicate_ratio

        # 连贯性和结构性
        coherence = metrics.coherence_score
        structure = metrics.structure_score

        # 加权平均
        score = (
            char_score * 0.30 +
            conf_score * 0.20 +
            dup_score * 0.20 +
            coherence * 0.15 +
            structure * 0.15
        )

        return score

    def _calculate_duplicate_ratio(self, lines: List[str]) -> float:
        """计算重复行比例"""
        if not lines:
            return 0.0

        unique_lines = set()
        duplicates = 0

        for line in lines:
            normalized = line.strip()
            if len(normalized) < 5:  # 忽略短行
                continue
            if normalized in unique_lines:
                duplicates += 1
            else:
                unique_lines.add(normalized)

        total = len([l for l in lines if len(l.strip()) >= 5])
        return duplicates / total if total > 0 else 0.0

    def _calculate_coherence(self, lines: List[str]) -> float:
        """
        计算文本连贯性

        基于：
        - 句子完整性（是否以标点结尾）
        - 行长度分布
        """
        if not lines:
            return 0.0

        # 句子完整性
        sentence_endings = '。！？.!?'
        complete_sentences = sum(1 for l in lines if l.strip() and l.strip()[-1] in sentence_endings)
        sentence_ratio = complete_sentences / len(lines) if lines else 0

        # 行长度一致性（标准差越小越好）
        lengths = [len(l) for l in lines if l.strip()]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            std_dev = variance ** 0.5
            # 标准差小于平均长度的 50% 得满分
            length_consistency = max(0, 1 - std_dev / (avg_len + 1))
        else:
            length_consistency = 0

        return (sentence_ratio * 0.6 + length_consistency * 0.4)

    def _calculate_structure_score(self, text: str, results: List) -> float:
        """
        计算结构评分

        基于：
        - 是否有明显的段落分隔
        - 是否有标题/小节
        """
        score = 0.0

        # 段落检测（连续空行）
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.3

        # 标题检测
        header_patterns = ['【', '】', '##', '**']
        has_headers = any(p in text for p in header_patterns)
        if has_headers:
            score += 0.3

        # 列表检测
        list_patterns = ['1.', '2.', '•', '-']
        has_lists = any(p in text for p in list_patterns)
        if has_lists:
            score += 0.2

        # 结果数量（越多说明检测越细致）
        if results and len(results) > 20:
            score += 0.2

        return min(score, 1.0)

    def _apply_layout(self, results: List) -> str:
        """应用布局分析"""
        from .layout_analyzer import format_with_layout

        # 转换为 layout_analyzer 需要的格式
        ocr_results = [
            (r.text, r.confidence, r.bbox)
            for r in results
        ]

        return format_with_layout(ocr_results)


def smart_ocr(
    image_path_or_pil,
    framework: str = 'livetext',
    language: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    便捷函数：智能 OCR 处理

    Args:
        image_path_or_pil: 图片路径或 PIL Image
        framework: OCR 框架
        language: 语言偏好
        verbose: 详细输出

    Returns:
        处理结果字典
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil)
    else:
        image = image_path_or_pil

    ocr = SmartOCR(
        framework=framework,
        language=language,
        verbose=verbose,
    )

    return ocr.process(image)
