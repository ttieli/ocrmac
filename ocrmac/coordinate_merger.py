"""坐标合并与去重模块 - 合并切片 OCR 结果并去除重复"""

import sys
from dataclasses import dataclass

if sys.version_info < (3, 9):
    from typing import List, Tuple, Optional
else:
    from typing import Optional
    List, Tuple = list, tuple

from .smart_slicer import SliceInfo


@dataclass
class MergedResult:
    """合并后的 OCR 结果"""
    text: str                   # 识别的文本
    confidence: float           # 置信度
    bbox: List[float]           # [x, y, w, h] 归一化到原图
    slice_index: int            # 来源切片索引
    original_bbox: List[float]  # 原始 bbox（切片内坐标）


class CoordinateMerger:
    """
    切片坐标合并器

    功能：
    1. 将切片内的归一化坐标映射到原图坐标
    2. 去除重叠区域的重复文本
    3. 按阅读顺序排序结果
    """

    def __init__(
        self,
        total_height: int,
        total_width: int,
        overlap_threshold: float = 0.5,
        text_similarity_threshold: float = 0.8,
    ):
        """
        初始化坐标合并器

        Args:
            total_height: 原图总高度
            total_width: 原图总宽度
            overlap_threshold: bbox 重叠阈值（超过此值认为是重复）
            text_similarity_threshold: 文本相似度阈值
        """
        self.total_height = total_height
        self.total_width = total_width
        self.overlap_threshold = overlap_threshold
        self.text_similarity_threshold = text_similarity_threshold

    def merge_results(
        self,
        slice_results: List[Tuple[SliceInfo, List[Tuple]]],
    ) -> List[MergedResult]:
        """
        合并多个切片的 OCR 结果

        Args:
            slice_results: [(slice_info, ocr_results), ...]
                ocr_results: [(text, confidence, [x, y, w, h]), ...]

        Returns:
            合并后的结果列表，坐标已映射到原图
        """
        all_results = []

        for slice_info, ocr_results in slice_results:
            for item in ocr_results:
                # 处理不同格式的 OCR 结果
                if len(item) == 3:
                    text, conf, bbox = item
                elif len(item) == 2:
                    text, bbox = item
                    conf = 1.0
                else:
                    continue

                # 确保 bbox 是列表
                if not isinstance(bbox, list):
                    bbox = list(bbox)

                # 映射坐标
                mapped_bbox = self._map_coordinates(
                    bbox,
                    slice_info.y_start,
                    slice_info.y_end
                )

                all_results.append(MergedResult(
                    text=text,
                    confidence=conf,
                    bbox=mapped_bbox,
                    slice_index=slice_info.index,
                    original_bbox=bbox,
                ))

        # 去除重叠区的重复文本
        deduplicated = self._deduplicate(all_results)

        # 按阅读顺序排序（从上到下，从左到右）
        sorted_results = self._sort_by_reading_order(deduplicated)

        return sorted_results

    def _map_coordinates(
        self,
        bbox: List[float],
        slice_y_start: int,
        slice_y_end: int
    ) -> List[float]:
        """
        将切片内坐标映射到原图坐标

        Args:
            bbox: [x, y, w, h] 切片内归一化坐标
            slice_y_start: 切片在原图中的起始 y
            slice_y_end: 切片在原图中的结束 y

        Returns:
            [x, y, w, h] 原图归一化坐标
        """
        x, y, w, h = bbox
        slice_height = slice_y_end - slice_y_start

        # 切片内的绝对像素坐标
        y_abs_in_slice = y * slice_height
        h_abs = h * slice_height

        # 映射到原图的绝对坐标
        y_abs = slice_y_start + y_abs_in_slice

        # 归一化到原图
        y_norm = y_abs / self.total_height
        h_norm = h_abs / self.total_height

        # x 和 w 不变（因为宽度没有切片）
        return [x, y_norm, w, h_norm]

    def _deduplicate(self, results: List[MergedResult]) -> List[MergedResult]:
        """
        去除重叠区的重复文本

        策略：
        1. 如果两个结果的 bbox 垂直重叠 > threshold
        2. 且文本相似度高
        3. 保留置信度更高的（或来自较早切片的）
        """
        if not results:
            return []

        # 按 y 坐标排序
        sorted_results = sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))

        deduplicated = []
        skip_indices = set()

        for i, current in enumerate(sorted_results):
            if i in skip_indices:
                continue

            # 检查后续结果是否与当前重复
            for j in range(i + 1, len(sorted_results)):
                if j in skip_indices:
                    continue

                other = sorted_results[j]

                # 如果 y 坐标差距太大，不可能重叠
                y_diff = other.bbox[1] - (current.bbox[1] + current.bbox[3])
                if y_diff > 0.02:  # 超过 2% 的高度差
                    break

                # 检查是否重叠
                overlap_ratio = self._calculate_overlap(current.bbox, other.bbox)

                if overlap_ratio > self.overlap_threshold:
                    # 检查文本相似度
                    similarity = self._text_similarity(current.text, other.text)

                    if similarity > self.text_similarity_threshold:
                        # 标记为重复，保留置信度更高的
                        if other.confidence > current.confidence:
                            skip_indices.add(i)
                            break
                        else:
                            skip_indices.add(j)

            if i not in skip_indices:
                deduplicated.append(current)

        return deduplicated

    def _calculate_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算两个 bbox 的重叠比例

        使用 IoU (Intersection over Union) 的变体
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 计算交集
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)

        if left >= right or top >= bottom:
            return 0.0

        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        min_area = min(area1, area2)

        if min_area <= 0:
            return 0.0

        # 使用较小面积作为分母，这样小 bbox 被大 bbox 包含时也能检测到
        return intersection / min_area

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        使用简化的字符重叠比例
        """
        if not text1 or not text2:
            return 0.0

        # 去除空白字符
        t1 = text1.strip()
        t2 = text2.strip()

        if t1 == t2:
            return 1.0

        # 检查是否一个是另一个的子串
        if t1 in t2 or t2 in t1:
            return 0.9

        # 计算字符重叠
        set1 = set(t1)
        set2 = set(t2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _sort_by_reading_order(self, results: List[MergedResult]) -> List[MergedResult]:
        """
        按阅读顺序排序（从上到下，从左到右）

        考虑到 Vision 坐标系 y=0 在底部，需要反转 y 排序
        """
        # Vision 坐标系：y 值越大越靠上
        # 所以按 y 降序排列（先处理 y 大的，即上方的）
        # 但由于我们已经映射到了像素坐标系（y=0 在顶部）
        # 所以直接按 y 升序排列
        return sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))


class TextMerger:
    """
    文本合并器

    将多个 OCR 结果合并为连续文本
    """

    @staticmethod
    def merge_to_text(results: List[MergedResult], preserve_lines: bool = True) -> str:
        """
        合并 OCR 结果为文本

        Args:
            results: OCR 结果列表
            preserve_lines: 是否保留行结构

        Returns:
            合并后的文本
        """
        if not results:
            return ""

        if not preserve_lines:
            return " ".join(r.text for r in results)

        # 按 y 坐标分组（同一行的文本）
        lines = TextMerger._group_by_lines(results)

        # 每行内按 x 坐标排序并合并
        text_lines = []
        for line_results in lines:
            sorted_line = sorted(line_results, key=lambda r: r.bbox[0])
            line_text = " ".join(r.text for r in sorted_line)
            text_lines.append(line_text)

        return "\n".join(text_lines)

    @staticmethod
    def _group_by_lines(
        results: List[MergedResult],
        y_tolerance: float = 0.01
    ) -> List[List[MergedResult]]:
        """
        按行分组

        Args:
            results: OCR 结果列表
            y_tolerance: y 坐标容忍度

        Returns:
            分组后的结果列表
        """
        if not results:
            return []

        sorted_results = sorted(results, key=lambda r: r.bbox[1])
        lines = []
        current_line = [sorted_results[0]]
        current_y = sorted_results[0].bbox[1]

        for result in sorted_results[1:]:
            y_center = result.bbox[1] + result.bbox[3] / 2
            current_y_center = current_y + current_line[0].bbox[3] / 2

            if abs(y_center - current_y_center) <= y_tolerance:
                current_line.append(result)
            else:
                lines.append(current_line)
                current_line = [result]
                current_y = result.bbox[1]

        if current_line:
            lines.append(current_line)

        return lines
