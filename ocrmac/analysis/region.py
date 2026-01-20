"""文档区域检测模块 - 自动检测和分割图片中的多个文档区域

适用场景：
- 一张照片中包含多张纸/文档
- 不同颜色背景的区块
- 多个表格并排
- 墙上贴的多张告示

检测方法：
1. 颜色聚类分割 - 基于 K-means 或颜色阈值
2. 轮廓检测 - 基于边缘和矩形拟合
3. 连通区域分析 - 基于二值化后的连通组件
"""

import sys
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

if sys.version_info < (3, 9):
    from typing import List, Tuple, Optional
else:
    from typing import Optional
    List, Tuple = list, tuple

# 尝试导入 OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class DetectedRegion:
    """检测到的区域"""
    index: int                  # 区域索引（按阅读顺序）
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    image: Image.Image          # 裁剪后的区域图片
    confidence: float           # 检测置信度
    region_type: str            # 区域类型: 'document', 'table', 'text', 'unknown'
    dominant_color: Optional[Tuple[int, int, int]] = None  # 主要颜色 (R, G, B)


class RegionDetector:
    """
    文档区域检测器

    自动检测图片中的多个独立文档区域，支持：
    - 不同颜色背景的文档
    - 物理纸张边界
    - 表格区块
    """

    # 配置参数
    MIN_REGION_RATIO = 0.02     # 最小区域面积占比（相对于整图）
    MAX_REGION_RATIO = 0.95     # 最大区域面积占比
    MIN_ASPECT_RATIO = 0.1      # 最小宽高比
    MAX_ASPECT_RATIO = 10.0     # 最大宽高比
    MERGE_OVERLAP_THRESHOLD = 0.5  # 重叠区域合并阈值

    def __init__(
        self,
        method: str = 'auto',
        min_regions: int = 1,
        max_regions: int = 10,
        color_clusters: int = 5,
    ):
        """
        初始化区域检测器

        Args:
            method: 检测方法 ('auto', 'color', 'contour', 'combined')
            min_regions: 最少检测区域数
            max_regions: 最多检测区域数
            color_clusters: 颜色聚类数量
        """
        self.method = method
        self.min_regions = min_regions
        self.max_regions = max_regions
        self.color_clusters = color_clusters

    def detect(self, image: Image.Image) -> List[DetectedRegion]:
        """
        检测图片中的文档区域

        Args:
            image: PIL Image 对象

        Returns:
            检测到的区域列表（按阅读顺序排序）
        """
        if not CV2_AVAILABLE:
            # 没有 OpenCV，返回整图作为单一区域
            return [DetectedRegion(
                index=0,
                bbox=(0, 0, image.width, image.height),
                image=image,
                confidence=1.0,
                region_type='unknown',
            )]

        arr = np.array(image)

        # 选择检测方法
        if self.method == 'auto':
            regions = self._detect_auto(arr)
        elif self.method == 'color':
            regions = self._detect_by_color(arr)
        elif self.method == 'contour':
            regions = self._detect_by_contour(arr)
        else:  # combined
            regions = self._detect_combined(arr)

        # 过滤无效区域
        regions = self._filter_regions(regions, arr.shape)

        # 合并重叠区域
        regions = self._merge_overlapping(regions)

        # 按阅读顺序排序（上→下，左→右）
        regions = self._sort_by_reading_order(regions)

        # 如果没有检测到有效区域，返回整图
        if not regions:
            return [DetectedRegion(
                index=0,
                bbox=(0, 0, image.width, image.height),
                image=image,
                confidence=1.0,
                region_type='unknown',
            )]

        # 裁剪区域图片
        result = []
        for i, (bbox, conf, rtype, color) in enumerate(regions):
            x, y, w, h = bbox
            cropped = image.crop((x, y, x + w, y + h))
            result.append(DetectedRegion(
                index=i,
                bbox=bbox,
                image=cropped,
                confidence=conf,
                region_type=rtype,
                dominant_color=color,
            ))

        return result

    def _detect_auto(self, arr: np.ndarray) -> List:
        """自动选择最佳检测方法"""
        # 先尝试颜色分割
        color_regions = self._detect_by_color(arr)

        # 如果颜色分割效果好（检测到多个区域），使用颜色分割
        if len(color_regions) >= 2:
            return color_regions

        # 否则尝试轮廓检测
        contour_regions = self._detect_by_contour(arr)

        if len(contour_regions) >= 2:
            return contour_regions

        # 使用组合方法
        return self._detect_combined(arr)

    def _detect_by_color(self, arr: np.ndarray) -> List:
        """基于颜色聚类的区域检测"""
        height, width = arr.shape[:2]

        # 转换到 LAB 色彩空间（对颜色差异更敏感）
        if len(arr.shape) == 3:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        else:
            lab = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB)

        # 重塑为像素列表
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # K-means 聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, self.color_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # 重塑标签为图像形状
        labels = labels.reshape(height, width)

        # 对每个聚类找区域
        regions = []
        for cluster_id in range(self.color_clusters):
            mask = (labels == cluster_id).astype(np.uint8) * 255

            # 形态学操作清理噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                total_area = height * width

                # 过滤太小或太大的区域
                if area < total_area * self.MIN_REGION_RATIO:
                    continue
                if area > total_area * self.MAX_REGION_RATIO:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # 计算主要颜色
                center_lab = centers[cluster_id]
                center_rgb = cv2.cvtColor(
                    np.array([[center_lab]], dtype=np.uint8),
                    cv2.COLOR_LAB2RGB
                )[0, 0]

                regions.append((
                    (x, y, w, h),
                    0.8,  # confidence
                    'document',
                    tuple(center_rgb.tolist()),
                ))

        return regions

    def _detect_by_contour(self, arr: np.ndarray) -> List:
        """基于轮廓检测的区域检测"""
        height, width = arr.shape[:2]

        # 转灰度
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 膨胀边缘，连接断开的线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area = height * width

            if area < total_area * self.MIN_REGION_RATIO:
                continue
            if area > total_area * self.MAX_REGION_RATIO:
                continue

            # 多边形拟合
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 获取边界矩形
            x, y, w, h = cv2.boundingRect(approx)

            # 检查宽高比
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
                continue

            # 计算区域主要颜色
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(arr, mask=mask)[:3]

            regions.append((
                (x, y, w, h),
                0.7,  # confidence
                'document',
                tuple(int(c) for c in mean_color),
            ))

        return regions

    def _detect_combined(self, arr: np.ndarray) -> List:
        """组合多种方法检测"""
        color_regions = self._detect_by_color(arr)
        contour_regions = self._detect_by_contour(arr)

        # 合并两种方法的结果
        all_regions = color_regions + contour_regions

        # 去重（基于 IoU）
        return self._merge_overlapping(all_regions)

    def _filter_regions(self, regions: List, shape: tuple) -> List:
        """过滤无效区域"""
        height, width = shape[:2]
        total_area = height * width

        filtered = []
        for region in regions:
            bbox, conf, rtype, color = region
            x, y, w, h = bbox
            area = w * h

            # 面积检查
            if area < total_area * self.MIN_REGION_RATIO:
                continue
            if area > total_area * self.MAX_REGION_RATIO:
                continue

            # 宽高比检查
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
                continue

            # 边界检查
            if x < 0 or y < 0 or x + w > width or y + h > height:
                continue

            filtered.append(region)

        return filtered

    def _merge_overlapping(self, regions: List) -> List:
        """合并重叠的区域，同时处理包含关系"""
        if len(regions) <= 1:
            return regions

        # 按面积降序排序
        regions = sorted(regions, key=lambda r: r[0][2] * r[0][3], reverse=True)

        merged = []
        used = set()

        for i, region_i in enumerate(regions):
            if i in used:
                continue

            bbox_i = region_i[0]

            # 检查是否与其他区域重叠或被包含
            for j, region_j in enumerate(regions):
                if i == j or j in used:
                    continue

                bbox_j = region_j[0]

                # 检查是否有包含关系（小区域被大区域包含）
                if self._is_contained(bbox_j, bbox_i):
                    used.add(j)
                    continue

                # 检查 IoU 重叠
                iou = self._calculate_iou(bbox_i, bbox_j)
                if iou > self.MERGE_OVERLAP_THRESHOLD:
                    used.add(j)

            merged.append(region_i)
            used.add(i)

        return merged

    def _is_contained(self, inner_bbox: tuple, outer_bbox: tuple, threshold: float = 0.8) -> bool:
        """
        检查 inner_bbox 是否大部分被 outer_bbox 包含

        Args:
            inner_bbox: 内部边界框 (x, y, w, h)
            outer_bbox: 外部边界框 (x, y, w, h)
            threshold: 包含阈值，inner 被 outer 覆盖的比例

        Returns:
            是否被包含
        """
        x1, y1, w1, h1 = inner_bbox
        x2, y2, w2, h2 = outer_bbox

        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return False

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        inner_area = w1 * h1

        # 如果 inner 的大部分被 outer 覆盖，则认为被包含
        return inner_area > 0 and (inter_area / inner_area) >= threshold

    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """计算两个边界框的 IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _sort_by_reading_order(self, regions: List) -> List:
        """按阅读顺序排序（上→下，左→右）"""
        if not regions:
            return regions

        # 计算行分组（y 坐标相近的区域归为一行）
        def get_row_key(region):
            bbox = region[0]
            y_center = bbox[1] + bbox[3] / 2
            return y_center

        # 先按 y 坐标排序
        sorted_regions = sorted(regions, key=get_row_key)

        # 分组：y 坐标差距小于高度 50% 的归为一行
        rows = []
        current_row = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            prev_region = current_row[-1]
            prev_y = prev_region[0][1] + prev_region[0][3] / 2
            curr_y = region[0][1] + region[0][3] / 2
            prev_h = prev_region[0][3]

            if abs(curr_y - prev_y) < prev_h * 0.5:
                # 同一行
                current_row.append(region)
            else:
                # 新行
                rows.append(current_row)
                current_row = [region]

        rows.append(current_row)

        # 每行内按 x 坐标排序
        result = []
        for row in rows:
            row_sorted = sorted(row, key=lambda r: r[0][0])
            result.extend(row_sorted)

        return result


def split_document_regions(image: Image.Image, method: str = 'auto') -> List[DetectedRegion]:
    """
    便捷函数：分割文档区域

    Args:
        image: PIL Image 对象
        method: 检测方法 ('auto', 'color', 'contour', 'combined')

    Returns:
        检测到的区域列表
    """
    detector = RegionDetector(method=method)
    return detector.detect(image)
