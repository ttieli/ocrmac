"""智能切片模块 - 避免切断文字行和代码块"""

import sys
from PIL import Image
import numpy as np

if sys.version_info < (3, 9):
    from typing import List, Generator, Optional
else:
    from typing import Generator, Optional
    List = list

from ..analysis.image import ImageProfile, ContentType
from ..types import SliceInfo


class SmartSlicer:
    """
    智能切片器

    特点：
    1. 基于水平投影分析找到安全切割点（文字行之间的空白）
    2. 避免切断文字行
    3. 感知代码块区域
    4. 支持重叠切片以避免边界文字丢失
    """

    def __init__(
        self,
        max_height: int = 5000,
        min_height: int = 2000,
        overlap: int = 400,
        prefer_blank_cut: bool = True,
    ):
        """
        初始化切片器

        Args:
            max_height: 最大切片高度
            min_height: 最小切片高度（避免切片过小）
            overlap: 切片重叠区域
            prefer_blank_cut: 是否优先在空白处切割
        """
        self.max_height = max_height
        self.min_height = min_height
        self.overlap = overlap
        self.prefer_blank_cut = prefer_blank_cut

    def slice(
        self, image: Image.Image, profile: Optional[ImageProfile] = None
    ) -> Generator[SliceInfo, None, None]:
        """
        智能切片

        Args:
            image: 输入图片
            profile: 图片特征（可选，如果提供会使用推荐参数）

        Yields:
            SliceInfo 对象
        """
        width, height = image.size

        # 使用 profile 中的推荐参数（如果提供）
        if profile is not None:
            max_height = profile.recommended_slice_height or self.max_height
            overlap = profile.recommended_overlap or self.overlap
        else:
            max_height = self.max_height
            overlap = self.overlap

        # 如果图片不需要切片
        if height <= max_height:
            yield SliceInfo(
                index=0,
                y_start=0,
                y_end=height,
                image=image,
                is_code_block=False,
            )
            return

        # 使用简单可靠的固定间隔切片
        # 每个切片最大 max_height，重叠 overlap
        slice_index = 0
        current_y = 0

        while current_y < height:
            # 计算切片结束位置
            end_y = min(current_y + max_height, height)

            # 裁剪切片
            slice_img = image.crop((0, current_y, width, end_y))

            # 检测此切片是否为代码块
            is_code = self._is_code_region(image, current_y, end_y)

            yield SliceInfo(
                index=slice_index,
                y_start=current_y,
                y_end=end_y,
                image=slice_img,
                is_code_block=is_code,
            )

            slice_index += 1

            # 如果已经到达图片底部，结束
            if end_y >= height:
                break

            # 下一个切片的起始位置（考虑重叠）
            current_y = end_y - overlap

    def _find_safe_cut_points(
        self, image: Image.Image, max_height: int, overlap: int
    ) -> List[int]:
        """
        找到安全切割点（基于水平投影分析）

        安全点 = 像素行几乎全白或全黑（行间空白或区块分隔）
        强制保证每个切片不超过 max_height
        """
        gray = np.array(image.convert('L'))
        height = gray.shape[0]

        # 计算每行的像素变化量（方差）
        row_variance = gray.var(axis=1)

        # 低变化量 = 空白行（纯白或纯黑）
        threshold = np.percentile(row_variance, 15)
        blank_rows = row_variance < threshold

        # 找到连续空白行区域
        blank_regions = self._find_blank_regions(blank_rows)

        # 生成切割点，确保每个切片不超过 max_height
        cut_points = []
        last_cut = 0

        # 遍历整个高度，按 max_height 间隔找切割点
        current_target = max_height

        while current_target < height:
            # 在 current_target 附近找最近的空白区域
            best_cut = None
            search_range = max_height * 0.3  # 在目标点前后 30% 范围内搜索

            for region_start, region_end in blank_regions:
                region_center = (region_start + region_end) // 2

                # 只考虑在搜索范围内的空白区域
                if current_target - search_range <= region_center <= current_target + search_range:
                    # 选择最接近目标的空白区域
                    if best_cut is None or abs(region_center - current_target) < abs(best_cut - current_target):
                        best_cut = region_center

            # 如果找到合适的空白区域，使用它；否则强制在目标位置切割
            if best_cut is not None:
                cut_points.append(best_cut)
                last_cut = best_cut
            else:
                cut_points.append(current_target)
                last_cut = current_target

            # 下一个目标（考虑重叠）
            current_target = last_cut + max_height - overlap

        # 添加最后的切割点（图片底部）
        cut_points.append(height)

        return sorted(set(cut_points))

    def _find_blank_regions(self, blank_rows: np.ndarray) -> List[tuple]:
        """
        找到连续的空白行区域

        Returns:
            [(start, end), ...] 空白区域列表
        """
        regions = []
        in_blank = False
        blank_start = 0

        for y, is_blank in enumerate(blank_rows):
            if is_blank:
                if not in_blank:
                    blank_start = y
                    in_blank = True
            else:
                if in_blank:
                    # 只保留足够宽的空白区域（至少 5 像素）
                    if y - blank_start >= 5:
                        regions.append((blank_start, y))
                    in_blank = False

        # 处理最后一个区域
        if in_blank:
            if len(blank_rows) - blank_start >= 5:
                regions.append((blank_start, len(blank_rows)))

        return regions

    def _generate_fixed_cut_points(
        self, height: int, max_height: int, overlap: int
    ) -> List[int]:
        """生成固定间隔的切割点（不考虑内容）"""
        cut_points = []
        current = max_height

        while current < height:
            cut_points.append(current)
            current += max_height - overlap

        cut_points.append(height)
        return cut_points

    def _is_code_region(self, image: Image.Image, y_start: int, y_end: int) -> bool:
        """
        检测指定区域是否主要是代码块

        代码块特征：大面积深色背景
        """
        region = image.crop((0, y_start, image.width, y_end))
        gray = np.array(region.convert('L'))

        # 代码块特征：大面积深色背景
        dark_ratio = np.mean(gray < 60)
        return dark_ratio > 0.4

    def get_slice_count(self, image: Image.Image, profile: Optional[ImageProfile] = None) -> int:
        """
        预估切片数量（不实际执行切片）

        Args:
            image: 输入图片
            profile: 图片特征

        Returns:
            预估的切片数量
        """
        width, height = image.size

        if profile is not None:
            max_height = profile.recommended_slice_height or self.max_height
            overlap = profile.recommended_overlap or self.overlap
        else:
            max_height = self.max_height
            overlap = self.overlap

        if height <= max_height:
            return 1

        # 估算切片数量
        effective_height = max_height - overlap
        return max(1, int(np.ceil((height - overlap) / effective_height)))


class SimpleSliceer:
    """
    简单切片器

    固定高度切片，不进行智能分析
    适用于快速处理或已知内容结构的场景
    """

    def __init__(self, slice_height: int = 5000, overlap: int = 400):
        self.slice_height = slice_height
        self.overlap = overlap

    def slice(self, image: Image.Image) -> Generator[SliceInfo, None, None]:
        """固定高度切片"""
        width, height = image.size

        if height <= self.slice_height:
            yield SliceInfo(
                index=0,
                y_start=0,
                y_end=height,
                image=image,
                is_code_block=False,
            )
            return

        current_y = 0
        slice_index = 0

        while current_y < height:
            end_y = min(current_y + self.slice_height, height)

            slice_img = image.crop((0, current_y, width, end_y))

            yield SliceInfo(
                index=slice_index,
                y_start=current_y,
                y_end=end_y,
                image=slice_img,
                is_code_block=False,
            )

            slice_index += 1

            if end_y >= height:
                break

            current_y = end_y - self.overlap
