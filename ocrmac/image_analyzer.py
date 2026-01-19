"""图片特征分析模块 - 自动检测图片特征并生成处理建议"""

import sys
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import numpy as np

if sys.version_info < (3, 9):
    from typing import Tuple, Optional
else:
    Tuple = tuple
    Optional = None.__class__


class ImageSource(Enum):
    """图片来源类型"""
    DIGITAL = "digital"      # 数字渲染（截图、PDF）
    PHOTO = "photo"          # 物理拍摄（相机、扫描）
    UNKNOWN = "unknown"


class ContentType(Enum):
    """内容类型"""
    DOCUMENT = "document"    # 纯文档（文字为主）
    TABLE = "table"          # 表格（结构化数据）
    MIXED = "mixed"          # 混合内容（文字+代码+图片）
    UNKNOWN = "unknown"


@dataclass
class ImageProfile:
    """图片特征画像"""
    width: int
    height: int
    source: ImageSource
    content_type: ContentType
    contrast_level: float      # 0-1, 1=高对比度
    noise_level: float         # 0-1, 0=无噪声
    needs_slicing: bool
    needs_preprocessing: bool
    has_code_blocks: bool
    has_tables: bool
    recommended_slice_height: int
    recommended_overlap: int


class ImageAnalyzer:
    """
    图片特征分析器

    自动检测图片的以下特征：
    - 来源（数字渲染 vs 物理拍摄）
    - 对比度水平
    - 噪声水平
    - 内容类型（文档/表格/混合）
    - 是否包含代码块

    基于检测结果生成处理建议。
    """

    # 阈值配置
    SLICE_THRESHOLD = 4000       # 超过此高度需要切片
    MAX_SAFE_HEIGHT = 5000       # 安全切片高度 (LiveText 在 6000px+ 会崩溃)
    CONTRAST_THRESHOLD = 0.5     # 低于此值需要预处理
    NOISE_THRESHOLD = 0.4        # 高于此值需要预处理

    def analyze(self, image: Image.Image) -> ImageProfile:
        """
        分析图片特征，生成处理建议

        Args:
            image: PIL Image 对象

        Returns:
            ImageProfile 对象，包含图片特征和处理建议
        """
        width, height = image.size

        # 1. 检测图片来源（数字 vs 物理拍摄）
        source = self._detect_source(image)

        # 2. 检测对比度
        contrast = self._calculate_contrast(image)

        # 3. 检测噪声水平
        noise = self._calculate_noise(image)

        # 4. 检测内容类型
        content_type = self._detect_content_type(image)

        # 5. 检测代码块
        has_code = self._detect_code_blocks(image)

        # 6. 生成处理建议
        needs_slicing = height > self.SLICE_THRESHOLD
        needs_preprocessing = (
            contrast < self.CONTRAST_THRESHOLD or
            noise > self.NOISE_THRESHOLD or
            source == ImageSource.PHOTO
        )

        # 7. 计算最优切片参数
        slice_height, overlap = self._calculate_slice_params(
            height, content_type, has_code
        )

        return ImageProfile(
            width=width,
            height=height,
            source=source,
            content_type=content_type,
            contrast_level=contrast,
            noise_level=noise,
            needs_slicing=needs_slicing,
            needs_preprocessing=needs_preprocessing,
            has_code_blocks=has_code,
            has_tables=(content_type == ContentType.TABLE),
            recommended_slice_height=slice_height,
            recommended_overlap=overlap,
        )

    def _detect_source(self, image: Image.Image) -> ImageSource:
        """
        检测图片来源：数字渲染 vs 物理拍摄

        判断依据:
        - 数字图片：边缘锐利、颜色纯净、无 EXIF 相机信息
        - 物理照片：边缘模糊、颜色渐变、有 EXIF 相机信息
        """
        # 方法1: 检查 EXIF 信息
        try:
            exif = image.getexif() if hasattr(image, 'getexif') else {}
            if exif and (271 in exif or 272 in exif):  # Make, Model
                return ImageSource.PHOTO
        except Exception:
            pass

        # 方法2: 边缘锐度分析
        edge_sharpness = self._calculate_edge_sharpness(image)
        if edge_sharpness > 0.7:
            return ImageSource.DIGITAL
        elif edge_sharpness < 0.4:
            return ImageSource.PHOTO

        return ImageSource.UNKNOWN

    def _calculate_contrast(self, image: Image.Image) -> float:
        """
        计算图片对比度 (0-1)

        使用灰度图的标准差作为对比度指标
        """
        # 采样以提高性能（对大图只取样本）
        sample = self._get_sample(image)
        gray = sample.convert('L')
        arr = np.array(gray, dtype=np.float32)

        # 使用标准差作为对比度指标
        std = arr.std()
        # 归一化到 0-1 (假设最大标准差约为 80)
        return min(std / 80.0, 1.0)

    def _calculate_noise(self, image: Image.Image) -> float:
        """
        计算噪声水平 (0-1)

        使用拉普拉斯算子检测高频噪声
        """
        sample = self._get_sample(image)
        gray = sample.convert('L')
        arr = np.array(gray, dtype=np.float32)

        # 简化的拉普拉斯算子（不依赖 scipy）
        # 使用 3x3 卷积核: [[0,1,0],[1,-4,1],[0,1,0]]
        laplacian = self._laplacian_filter(arr)
        noise_estimate = laplacian.var()

        # 归一化 (经验值)
        return min(noise_estimate / 800.0, 1.0)

    def _laplacian_filter(self, arr: np.ndarray) -> np.ndarray:
        """简化的拉普拉斯滤波器（不依赖 scipy）"""
        # 拉普拉斯核
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        # 简单的卷积实现（边缘填充）
        padded = np.pad(arr, 1, mode='edge')
        result = np.zeros_like(arr)

        for i in range(3):
            for j in range(3):
                result += kernel[i, j] * padded[i:i+arr.shape[0], j:j+arr.shape[1]]

        return result

    def _calculate_edge_sharpness(self, image: Image.Image) -> float:
        """
        计算边缘锐度

        数字图片边缘锐利，照片边缘模糊
        """
        sample = self._get_sample(image)
        gray = sample.convert('L')
        arr = np.array(gray, dtype=np.float32)

        # Sobel 边缘检测（简化版）
        sx = self._sobel_x(arr)
        sy = self._sobel_y(arr)
        edge_magnitude = np.sqrt(sx**2 + sy**2)

        # 锐度 = 边缘强度的集中程度
        return min(edge_magnitude.std() / 40.0, 1.0)

    def _sobel_x(self, arr: np.ndarray) -> np.ndarray:
        """Sobel X 方向滤波"""
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        padded = np.pad(arr, 1, mode='edge')
        result = np.zeros_like(arr)
        for i in range(3):
            for j in range(3):
                result += kernel[i, j] * padded[i:i+arr.shape[0], j:j+arr.shape[1]]
        return result

    def _sobel_y(self, arr: np.ndarray) -> np.ndarray:
        """Sobel Y 方向滤波"""
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        padded = np.pad(arr, 1, mode='edge')
        result = np.zeros_like(arr)
        for i in range(3):
            for j in range(3):
                result += kernel[i, j] * padded[i:i+arr.shape[0], j:j+arr.shape[1]]
        return result

    def _get_sample(self, image: Image.Image, max_size: int = 1000) -> Image.Image:
        """
        获取图片样本用于分析（避免处理过大的图片）
        """
        width, height = image.size
        if max(width, height) <= max_size:
            return image

        # 缩放到合适大小
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _detect_content_type(self, image: Image.Image) -> ContentType:
        """
        检测内容类型

        基于颜色分布和结构特征判断
        """
        sample = self._get_sample(image)
        arr = np.array(sample)

        # 检测深色区域比例（可能是代码块）
        if len(arr.shape) == 3:
            gray = np.mean(arr, axis=2)
        else:
            gray = arr

        dark_ratio = np.mean(gray < 50)

        # 检测是否有规则的网格结构（可能是表格）
        has_grid = self._detect_grid_structure(sample)

        if has_grid:
            return ContentType.TABLE
        elif dark_ratio > 0.15:
            return ContentType.MIXED  # 有代码块
        else:
            return ContentType.DOCUMENT

    def _detect_grid_structure(self, image: Image.Image) -> bool:
        """
        检测是否有网格结构（表格特征）

        基于水平和垂直线条检测
        """
        gray = np.array(image.convert('L'), dtype=np.float32)

        # 检测水平线：行方差很小的连续区域
        row_variance = gray.var(axis=1)
        low_variance_rows = row_variance < np.percentile(row_variance, 10)

        # 统计连续的低方差行数量
        consecutive_count = 0
        max_consecutive = 0
        for is_low in low_variance_rows:
            if is_low:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0

        # 如果有多条水平线，可能是表格
        horizontal_lines = np.sum(np.diff(low_variance_rows.astype(int)) == 1)

        return horizontal_lines >= 3 and max_consecutive >= 2

    def _detect_code_blocks(self, image: Image.Image) -> bool:
        """
        检测是否包含代码块（深色背景区域）
        """
        sample = self._get_sample(image)
        arr = np.array(sample.convert('L'))

        # 检测连续的深色水平带
        row_means = arr.mean(axis=1)
        dark_rows = row_means < 60

        # 检测是否有连续的深色区域（代码块通常较高）
        consecutive = 0
        max_consecutive = 0
        for is_dark in dark_rows:
            if is_dark:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        # 代码块至少占图片 5% 高度
        min_code_height = len(dark_rows) * 0.05
        return max_consecutive > min_code_height

    def _calculate_slice_params(
        self, height: int, content_type: ContentType, has_code: bool
    ) -> Tuple[int, int]:
        """
        计算最优切片参数

        Args:
            height: 图片高度
            content_type: 内容类型
            has_code: 是否有代码块

        Returns:
            (slice_height, overlap) 切片高度和重叠区域
        """
        if height <= self.SLICE_THRESHOLD:
            return height, 0

        # 基础切片高度
        slice_height = self.MAX_SAFE_HEIGHT

        # 有代码块时减小切片高度，避免切断代码块
        if has_code:
            slice_height = 4000

        # 表格内容使用较大重叠，避免切断表格行
        if content_type == ContentType.TABLE:
            overlap = 600
        else:
            overlap = 400

        return slice_height, overlap
