"""自适应图像预处理模块 - 根据图片特征自动选择预处理策略"""

import sys
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

if sys.version_info < (3, 9):
    from typing import Optional
else:
    Optional = None.__class__

from .image_analyzer import ImageProfile, ImageSource

# 尝试导入 OpenCV（可选依赖）
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class AdaptivePreprocessor:
    """
    自适应图像预处理器

    根据图片特征自动选择最优预处理策略：
    - 物理照片：对比度增强 + 降噪 + 锐化
    - 数字图片：通常无需预处理，或仅轻度增强
    """

    def process(self, image: Image.Image, profile: ImageProfile) -> Image.Image:
        """
        根据图片特征自动选择预处理策略

        Args:
            image: 原始 PIL Image
            profile: 图片特征分析结果

        Returns:
            预处理后的 PIL Image
        """
        if not profile.needs_preprocessing:
            # 高质量数字图片，无需预处理
            return image

        # 根据来源选择预处理流水线
        if profile.source == ImageSource.PHOTO:
            return self._process_photo(image, profile)
        else:
            return self._process_digital(image, profile)

    def _process_photo(self, image: Image.Image, profile: ImageProfile) -> Image.Image:
        """
        物理照片预处理流水线

        处理步骤：
        1. 对比度增强（如果对比度低）
        2. 降噪（如果噪声高）
        3. 锐化
        """
        result = image

        # 1. 如果对比度低，进行增强
        if profile.contrast_level < 0.6:
            result = self._enhance_contrast(result)

        # 2. 如果噪声高，进行降噪
        if profile.noise_level > 0.4:
            result = self._denoise(result)

        # 3. 轻度锐化（提升文字清晰度）
        result = self._sharpen(result, strength=0.5)

        return result

    def _process_digital(self, image: Image.Image, profile: ImageProfile) -> Image.Image:
        """
        数字图片预处理（通常很少需要）

        仅在对比度特别低时进行轻度增强
        """
        if profile.contrast_level < 0.4:
            return self._enhance_contrast(image, strength=0.5)
        return image

    def _enhance_contrast(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """
        对比度增强

        优先使用 OpenCV 的 CLAHE（自适应直方图均衡化）
        如果 OpenCV 不可用，使用 PIL 的直方图均衡化
        """
        if CV2_AVAILABLE:
            return self._enhance_contrast_cv2(image, strength)
        else:
            return self._enhance_contrast_pil(image, strength)

    def _enhance_contrast_cv2(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """使用 OpenCV CLAHE 进行对比度增强"""
        arr = np.array(image)

        if len(arr.shape) == 2:
            # 灰度图
            clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
            enhanced = clahe.apply(arr)
            return Image.fromarray(enhanced)
        else:
            # 彩色图片：在 LAB 空间的 L 通道增强
            if arr.shape[2] == 4:  # RGBA
                rgb = arr[:, :, :3]
                alpha = arr[:, :, 3]
            else:
                rgb = arr
                alpha = None

            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            lab_enhanced = cv2.merge([l_enhanced, a, b])
            rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

            if alpha is not None:
                result = np.dstack([rgb_enhanced, alpha])
                return Image.fromarray(result, mode='RGBA')
            else:
                return Image.fromarray(rgb_enhanced)

    def _enhance_contrast_pil(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """使用 PIL 进行对比度增强（备选方案）"""
        # 方法1: 自动对比度
        result = ImageOps.autocontrast(image, cutoff=1)

        # 方法2: 增强对比度
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.0 + 0.3 * strength)

        return result

    def _denoise(self, image: Image.Image) -> Image.Image:
        """
        降噪处理

        优先使用 OpenCV 的 fastNlMeansDenoising
        如果不可用，使用 PIL 的中值滤波
        """
        if CV2_AVAILABLE:
            return self._denoise_cv2(image)
        else:
            return self._denoise_pil(image)

    def _denoise_cv2(self, image: Image.Image) -> Image.Image:
        """使用 OpenCV 进行降噪"""
        arr = np.array(image)

        if len(arr.shape) == 2:
            # 灰度图
            denoised = cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)
        elif arr.shape[2] == 3:
            # RGB
            denoised = cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)
        else:
            # RGBA - 分离 alpha 通道
            rgb = arr[:, :, :3]
            alpha = arr[:, :, 3]
            denoised_rgb = cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
            denoised = np.dstack([denoised_rgb, alpha])
            return Image.fromarray(denoised, mode='RGBA')

        return Image.fromarray(denoised)

    def _denoise_pil(self, image: Image.Image) -> Image.Image:
        """使用 PIL 进行降噪（备选方案）"""
        # 使用中值滤波降噪
        return image.filter(ImageFilter.MedianFilter(size=3))

    def _sharpen(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """
        锐化处理

        Args:
            image: 输入图片
            strength: 锐化强度 (0-1)
        """
        if strength <= 0:
            return image

        # 使用 UnsharpMask 进行锐化
        sharpened = image.filter(ImageFilter.UnsharpMask(
            radius=2,
            percent=int(100 * strength),
            threshold=3
        ))

        return sharpened


class ManualPreprocessor:
    """
    手动预处理器

    提供独立的预处理方法，供用户手动调用
    """

    @staticmethod
    def grayscale(image: Image.Image) -> Image.Image:
        """转换为灰度图"""
        return image.convert('L')

    @staticmethod
    def binarize(image: Image.Image, threshold: int = 128) -> Image.Image:
        """
        二值化

        Args:
            image: 输入图片
            threshold: 阈值 (0-255)
        """
        gray = image.convert('L')
        return gray.point(lambda x: 255 if x > threshold else 0, mode='1')

    @staticmethod
    def auto_binarize(image: Image.Image) -> Image.Image:
        """
        自动二值化（Otsu 方法）
        """
        gray = image.convert('L')
        arr = np.array(gray)

        # Otsu 阈值计算
        threshold = ManualPreprocessor._otsu_threshold(arr)

        return gray.point(lambda x: 255 if x > threshold else 0, mode='1')

    @staticmethod
    def _otsu_threshold(arr: np.ndarray) -> int:
        """Otsu 阈值计算"""
        # 计算直方图
        hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()

        # 计算累积和
        cum_sum = np.cumsum(hist)
        cum_mean = np.cumsum(hist * np.arange(256))

        # 全局均值
        global_mean = cum_mean[-1]

        # 计算类间方差
        variance = np.zeros(256)
        for t in range(256):
            if cum_sum[t] == 0 or cum_sum[t] == 1:
                continue
            w0 = cum_sum[t]
            w1 = 1 - w0
            m0 = cum_mean[t] / w0
            m1 = (global_mean - cum_mean[t]) / w1
            variance[t] = w0 * w1 * (m0 - m1) ** 2

        return int(np.argmax(variance))

    @staticmethod
    def rotate(image: Image.Image, angle: float) -> Image.Image:
        """
        旋转图片

        Args:
            image: 输入图片
            angle: 旋转角度（度）
        """
        return image.rotate(angle, expand=True, fillcolor='white')

    @staticmethod
    def deskew(image: Image.Image) -> Image.Image:
        """
        自动校正倾斜

        检测文本行角度并校正
        """
        if not CV2_AVAILABLE:
            return image

        gray = np.array(image.convert('L'))

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None or len(lines) == 0:
            return image

        # 计算平均角度
        angles = []
        for line in lines[:20]:  # 只取前 20 条线
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:  # 过滤异常角度
                angles.append(angle)

        if not angles:
            return image

        avg_angle = np.median(angles)

        # 如果角度很小，不需要校正
        if abs(avg_angle) < 0.5:
            return image

        return ManualPreprocessor.rotate(image, -avg_angle)
