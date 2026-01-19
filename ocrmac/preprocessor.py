"""自适应图像预处理模块 - 根据图片特征自动选择预处理策略

预处理流水线专为提升 OCR 准确率设计，补充 Apple Vision/LiveText 未做的处理：
1. 自适应二值化 (Sauvola) - 处理光照不均匀
2. 颜色通道分离增强 - 针对彩色文字（如红字粉底）
3. 背景归一化 - 去除背景干扰
4. 分辨率提升 - 小图放大到最佳 DPI
5. 伽马校正 - 曝光校正
6. CLAHE 对比度增强 - 局部自适应对比度
7. 去噪 + 锐化 - 提升文字清晰度
"""

import sys
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

if sys.version_info < (3, 9):
    from typing import Optional, Tuple
else:
    Optional = None.__class__
    Tuple = tuple

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
    - 物理照片：去噪 + 对比度增强 + 锐化 + 可选二值化
    - 数字图片：轻度增强（如果需要）
    - 低对比度彩色：颜色通道分离 + 增强
    """

    # 最佳 OCR 分辨率（宽度）
    MIN_WIDTH_FOR_OCR = 1500

    def __init__(
        self,
        enable_upscale: bool = True,
        enable_color_enhancement: bool = True,
        enable_binarization: bool = False,  # 默认关闭，因为 Vision 内部会做
        aggressive_mode: bool = False,
    ):
        """
        初始化预处理器

        Args:
            enable_upscale: 是否启用分辨率提升
            enable_color_enhancement: 是否启用颜色增强（针对彩色文字）
            enable_binarization: 是否启用二值化（一般不需要）
            aggressive_mode: 激进模式，应用更强的预处理
        """
        self.enable_upscale = enable_upscale
        self.enable_color_enhancement = enable_color_enhancement
        self.enable_binarization = enable_binarization
        self.aggressive_mode = aggressive_mode

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

        result = image

        # 步骤 0: 分辨率提升（如果图片太小）
        if self.enable_upscale:
            result = self._upscale_if_needed(result)

        # 根据来源选择预处理流水线
        if profile.source == ImageSource.PHOTO:
            result = self._process_photo(result, profile)
        else:
            result = self._process_digital(result, profile)

        return result

    def _upscale_if_needed(self, image: Image.Image) -> Image.Image:
        """
        如果图片分辨率太低，进行放大

        OCR 对于分辨率要求较高，建议至少 300 DPI
        对于屏幕截图，宽度至少 1500px 效果较好
        """
        width, height = image.size

        if width >= self.MIN_WIDTH_FOR_OCR:
            return image

        # 计算放大比例
        scale = self.MIN_WIDTH_FOR_OCR / width

        # 限制最大放大比例
        scale = min(scale, 2.0)

        if scale <= 1.0:
            return image

        new_size = (int(width * scale), int(height * scale))

        # 使用 LANCZOS 重采样（高质量）
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _process_photo(self, image: Image.Image, profile: ImageProfile) -> Image.Image:
        """
        物理照片预处理流水线

        处理步骤：
        1. 伽马校正（如果曝光不足/过曝）
        2. 颜色增强（如果是彩色文字）
        3. 对比度增强（CLAHE）
        4. 降噪
        5. 锐化
        6. 可选：自适应二值化
        """
        result = image

        # 1. 伽马校正
        result = self._gamma_correction(result, profile)

        # 2. 颜色增强（针对彩色文字，如红字粉底）
        if self.enable_color_enhancement and profile.contrast_level < 0.6:
            result = self._enhance_color_text(result)

        # 3. 对比度增强（CLAHE）
        if profile.contrast_level < 0.7:
            strength = 1.5 if self.aggressive_mode else 1.0
            result = self._enhance_contrast(result, strength=strength)

        # 4. 降噪
        if profile.noise_level > 0.3:
            result = self._denoise(result)

        # 5. 锐化
        strength = 0.8 if self.aggressive_mode else 0.5
        result = self._sharpen(result, strength=strength)

        # 6. 可选：自适应二值化
        if self.enable_binarization:
            result = self._adaptive_binarize(result)

        return result

    def _process_digital(self, image: Image.Image, profile: ImageProfile) -> Image.Image:
        """
        数字图片预处理

        数字截图通常质量较高，仅在需要时轻度增强
        """
        result = image

        # 针对低对比度场景增强
        if profile.contrast_level < 0.5:
            # 颜色增强
            if self.enable_color_enhancement:
                result = self._enhance_color_text(result)

            # 对比度增强
            result = self._enhance_contrast(result, strength=0.8)

            # 轻度锐化
            result = self._sharpen(result, strength=0.3)

        elif profile.contrast_level < 0.7:
            # 轻度增强
            result = self._enhance_contrast(result, strength=0.5)

        return result

    def _gamma_correction(self, image: Image.Image, profile: ImageProfile) -> Image.Image:
        """
        伽马校正

        根据图片亮度自动调整伽马值：
        - 太暗：gamma < 1（提亮）
        - 太亮：gamma > 1（压暗）
        """
        arr = np.array(image)

        # 计算平均亮度
        if len(arr.shape) == 3:
            gray = np.mean(arr, axis=2)
        else:
            gray = arr

        mean_brightness = np.mean(gray) / 255.0

        # 目标亮度 0.5
        if mean_brightness < 0.3:
            gamma = 0.7  # 提亮
        elif mean_brightness > 0.7:
            gamma = 1.3  # 压暗
        else:
            return image  # 不需要校正

        # 应用伽马校正
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        if len(arr.shape) == 3:
            corrected = np.zeros_like(arr)
            for i in range(arr.shape[2]):
                corrected[:, :, i] = table[arr[:, :, i]]
        else:
            corrected = table[arr]

        return Image.fromarray(corrected)

    def _enhance_color_text(self, image: Image.Image) -> Image.Image:
        """
        颜色文字增强

        针对彩色文字（如红字粉底、蓝字白底）进行专门增强：
        1. 转换到 LAB 色彩空间
        2. 增强 a/b 通道的对比度（色彩对比）
        3. 归一化背景
        """
        if CV2_AVAILABLE:
            return self._enhance_color_text_cv2(image)
        else:
            return self._enhance_color_text_pil(image)

    def _enhance_color_text_cv2(self, image: Image.Image) -> Image.Image:
        """使用 OpenCV 增强彩色文字"""
        arr = np.array(image)

        if len(arr.shape) == 2:
            # 灰度图，直接返回
            return image

        # 处理 RGBA
        has_alpha = arr.shape[2] == 4
        if has_alpha:
            rgb = arr[:, :, :3]
            alpha = arr[:, :, 3]
        else:
            rgb = arr
            alpha = None

        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 增强 L 通道（亮度）- 使用 CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # 增强 a/b 通道（色彩）- 拉伸对比度
        a_enhanced = self._stretch_channel(a)
        b_enhanced = self._stretch_channel(b)

        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        if alpha is not None:
            result = np.dstack([rgb_enhanced, alpha])
            return Image.fromarray(result, mode='RGBA')
        else:
            return Image.fromarray(rgb_enhanced)

    def _stretch_channel(self, channel: np.ndarray) -> np.ndarray:
        """拉伸单通道对比度"""
        min_val = np.percentile(channel, 2)
        max_val = np.percentile(channel, 98)

        if max_val - min_val < 10:
            return channel

        stretched = (channel.astype(float) - min_val) / (max_val - min_val) * 255
        return np.clip(stretched, 0, 255).astype(np.uint8)

    def _enhance_color_text_pil(self, image: Image.Image) -> Image.Image:
        """使用 PIL 增强彩色文字（备选方案）"""
        # 增强饱和度
        enhancer = ImageEnhance.Color(image)
        result = enhancer.enhance(1.5)

        # 增强对比度
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.3)

        return result

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
        result = ImageOps.autocontrast(image, cutoff=2)

        # 方法2: 增强对比度
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.0 + 0.5 * strength)

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
            percent=int(150 * strength),
            threshold=3
        ))

        return sharpened

    def _adaptive_binarize(self, image: Image.Image) -> Image.Image:
        """
        自适应二值化 (Sauvola 方法)

        对于光照不均的图片效果更好
        """
        if CV2_AVAILABLE:
            return self._adaptive_binarize_cv2(image)
        else:
            return self._adaptive_binarize_pil(image)

    def _adaptive_binarize_cv2(self, image: Image.Image) -> Image.Image:
        """使用 OpenCV 进行自适应二值化"""
        # 转换为灰度
        gray = np.array(image.convert('L'))

        # 使用自适应阈值（Gaussian）
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,  # 邻域大小
            C=10  # 常数
        )

        return Image.fromarray(binary)

    def _adaptive_binarize_pil(self, image: Image.Image) -> Image.Image:
        """使用 PIL 进行自适应二值化（简化版 Sauvola）"""
        gray = image.convert('L')
        arr = np.array(gray, dtype=np.float32)

        # 计算局部均值和标准差
        window_size = 15
        half_window = window_size // 2

        # 使用滑动窗口计算局部统计量
        padded = np.pad(arr, half_window, mode='reflect')

        height, width = arr.shape
        result = np.zeros_like(arr, dtype=np.uint8)

        # 简化版：使用分块处理
        block_size = 50
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                y_end = min(y + block_size, height)
                x_end = min(x + block_size, width)

                block = arr[y:y_end, x:x_end]
                local_mean = np.mean(block)
                local_std = np.std(block)

                # Sauvola 公式
                k = 0.5
                R = 128
                threshold = local_mean * (1 + k * (local_std / R - 1))

                result[y:y_end, x:x_end] = np.where(
                    block > threshold, 255, 0
                ).astype(np.uint8)

        return Image.fromarray(result)


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

    @staticmethod
    def remove_background(image: Image.Image, threshold: int = 200) -> Image.Image:
        """
        背景去除/归一化

        将浅色背景统一为白色
        """
        arr = np.array(image)

        if len(arr.shape) == 2:
            # 灰度图
            arr[arr > threshold] = 255
        else:
            # 彩色图
            # 计算每个像素的亮度
            if arr.shape[2] >= 3:
                brightness = np.mean(arr[:, :, :3], axis=2)
                mask = brightness > threshold
                arr[mask] = 255

        return Image.fromarray(arr)

    @staticmethod
    def enhance_red_text(image: Image.Image) -> Image.Image:
        """
        专门增强红色文字（如红字粉底）

        通过提取红色通道并增强对比度
        """
        arr = np.array(image)

        if len(arr.shape) == 2:
            return image

        # 提取 RGB 通道
        r = arr[:, :, 0].astype(float)
        g = arr[:, :, 1].astype(float)
        b = arr[:, :, 2].astype(float)

        # 计算红色占比 (r - max(g, b))
        red_diff = r - np.maximum(g, b)

        # 归一化
        red_diff = np.clip(red_diff, 0, 255)

        # 反转（红色文字变黑，背景变白）
        result = 255 - red_diff

        # 增强对比度
        result = np.clip((result - 128) * 1.5 + 128, 0, 255).astype(np.uint8)

        return Image.fromarray(result)
