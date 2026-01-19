"""图像预处理模块"""

from .image import AdaptivePreprocessor, ManualPreprocessor

# 别名，方便使用
ImagePreprocessor = AdaptivePreprocessor

__all__ = ['AdaptivePreprocessor', 'ManualPreprocessor', 'ImagePreprocessor']
