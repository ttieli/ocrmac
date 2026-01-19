"""共享类型定义 - 避免循环依赖"""

import sys
from dataclasses import dataclass
from PIL import Image

if sys.version_info < (3, 9):
    from typing import List
else:
    List = list


@dataclass
class SliceInfo:
    """切片信息"""
    index: int              # 切片索引
    y_start: int            # 在原图中的起始 y 坐标
    y_end: int              # 在原图中的结束 y 坐标
    image: Image.Image      # 切片图片
    is_code_block: bool     # 是否主要是代码块区域
