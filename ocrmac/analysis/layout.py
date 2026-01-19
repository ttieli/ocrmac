"""文档布局分析模块 - 段落、标题、列表检测"""

import re
import sys

if sys.version_info < (3, 9):
    from typing import List, Dict, Tuple
else:
    List, Dict, Tuple = list, dict, tuple


class ParagraphDetector:
    """段落检测器 - 基于行间距分析"""

    def __init__(self, line_spacing_threshold=1.5, indent_threshold=0.05):
        """
        初始化段落检测器

        Args:
            line_spacing_threshold: 行间距阈值（相对于平均行高的倍数）
            indent_threshold: 段首缩进阈值（相对于图片宽度）
        """
        self.line_spacing_threshold = line_spacing_threshold
        self.indent_threshold = indent_threshold

    def detect_paragraphs(self, ocr_results: List[Tuple]) -> List[Dict]:
        """
        检测段落

        Args:
            ocr_results: OCR 结果列表 [(text, confidence, [x, y, w, h]), ...]
                注意: 坐标已被转换为 y=0 在顶部的系统

        Returns:
            段落列表: [{'lines': [...], 'bbox': [...], 'text': '...'}, ...]
        """
        if not ocr_results:
            return []

        # 按 y 坐标排序（从上到下）
        sorted_lines = sorted(ocr_results, key=lambda r: r[2][1])

        # 计算平均行高
        avg_height = self._calculate_avg_height(sorted_lines)

        # 分段
        paragraphs = []
        current_paragraph = {
            'lines': [sorted_lines[0]],
            'bbox': sorted_lines[0][2],
        }

        for i in range(1, len(sorted_lines)):
            prev_line = sorted_lines[i - 1]
            curr_line = sorted_lines[i]

            prev_y_bottom = prev_line[2][1] + prev_line[2][3]
            curr_y_top = curr_line[2][1]
            spacing = curr_y_top - prev_y_bottom

            if self._should_split_paragraph(spacing, avg_height):
                current_paragraph['text'] = self._combine_lines(current_paragraph['lines'])
                paragraphs.append(current_paragraph)

                current_paragraph = {
                    'lines': [curr_line],
                    'bbox': curr_line[2],
                }
            else:
                current_paragraph['lines'].append(curr_line)

        if current_paragraph['lines']:
            current_paragraph['text'] = self._combine_lines(current_paragraph['lines'])
            paragraphs.append(current_paragraph)

        return paragraphs

    def _calculate_avg_height(self, lines):
        """计算平均行高"""
        if not lines:
            return 0.05
        heights = [line[2][3] for line in lines]
        return sum(heights) / len(heights)

    def _should_split_paragraph(self, spacing, avg_height):
        """判断是否应该分段"""
        return spacing > avg_height * self.line_spacing_threshold

    def _combine_lines(self, lines):
        """合并行文本"""
        return '\n'.join(line[0] for line in lines)


class HeadingDetector:
    """标题检测器 - 基于字体大小、位置和文本模式"""

    def __init__(self, size_threshold=1.3, max_length=100):
        """
        初始化标题检测器

        Args:
            size_threshold: 字体大小阈值（相对于平均字高）
            max_length: 标题最大长度（字符数）
        """
        self.size_threshold = size_threshold
        self.max_length = max_length
        self.cn_header_patterns = [
            (r'^【[^】]+】$', 2),
            (r'^[一二三四五六七八九十]+[、\.]\s*.+', 2),
            (r'^\d+[\.\、]\s*.+', 3),
        ]

    def detect_headings(self, paragraphs: List[Dict], avg_height: float) -> List[Dict]:
        """检测标题并标注层级"""
        for para in paragraphs:
            lines = para.get('lines', [])
            if not lines:
                continue

            first_line = lines[0]
            text = first_line[0].strip()
            bbox = first_line[2]
            height = bbox[3]

            is_heading = False
            level = 0

            for pattern, lvl in self.cn_header_patterns:
                if re.match(pattern, text):
                    is_heading = True
                    level = lvl
                    break

            if not is_heading and height > avg_height * self.size_threshold:
                is_heading = True
                if height >= avg_height * 2.0:
                    level = 1
                elif height >= avg_height * 1.6:
                    level = 2
                else:
                    level = 3

            if not is_heading and len(lines) == 1 and len(text) < self.max_length:
                if height > avg_height * 1.1:
                    is_heading = True
                    level = 3

            para['is_heading'] = is_heading
            para['heading_level'] = level

        return paragraphs


class ListDetector:
    """列表检测器 - 基于符号和缩进"""

    def __init__(self):
        self.bullet_patterns = ['•', '-', '*', '▪', '◦']
        self.number_pattern = r'^\s*\d+[\.\)]\s+'

    def detect_lists(self, paragraphs: List[Dict]) -> List[Dict]:
        """检测列表项"""
        for para in paragraphs:
            text = para.get('text', '')
            lines = text.split('\n')
            first_line = lines[0].strip() if lines else ''

            is_list = False
            list_type = None

            if any(first_line.startswith(bullet) for bullet in self.bullet_patterns):
                is_list = True
                list_type = 'unordered'
            elif re.match(self.number_pattern, first_line):
                is_list = True
                list_type = 'ordered'

            para['is_list'] = is_list
            para['list_type'] = list_type

        return paragraphs


class LayoutAnalyzer:
    """综合布局分析器"""

    def __init__(
        self,
        line_spacing_threshold=1.5,
        heading_size_threshold=1.3,
    ):
        """
        初始化布局分析器

        Args:
            line_spacing_threshold: 段落间距阈值
            heading_size_threshold: 标题字体大小阈值
        """
        self.para_detector = ParagraphDetector(line_spacing_threshold)
        self.heading_detector = HeadingDetector(heading_size_threshold)
        self.list_detector = ListDetector()

    def analyze(self, ocr_results: List[Tuple]) -> Dict:
        """
        完整的布局分析

        Args:
            ocr_results: OCR 结果列表

        Returns:
            布局分析结果字典
        """
        paragraphs = self.para_detector.detect_paragraphs(ocr_results)
        avg_height = self.para_detector._calculate_avg_height(ocr_results)
        paragraphs = self.heading_detector.detect_headings(paragraphs, avg_height)
        paragraphs = self.list_detector.detect_lists(paragraphs)

        return {
            'paragraphs': paragraphs,
            'avg_height': avg_height,
            'total_paragraphs': len(paragraphs),
        }

    def to_markdown(self, layout_result: Dict) -> str:
        """将布局分析结果转换为 Markdown"""
        paragraphs = layout_result['paragraphs']
        output = []

        for para in paragraphs:
            text = para.get('text', '')

            if para.get('is_heading'):
                level = para.get('heading_level', 3)
                prefix = '#' * level
                output.append(f"{prefix} {text}")
                output.append("")
            elif para.get('is_list'):
                output.append(text)
                output.append("")
            else:
                output.append(text)
                output.append("")

        return '\n'.join(output)


def format_with_layout(ocr_results: List[Tuple], image_height: int = 0) -> str:
    """
    使用布局分析格式化 OCR 结果

    Args:
        ocr_results: OCR 结果列表 [(text, confidence, [x, y, w, h]), ...]
        image_height: 图片高度（像素）

    Returns:
        格式化的 Markdown 文本（未清理，调用方需自行使用 TextCleaner）
    """
    if not ocr_results:
        return ""

    analyzer = LayoutAnalyzer(line_spacing_threshold=1.8)
    layout = analyzer.analyze(ocr_results)
    markdown = analyzer.to_markdown(layout)

    return markdown
