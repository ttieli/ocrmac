"""文档布局分析模块 - 段落、标题、列表检测"""

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

        Returns:
            段落列表: [{'lines': [...], 'bbox': [...], 'text': '...'}, ...]
        """
        if not ocr_results:
            return []

        # 按 y 坐标排序（从上到下 = y_top 从大到小，因为 Vision 坐标系 y=0 在底部）
        sorted_lines = sorted(ocr_results, key=lambda r: self._get_y_top(r[2]), reverse=True)

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

            prev_y_bottom = self._get_y_bottom(prev_line[2])
            curr_y_top = self._get_y_top(curr_line[2])

            # 计算行间距（从上往下，prev_bottom > curr_top，所以间距 = prev - curr）
            spacing = prev_y_bottom - curr_y_top

            # 判断是否需要分段
            if self._should_split_paragraph(spacing, avg_height):
                # 结束当前段落
                current_paragraph['text'] = self._combine_lines(current_paragraph['lines'])
                paragraphs.append(current_paragraph)

                # 开始新段落
                current_paragraph = {
                    'lines': [curr_line],
                    'bbox': curr_line[2],
                }
            else:
                # 继续当前段落
                current_paragraph['lines'].append(curr_line)

        # 添加最后一个段落
        if current_paragraph['lines']:
            current_paragraph['text'] = self._combine_lines(current_paragraph['lines'])
            paragraphs.append(current_paragraph)

        return paragraphs

    def _get_y_top(self, bbox):
        """获取 bbox 顶部 y 坐标（Vision 坐标系）"""
        x, y, w, h = bbox
        return y + h  # Vision 坐标系 y 是底部，y+h 是顶部

    def _get_y_bottom(self, bbox):
        """获取 bbox 底部 y 坐标（Vision 坐标系）"""
        x, y, w, h = bbox
        return y  # Vision 坐标系 y 就是底部

    def _calculate_avg_height(self, lines):
        """计算平均行高"""
        if not lines:
            return 0.05  # 默认值
        heights = [line[2][3] for line in lines]  # h
        return sum(heights) / len(heights)

    def _should_split_paragraph(self, spacing, avg_height):
        """判断是否应该分段"""
        return spacing > avg_height * self.line_spacing_threshold

    def _combine_lines(self, lines):
        """合并行文本"""
        return '\n'.join(line[0] for line in lines)


class HeadingDetector:
    """标题检测器 - 基于字体大小和位置"""

    def __init__(self, size_threshold=1.3, max_length=100):
        """
        初始化标题检测器

        Args:
            size_threshold: 字体大小阈值（相对于平均字高）
            max_length: 标题最大长度（字符数）
        """
        self.size_threshold = size_threshold
        self.max_length = max_length

    def detect_headings(self, paragraphs: List[Dict], avg_height: float) -> List[Dict]:
        """
        检测标题并标注层级

        Args:
            paragraphs: 段落列表
            avg_height: 平均行高

        Returns:
            标注后的段落列表（增加 'is_heading' 和 'level' 字段）
        """
        for para in paragraphs:
            lines = para.get('lines', [])
            if not lines:
                continue

            first_line = lines[0]
            text = first_line[0]
            bbox = first_line[2]
            height = bbox[3]

            # 判断是否为标题
            is_heading = False
            level = 0

            # 规则1: 字体明显大于平均
            if height > avg_height * self.size_threshold:
                is_heading = True
                # 根据大小分级
                if height >= avg_height * 2.0:
                    level = 1
                elif height >= avg_height * 1.6:
                    level = 2
                else:
                    level = 3

            # 规则2: 文本较短且独立成段（且字体略大于平均）
            if len(lines) == 1 and len(text) < self.max_length:
                if not is_heading and height > avg_height * 1.1:
                    is_heading = True
                    level = 3

            para['is_heading'] = is_heading
            para['heading_level'] = level

        return paragraphs


class ListDetector:
    """列表检测器 - 基于符号和缩进"""

    def __init__(self):
        """初始化列表检测器"""
        self.bullet_patterns = ['•', '-', '*', '▪', '◦']
        self.number_pattern = r'^\s*\d+[\.\)]\s+'

    def detect_lists(self, paragraphs: List[Dict]) -> List[Dict]:
        """
        检测列表项

        Args:
            paragraphs: 段落列表

        Returns:
            标注后的段落列表（增加 'is_list' 和 'list_type' 字段）
        """
        import re

        for para in paragraphs:
            text = para.get('text', '')
            lines = text.split('\n')

            # 检测第一行
            first_line = lines[0].strip() if lines else ''

            is_list = False
            list_type = None

            # 检测无序列表
            if any(first_line.startswith(bullet) for bullet in self.bullet_patterns):
                is_list = True
                list_type = 'unordered'

            # 检测有序列表
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
        # 1. 段落检测
        paragraphs = self.para_detector.detect_paragraphs(ocr_results)

        # 2. 计算平均行高
        avg_height = self.para_detector._calculate_avg_height(ocr_results)

        # 3. 标题检测
        paragraphs = self.heading_detector.detect_headings(paragraphs, avg_height)

        # 4. 列表检测
        paragraphs = self.list_detector.detect_lists(paragraphs)

        return {
            'paragraphs': paragraphs,
            'avg_height': avg_height,
            'total_paragraphs': len(paragraphs),
        }

    def to_markdown(self, layout_result: Dict) -> str:
        """
        将布局分析结果转换为结构化 Markdown

        Args:
            layout_result: analyze() 的返回结果

        Returns:
            Markdown 格式文本
        """
        paragraphs = layout_result['paragraphs']
        output = []

        for para in paragraphs:
            text = para.get('text', '')

            # 标题
            if para.get('is_heading'):
                level = para.get('heading_level', 3)
                prefix = '#' * level
                output.append(f"{prefix} {text}")
                output.append("")  # 空行

            # 列表
            elif para.get('is_list'):
                list_type = para.get('list_type')
                if list_type == 'unordered':
                    # 保持原有列表格式
                    output.append(text)
                else:
                    # 有序列表
                    output.append(text)
                output.append("")

            # 普通段落
            else:
                output.append(text)
                output.append("")  # 段落间空行

        return '\n'.join(output)
