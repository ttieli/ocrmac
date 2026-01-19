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
                注意: 坐标已被转换为 y=0 在顶部的系统

        Returns:
            段落列表: [{'lines': [...], 'bbox': [...], 'text': '...'}, ...]
        """
        if not ocr_results:
            return []

        # 按 y 坐标排序（从上到下 = y 从小到大，因为坐标已转换为 y=0 在顶部）
        sorted_lines = sorted(ocr_results, key=lambda r: r[2][1])  # 直接用 y 值排序

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

            # 坐标系：y=0 在顶部，y 增大向下
            # prev_line 在上面（y 较小），curr_line 在下面（y 较大）
            prev_y_bottom = prev_line[2][1] + prev_line[2][3]  # y + h
            curr_y_top = curr_line[2][1]  # y

            # 计算行间距 = 当前行顶部 - 上一行底部
            spacing = curr_y_top - prev_y_bottom

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
        # 中文标题模式
        self.cn_header_patterns = [
            (r'^【[^】]+】$', 2),  # 【标题】 -> h2
            (r'^[一二三四五六七八九十]+[、\.]\s*.+', 2),  # 一、标题 -> h2
            (r'^\d+[\.\、]\s*.+', 3),  # 1. 标题 -> h3
        ]

    def detect_headings(self, paragraphs: List[Dict], avg_height: float) -> List[Dict]:
        """
        检测标题并标注层级

        Args:
            paragraphs: 段落列表
            avg_height: 平均行高

        Returns:
            标注后的段落列表（增加 'is_heading' 和 'level' 字段）
        """
        import re

        for para in paragraphs:
            lines = para.get('lines', [])
            if not lines:
                continue

            first_line = lines[0]
            text = first_line[0].strip()
            bbox = first_line[2]
            height = bbox[3]

            # 判断是否为标题
            is_heading = False
            level = 0

            # 规则1: 中文标题模式 【标题】
            for pattern, lvl in self.cn_header_patterns:
                if re.match(pattern, text):
                    is_heading = True
                    level = lvl
                    break

            # 规则2: 字体明显大于平均
            if not is_heading and height > avg_height * self.size_threshold:
                is_heading = True
                # 根据大小分级
                if height >= avg_height * 2.0:
                    level = 1
                elif height >= avg_height * 1.6:
                    level = 2
                else:
                    level = 3

            # 规则3: 文本较短且独立成段（且字体略大于平均）
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


class TextCleaner:
    """文本清理器 - 去重、合并、格式化"""

    def __init__(self):
        """初始化文本清理器"""
        self.duplicate_threshold = 0.8  # 文本相似度阈值

    def clean_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        lines = text.split('\n')

        # 1. 去除重复行
        lines = self._remove_duplicate_lines(lines)

        # 2. 合并断句
        lines = self._merge_fragments(lines)

        # 3. 转换中文标题格式 【标题】 → ## 标题
        lines = self._convert_chinese_headers(lines)

        # 4. 格式化数字列表
        lines = self._format_numbered_lists(lines)

        # 5. 添加段落间空行
        lines = self._add_paragraph_spacing(lines)

        # 6. 清理空行
        lines = self._clean_empty_lines(lines)

        return '\n'.join(lines)

    def _remove_duplicate_lines(self, lines: List[str]) -> List[str]:
        """去除重复或高度相似的行"""
        if not lines:
            return []

        result = [lines[0]]
        for line in lines[1:]:
            # 检查是否与之前的行重复
            is_duplicate = False
            for prev_line in result[-5:]:  # 只检查最近5行
                similarity = self._calculate_similarity(line.strip(), prev_line.strip())
                if similarity > self.duplicate_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                result.append(line)

        return result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0
        if text1 == text2:
            return 1.0
        if text1 in text2 or text2 in text1:
            return 0.9

        # 字符级别的 Jaccard 相似度
        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _merge_fragments(self, lines: List[str]) -> List[str]:
        """合并断句（句子被切断的情况）"""
        if not lines:
            return []

        result = []
        buffer = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append(line)
                continue

            # 检查是否应该与上一行合并
            if buffer:
                # 如果上一行没有结束符，且当前行不是新段落开头
                if self._should_merge(buffer, stripped):
                    buffer += stripped
                    continue
                else:
                    result.append(buffer)
                    buffer = ""

            buffer = stripped

        if buffer:
            result.append(buffer)

        # 第二轮：修复明显的断词
        result = self._fix_broken_words(result)

        return result

    def _fix_broken_words(self, lines: List[str]) -> List[str]:
        """修复明显的断词（如"筛" + "选器"）"""
        if len(lines) < 2:
            return lines

        result = []
        i = 0

        while i < len(lines):
            current = lines[i]
            stripped = current.strip()

            # 如果是空行，直接添加
            if not stripped:
                result.append(current)
                i += 1
                continue

            # 不合并特殊行：标题、短行
            # 但列表项内容（以 **N.** 开头）只有在很短时才算特殊行
            is_list_content = stripped.startswith('**') and not self._is_list_item_title(stripped)

            is_special_line = (
                stripped.startswith('#') or           # Markdown 标题
                stripped.startswith('◎') or           # 特殊符号
                stripped.endswith('】') or            # 中文标题结尾
                stripped.endswith('？') or            # 问号结尾
                stripped.endswith('：') or            # 冒号结尾（通常是小标题）
                self._is_list_item_title(stripped) or # 列表项短标题
                (len(stripped) < 25 and not is_list_content)  # 很短的行可能是标题（但排除列表项内容）
            )

            # 检查当前行是否以中文字符结尾且不是句末，且不是特殊行
            should_try_merge = (
                stripped and
                not is_special_line and
                self._is_chinese_char(stripped[-1]) and
                not self._ends_with_sentence_end(stripped)
            )

            if should_try_merge:
                # 跳过空行找下一个非空行
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1

                if j < len(lines):
                    next_line = lines[j].strip()

                    # 如果下一行以中文字符开头，且不是特殊开头
                    if (next_line and
                        self._is_chinese_char(next_line[0]) and
                        not self._is_paragraph_start(next_line) and
                        not next_line.startswith('•') and
                        not next_line.startswith('*')):

                        # 合并
                        result.append(stripped + next_line)
                        i = j + 1
                        continue

            result.append(current)
            i += 1

        return result

    def _ends_with_sentence_end(self, text: str) -> bool:
        """检查是否以句末标点结尾"""
        sentence_endings = '。！？.!?'
        return text and text[-1] in sentence_endings

    def _is_chinese_char(self, char: str) -> bool:
        """检查是否是中文字符"""
        return '\u4e00' <= char <= '\u9fff'

    def _is_list_item_title(self, text: str) -> bool:
        """判断是否是列表项标题（如 **1.** 标题内容）"""
        import re
        # 匹配 **数字.** 开头的行
        match = re.match(r'^\*\*\d+\.\*\*\s*(.+)$', text)
        if match:
            content = match.group(1)
            # 如果内容部分较短（< 15字），认为是标题
            return len(content) < 15
        return False

    def _should_merge(self, prev: str, curr: str) -> bool:
        """判断是否应该合并两行"""
        # 上一行以不完整方式结尾
        incomplete_endings = ['，', '、', '：', '；', '"', '"', '（', '的', '是', '和', '与', '在', '了', '为', '对', '向', '被', '把', '让', '给', '从', '到', '以', '因', '但', '而', '或', '及', '筛', '防', '密', '告']
        # 当前行以小写或延续符号开头
        continuation_starts = ['"', '"', '）', '。', '！', '？', '选器', '御', '码', '诉']

        # 如果上一行以这些字符结尾，可能需要合并
        if any(prev.endswith(c) for c in incomplete_endings):
            return True

        # 如果当前行以这些字符开头，可能需要合并
        if any(curr.startswith(c) for c in continuation_starts):
            return True

        # 如果上一行很短（少于5个字符），且当前行不是新段落开始
        if len(prev) < 5 and not self._is_paragraph_start(curr):
            return True

        # 如果当前行很短（少于3个字符），可能是断句
        if len(curr) < 3:
            return True

        return False

    def _is_paragraph_start(self, text: str) -> bool:
        """判断是否是段落开始"""
        import re
        # 以数字列表开头
        if re.match(r'^\d+\.', text):
            return True
        # 以【标题】开头
        if text.startswith('【'):
            return True
        # 以 Markdown 标题开头
        if text.startswith('#'):
            return True
        # 以常见段落开头词开头
        para_starters = ['所以', '因此', '但是', '然而', '首先', '其次', '最后', '总之', '例如', '比如']
        if any(text.startswith(s) for s in para_starters):
            return True
        return False

    def _clean_empty_lines(self, lines: List[str]) -> List[str]:
        """清理多余的空行"""
        result = []
        prev_empty = False

        for line in lines:
            is_empty = not line.strip()

            if is_empty:
                if not prev_empty:  # 最多保留一个空行
                    result.append(line)
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        return result

    def _convert_chinese_headers(self, lines: List[str]) -> List[str]:
        """转换中文标题格式 【标题】 → ## 标题"""
        import re
        result = []

        for line in lines:
            stripped = line.strip()
            # 匹配 【xxx】 格式的标题
            match = re.match(r'^【(.+)】$', stripped)
            if match:
                title = match.group(1)
                result.append(f"\n## {title}\n")
            else:
                result.append(line)

        return result

    def _format_numbered_lists(self, lines: List[str]) -> List[str]:
        """格式化数字列表项"""
        import re
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            # 匹配 1.xxx 2.xxx 等格式（数字后直接跟文字，无空格）
            match = re.match(r'^(\d+)\.([^\s\d])', stripped)
            if match:
                num = match.group(1)
                rest = stripped[len(num) + 1:]  # 去掉 "1." 部分
                # 在列表项前添加空行（如果前一行不是空行）
                if result and result[-1].strip():
                    result.append('')
                result.append(f"**{num}.** {rest}")
            else:
                result.append(line)

        return result

    def _add_paragraph_spacing(self, lines: List[str]) -> List[str]:
        """在段落之间添加空行（基于句子结束符）"""
        result = []
        sentence_endings = '。！？'

        for i, line in enumerate(lines):
            result.append(line)
            stripped = line.strip()

            # 跳过空行和特殊格式行
            if not stripped:
                continue

            # 如果当前行以句号等结尾
            if stripped[-1] in sentence_endings:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # 下一行不为空，且不是已有格式
                    if (next_line and
                        not next_line.startswith('#') and
                        not next_line.startswith('-') and
                        not next_line.startswith('*') and
                        not next_line.startswith('**') and  # 不在加粗列表项前加空行
                        not next_line[0].isdigit()):
                        # 检查是否是新段落的开始（通常是较长的句子）
                        if len(next_line) > 15:
                            result.append('')

        return result


def format_with_layout(ocr_results: List[Tuple], image_height: int = 0) -> str:
    """
    使用布局分析格式化 OCR 结果

    Args:
        ocr_results: OCR 结果列表 [(text, confidence, [x, y, w, h]), ...]
        image_height: 图片高度（像素），用于计算间距

    Returns:
        格式化的 Markdown 文本
    """
    if not ocr_results:
        return ""

    # 布局分析
    analyzer = LayoutAnalyzer(line_spacing_threshold=1.8)
    layout = analyzer.analyze(ocr_results)

    # 转换为 Markdown
    markdown = analyzer.to_markdown(layout)

    # 文本清理
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(markdown)

    return cleaned
