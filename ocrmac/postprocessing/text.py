"""文本清理和格式化模块"""

import re
import sys

if sys.version_info < (3, 9):
    from typing import List
else:
    List = list


class TextCleaner:
    """
    文本清理器 - 去重、合并、格式化

    处理流程：
    1. 去除重复行
    2. 合并断句
    3. 转换中文标题格式
    4. 格式化数字列表
    5. 添加段落间空行
    6. 清理多余空行
    """

    def __init__(self, duplicate_threshold: float = 0.8):
        """
        初始化文本清理器

        Args:
            duplicate_threshold: 文本相似度阈值，超过此值认为是重复行
        """
        self.duplicate_threshold = duplicate_threshold

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

    # ==================== 去重处理 ====================

    def _remove_duplicate_lines(self, lines: List[str]) -> List[str]:
        """
        去除重复或高度相似的行

        特别处理跨切片重复：
        - 如果新行是旧行的扩展（更完整），替换旧行
        - 如果新行是旧行的子集，跳过新行
        """
        if not lines:
            return []

        result = [lines[0]]
        for line in lines[1:]:
            curr = line.strip()
            if not curr:
                result.append(line)
                continue

            is_duplicate = False
            replace_index = -1

            # 检查最近的行
            for i, prev_line in enumerate(result[-10:]):  # 扩大检查范围到10行
                prev = prev_line.strip()
                if not prev:
                    continue

                # 完全相同
                if curr == prev:
                    is_duplicate = True
                    break

                # 检查是否一个是另一个的前缀/子串
                relation = self._check_text_relation(prev, curr)

                if relation == 'same':
                    is_duplicate = True
                    break
                elif relation == 'curr_extends_prev':
                    # 当前行是前一行的扩展，需要替换
                    actual_index = len(result) - 10 + i
                    if actual_index >= 0:
                        replace_index = actual_index
                    is_duplicate = True  # 标记为重复，但会替换
                    break
                elif relation == 'prev_extends_curr':
                    # 前一行已经是更完整的版本，跳过当前行
                    is_duplicate = True
                    break
                elif relation == 'high_similarity':
                    # 高相似度，保留较长的
                    if len(curr) > len(prev):
                        actual_index = len(result) - 10 + i
                        if actual_index >= 0:
                            replace_index = actual_index
                    is_duplicate = True
                    break

            if replace_index >= 0:
                # 替换为更完整的版本
                result[replace_index] = line
            elif not is_duplicate:
                result.append(line)

        return result

    def _check_text_relation(self, text1: str, text2: str) -> str:
        """
        检查两个文本的关系

        Returns:
            'same': 完全相同
            'curr_extends_prev': text2 是 text1 的扩展
            'prev_extends_curr': text1 是 text2 的扩展
            'high_similarity': 高相似度但不是子集关系
            'different': 不同
        """
        if text1 == text2:
            return 'same'

        # 检查前缀关系（更精确的扩展检测）
        # text2 以 text1 开头，说明 text2 是扩展
        if text2.startswith(text1) and len(text2) > len(text1):
            return 'curr_extends_prev'

        # text1 以 text2 开头，说明 text1 是扩展
        if text1.startswith(text2) and len(text1) > len(text2):
            return 'prev_extends_curr'

        # 检查是否有大量重叠（可能是中间部分重复）
        # 比如 "ABC" 和 "BCD" 的情况
        overlap = self._find_overlap(text1, text2)
        if overlap:
            overlap_ratio = len(overlap) / min(len(text1), len(text2))
            if overlap_ratio > 0.7:
                # 高重叠，保留较长的
                if len(text2) > len(text1):
                    return 'curr_extends_prev'
                else:
                    return 'prev_extends_curr'

        # 计算相似度
        similarity = self._calculate_similarity(text1, text2)
        if similarity > self.duplicate_threshold:
            return 'high_similarity'

        return 'different'

    def _find_overlap(self, text1: str, text2: str) -> str:
        """
        查找两个文本之间的重叠部分

        检查 text1 的后缀是否是 text2 的前缀
        """
        min_overlap = 10  # 最小重叠长度
        max_check = min(len(text1), len(text2), 100)  # 最多检查100字符

        for i in range(min_overlap, max_check + 1):
            suffix = text1[-i:]
            if text2.startswith(suffix):
                return suffix

        return ""

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（Jaccard）"""
        if not text1 or not text2:
            return 0.0
        if text1 == text2:
            return 1.0
        if text1 in text2 or text2 in text1:
            return 0.9

        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    # ==================== 断句合并 ====================

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

            if buffer:
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

            if not stripped:
                result.append(current)
                i += 1
                continue

            # 判断是否是特殊行（不应合并）
            is_list_content = stripped.startswith('**') and not self._is_list_item_title(stripped)

            is_special_line = (
                stripped.startswith('#') or
                stripped.startswith('◎') or
                stripped.endswith('】') or
                stripped.endswith('？') or
                stripped.endswith('：') or
                self._is_list_item_title(stripped) or
                (len(stripped) < 25 and not is_list_content)
            )

            should_try_merge = (
                stripped and
                not is_special_line and
                self._is_chinese_char(stripped[-1]) and
                not self._ends_with_sentence_end(stripped)
            )

            if should_try_merge:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1

                if j < len(lines):
                    next_line = lines[j].strip()

                    if (next_line and
                        self._is_chinese_char(next_line[0]) and
                        not self._is_paragraph_start(next_line) and
                        not next_line.startswith('•') and
                        not next_line.startswith('*')):

                        result.append(stripped + next_line)
                        i = j + 1
                        continue

            result.append(current)
            i += 1

        return result

    def _should_merge(self, prev: str, curr: str) -> bool:
        """判断是否应该合并两行"""
        incomplete_endings = [
            '，', '、', '：', '；', '"', '"', '（',
            '的', '是', '和', '与', '在', '了', '为', '对', '向',
            '被', '把', '让', '给', '从', '到', '以', '因', '但',
            '而', '或', '及', '筛', '防', '密', '告'
        ]
        continuation_starts = ['"', '"', '）', '。', '！', '？', '选器', '御', '码', '诉']

        if any(prev.endswith(c) for c in incomplete_endings):
            return True

        if any(curr.startswith(c) for c in continuation_starts):
            return True

        if len(prev) < 5 and not self._is_paragraph_start(curr):
            return True

        if len(curr) < 3:
            return True

        return False

    # ==================== 格式转换 ====================

    def _convert_chinese_headers(self, lines: List[str]) -> List[str]:
        """转换中文标题格式 【标题】 → ## 标题"""
        result = []

        for line in lines:
            stripped = line.strip()
            match = re.match(r'^【(.+)】$', stripped)
            if match:
                title = match.group(1)
                result.append(f"\n## {title}\n")
            else:
                result.append(line)

        return result

    def _format_numbered_lists(self, lines: List[str]) -> List[str]:
        """格式化数字列表项 1.xxx → **1.** xxx"""
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            match = re.match(r'^(\d+)\.([^\s\d])', stripped)
            if match:
                num = match.group(1)
                rest = stripped[len(num) + 1:]
                if result and result[-1].strip():
                    result.append('')
                result.append(f"**{num}.** {rest}")
            else:
                result.append(line)

        return result

    # ==================== 段落处理 ====================

    def _add_paragraph_spacing(self, lines: List[str]) -> List[str]:
        """在段落之间添加空行"""
        result = []
        sentence_endings = '。！？'

        for i, line in enumerate(lines):
            result.append(line)
            stripped = line.strip()

            if not stripped:
                continue

            if stripped[-1] in sentence_endings:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if (next_line and
                        not next_line.startswith('#') and
                        not next_line.startswith('-') and
                        not next_line.startswith('*') and
                        not next_line.startswith('**') and
                        not next_line[0].isdigit()):
                        if len(next_line) > 15:
                            result.append('')

        return result

    def _clean_empty_lines(self, lines: List[str]) -> List[str]:
        """清理多余的空行（最多保留一个）"""
        result = []
        prev_empty = False

        for line in lines:
            is_empty = not line.strip()

            if is_empty:
                if not prev_empty:
                    result.append(line)
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        return result

    # ==================== 辅助方法 ====================

    def _ends_with_sentence_end(self, text: str) -> bool:
        """检查是否以句末标点结尾"""
        sentence_endings = '。！？.!?'
        return text and text[-1] in sentence_endings

    def _is_chinese_char(self, char: str) -> bool:
        """检查是否是中文字符"""
        return '\u4e00' <= char <= '\u9fff'

    def _is_list_item_title(self, text: str) -> bool:
        """判断是否是列表项标题（如 **1.** 标题内容）"""
        match = re.match(r'^\*\*\d+\.\*\*\s*(.+)$', text)
        if match:
            content = match.group(1)
            return len(content) < 15
        return False

    def _is_paragraph_start(self, text: str) -> bool:
        """判断是否是段落开始"""
        if re.match(r'^\d+\.', text):
            return True
        if text.startswith('【'):
            return True
        if text.startswith('#'):
            return True
        para_starters = ['所以', '因此', '但是', '然而', '首先', '其次', '最后', '总之', '例如', '比如']
        if any(text.startswith(s) for s in para_starters):
            return True
        return False
