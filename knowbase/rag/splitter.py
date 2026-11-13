"""中文递归切分器。

LangChain 自带的 RecursiveCharacterTextSplitter 默认只认英文标点。这里按
"段落 -> 换行 -> 句末标点 -> 分号 -> 逗号"逐级向下切分,优先在语义边界断开,
并保留分隔符(句号等)留在句尾,避免切出以标点开头的碎片。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_with_regex(text: str, separator: str, keep_separator: bool) -> list[str]:
    """按正则分隔符切分;keep_separator=True 时把分隔符并回前一段末尾。空串分隔符表示逐字切分(兜底)。"""
    if separator:
        if keep_separator:
            parts = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(parts[0::2], parts[1::2], strict=False)]
            if len(parts) % 2 == 1:
                splits += parts[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """面向中文的递归切分器,接口与父类一致(chunk_size / chunk_overlap)。"""

    def __init__(self, separators: list[str] | None = None, keep_separator: bool = True, **kwargs: Any) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", "。|！|？", r"\.\s|\!\s|\?\s", r"；|;\s", r"，|,\s"]
        self._is_separator_regex = True

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """选出第一个在文本中出现的分隔符切分;过长的片段用更细的分隔符递归处理,短片段合并到目标大小。"""
        final: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if re.search(s, text):
                separator = s
                new_separators = separators[i + 1 :]
                break
        splits = _split_with_regex(text, separator, self._keep_separator)
        good: list[str] = []
        joiner = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good.append(s)
            else:
                if good:
                    final.extend(self._merge_splits(good, joiner))
                    good = []
                final.extend(self._split_text(s, new_separators) if new_separators else [s])
        if good:
            final.extend(self._merge_splits(good, joiner))
        # 收尾:压缩连续空行,丢弃空块
        return [re.sub(r"\n{2,}", "\n", c.strip()) for c in final if c.strip()]
