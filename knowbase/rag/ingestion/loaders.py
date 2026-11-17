"""文档加载:返回带「章节标题 / 页码」的 Section,供切分与引用溯源使用。

默认覆盖 md / txt / pdf / docx。需要 OCR 的扫描件、图片、PPT 等格式,
可通过 register_loader(".pptx", fn) 接入;默认不引入相关依赖,以保持镜像轻量。

约定:加载器只负责"取出文字与结构",不做切块;读不到文字时由 load_file 统一报错,
避免把空文档当作入库成功。
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Section:
    """文档中的一个连续片段,是切块与引用的最小结构单元。"""

    text: str
    heading: str | None = None  # 如 "第三章 休假 > 年假"
    page: int | None = None


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _load_markdown(path: str) -> list[Section]:
    """按 # 标题层级切成章节,heading 为"一级 > 二级"的路径,用于引用展示。"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(text=body, heading=" > ".join(h for _, h in stack) or None))
        buf.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            buf.append(line)
    flush()
    return sections


def _load_text(path: str) -> list[Section]:
    """纯文本:整篇作为一个无标题章节。errors=ignore 容忍个别坏字节。"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [Section(text=text)] if text else []


def _load_pdf(path: str) -> list[Section]:
    """PDF:每页一个 Section 并记录页码。纯图片页没有文字层会被跳过(需 OCR 加载器)。"""
    from pypdf import PdfReader

    out = []
    for i, page in enumerate(PdfReader(path).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(Section(text=text, page=i))
    return out


def _load_docx(path: str) -> list[Section]:
    """Word:依据 Heading N 样式识别章节层级(未使用标题样式的文档会成为单个无标题章节)。"""
    import docx

    d = docx.Document(path)
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(text=body, heading=" > ".join(h for _, h in stack) or None))
        buf.clear()

    for p in d.paragraphs:
        style = (p.style.name or "") if p.style is not None else ""
        m = re.match(r"Heading (\d)", style)
        if m and p.text.strip():
            flush()
            level = int(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, p.text.strip()))
        elif p.text.strip():
            buf.append(p.text.strip())
    flush()
    return sections


_LOADERS: dict[str, Callable[[str], list[Section]]] = {
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".txt": _load_text,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS)


def register_loader(ext: str, fn: Callable[[str], list[Section]]) -> None:
    """注册/覆盖某扩展名的加载器,用于接入 OCR、PPT 等自定义格式。"""
    _LOADERS[ext.lower()] = fn


def load_file(path: str) -> list[Section]:
    """按扩展名选择加载器读取文件。

    Raises:
        ValueError: 扩展名不支持,或文档中没有可提取的文字。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"不支持的文件类型: {ext}(支持 {', '.join(SUPPORTED_EXTENSIONS)})")
    sections = _LOADERS[ext](path)
    if not sections:
        raise ValueError("文档中没有可提取的文字(扫描件请接入 OCR 加载器)")
    return sections
