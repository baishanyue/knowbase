"""父子块切分:检索用小块(命中精准),送给 LLM 用大块(上下文完整)。

为什么分父子:
    小块语义集中,向量/关键词匹配更准;但单个小块往往缺少上下文,直接喂给 LLM 容易断章取义。
    所以"用子块找到位置、用父块提供上下文"。

为什么先按章节再切块:
    保证每个父块只属于一个章节,引用才能精确到「文档 · 章节 · 页码」,
    员工点开出处就能直接定位到制度原文。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowbase.rag.ingestion.loaders import Section
from knowbase.rag.splitter import ChineseRecursiveTextSplitter


@dataclass
class ParentChunk:
    """一个父块及其子块。parent_id 形如 d{文档}v{版本}p{序号},版本号写进 ID,新旧版本的块互不冲突。"""

    parent_id: str
    content: str
    section: str | None
    page: int | None
    seq: int
    children: list[tuple[str, str]] = field(default_factory=list)  # (child_id, text)


def build_chunks(
    sections: list[Section], document_id: int, version: int, parent_size: int, child_size: int, overlap: int
) -> list[ParentChunk]:
    """把文档章节切成父子块。

    Args:
        sections: 文档章节(含标题、正文、页码)。
        document_id / version: 用于生成稳定且可追溯的块 ID。
        parent_size / child_size: 父/子块目标字符数。
        overlap: 相邻块重叠字符数,减少在边界处丢失语义;子块重叠上限为其大小的 1/4。
    """
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_size, chunk_overlap=overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_size, chunk_overlap=min(overlap, child_size // 4))
    out: list[ParentChunk] = []
    seq = 0
    for sec in sections:
        for ptext in parent_splitter.split_text(sec.text):
            # 把章节标题带进块内文本,检索时标题里的关键词(如"年假")也能命中
            content = f"{sec.heading}\n{ptext}" if sec.heading else ptext
            pid = f"d{document_id}v{version}p{seq}"
            parent = ParentChunk(pid, content, sec.heading, sec.page, seq)
            for k, ctext in enumerate(child_splitter.split_text(content)):
                parent.children.append((f"{pid}c{k}", ctext))
            out.append(parent)
            seq += 1
    return out
