"""检索编排:策略扩展 -> 召回 -> 取父块(二次校验权限/状态) -> 重排 -> 相关度阈值 -> TopM。

设计要点:
    - 支持直接 / HyDE / 子查询 / 回溯四种策略;扩展问题与原问题一并召回,
      LLM 改写跑偏时也不会丢失原意;
    - 所有检索统一在此处传入用户权限,避免不同策略路径遗漏过滤;
    - 相关度低于阈值时返回空,由上层拒答,绝不带着空上下文去调用 LLM(那样模型会编造);
    - 权限纵深防御:向量库按标签过滤一次,取父块时再按关系库中的文档 ACL 与状态校验一次,
      即使向量库数据陈旧(文档刚改权限/刚下线)也不会泄露;
    - 多路召回按 parent_id 去重,保留最高分。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from knowbase.core.security import Principal
from knowbase.db.models import DocParent, Document
from knowbase.rag import prompts
from knowbase.rag.llm import LLM, LLMError
from knowbase.rag.retrievers import Hit, Reranker, Retriever

log = logging.getLogger(__name__)


@dataclass
class Context:
    """送给 LLM 的一个资料片段,同时也是答案引用的来源。"""

    n: int  # 引用编号(从 1 开始)
    parent_id: str
    document_id: int
    title: str
    version: int
    section: str | None
    page: int | None
    text: str
    score: float

    def to_citation(self, cited: bool = True) -> dict:
        """转成对外的引用结构;snippet 只给前 160 字,供前端预览,不外泄整段正文。"""
        return {
            "n": self.n,
            "type": "doc",
            "document_id": self.document_id,
            "title": self.title,
            "version": self.version,
            "section": self.section,
            "page": self.page,
            "score": round(self.score, 3),
            "snippet": self.text[:160],
            "cited": cited,
        }


class RetrievalService:
    """检索服务。无状态,依赖(检索器、重排器、LLM、数据库)均由构造函数注入。"""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None,
        llm: LLM,
        session_factory: sessionmaker[Session],
        k: int,
        top_m: int,
        min_relevance: float,
    ) -> None:
        self.retriever, self.reranker, self.llm, self._sf = retriever, reranker, llm, session_factory
        self.k, self.top_m, self.min_relevance = k, top_m, min_relevance

    # ---- 策略扩展 ----
    def expand(self, query: str, strategy: str) -> list[str]:
        """按策略把一个问题扩展成一组检索问题(第一个始终是原问题)。

        任何 LLM 失败都回退为只用原问题——检索增强是锦上添花,不能因此让问答失败。
        """
        try:
            if strategy == "假设问题检索":
                hypo = self.llm.complete([{"role": "user", "content": prompts.HYDE_PROMPT.format(query=query)}], max_tokens=200)
                return [query, hypo] if hypo else [query]
            if strategy == "子查询检索":
                text = self.llm.complete(
                    [{"role": "user", "content": prompts.SUBQUERY_PROMPT.format(query=query)}], max_tokens=200
                )
                subs = [re.sub(r"^\s*[\d\-\.\)、\*]+\s*", "", s).strip() for s in text.splitlines()]
                return [query] + [s for s in subs if s][:3]
            if strategy == "回溯问题检索":
                simple = self.llm.complete(
                    [{"role": "user", "content": prompts.BACKTRACK_PROMPT.format(query=query)}], max_tokens=100
                )
                return [query, simple] if simple else [query]
        except LLMError:
            log.warning("检索策略 %s 的 LLM 扩展失败,回退到原问题", strategy)
        return [query]

    # ---- 主流程 ----
    def retrieve(
        self, principal: Principal, query: str, strategy: str = "直接检索", category: str | None = None
    ) -> list[Context]:
        """执行完整检索流程,返回按相关度降序、已通过阈值与权限校验的资料片段(可能为空)。

        Args:
            principal: 当前用户;其 acl_tags 同时用于检索过滤与二次校验。
            query: 用于检索的问题(多轮追问时已被改写补全)。
            strategy: 检索策略名,须在白名单内。
            category: 可选的知识分类过滤。
        """
        merged: dict[str, Hit] = {}
        for q in self.expand(query, strategy):
            for h in self.retriever.search(q, principal.acl_tags, category, self.k):
                if h.parent_id not in merged or merged[h.parent_id].score < h.score:
                    merged[h.parent_id] = h
        if not merged:
            return []

        # 取父块正文并校验:文档必须仍处于 indexed 状态(下线/归档/重建中的文档不应被检索到)
        with self._sf() as db:
            rows = db.execute(
                select(DocParent, Document)
                .join(Document, Document.id == DocParent.document_id)
                .where(DocParent.parent_id.in_(list(merged)), Document.status == "indexed")
            ).all()

        cands = []
        for parent, doc in rows:
            if not principal.can_access(doc.acl_tags):  # 纵深防御:库里的 ACL 为准
                log.warning("向量库返回了无权文档,已丢弃 doc=%s user=%s", doc.id, principal.username)
                continue
            cands.append((parent, doc, merged[parent.parent_id].score))
        if not cands:
            return []

        # 有重排器用重排分;没有(开发/CI)则用检索原始分。两者都归一到 0~1,可与 min_relevance 比较
        scores = self.reranker.score(query, [p.content for p, _, _ in cands]) if self.reranker else [s for _, _, s in cands]
        ranked = sorted(zip(cands, scores, strict=True), key=lambda x: x[1], reverse=True)
        out: list[Context] = []
        for (parent, doc, _), score in ranked:
            if score < self.min_relevance:
                continue
            out.append(
                Context(
                    len(out) + 1,
                    parent.parent_id,
                    doc.id,
                    doc.title,
                    doc.version,
                    parent.section,
                    parent.page,
                    parent.content,
                    float(score),
                )
            )
            if len(out) >= self.top_m:
                break
        log.info("检索完成 strategy=%s 候选=%d 采用=%d", strategy, len(cands), len(out))
        return out
