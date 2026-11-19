"""FAQ 层:高频问题优先命中,成本低、答案由运营掌控。

设计要点: 命中前先做部门权限过滤;索引随库内容自动刷新(以 count + max(updated_at) 为签名,
多进程安全);置信度经归一化(见 rag/bm25.py),并要求领先第二名一定差距以避免误命中;
命中计数用于运营分析。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from knowbase.core.security import Principal
from knowbase.db.models import FaqEntry
from knowbase.rag.bm25 import BM25, tokenize

log = logging.getLogger(__name__)


@dataclass
class FaqHit:
    """FAQ 命中结果。confidence 为归一化后的 0~1 置信度。"""

    entry_id: int
    question: str
    answer: str
    confidence: float
    category: str


class FaqMatcher:
    """FAQ 匹配器:内存 BM25 索引 + 按用户权限过滤。线程安全,可被多个请求并发使用。"""

    def __init__(self, threshold: float, margin: float) -> None:
        self.threshold, self.margin = threshold, margin
        self._lock = threading.Lock()
        self._sig: tuple | None = None
        self._bm25: BM25 | None = None
        self._entries: list[FaqEntry] = []

    def _signature(self, db: Session) -> tuple:
        """库内容签名:(启用条数, 最近更新时间)。任何增删改都会改变它,据此判断索引是否过期。"""
        row = db.execute(select(func.count(FaqEntry.id), func.max(FaqEntry.updated_at)).where(FaqEntry.status == "active")).one()
        return (row[0], row[1])

    def _refresh(self, db: Session) -> None:
        """签名变化时重建索引。

        先无锁快速判断(绝大多数请求直接返回),需要重建时加锁并再判断一次(双重检查),
        避免并发请求同时重建。索引放在进程内,多进程各自按同一签名独立刷新,无需跨进程通信。
        """
        sig = self._signature(db)
        if sig == self._sig:
            return
        with self._lock:
            if sig == self._sig:
                return
            entries = list(db.scalars(select(FaqEntry).where(FaqEntry.status == "active").order_by(FaqEntry.id)))
            for e in entries:
                db.expunge(e)  # 脱离会话:缓存的对象在会话关闭后仍可读取
            self._entries = entries
            self._bm25 = BM25([tokenize(e.question) for e in entries])
            self._sig = sig
            log.info("FAQ 索引已重建: %d 条", len(entries))

    def match(self, db: Session, query: str, principal: Principal, category: str | None = None) -> FaqHit | None:
        """在用户有权限的 FAQ 中寻找最匹配的一条。

        命中需同时满足:置信度 >= threshold,且领先第二名至少 margin。
        第二个条件用来避免两条相近 FAQ 互相"抢答"导致答非所问,宁可交给 RAG 处理。

        Returns:
            命中则返回 FaqHit(并累加命中计数),否则 None。
        """
        self._refresh(db)
        if not self._bm25 or not self._entries:
            return None
        # 权限与分类过滤放在排序之前:无权访问的 FAQ 既不会命中,也不会影响"领先第二名"的判断
        mask = [principal.can_access(e.acl_tags) and (category is None or e.category == category) for e in self._entries]
        ranked = self._bm25.search(query, mask=mask, top_k=2)
        if not ranked:
            return None
        best = ranked[0]
        second = ranked[1].confidence if len(ranked) > 1 else 0.0
        if best.confidence < self.threshold or (best.confidence - second) < self.margin:
            log.info("FAQ 未命中: best=%.2f second=%.2f", best.confidence, second)
            return None
        e = self._entries[best.index]
        db.execute(update(FaqEntry).where(FaqEntry.id == e.id).values(hit_count=FaqEntry.hit_count + 1))
        return FaqHit(e.id, e.question, e.answer, round(best.confidence, 3), e.category)
