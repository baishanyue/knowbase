"""分词与 BM25(带归一化置信度),供 FAQ 层与内存检索器共用。

为什么要"归一化置信度":
    原始 BM25 得分的量级取决于语料大小和词的稀有程度,无法设定通用阈值。这里用
        置信度 = 候选的 BM25 得分 / "查询与候选完全一致时"的理想得分
    把分数压到约 0~1,与语料规模无关,阈值才有稳定含义。

跨领域问题的处理:
    查询里"库中从未出现的词"信息量大却不可能被任何文档匹配,会按 UNSEEN_W 的权重计入
    理想得分的分母,使"今天北京天气怎么样"这类只有个别词碰巧命中的问题置信度被拉低。

为什么自研而不用 rank_bm25:
    需要上述归一化与"按权限掩码过滤后再排名"的能力,代码量很小,且无额外依赖。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

try:  # jieba 是首选;缺失时退化为字符二元组,功能可用但召回略差
    import jieba

    jieba.setLogLevel(60)
except Exception:  # noqa: BLE001
    jieba = None

# fmt: off
_STOP = frozenset(
    {
        "的", "了", "是", "在", "我", "你", "他", "她", "它", "吗", "呢", "吧", "啊", "呀", "么",
        "请问", "请", "想", "要", "能", "可以", "怎么", "如何", "怎样", "什么", "哪些", "有", "和",
        "与", "或", "及", "一下", "一个", "这个", "那个", "我们", "公司", "如果", "需要", "应该",
        "是否", "能否", "咨询", "问", "个", "把", "被", "就", "都", "也", "还", "并", "但", "而", "之", "其",
    }
)
# fmt: on
UNSEEN_W = 0.5  # 库中未出现的查询词在置信度上限中的权重(0=忽略,1=按最稀有词计);经评估集标定
_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文分词并去停用词。jieba 不可用时退化为字符二元组(召回略差但功能可用)。"""
    text = (text or "").lower()
    if jieba is not None:
        toks = [t.strip() for t in jieba.lcut(text)]
    else:  # pragma: no cover
        toks = []
        for w in _WORD_RE.findall(text):
            toks.extend([w[i : i + 2] for i in range(max(len(w) - 1, 1))] if re.search(r"[\u4e00-\u9fff]", w) else [w])
    return [t for t in toks if t and _WORD_RE.fullmatch(t) and t not in _STOP]


@dataclass
class ScoredIndex:
    """一次检索的单条结果:文档下标与两种分数。"""

    index: int
    score: float  # 原始 BM25
    confidence: float  # 归一化 0~1


class BM25:
    """内存 BM25 索引。构造时一次性建好词频与 idf,查询是纯内存计算。"""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.n = len(docs_tokens)
        self.avgdl = (sum(len(d) for d in docs_tokens) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df: Counter[str] = Counter()
        for d in docs_tokens:
            df.update(set(d))
        # 恒为正的 idf 变体,避免小语料出现负权重
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _score_doc(self, q_tokens: list[str], i: int) -> float:
        """查询对第 i 篇文档的原始 BM25 得分(查询词去重,避免重复词放大得分)。"""
        dl = len(self.docs[i]) or 1
        tf = self.tf[i]
        s = 0.0
        for t in set(q_tokens):
            f = tf.get(t, 0)
            if not f:
                continue
            idf = self.idf.get(t, 0.0)
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return s

    def upper_bound(self, q_tokens: list[str]) -> float:
        """完全匹配时的理想得分,用作置信度的分母。

        假设:每个查询词在一篇与查询等长的文档里恰好出现 1 次(FAQ 问题与提问一字不差时置信度≈1.0)。
        库中从未出现的词按"最稀有词的 idf × UNSEEN_W"计入,原因见模块说明。
        """
        uniq = set(q_tokens)
        idf_unseen = UNSEEN_W * math.log(1 + (self.n + 0.5) / 0.5)
        norm = 1 - self.b + self.b * len(uniq) / (self.avgdl or 1)
        sat = (self.k1 + 1) / (1 + self.k1 * norm)
        return sum(self.idf.get(t, idf_unseen) for t in uniq) * sat

    def search(self, query: str, mask: list[bool] | None = None, top_k: int = 5) -> list[ScoredIndex]:
        """检索并返回按原始得分降序的 top_k 条。

        Args:
            query: 查询文本。
            mask: mask[i]=False 的文档不参与排名。用于权限过滤:无权文档既不能命中,
                也不会影响调用方"领先第二名"的判断,避免通过置信度差间接探测到隐藏文档。
            top_k: 返回条数上限。
        """
        q = tokenize(query)
        if not q or not self.n:
            return []
        ub = self.upper_bound(q)
        if ub <= 0:
            return []
        out = []
        for i in range(self.n):
            if mask is not None and not mask[i]:
                continue
            s = self._score_doc(q, i)
            if s > 0:
                out.append(ScoredIndex(i, s, min(s / ub, 1.0)))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]
