"""检索后端:Milvus(生产,稠密+稀疏混合检索)与 Memory(开发/演示/CI,纯 BM25,无需 GPU 与模型)。

两个后端实现同一个 Retriever 协议,上层(RetrievalService)完全不感知差异,
这也是单元测试与 CI 不依赖向量库和模型文件的原因。

设计要点:
    - 权限过滤在检索层强制执行(array_contains_any(acl_tags, 用户标签)),
      过滤值来自服务端身份并经白名单校验,不拼接任何用户原始输入;
    - 父块正文存放在关系库,向量库只保存子块向量与权限/分类等过滤字段;
    - 入库携带 document_id,支持按文档整体删除或替换(版本更新、权限变更、下线);
    - 向量库只返回 (parent_id, 分数),业务字段以关系库为准,避免两处数据不一致。

未经真实环境验证的部分:
    MilvusRetriever 与 CrossEncoderReranker 依赖 GPU/模型/向量库,自动化测试无法覆盖,
    上线前须在目标环境用 KNOWBASE_RETRIEVER_BACKEND=milvus 跑一遍评估脚本并标定阈值。
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from knowbase.db.models import DocParent, Document
from knowbase.rag.bm25 import BM25, tokenize

log = logging.getLogger(__name__)
_CAT_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")


@dataclass
class ChunkRecord:
    """待写入索引的一个子块,附带检索时用于过滤的元数据。"""

    child_id: str
    parent_id: str
    document_id: int
    text: str
    category: str
    acl_tags: list[str]
    version: int


@dataclass
class Hit:
    """一次召回命中:父块 ID、所属文档与 0~1 的相关度分。"""

    parent_id: str
    document_id: int
    score: float  # 0~1


class Retriever(Protocol):
    """检索后端协议。"""

    def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        """写入一批子块(幂等,相同 child_id 覆盖)。"""
        ...

    def delete_document(self, document_id: int) -> None:
        """删除某文档的全部子块(版本归档、下线、重建索引前调用)。"""
        ...

    def search(self, query: str, acl_tags: list[str], category: str | None, k: int) -> list[Hit]:
        """召回最多 k 个父块;只返回 acl_tags 有交集的内容。"""
        ...

    def ping(self) -> bool:
        """健康检查。"""
        ...


# ----------------------------------------------------------------------------- Memory
class MemoryRetriever:
    """以父块为粒度做 BM25。

    索引由关系库内容派生(签名变化即重建),所以重启不丢、也无需单独的入库步骤。
    仅适合开发/演示/评估;多实例或大数据量请用 Milvus。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory
        self._lock = threading.Lock()
        self._sig: tuple | None = None
        self._bm25: BM25 | None = None
        self._meta: list[tuple[str, int, list[str], str]] = []  # parent_id, doc_id, acl, category

    def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        """空实现:索引直接由关系库派生,无需写入。"""
        return None

    def delete_document(self, document_id: int) -> None:
        """空实现:文档状态变化后,下一次检索会按新签名自动重建索引。"""
        return None

    def _refresh(self) -> None:
        """签名(父块数、最近更新时间、最大文档 ID)变化时重建内存索引。"""
        with self._sf() as db:
            row = db.execute(
                select(func.count(DocParent.parent_id), func.max(Document.updated_at), func.max(Document.id))
                .join(Document, Document.id == DocParent.document_id)
                .where(Document.status == "indexed")
            ).one()
            sig = tuple(row)
            if sig == self._sig:
                return
            with self._lock:
                rows = db.execute(
                    select(DocParent.parent_id, DocParent.content, Document.id, Document.acl_tags, Document.category)
                    .join(Document, Document.id == DocParent.document_id)
                    .where(Document.status == "indexed")
                    .order_by(DocParent.parent_id)
                ).all()
                self._meta = [(r[0], r[2], list(r[3] or []), r[4]) for r in rows]
                self._bm25 = BM25([tokenize(r[1]) for r in rows])
                self._sig = sig
                log.info("内存索引已重建: %d 个父块", len(rows))

    def search(self, query: str, acl_tags: list[str], category: str | None, k: int) -> list[Hit]:
        """BM25 检索。权限与分类以掩码形式在排名前过滤,得分为归一化置信度(0~1)。"""
        self._refresh()
        if not self._bm25 or not self._meta:
            return []
        allowed = set(acl_tags)
        mask = [bool(allowed & set(m[2])) and (category is None or m[3] == category) for m in self._meta]
        ranked = self._bm25.search(query, mask=mask, top_k=k)
        return [Hit(self._meta[r.index][0], self._meta[r.index][1], r.confidence) for r in ranked]

    def ping(self) -> bool:
        return True


# ----------------------------------------------------------------------------- Milvus
class MilvusRetriever:
    """BGE-M3 稠密+稀疏混合检索。

    稠密向量擅长语义相近,稀疏向量擅长关键词精确(制度编号、专有名词),两路用
    WeightedRanker(稠密 1.0, 稀疏 0.7)融合;权重是经验起点,请用评估集调优。
    """

    def __init__(self, uri: str, token: str | None, db_name: str, collection: str, model_dir: str, embedding_model: str) -> None:
        import os

        import torch
        from milvus_model.hybrid import BGEM3EmbeddingFunction
        from pymilvus import MilvusClient

        self.collection = collection
        # 有 GPU 用 GPU 并启用半精度;否则退回 CPU(可用但较慢,适合小规模验证)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed = BGEM3EmbeddingFunction(
            model_name_or_path=os.path.join(model_dir, embedding_model), use_fp16=(device == "cuda"), device=device
        )
        self.dense_dim = self.embed.dim["dense"]
        self.client = MilvusClient(uri=uri, token=token or "")
        try:
            if db_name not in self.client.list_databases():
                self.client.create_database(db_name)
            self.client.using_database(db_name)
        except Exception as e:  # noqa: BLE001  (Milvus 2.4 之前没有多库,忽略)
            log.warning("Milvus 切库失败,使用默认库: %s", e)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """集合不存在则按固定 schema 创建(含两路向量索引),并加载到内存。幂等。"""
        from pymilvus import DataType

        if not self.client.has_collection(self.collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field("text", DataType.VARCHAR, max_length=8192)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("parent_id", DataType.VARCHAR, max_length=100)
            schema.add_field("document_id", DataType.INT64)
            schema.add_field("category", DataType.VARCHAR, max_length=32)
            schema.add_field("version", DataType.INT64)
            schema.add_field("acl_tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=16, max_length=64)
            idx = self.client.prepare_index_params()
            idx.add_index(
                "dense_vector", index_name="dense_index", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128}
            )
            idx.add_index(
                "sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )
            self.client.create_collection(self.collection, schema=schema, index_params=idx)
            log.info("已创建集合 %s", self.collection)
        self.client.load_collection(self.collection)

    @staticmethod
    def _sparse_to_dict(matrix, i: int) -> dict[int, float]:
        """把稀疏矩阵的第 i 行转成 Milvus 要求的 {维度: 权重} 字典。"""
        row = matrix[[i]]
        return {int(c): float(v) for c, v in zip(row.indices, row.data, strict=True)}

    def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        """编码并 upsert 子块。"""
        for start in range(0, len(chunks), 64):  # 分批,避免一次编码过多占满显存
            batch = chunks[start : start + 64]
            emb = self.embed([c.text for c in batch])
            data = [
                {
                    "id": c.child_id,
                    "text": c.text[:8000],
                    "dense_vector": emb["dense"][i],
                    "sparse_vector": self._sparse_to_dict(emb["sparse"], i),
                    "parent_id": c.parent_id,
                    "document_id": c.document_id,
                    "category": c.category,
                    "version": c.version,
                    "acl_tags": c.acl_tags,
                }
                for i, c in enumerate(batch)
            ]
            self.client.upsert(collection_name=self.collection, data=data)

    def delete_document(self, document_id: int) -> None:
        """按文档删除全部子块。int() 强制转换,杜绝过滤表达式注入。"""
        self.client.delete(collection_name=self.collection, filter=f"document_id == {int(document_id)}")

    @staticmethod
    def build_filter(acl_tags: list[str], category: str | None) -> str:
        """构造 Milvus 过滤表达式:权限标签必有,分类可选。

        这是权限隔离的第一道关口。标签来自服务端 Principal 且已过白名单,category 再次校验;
        任何不合规输入都在这里抛错而不是被拼进表达式。
        """
        tags = ", ".join(f'"{t}"' for t in acl_tags)  # acl_tags 来自服务端 Principal,且已过白名单
        expr = f"array_contains_any(acl_tags, [{tags}])"
        if category:
            if not _CAT_RE.match(category):
                raise ValueError("非法的 category")
            expr += f' and category == "{category}"'
        return expr

    def search(self, query: str, acl_tags: list[str], category: str | None, k: int) -> list[Hit]:
        """混合检索:稠密、稀疏两路各召回 k 个,加权融合后按 parent_id 去重取最高分。"""
        from pymilvus import AnnSearchRequest, WeightedRanker

        emb = self.embed([query])
        expr = self.build_filter(acl_tags, category)
        dense = AnnSearchRequest(
            data=[emb["dense"][0]],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=expr,
        )
        sparse = AnnSearchRequest(
            data=[self._sparse_to_dict(emb["sparse"], 0)],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=expr,
        )
        res = self.client.hybrid_search(
            self.collection,
            reqs=[dense, sparse],
            ranker=WeightedRanker(1.0, 0.7),
            limit=k,
            output_fields=["parent_id", "document_id"],
        )[0]
        hits: dict[str, Hit] = {}
        for r in res:
            ent = r["entity"]
            score = max(0.0, min(float(r["distance"]) / 1.7, 1.0))  # 原始融合分归一到 0~1(仅作无重排时的兜底)
            pid = ent["parent_id"]
            if pid not in hits or hits[pid].score < score:
                hits[pid] = Hit(pid, int(ent["document_id"]), score)
        return sorted(hits.values(), key=lambda h: h.score, reverse=True)

    def ping(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception:  # noqa: BLE001
            return False


# ----------------------------------------------------------------------------- Reranker
class Reranker(Protocol):
    """重排器协议:对(问题, 候选文本)逐一打 0~1 的相关度分。"""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """bge-reranker 交叉编码器重排,输出过 sigmoid 归一到 0~1,可直接与 min_relevance 阈值比较。"""

    def __init__(self, model_path: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_path, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """交叉编码器逐对打分。sigmoid 让分数落在 0~1,才能与 min_relevance 阈值比较。"""
        import math

        raw = self.model.predict([[query, t] for t in texts])
        return [1 / (1 + math.exp(-float(x))) for x in raw]
