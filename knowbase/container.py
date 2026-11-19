"""依赖装配(组合根):把配置翻译成具体实现,是全项目唯一"new 出各组件"的地方。

好处:
    - 业务代码只依赖协议(LLM / Retriever / Reranker / KV),不关心具体实现;
    - 测试时可向 build_container 直接注入假的 LLM / Retriever,无需 monkeypatch;
    - 重模型(BGE-M3 / reranker)与向量库连接只在选择 milvus 后端时才加载,
      开发与 CI 不需要 GPU 与模型文件。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from knowbase.cache.kv import KV, make_kv
from knowbase.core.config import Settings
from knowbase.db.session import init_db, make_engine, make_session_factory
from knowbase.rag.ingestion.pipeline import IngestionService
from knowbase.rag.intent import IntentRouter, StrategySelector
from knowbase.rag.llm import LLM, OpenAICompatLLM
from knowbase.rag.retrieval_service import RetrievalService
from knowbase.rag.retrievers import CrossEncoderReranker, MemoryRetriever, MilvusRetriever, Reranker, Retriever
from knowbase.services.chat_service import ChatService
from knowbase.services.faq_service import FaqMatcher

log = logging.getLogger(__name__)


@dataclass
class Container:
    """应用级依赖集合。随应用生命周期创建一次,通过 FastAPI 的 app.state 共享。"""

    settings: Settings
    engine: Engine
    sf: sessionmaker[Session]
    kv: KV
    retriever: Retriever
    llm: LLM
    faq: FaqMatcher
    ingestion: IngestionService
    chat: ChatService


def build_container(
    settings: Settings, llm: LLM | None = None, retriever: Retriever | None = None, reranker: Reranker | None = None
) -> Container:
    """按配置构建全部依赖。

    Args:
        settings: 应用配置。
        llm / retriever / reranker: 可选的外部注入,主要供测试使用;不传则按配置创建。
    """
    engine = make_engine(settings.database_url)
    init_db(engine)
    sf = make_session_factory(engine)
    kv = make_kv(settings.redis_url)

    if retriever is None:
        if settings.retriever_backend == "milvus":
            retriever = MilvusRetriever(
                settings.milvus_uri,
                settings.milvus_token,
                settings.milvus_db,
                settings.milvus_collection,
                settings.model_dir,
                settings.embedding_model,
            )
            if reranker is None and settings.enable_reranker:
                reranker = CrossEncoderReranker(os.path.join(settings.model_dir, settings.reranker_model))
        else:
            retriever = MemoryRetriever(sf)
    llm = llm or OpenAICompatLLM(
        settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_s, settings.llm_max_retries
    )

    faq = FaqMatcher(settings.faq_threshold, settings.faq_margin)
    retrieval = RetrievalService(
        retriever, reranker, llm, sf, settings.retrieval_k, settings.context_top_m, settings.min_relevance
    )
    chat = ChatService(
        settings,
        sf,
        faq,
        IntentRouter(settings.company_name, settings.intent_model_dir),
        StrategySelector(llm, settings.enable_strategy_llm),
        retrieval,
        llm,
    )
    ingestion = IngestionService(
        sf,
        retriever,
        settings.upload_dir,
        settings.parent_chunk_size,
        settings.child_chunk_size,
        settings.chunk_overlap,
        settings.max_upload_mb,
    )
    return Container(settings, engine, sf, kv, retriever, llm, faq, ingestion, chat)
