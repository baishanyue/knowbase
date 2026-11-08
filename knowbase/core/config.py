"""集中配置:全部来自环境变量或 .env(前缀 KNOWBASE_),代码与仓库中不包含任何密钥。

设计原则:
    - 所有可调参数都在这里声明并带默认值,业务代码不直接读取环境变量;
    - 默认值面向"本地开发可直接跑起来",生产环境(KNOWBASE_ENV=prod)由 `_check_prod`
      强制校验关键项,配置不安全时拒绝启动,而不是带着隐患上线;
    - 阈值类参数(min_relevance、faq_threshold 等)需要结合企业真实文档用评估脚本标定,
      默认值只是起点。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 仅用于本地开发的占位密钥。prod 启动校验会拒绝它,防止误用默认值签发令牌。
_DEV_SECRET = "dev-only-secret-change-me"  # noqa: S105


class Settings(BaseSettings):
    """应用配置。字段名对应环境变量 KNOWBASE_<大写字段名>。"""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="KNOWBASE_", extra="ignore")

    # ---- 基础 ----
    app_name: str = "知底 KnowBase · 企业知识库智能问答平台"
    company_name: str = "示例科技"  # 出现在助手自我介绍与提示词中
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False  # 生产建议开启,便于日志平台按字段检索
    cors_origins: list[str] = []  # 默认不开放跨域;前后端分离部署时填写前端域名白名单

    # ---- 安全 ----
    secret_key: str = _DEV_SECRET  # JWT 签名密钥,prod 要求 >=32 位随机串
    access_token_expire_minutes: int = 480
    password_hash_iterations: int = 600_000  # PBKDF2 迭代数;测试环境可调小以加速
    rate_limit_per_minute: int = 30  # 每用户每分钟问答次数上限
    mask_pii: bool = True  # 问题进入系统时脱敏手机号/身份证/银行卡/邮箱,再检索、调 LLM、落库
    metrics_token: str | None = None  # /metrics 的 Bearer 令牌;prod 未配置则该端点不可用

    # ---- 存储 ----
    database_url: str = "sqlite:///./knowbase.db"
    redis_url: str | None = None  # 不配置则退化为进程内存(仅单实例/开发)
    upload_dir: str = "./storage/uploads"
    max_upload_mb: int = 30

    # ---- 向量库 / 检索 ----
    retriever_backend: Literal["milvus", "memory"] = "memory"  # memory 仅用于开发/CI
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_db: str = "knowbase"
    milvus_collection: str = "knowbase_chunks"
    model_dir: str = "./models"
    embedding_model: str = "bge-m3"
    reranker_model: str = "bge-reranker-large"
    enable_reranker: bool = True
    parent_chunk_size: int = 1200  # 父块:送给 LLM 的上下文粒度
    child_chunk_size: int = 300  # 子块:参与向量检索的粒度,小块召回更准
    chunk_overlap: int = 50
    retrieval_k: int = 8  # 每路召回的子块数
    context_top_m: int = 3  # 最终送给 LLM 的父块数
    min_relevance: float = 0.30  # 低于该相关度(0~1)视为"知识库里没有",触发拒答

    # ---- FAQ(高频问题快速通道)----
    faq_threshold: float = 0.60  # 归一化 BM25 置信度阈值
    faq_margin: float = 0.10  # 第一名需领先第二名的最小差距,避免两个相近问题互相抢答

    # ---- LLM(任意 OpenAI 兼容接口:公有云网关 / vLLM / Ollama ...)----
    llm_base_url: str = ""  # 留空则使用 SDK 默认地址
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    enable_strategy_llm: bool = True  # 关闭则一律"直接检索",省一次 LLM 调用
    enable_query_rewrite: bool = True  # 多轮追问时先把问题补全再检索
    allow_general_knowledge: bool = False  # 企业场景默认不让模型脱离知识库回答
    history_turns: int = 5  # 带入提示词与追问改写的历史轮数

    # ---- 意图分类(可选的 BERT 模型目录,不配置则只用规则)----
    intent_model_dir: str | None = None

    # ---- 兜底 ----
    helpdesk_contact: str = "HR 热线 8001 / IT 服务台 8000"  # 拒答与降级时展示的人工渠道

    @model_validator(mode="after")
    def _check_prod(self) -> Settings:
        """生产环境强校验:任何一项不满足都直接拒绝启动。"""
        if self.env == "prod":
            if self.secret_key == _DEV_SECRET or len(self.secret_key) < 32:
                raise ValueError("生产环境必须设置长度>=32 的 KNOWBASE_SECRET_KEY")
            if self.database_url.startswith("sqlite"):
                raise ValueError("生产环境请使用 MySQL/PostgreSQL,不要用 SQLite")
            if self.retriever_backend == "memory":
                raise ValueError("生产环境请使用 KNOWBASE_RETRIEVER_BACKEND=milvus")
            if not self.llm_api_key or not self.llm_model:
                raise ValueError("生产环境必须设置 KNOWBASE_LLM_API_KEY 与 KNOWBASE_LLM_MODEL")
        return self


@lru_cache
def get_settings() -> Settings:
    """进程内单例配置;测试可直接构造 Settings 绕过缓存。"""
    return Settings()
