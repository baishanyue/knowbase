"""关系库数据模型(SQLAlchemy 2.0 声明式)。

分组:
    身份:Department、User
    知识:FaqEntry(高频问答)、Document(文档与版本)、DocParent(父块正文)
    交互:ChatSession、ChatTurn(问答轮次,兼作审计留痕)、Feedback(点赞点踩)
    合规:AuditLog

设计要点:
    - 时间统一存 UTC 的 naive datetime(见 utcnow),展示层再转换时区,避免跨库时区歧义;
    - acl_tags 用 JSON 列保存标签数组,读取时与用户标签求交集;
    - 父块正文放关系库、子块向量放向量库:更新文档只需重建索引,正文不在向量库冗余存储;
    - 常用查询路径(按会话取历史、按路由统计、按时间过滤)都建了索引。
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """当前 UTC 时间(去掉时区信息,与各数据库的 DATETIME 类型兼容)。"""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """所有模型的基类。"""


class Department(Base):
    """部门。code 同时是权限标签的组成部分(dept:<code>),创建后不应随意改名。"""

    __tablename__ = "departments"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)  # hr / it / finance ...
    name: Mapped[str] = mapped_column(String(64))


class User(Base):
    """账号。只存密码哈希,绝不存明文。"""

    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    display_name: Mapped[str] = mapped_column(String(64), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id"))
    role: Mapped[str] = mapped_column(String(16), default="employee")  # employee | kb_admin | admin
    is_active: Mapped[bool] = mapped_column(default=True)  # 停用即刻生效,令牌随之失效
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    department: Mapped[Department] = relationship()


class FaqEntry(Base):
    """高频问题库。运营在后台维护;命中则不走 RAG,成本低、答案可控。"""

    __tablename__ = "faq_entries"
    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(String(500))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32), default="general")  # hr / it / admin / finance
    acl_tags: Mapped[list] = mapped_column(JSON, default=lambda: ["public"])
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | disabled
    hit_count: Mapped[int] = mapped_column(Integer, default=0)  # 命中次数,运营评估 FAQ 价值
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Document(Base):
    """知识文档(制度、手册……)。

    版本规则:同标题重新上传 -> version + 1;新版本入库成功后,旧版本归档(archived)并移出索引。
    checksum 用于识别"内容完全相同的重复上传",避免无谓的重复入库。
    """

    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(32), default="general")
    version: Mapped[int] = mapped_column(Integer, default=1)
    checksum: Mapped[str] = mapped_column(String(64))  # 文件内容 SHA-256
    acl_tags: Mapped[list] = mapped_column(JSON, default=lambda: ["public"])
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | indexed | failed | archived
    error: Mapped[str | None] = mapped_column(Text, nullable=True)  # 失败原因(不含堆栈)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    storage_path: Mapped[str] = mapped_column(String(500), default="")
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class DocParent(Base):
    """父块:送给 LLM 的上下文粒度。正文存放在关系库,向量库只保存子块。

    删除文档时级联删除其父块;取回时会再次校验所属文档的状态与权限。
    """

    __tablename__ = "doc_parents"
    parent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(String(300), nullable=True)  # 章节路径,用于引用展示
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 页码(PDF 等分页文档)
    seq: Mapped[int] = mapped_column(Integer, default=0)  # 在文档内的顺序


class ChatSession(Base):
    """会话。必须归属某个用户,所有访问都要校验归属。"""

    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ChatTurn(Base):
    """一轮问答。既是会话历史,也是审计留痕(路由、引用、耗时、模型、请求 ID)。

    route 取值:chitchat | faq | rag | refused | general | error。
    其中 refused 就是"知识缺口"的数据来源;error 轮次不参与后续对话上下文。
    """

    __tablename__ = "chat_turns"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    question: Mapped[str] = mapped_column(Text)  # 已脱敏的问题文本
    answer: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(16), index=True)
    strategy: Mapped[str | None] = mapped_column(String(16), nullable=True)  # 检索策略
    citations: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_id: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    # 取会话历史时按 (session_id, created_at) 排序,联合索引覆盖该路径
    __table_args__ = (Index("ix_turn_session_created", "session_id", "created_at"),)


class Feedback(Base):
    """对某轮回答的评价。同一用户对同一轮只保留一条(可修改)。"""

    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("chat_turns.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    rating: Mapped[int] = mapped_column(Integer)  # 1 = 有帮助, -1 = 无帮助
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("turn_id", "user_id", name="uq_feedback_turn_user"),)


class AuditLog(Base):
    """安全相关操作审计:登录、FAQ/文档变更、用户管理、越权尝试、数据导出等。

    只追加、不修改;detail 中不得写入密码、令牌或完整问题原文。
    """

    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    request_id: Mapped[str] = mapped_column(String(32), default="")
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)  # 登录失败时可能无用户
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(48), index=True)  # 如 auth.login / faq.create / gaps.export
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
