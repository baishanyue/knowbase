"""接口层的请求/响应数据模型(Pydantic)。

约定:
    - 入参模型(*In)负责边界校验:长度上限、字符白名单,把"脏数据"挡在业务层之外;
    - 出参模型(*Out)只暴露必要字段,避免把内部字段(如密码哈希、ACL 内部结构)带出去;
    - 带 `from_attributes=True` 的模型可直接由 ORM 对象构造。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# 分类/部门编码的统一白名单:小写字母、数字、下划线、连字符,最长 32。
# 这些值会参与检索过滤与权限标签,必须限制字符集,杜绝注入。
_CODE_PATTERN = r"^[a-z0-9_\-]{1,32}$"


class LoginIn(BaseModel):
    """登录请求。长度上限用于防止超大密码拖慢哈希计算(DoS)。"""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenOut(BaseModel):
    """登录成功返回的访问令牌。"""

    access_token: str
    token_type: str = "bearer"  # noqa: S105 协议常量,非凭据
    expires_in: int  # 有效期(秒)


class MeOut(BaseModel):
    """当前登录用户信息。"""

    user_id: int
    username: str
    department: str
    role: str


class ChatIn(BaseModel):
    """问答请求。"""

    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=36, description="续接会话;缺省则新建")
    category: str | None = Field(default=None, max_length=32, description="可选:限定知识分类,如 hr / it")


class ChatOut(BaseModel):
    """一次性问答结果。"""

    session_id: str
    answer: str
    route: str  # faq / rag / refused / chitchat / general
    strategy: str | None = None  # RAG 时采用的检索策略
    sources: list[dict] = []  # 引用来源,含文档标题/章节/页码/版本与是否被答案引用
    turn_id: int  # 用于提交点赞/点踩反馈
    latency_ms: int
    rewritten_query: str | None = Field(default=None, description="追问被补全后的检索问题;未改写为空")


class TurnOut(BaseModel):
    """历史记录中的一轮问答。"""

    model_config = ConfigDict(from_attributes=True)
    id: int
    question: str
    answer: str
    route: str
    citations: list = []
    created_at: datetime


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    created_at: datetime


class FeedbackIn(BaseModel):
    """对某一轮回答的评价:1=有帮助,-1=无帮助。点踩会进入知识缺口分析。"""

    turn_id: int
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=500)


class FaqIn(BaseModel):
    """FAQ 维护入参。acl_tags 决定哪些部门可见,默认公开。"""

    question: str = Field(min_length=2, max_length=500)
    answer: str = Field(min_length=1, max_length=5000)
    category: str = Field(default="general", pattern=_CODE_PATTERN)
    acl_tags: list[str] = ["public"]
    status: Literal["active", "disabled"] = "active"


class FaqOut(FaqIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    hit_count: int  # 被命中次数,运营据此判断 FAQ 价值
    updated_at: datetime


class DocumentOut(BaseModel):
    """文档及其入库状态。status: pending(待/处理中)/ indexed(已入库)/ failed / archived(旧版本归档)。"""

    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    filename: str
    category: str
    version: int
    acl_tags: list[str]
    status: str
    error: str | None  # 入库失败原因(已脱敏,不含堆栈)
    chunk_count: int
    created_at: datetime


class UserIn(BaseModel):
    """创建用户(仅系统管理员)。密码至少 8 位;用户名限制字符集以便审计检索。"""

    username: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    password: str = Field(min_length=8, max_length=128)
    display_name: str = ""
    department: str = Field(pattern=_CODE_PATTERN)
    role: Literal["employee", "kb_admin", "admin"] = "employee"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    display_name: str
    role: str
    is_active: bool
