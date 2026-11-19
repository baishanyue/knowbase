"""安全审计:登录、权限拒绝、FAQ/文档/用户变更、数据导出。

与 chat_turns 的分工:问答本身的留痕在 chat_turns(路由/引用/耗时/模型);
审计表记录"谁在什么时候对系统做了什么",二者合起来满足合规追溯要求。

约定:detail 只放定位问题所需的最少信息(ID、数量、动作类型),
严禁写入密码、令牌或问题原文。
"""

from __future__ import annotations

import logging

from knowbase.core.logging import request_id_var
from knowbase.core.security import Principal
from knowbase.db.models import AuditLog
from knowbase.db.session import session_scope

log = logging.getLogger(__name__)


def audit(
    container,
    action: str,
    principal: Principal | None = None,
    detail: dict | None = None,
    ip: str | None = None,
    username: str | None = None,
) -> None:
    """写一条审计记录。

    Args:
        container: 依赖容器(取数据库会话工厂)。
        action: 动作标识,形如 `模块.动作`,如 faq.create、auth.login_failed。
        principal: 操作者;登录失败等未认证场景可为空,此时用 username 记录尝试的账号。
        detail: 补充信息(不含敏感内容)。
        ip: 客户端 IP。
    """
    try:
        with session_scope(container.sf) as db:
            db.add(
                AuditLog(
                    request_id=request_id_var.get(),
                    user_id=principal.user_id if principal else None,
                    username=principal.username if principal else username,
                    action=action,
                    detail=detail or {},
                    ip=ip,
                )
            )
    except Exception:  # noqa: BLE001  审计失败不应影响主流程,但必须留下告警
        log.exception("写审计日志失败 action=%s", action)
