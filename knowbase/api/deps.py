"""鉴权与权限的 FastAPI 依赖。

分层:
    current_principal  必须登录 -> 返回 Principal
    require_kb_admin   在登录基础上要求知识库管理员及以上
    require_admin      在登录基础上要求系统管理员
    rate_limited       在登录基础上叠加按用户的问答限流

越权访问(401 之后的 403)会写入审计日志 authz.denied,便于发现探测行为。
"""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from knowbase.container import Container
from knowbase.core.security import Principal, decode_access_token
from knowbase.db.models import User
from knowbase.services.audit import audit

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    """取出应用级依赖容器(在 lifespan 或测试中挂到 app.state)。"""
    return request.app.state.container


def principal_from_token(token: str, c: Container) -> Principal:
    """校验令牌并还原身份;HTTP 与 WebSocket 共用。

    三道检查依次为:签名与有效期 -> 是否已登出吊销 -> 账号当前是否可用。
    对外错误信息保持笼统,不区分"过期""被吊销""签名错误",避免给攻击者提供判断依据。
    """
    try:
        claims = decode_access_token(token, c.settings.secret_key)
    except jwt.PyJWTError:
        raise HTTPException(401, "登录已失效,请重新登录") from None
    if c.kv.has_flag(f"revoked:{claims['jti']}"):
        raise HTTPException(401, "登录已失效,请重新登录")
    with c.sf() as db:  # 以数据库为准:停用账号/调岗/改角色即时生效,不必等 token 过期
        user = db.get(User, int(claims["sub"]))
        if user is None or not user.is_active:
            raise HTTPException(401, "账号不可用")
        return Principal(user.id, user.username, user.department.code, user.role)


def current_principal(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer), c: Container = Depends(get_container)
) -> Principal:
    """从 Authorization: Bearer 头解析当前用户;缺失返回 401 并带 WWW-Authenticate 头。"""
    if cred is None:
        raise HTTPException(401, "未登录", headers={"WWW-Authenticate": "Bearer"})
    return principal_from_token(cred.credentials, c)


def require_kb_admin(
    request: Request, p: Principal = Depends(current_principal), c: Container = Depends(get_container)
) -> Principal:
    """要求知识库管理员(kb_admin)或系统管理员(admin)。"""
    if not p.is_kb_admin:
        audit(c, "authz.denied", p, {"path": request.url.path})
        raise HTTPException(403, "需要知识库管理员权限")
    return p


def require_admin(
    request: Request, p: Principal = Depends(current_principal), c: Container = Depends(get_container)
) -> Principal:
    """要求系统管理员(admin)。"""
    if not p.is_admin:
        audit(c, "authz.denied", p, {"path": request.url.path})
        raise HTTPException(403, "需要系统管理员权限")
    return p


def rate_limited(p: Principal = Depends(current_principal), c: Container = Depends(get_container)) -> Principal:
    """按用户做每分钟问答限流,超限返回 429。限流键含 user_id,不同用户互不影响。"""
    if c.kv.incr_window(f"rl:chat:{p.user_id}", 60) > c.settings.rate_limit_per_minute:
        raise HTTPException(429, "请求过于频繁,请稍后再试")
    return p
