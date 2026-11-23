"""登录/登出/当前用户。

企业接入 SSO(企业微信/钉钉/飞书/OIDC)时,只需新增一个 /auth/sso/callback 路由:校验第三方身份后按同样方式签发本系统 JWT,
其余权限、审计、ACL 逻辑完全不变。
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from knowbase.api.deps import current_principal, get_container
from knowbase.api.schemas import LoginIn, MeOut, TokenOut
from knowbase.container import Container
from knowbase.core.security import Principal, create_access_token, decode_access_token, verify_password
from knowbase.db.models import User
from knowbase.services.audit import audit

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)
# 用户不存在时仍对这个假哈希做一次校验,使"用户不存在"与"密码错误"耗时相近,防止通过响应时间枚举用户名
_DUMMY_HASH = "pbkdf2_sha256$1000$" + "00" * 16 + "$" + "00" * 32


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, request: Request, c: Container = Depends(get_container)):
    """账号密码登录,成功返回 JWT。

    防爆破:同一用户名 5 分钟内最多尝试 10 次,超过即锁定并记审计(按用户名而非 IP,
    因为企业内网常经同一出口 IP)。失败原因统一为"用户名或密码错误",不泄露账号是否存在。
    """
    ip = request.client.host if request.client else None
    if c.kv.incr_window(f"rl:login:{body.username.lower()}", 300) > 10:
        audit(c, "auth.locked", ip=ip, username=body.username)
        raise HTTPException(429, "登录尝试过多,请 5 分钟后再试")
    with c.sf() as db:
        user = db.scalars(select(User).where(User.username == body.username)).first()
        ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
        if not (user and ok and user.is_active):
            audit(c, "auth.login_failed", ip=ip, username=body.username)
            raise HTTPException(401, "用户名或密码错误")
        principal = Principal(user.id, user.username, user.department.code, user.role)
    token, _ = create_access_token(principal, c.settings.secret_key, c.settings.access_token_expire_minutes)
    audit(c, "auth.login", principal, ip=ip)
    return TokenOut(access_token=token, expires_in=c.settings.access_token_expire_minutes * 60)


@router.post("/logout", status_code=204)
def logout(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
    p: Principal = Depends(current_principal),
    c: Container = Depends(get_container),
):
    """登出:把当前令牌的 jti 加入吊销名单,有效期与令牌剩余寿命一致(过期后自动清理)。"""
    claims = decode_access_token(cred.credentials, c.settings.secret_key)
    c.kv.set_flag(f"revoked:{claims['jti']}", max(int(claims["exp"] - time.time()), 1))
    audit(c, "auth.logout", p)


@router.get("/me", response_model=MeOut)
def me(p: Principal = Depends(current_principal)):
    """返回当前登录用户的基本信息(前端据此展示部门与角色)。"""
    return MeOut(user_id=p.user_id, username=p.username, department=p.department, role=p.role)
