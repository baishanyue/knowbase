"""认证与授权的基础能力:密码哈希、JWT、身份主体(Principal)、权限标签。

权限模型(整个系统的安全基石):
    - 用户属于且仅属于一个部门,身份标签固定为 {"public", "dept:<部门>"};
    - 文档与 FAQ 带 acl_tags,与用户标签"有交集"即可见;
    - 过滤在检索层强制执行,标签只来自服务端签发的令牌,绝不信任客户端传参;
    - 取回正文时还会按关系库再校验一次(纵深防御,见 retrieval_service)。

选型说明:
    - 密码用 PBKDF2-HMAC-SHA256(标准库即可,无额外依赖,迭代数可配置);
    - 令牌用 HS256 JWT,带 jti 以支持登出吊销;多服务、跨团队场景可升级为 RS256 + 公钥分发。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
import uuid
from dataclasses import dataclass

import jwt

ROLES = ("employee", "kb_admin", "admin")  # 普通员工 / 知识库管理员 / 系统管理员,权限依次包含

# 权限标签白名单。标签会被拼接进 Milvus 过滤表达式,必须严格限制字符集以杜绝注入。
_TAG_RE = re.compile(r"^(public|dept:[a-z0-9_\-]{1,32})$")


def validate_acl_tags(tags: list[str]) -> list[str]:
    """校验并规范化权限标签:去空白、去重、排序。

    Raises:
        ValueError: 标签为空,或含不符合白名单格式的项。
    """
    cleaned = sorted({t.strip() for t in tags if t and t.strip()})
    if not cleaned:
        raise ValueError("acl_tags 不能为空(至少 public 或 dept:<code>)")
    bad = [t for t in cleaned if not _TAG_RE.match(t)]
    if bad:
        raise ValueError(f"非法的 acl 标签: {bad}")
    return cleaned


def hash_password(password: str, iterations: int = 600_000) -> str:
    """生成密码哈希,格式 `pbkdf2_sha256$迭代数$盐(hex)$哈希(hex)`。

    迭代数写进存储串,日后调高迭代数时旧密码仍可校验,登录成功后可择机升级。
    每个密码独立随机盐,防止彩虹表与"相同密码相同哈希"。
    """
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码。格式异常或算法不符一律返回 False(不抛异常,避免泄露内部细节)。"""
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)  # 恒定时间比较,防计时攻击
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class Principal:
    """已认证的调用者身份。不可变,可安全地在各层之间传递。"""

    user_id: int
    username: str
    department: str
    role: str

    @property
    def acl_tags(self) -> list[str]:
        """该用户可见的权限标签集合;检索过滤与二次校验都以它为准。"""
        return ["public", f"dept:{self.department}"]

    @property
    def is_kb_admin(self) -> bool:
        return self.role in ("kb_admin", "admin")

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def can_access(self, acl_tags: list[str] | None) -> bool:
        """资源标签与用户标签有交集才可访问;资源无标签视为不可访问(默认拒绝)。"""
        return bool(set(acl_tags or []) & set(self.acl_tags))


def create_access_token(principal: Principal, secret: str, expire_minutes: int) -> tuple[str, str]:
    """签发访问令牌。

    Returns:
        (令牌, jti)。jti 用于登出时加入吊销名单。
    """
    jti = uuid.uuid4().hex
    now = int(time.time())
    payload = {
        "sub": str(principal.user_id),
        "usr": principal.username,
        "dep": principal.department,
        "role": principal.role,
        "jti": jti,
        "iat": now,
        "exp": now + expire_minutes * 60,
    }
    return jwt.encode(payload, secret, algorithm="HS256"), jti


def decode_access_token(token: str, secret: str) -> dict:
    """解码并校验令牌。

    显式限定算法列表(防止 alg=none 与算法混淆攻击),并强制要求 exp / sub / jti 存在。

    Raises:
        jwt.PyJWTError: 签名错误、过期或缺少必要声明。
    """
    return jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub", "jti"]})


def principal_from_claims(claims: dict) -> Principal:
    """由令牌声明还原身份。账号是否仍有效由调用方(deps)再查库确认。"""
    return Principal(
        user_id=int(claims["sub"]),
        username=claims["usr"],
        department=claims["dep"],
        role=claims["role"],
    )
