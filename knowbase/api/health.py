"""健康探针(不需要登录,不返回敏感信息)。

    /health/live   存活:进程能响应即可,供 livenessProbe;依赖故障时不应让它失败,否则会被无谓重启。
    /health/ready  就绪:数据库、缓存、检索器都可用才返回 200,否则 503,供 readinessProbe 摘除流量。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text

from knowbase.api.deps import get_container
from knowbase.container import Container

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live():
    """存活探针。"""
    return {"status": "ok"}


@router.get("/health/ready")
def ready(response: Response, c: Container = Depends(get_container)):
    """就绪探针:逐项检查依赖,任一失败则整体降级为 503,并在响应体中给出各项结果便于排障。"""
    checks: dict[str, bool] = {}
    try:
        with c.sf() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:  # noqa: BLE001
        checks["database"] = False
    checks["cache"] = c.kv.ping()
    checks["retriever"] = c.retriever.ping()
    ok = all(checks.values())
    if not ok:
        response.status_code = 503
    return {"status": "ready" if ok else "degraded", "checks": checks}
