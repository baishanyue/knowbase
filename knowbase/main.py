"""应用入口与装配。

运行: uvicorn knowbase.main:app --host 0.0.0.0 --port 8080

设计要点:
- 工厂函数 create_app 便于测试注入依赖;
- CORS 默认关闭,仅按配置白名单开启;
- 请求 ID 贯穿日志与响应头,错误响应结构统一;
- 重资源(模型、数据库连接)在 lifespan 中创建,避免导入期副作用。
"""

from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from knowbase.api import admin, auth, chat, health
from knowbase.container import Container, build_container
from knowbase.core.config import Settings, get_settings
from knowbase.core.logging import request_id_var, setup_logging
from knowbase.core.metrics import metrics

log = logging.getLogger("knowbase")
_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """应用工厂。

    Args:
        settings: 配置;缺省时从环境读取。
        container: 测试时可直接注入已装配好的依赖,跳过 lifespan 中的构建。
    """
    settings = settings or (container.settings if container else get_settings())
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期:启动时创建依赖容器(模型、数据库连接等重资源),避免导入期副作用。"""
        app.state.container = container or build_container(settings)
        log.info("启动完成 env=%s retriever=%s", settings.env, settings.retriever_backend)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if settings.env == "prod" else "/docs",
        redoc_url=None,
    )
    if container is not None:  # 测试场景: 不经过 lifespan 也能直接用
        app.state.container = container

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """请求中间件:注入/回传请求 ID、统一安全响应头、记录访问日志与指标。"""
        # 沿用上游网关传来的请求 ID 便于全链路追踪;没有则自己生成
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.time()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        elapsed = time.time() - t0
        # 指标的 path 用"路由模板"(如 /sessions/{session_id}/history)而非真实路径,
        # 防止 ID 进入标签造成基数爆炸;未匹配到路由的请求统一归为 unmatched。
        route = request.scope.get("route")
        template = getattr(route, "path", "unmatched")
        if not template.startswith(("/health", "/metrics")):
            labels = {"method": request.method, "path": template, "status": str(response.status_code)}
            metrics.inc("knowbase_http_requests_total", labels)
            metrics.observe("knowbase_http_request_seconds", elapsed, {"method": request.method, "path": template})
            logging.getLogger("access").info(
                "%s %s -> %s %.0fms", request.method, request.url.path, response.status_code, elapsed * 1000
            )
        return response

    @app.exception_handler(HTTPException)
    async def http_exc(request: Request, exc: HTTPException):
        """HTTPException -> 统一错误结构 {"error": {code, message, request_id}}。"""
        return JSONResponse(
            {"error": {"code": exc.status_code, "message": exc.detail, "request_id": request_id_var.get()}},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exc(request: Request, exc: RequestValidationError):
        """参数校验失败 -> 422,只返回字段位置与原因,不回显用户提交的值(可能含敏感内容)。"""
        return JSONResponse(
            {
                "error": {
                    "code": 422,
                    "message": "请求参数不合法",
                    "details": [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()],
                    "request_id": request_id_var.get(),
                }
            },
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        """兜底:未预期异常只在服务端记录堆栈,对外只给通用 500 与请求 ID,不泄露内部信息。"""
        log.exception("未处理异常 %s %s", request.method, request.url.path)
        return JSONResponse(
            {"error": {"code": 500, "message": "服务内部错误", "request_id": request_id_var.get()}}, status_code=500
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics(request: Request):
        """Prometheus 抓取端点。

        访问控制:配置了 KNOWBASE_METRICS_TOKEN 时要求 Bearer 令牌匹配;未配置时
        仅非 prod 环境开放,prod 返回 404(不暴露端点存在),避免运行指标被未授权读取。
        """
        token = settings.metrics_token
        if token:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied, f"Bearer {token}"):
                raise HTTPException(401, "未授权")
        elif settings.env == "prod":
            raise HTTPException(404, "Not Found")
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    if os.path.isdir(_FRONTEND):
        app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            """演示用前端首页。"""
            return FileResponse(os.path.join(_FRONTEND, "index.html"))

    return app


def _default_app() -> FastAPI:
    """按环境变量创建默认应用(供 uvicorn knowbase.main:app 使用)。"""
    return create_app()


# 测试设置 KNOWBASE_AUTOSTART=0,避免导入模块时就按默认配置创建应用、连接数据库
app = _default_app() if os.getenv("KNOWBASE_AUTOSTART", "1") == "1" else None
