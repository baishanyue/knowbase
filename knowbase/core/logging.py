"""日志初始化:统一格式、请求 ID 贯穿、可选 JSON 输出。

为什么需要 request_id:
    一次问答会经过中间件、鉴权、检索、LLM、落库多个环节。把同一个请求 ID 写进每一行日志
    (并通过响应头 X-Request-ID 返回给调用方),排障时按 ID 就能把整条链路串起来。

为什么用 ContextVar:
    请求在线程池/协程间切换时,普通全局变量会串号;ContextVar 能保证每个请求读到自己的值。

安全约束:
    日志中不得输出密钥、令牌、完整问题原文中的个人信息;业务代码只记录数量与 ID。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys

# 当前请求 ID;在请求上下文之外(如启动阶段)显示为 "-"。
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    """给每条日志记录附上当前请求 ID,供格式化器引用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    """单行 JSON 格式,便于 ELK / Loki 等日志平台按字段检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)  # ensure_ascii=False:中文日志保持可读


def setup_logging(level: str = "INFO", as_json: bool = False) -> None:
    """初始化根日志器。重复调用是安全的(会替换而不是叠加 handler)。

    Args:
        level: 日志级别名称,如 INFO / DEBUG。
        as_json: True 输出 JSON(生产建议);False 输出便于本地阅读的文本。
    """
    handler = logging.StreamHandler(sys.stdout)  # 输出到 stdout,由容器运行时负责采集
    handler.addFilter(_RequestIdFilter())
    if as_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]  # 替换而非追加,避免热重载或测试中重复初始化导致日志重复
    root.setLevel(level.upper())
    # 第三方库的 INFO 日志(每次 HTTP 请求一行)噪声很大,统一降到 WARNING
    for noisy in ("httpx", "httpcore", "urllib3", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
