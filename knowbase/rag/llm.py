"""LLM 客户端(OpenAI 兼容协议,可对接公有云网关、vLLM、Ollama 等)。

设计要点:
    - 上层只依赖 LLM 协议(complete / stream),换模型供应商或在测试中注入假实现都不用改业务代码;
    - 统一封装超时与重试(重试由 SDK 完成);
    - 失败一律抛出 LLMError,由上层决定降级策略,绝不把异常文本当作答案返回给员工;
    - 日志只记录错误摘要,不记录提示词内容(其中可能含制度原文与员工提问)。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """LLM 调用失败(超时、鉴权失败、服务不可用等)。上层据此降级或返回 502。"""


class LLM(Protocol):
    """LLM 协议。temperature 默认取低值:企业问答要求答案稳定、贴合资料,而非发散。"""

    model: str

    def complete(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 600) -> str:
        """一次性返回完整文本(用于策略选择、改写等短任务)。失败抛 LLMError。"""
        ...

    def stream(self, messages: list[dict], temperature: float = 0.1) -> Iterator[str]:
        """流式返回文本片段(用于生成最终答案)。失败抛 LLMError。"""
        ...


class OpenAICompatLLM:
    """基于 openai SDK 的实现,兼容任何 OpenAI 协议的服务端。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float, max_retries: int) -> None:
        from openai import OpenAI  # 延迟导入:测试与不调用 LLM 的场景无需加载 SDK

        self.model = model
        # 私有化部署(vLLM/Ollama)通常不校验密钥,但 SDK 要求非空,故用占位值 "EMPTY"
        self._client = OpenAI(base_url=base_url or None, api_key=api_key or "EMPTY", timeout=timeout_s, max_retries=max_retries)

    def complete(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 600) -> str:
        try:
            r = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return (r.choices[0].message.content or "").strip() if r.choices else ""
        except Exception as e:  # noqa: BLE001
            log.error("LLM complete 失败: %s", e)
            raise LLMError(str(e)) from e

    def stream(self, messages: list[dict], temperature: float = 0.1) -> Iterator[str]:
        try:
            resp = self._client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, stream=True)
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:  # noqa: BLE001
            log.error("LLM stream 失败: %s", e)
            raise LLMError(str(e)) from e
