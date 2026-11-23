"""问答接口: 普通 JSON / SSE 流式 / WebSocket 流式,三者共用 ChatService.stream。

设计要点:
- 同步的 LLM 流在线程池中迭代,不阻塞事件循环;
- WebSocket 连接可复用多轮;
- 会话必须属于当前用户,他人会话一律按不存在处理;
- 统一 /api/v1 前缀、统一错误结构,带限流与鉴权。
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import iterate_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from knowbase.api.deps import current_principal, get_container, principal_from_token, rate_limited
from knowbase.api.schemas import ChatIn, ChatOut, FeedbackIn, SessionOut, TurnOut
from knowbase.container import Container
from knowbase.core.logging import request_id_var
from knowbase.core.security import Principal
from knowbase.db.models import ChatSession, ChatTurn, Feedback
from knowbase.db.session import session_scope
from knowbase.services.chat_service import SessionNotFound

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _sse(event: dict) -> str:
    """按 SSE 协议编码一个事件。ensure_ascii=False 保持中文可读;每个事件以空行结束。"""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, p: Principal = Depends(rate_limited), c: Container = Depends(get_container)):
    """一次性问答:返回完整答案与引用。LLM 失败时返回 502(而不是把兜底话术当作成功答案)。"""
    try:
        r = c.chat.answer(p, body.question, body.session_id, body.category, request_id_var.get())
    except SessionNotFound:
        raise HTTPException(404, "会话不存在") from None
    if "error" in r:
        raise HTTPException(502, "问答服务暂时不可用,请稍后重试")
    return ChatOut(**{k: r.get(k) for k in ChatOut.model_fields})


@router.post("/chat/stream")
def chat_stream(body: ChatIn, p: Principal = Depends(rate_limited), c: Container = Depends(get_container)):
    """SSE: event: meta | token | sources | end | error。前端用 fetch + ReadableStream 解析。"""
    if body.session_id:  # 先校验会话归属,避免响应开始后才报错
        try:
            c.chat.get_history(p, body.session_id, limit=1)
        except SessionNotFound:
            raise HTTPException(404, "会话不存在") from None
    # 请求 ID 必须在生成器外取出:生成器在线程池里迭代,那里读不到本请求的 ContextVar
    rid = request_id_var.get()

    def gen() -> Iterator[str]:
        try:
            for ev in c.chat.stream(p, body.question, body.session_id, body.category, rid):
                yield _sse(ev)
        except Exception:  # noqa: BLE001
            yield _sse({"type": "error", "message": "服务内部错误"})

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.websocket("/chat/ws")
async def chat_ws(ws: WebSocket, token: str = Query(...)):
    """WebSocket 流式问答。

    浏览器的 WebSocket 无法自定义请求头,所以令牌通过查询参数传递(注意:反向代理日志应脱敏该参数)。
    鉴权失败以自定义关闭码 4401 断开;同一连接可连续多轮,限流与 HTTP 接口共用同一个计数。
    """
    c: Container = ws.app.state.container
    try:
        p = principal_from_token(token, c)
    except HTTPException:
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        while True:
            try:
                data = json.loads(await ws.receive_text())
                body = ChatIn(**data)
            except (ValueError, TypeError):
                await ws.send_json({"type": "error", "message": "消息格式错误"})
                continue
            if c.kv.incr_window(f"rl:chat:{p.user_id}", 60) > c.settings.rate_limit_per_minute:
                await ws.send_json({"type": "error", "message": "请求过于频繁,请稍后再试"})
                continue
            try:
                async for ev in iterate_in_threadpool(c.chat.stream(p, body.question, body.session_id, body.category)):
                    await ws.send_json(ev)
            except SessionNotFound:
                await ws.send_json({"type": "error", "message": "会话不存在"})
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------- 会话
@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(p: Principal = Depends(current_principal), c: Container = Depends(get_container)):
    """新建空会话。"""
    return c.chat.create_session(p)


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(p: Principal = Depends(current_principal), c: Container = Depends(get_container)):
    """当前用户最近 50 个会话(只含本人)。"""
    with c.sf() as db:
        return list(
            db.scalars(
                select(ChatSession).where(ChatSession.user_id == p.user_id).order_by(ChatSession.created_at.desc()).limit(50)
            )
        )


@router.get("/sessions/{session_id}/history", response_model=list[TurnOut])
def history(session_id: str, p: Principal = Depends(current_principal), c: Container = Depends(get_container)):
    """某会话的历史轮次;非本人会话与不存在一视同仁返回 404。"""
    try:
        return c.chat.get_history(p, session_id)
    except SessionNotFound:
        raise HTTPException(404, "会话不存在") from None


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, p: Principal = Depends(current_principal), c: Container = Depends(get_container)):
    """删除本人会话及其全部问答记录。"""
    try:
        c.chat.delete_session(p, session_id)
    except SessionNotFound:
        raise HTTPException(404, "会话不存在") from None


# ---------------------------------------------------------------- 反馈
@router.post("/feedback", status_code=204)
def feedback(body: FeedbackIn, p: Principal = Depends(current_principal), c: Container = Depends(get_container)):
    """对本人的某轮回答点赞/点踩;重复提交视为修改。点踩会进入知识缺口分析。"""
    with session_scope(c.sf) as db:
        turn = db.get(ChatTurn, body.turn_id)
        if turn is None or turn.user_id != p.user_id:
            raise HTTPException(404, "记录不存在")
        fb = db.scalars(select(Feedback).where(Feedback.turn_id == turn.id, Feedback.user_id == p.user_id)).first()
        if fb:
            fb.rating, fb.comment = body.rating, body.comment
        else:
            db.add(Feedback(turn_id=turn.id, user_id=p.user_id, rating=body.rating, comment=body.comment))
