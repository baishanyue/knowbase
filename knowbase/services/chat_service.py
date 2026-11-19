"""问答编排服务:把"一个问题"变成"一次可追溯的回答"。

处理链路(按成本由低到高,命中即返回):
    脱敏 -> 寒暄规则 -> FAQ(BM25) -> [追问改写 -> 检索策略 -> 混合检索 -> 重排 -> 阈值]
         -> 严格依据资料的 LLM 生成(带 [n] 引用) -> 拒答/人工兜底 -> 落库留痕

设计要点:
    - 事件化流式输出(meta / token / sources / end / error),SSE 与 WebSocket 共用同一个生成器;
    - 检索不到可靠资料时明确拒答并给出人工渠道,route=refused 把"答不出"沉淀为知识缺口数据;
    - LLM 异常不会被当作答案写进历史,错误轮次(route=error)也不参与后续上下文;
    - 数据库会话只在"读历史/FAQ"与"落库"两个短阶段持有,流式生成期间不占用连接,
      避免慢 LLM 把连接池耗尽;
    - 问题在入口处脱敏,之后的检索、LLM 调用、落库、日志看到的都是脱敏文本。
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from knowbase.core.config import Settings
from knowbase.core.metrics import metrics
from knowbase.core.pii import mask_pii
from knowbase.core.security import Principal
from knowbase.db.models import ChatSession, ChatTurn
from knowbase.db.session import session_scope
from knowbase.rag import prompts
from knowbase.rag.intent import Intent, IntentRouter, StrategySelector
from knowbase.rag.llm import LLM, LLMError
from knowbase.rag.retrieval_service import Context, RetrievalService
from knowbase.rag.rewriter import QueryRewriter
from knowbase.services.faq_service import FaqMatcher

log = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")  # 提取答案中的 [n] 引用编号
_NO_INFO = "知识库中暂无相关信息"  # 与系统提示词约定的"资料不足"固定话术,模型输出以它开头即视为拒答
_CAT_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")  # 分类参数白名单,防止拼接进检索过滤表达式


class SessionNotFound(Exception):
    """会话不存在,或不属于当前用户。两种情况对外一视同仁,避免泄露会话 ID 是否存在。"""


@dataclass
class _Turn:
    """一次问答的上下文,在各内部方法间传递,避免长参数列表。"""

    principal: Principal
    sid: str
    question: str  # 已脱敏
    t0: float
    request_id: str
    strategy: str | None = None
    rewritten: str | None = None  # 追问被改写时的检索用问题;未改写为 None

    def elapsed_ms(self) -> int:
        return int((time.time() - self.t0) * 1000)


class ChatService:
    """问答编排。无状态,可被多个请求并发使用;所有依赖通过构造函数注入,便于测试替换。"""

    def __init__(
        self,
        settings: Settings,
        sf: sessionmaker[Session],
        faq: FaqMatcher,
        intent: IntentRouter,
        strategy: StrategySelector,
        retrieval: RetrievalService,
        llm: LLM,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.s, self._sf = settings, sf
        self.faq, self.intent, self.strategy, self.retrieval, self.llm = faq, intent, strategy, retrieval, llm
        # 未显式注入时按配置构造,保持与旧调用方兼容
        self.rewriter = rewriter or QueryRewriter(llm, settings.enable_query_rewrite)

    # ---------------------------------------------------------------- 会话
    def create_session(self, principal: Principal, title: str = "新会话") -> ChatSession:
        """为当前用户新建会话。标题截断到 200 字以内,与表结构一致。"""
        with session_scope(self._sf) as db:
            s = ChatSession(id=str(uuid.uuid4()), user_id=principal.user_id, title=title[:200])
            db.add(s)
        return s

    def _get_owned_session(self, db: Session, principal: Principal, session_id: str) -> ChatSession:
        """取会话并校验归属。越权访问与不存在返回同样的异常,防止枚举他人会话。"""
        s = db.get(ChatSession, session_id)
        if s is None or s.user_id != principal.user_id:
            raise SessionNotFound(session_id)
        return s

    def get_history(self, principal: Principal, session_id: str, limit: int | None = None) -> list[ChatTurn]:
        """返回会话最近 limit 轮(默认 50),按时间正序。"""
        with session_scope(self._sf) as db:
            self._get_owned_session(db, principal, session_id)
            q = select(ChatTurn).where(ChatTurn.session_id == session_id).order_by(ChatTurn.id.desc()).limit(limit or 50)
            return list(reversed(db.scalars(q).all()))

    def delete_session(self, principal: Principal, session_id: str) -> None:
        """删除会话及其全部轮次。仅限本人。"""
        with session_scope(self._sf) as db:
            s = self._get_owned_session(db, principal, session_id)
            for t in db.scalars(select(ChatTurn).where(ChatTurn.session_id == s.id)):
                db.delete(t)
            db.delete(s)

    # ---------------------------------------------------------------- 主流程
    def stream(
        self,
        principal: Principal,
        question: str,
        session_id: str | None = None,
        category: str | None = None,
        request_id: str = "",
    ) -> Iterator[dict]:
        """流式问答,逐个产出事件字典。

        事件类型:
            meta     会话 ID(最先产出,便于前端立即绑定会话)
            token    答案文本片段
            sources  引用来源列表(文档/FAQ 的标题、章节、页码、版本、是否被答案引用)
            end      结束,含 turn_id / route / strategy / latency_ms / rewritten_query
            error    LLM 调用失败(此时答案不入历史上下文)

        Args:
            principal: 当前用户身份;其权限标签决定能检索到哪些文档/FAQ。
            question: 员工原始问题。
            session_id: 续接已有会话;None 表示新开会话。
            category: 限定知识分类(如 hr/it),不合法的值被忽略。
            request_id: 请求 ID,写入问答记录便于日志串联。

        Raises:
            SessionNotFound: session_id 不存在或不属于当前用户。
        """
        t0 = time.time()
        question = self._sanitize(question)
        if category and not _CAT_RE.match(category):
            category = None

        # ---- 阶段 1:会话 / 历史 / 寒暄 / FAQ(短连接,用完即释放)----
        with session_scope(self._sf) as db:
            if session_id:
                sess = self._get_owned_session(db, principal, session_id)
            else:
                sess = ChatSession(id=str(uuid.uuid4()), user_id=principal.user_id, title=question[:30] or "新会话")
                db.add(sess)
                db.flush()
            sid = sess.id
            # 历史只取成功的轮次,按时间正序;错误轮次的"答案"是兜底话术,不应成为后续问答的上下文
            history = [
                {"question": t.question, "answer": t.answer}
                for t in reversed(
                    db.scalars(
                        select(ChatTurn)
                        .where(ChatTurn.session_id == sid, ChatTurn.route != "error")
                        .order_by(ChatTurn.id.desc())
                        .limit(self.s.history_turns)
                    ).all()
                )
            ]
            intent, chit_reply = self.intent.classify(question)
            faq_hit = None if intent == Intent.CHITCHAT else self.faq.match(db, question, principal, category)

        yield {"type": "meta", "session_id": sid}
        turn = _Turn(principal, sid, question, t0, request_id)

        if intent == Intent.CHITCHAT:
            yield from self._finish_simple(turn, chit_reply or "", "chitchat", [])
            return
        if faq_hit:
            cites = [
                {
                    "n": 1,
                    "type": "faq",
                    "faq_id": faq_hit.entry_id,
                    "title": faq_hit.question,
                    "score": faq_hit.confidence,
                    "cited": True,
                }
            ]
            yield from self._finish_simple(turn, faq_hit.answer, "faq", cites)
            return

        # ---- 阶段 2:通用常识(默认关闭;企业场景宁可拒答也不让模型自由发挥)----
        if intent == Intent.GENERAL and self.s.allow_general_knowledge:
            yield from self._generate(turn, prompts.build_general_messages(self.s.company_name, question), [], "general")
            return

        # ---- 阶段 3:RAG ----
        # 追问先补全再检索;生成阶段仍使用用户原问题 + 历史,答案语气与上下文保持自然
        search_q = self.rewriter.rewrite(question, history)
        turn.rewritten = search_q if search_q != question else None
        turn.strategy = self.strategy.select(search_q)
        contexts = self.retrieval.retrieve(principal, search_q, turn.strategy, category)
        if not contexts:
            msg = f"{_NO_INFO}。你可以联系:{self.s.helpdesk_contact}。"
            yield from self._finish_simple(turn, msg, "refused", [])
            return
        messages = prompts.build_rag_messages(
            self.s.company_name, question, [{"title": c.title, "section": c.section, "text": c.text} for c in contexts], history
        )
        yield from self._generate(turn, messages, contexts, "rag")

    def answer(
        self,
        principal: Principal,
        question: str,
        session_id: str | None = None,
        category: str | None = None,
        request_id: str = "",
    ) -> dict:
        """非流式问答:消费 `stream` 的事件并聚合成一个结果字典。

        Returns:
            含 answer / sources / session_id / turn_id / route / strategy / latency_ms /
            rewritten_query;LLM 失败时额外带 error 字段,调用方据此返回 502。
        """
        out: dict = {"answer": "", "sources": [], "session_id": session_id}
        for ev in self.stream(principal, question, session_id, category, request_id):
            if ev["type"] == "meta":
                out["session_id"] = ev["session_id"]
            elif ev["type"] == "token":
                out["answer"] += ev["text"]
            elif ev["type"] == "sources":
                out["sources"] = ev["items"]
            elif ev["type"] == "end":
                out.update(
                    turn_id=ev["turn_id"],
                    route=ev["route"],
                    strategy=ev.get("strategy"),
                    latency_ms=ev["latency_ms"],
                    rewritten_query=ev.get("rewritten_query"),
                )
            elif ev["type"] == "error":
                out["error"] = ev["message"]
        return out

    # ---------------------------------------------------------------- 内部
    def _sanitize(self, question: str) -> str:
        """去首尾空白并按配置脱敏。只记录命中数量,日志与指标中不出现原文。"""
        question = question.strip()
        if not self.s.mask_pii:
            return question
        masked = mask_pii(question)
        for label, n in masked.counts.items():
            metrics.inc("knowbase_pii_masked_total", {"type": label}, n)
        if masked.total:
            log.info("问题中已脱敏 %d 处个人信息", masked.total)
        return masked.text

    def _end_event(self, turn: _Turn, turn_id: int, route: str) -> dict:
        return {
            "type": "end",
            "turn_id": turn_id,
            "route": route,
            "strategy": turn.strategy,
            "latency_ms": turn.elapsed_ms(),
            "rewritten_query": turn.rewritten,
        }

    def _finish_simple(self, turn: _Turn, answer: str, route: str, cites: list[dict]) -> Iterator[dict]:
        """不经过 LLM 的收尾(寒暄 / FAQ / 拒答):一次性输出答案并落库。"""
        yield {"type": "token", "text": answer}
        if cites:
            yield {"type": "sources", "items": cites}
        turn_id = self._save(turn, answer, route, cites)
        yield self._end_event(turn, turn_id, route)

    def _generate(self, turn: _Turn, messages: list[dict], contexts: list[Context], route: str) -> Iterator[dict]:
        """调用 LLM 流式生成,并做引用提取、拒答识别与落库。"""
        collected: list[str] = []
        try:
            for token in self.llm.stream(messages):
                collected.append(token)
                yield {"type": "token", "text": token}
        except LLMError:
            # 已输出的部分保留给前端展示;没有任何输出时给出兜底话术。记为 error 轮次,不进入后续上下文。
            partial = "".join(collected)
            fallback = f"抱歉,服务暂时繁忙,请稍后重试,或联系:{self.s.helpdesk_contact}。"
            if not partial:
                yield {"type": "token", "text": fallback}
            turn_id = self._save(turn, partial or fallback, "error", [])
            yield {"type": "error", "message": "LLM 调用失败", "turn_id": turn_id}
            return

        answer = "".join(collected).strip()
        if answer.startswith(_NO_INFO):
            # 模型自己判断"资料不足":同样记为拒答/知识缺口,并补上人工渠道;不再展示无用的引用
            if self.s.helpdesk_contact not in answer:
                tail = f"。你可以联系:{self.s.helpdesk_contact}。"
                yield {"type": "token", "text": tail}
                answer += tail
            route, contexts = "refused", []
        # 标注"哪些来源真正被答案引用",前端据此高亮;未被引用的来源仍保留,供员工核对
        cited_ns = {int(n) for n in _CITE_RE.findall(answer)}
        cites = [c.to_citation(cited=(c.n in cited_ns)) for c in contexts]
        if cites:
            yield {"type": "sources", "items": cites}
        turn_id = self._save(turn, answer, route, cites)
        yield self._end_event(turn, turn_id, route)

    def _save(self, turn: _Turn, answer: str, route: str, cites: list[dict]) -> int:
        """落库一轮问答并记录指标。返回 turn_id。"""
        with session_scope(self._sf) as db:
            row = ChatTurn(
                session_id=turn.sid,
                user_id=turn.principal.user_id,
                question=turn.question,
                answer=answer,
                route=route,
                strategy=turn.strategy,
                citations=cites,
                latency_ms=turn.elapsed_ms(),
                model=getattr(self.llm, "model", None),
                request_id=turn.request_id,
            )
            db.add(row)
            db.flush()
            tid = row.id
        metrics.inc("knowbase_chat_total", {"route": route})
        metrics.observe("knowbase_chat_seconds", time.time() - turn.t0, {"route": route})
        log.info("问答完成 route=%s strategy=%s user=%s turn=%s", route, turn.strategy, turn.principal.username, tid)
        return tid
