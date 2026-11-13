"""多轮追问改写:把依赖上下文的追问补全为可独立检索的问题。

问题场景:
    员工先问"年假怎么申请?",接着追问"那病假呢?"。后一句单独拿去检索几乎找不到资料,
    因为向量/BM25 看不到"那病假呢"里省略的主语和场景。

做法(两级,优先保证可用而不是完美):
    1. 只有"有历史 + 看起来像追问"才改写,独立完整的问题原样放行,避免无谓的 LLM 调用和语义漂移;
    2. 优先让 LLM 改写;LLM 不可用、超时或输出不合格时,退化为"上一问 + 本问"的拼接,
       拼接对关键词检索同样有效,且零成本、不会失败。

安全约束:
    改写结果只用于"检索",不会替代用户原问题展示或落库;LLM 输出会做长度、换行校验,
    防止模型把长篇说明或提示词注入内容带进检索链路。
"""

from __future__ import annotations

import logging
import re

from knowbase.rag import prompts
from knowbase.rag.llm import LLM, LLMError

log = logging.getLogger(__name__)

# 追问的典型起手词/指代词。命中即认为该问题依赖上文。
_FOLLOWUP_HEAD = re.compile(r"^(那|那么|还有|另外|然后|所以|它|他|她|这个|那个|这些|那些|上述|刚才|前面|同样|这种)")
# 短句 + 语气词结尾("病假呢"、"需要审批吗"),通常是省略主语的追问。
_SHORT_TAIL = re.compile(r"(呢|吗|么)[??]?$")

_MAX_REWRITE_LEN = 120  # 改写结果的长度上限,超过视为模型"跑题"
_SHORT_QUESTION_LEN = 12


def looks_like_followup(question: str) -> bool:
    """启发式判断一个问题是否依赖上文。

    刻意偏保守:误判为"追问"的代价只是多一次改写,误判为"独立问题"则会漏掉上下文,
    所以对"很短"的问题一律放宽为追问。
    """
    q = question.strip()
    if not q:
        return False
    if _FOLLOWUP_HEAD.match(q):
        return True
    return len(q) <= _SHORT_QUESTION_LEN and bool(_SHORT_TAIL.search(q))


class QueryRewriter:
    """追问改写器。无历史或不像追问时直接返回原问题。"""

    def __init__(self, llm: LLM, enabled: bool = True) -> None:
        self.llm = llm
        self.enabled = enabled

    def rewrite(self, question: str, history: list[dict]) -> str:
        """返回用于检索的问题。

        Args:
            question: 员工当前的原始问题(已脱敏)。
            history: 最近若干轮 [{"question":..., "answer":...}],按时间正序。

        Returns:
            补全后的独立问题;无需改写或改写失败且无法拼接时返回原问题。
        """
        if not self.enabled or not history or not looks_like_followup(question):
            return question

        last_q = history[-1]["question"].strip()
        try:
            out = self.llm.complete(
                [{"role": "user", "content": prompts.REWRITE_PROMPT.format(history=self._render(history), query=question)}],
                max_tokens=80,
            )
        except LLMError:
            return self._fallback(last_q, question)

        cleaned = self._clean(out)
        if cleaned is None:
            log.warning("追问改写输出不合格,退化为拼接: %r", out[:80])
            return self._fallback(last_q, question)
        return cleaned

    @staticmethod
    def _render(history: list[dict]) -> str:
        """取最近两轮作为改写依据;答案截断,控制提示词长度并减少无关内容。"""
        return "\n".join(f"员工: {h['question']}\n助手: {h['answer'][:120]}" for h in history[-2:])

    @staticmethod
    def _clean(raw: str) -> str | None:
        """校验并清洗模型输出;不合格返回 None。"""
        text = raw.strip().strip("\"'「」『』“”")
        if not text or "\n" in text or len(text) > _MAX_REWRITE_LEN:
            return None
        return text

    @staticmethod
    def _fallback(last_question: str, question: str) -> str:
        """拼接上一问与本问:粗糙但对检索有效,且保证不会失败。"""
        return f"{last_question} {question}"[:_MAX_REWRITE_LEN * 2]
