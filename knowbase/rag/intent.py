"""意图识别与检索策略选择。

意图识别(是寒暄?要查知识库?还是通用常识?):
    - 寒暄由正则规则处理,零成本、零延迟;
    - BERT 二分类器是可选插件,需用企业自有数据训练,默认不启用;
    - 默认策略:除寒暄外一律查知识库——企业场景宁可拒答,也不让模型脱离资料自由发挥。

检索策略选择(直接 / 假设问题 / 子查询 / 回溯):
    - 由 LLM 在白名单内选择,输出不在白名单内就回退"直接检索";
    - 短问题直接走"直接检索",省一次 LLM 调用与约几百毫秒延迟。
"""

from __future__ import annotations

import logging
import os
import re
from enum import StrEnum

from knowbase.rag import prompts
from knowbase.rag.llm import LLM, LLMError

log = logging.getLogger(__name__)


class Intent(StrEnum):
    CHITCHAT = "chitchat"  # 寒暄
    KNOWLEDGE = "knowledge"  # 需要查知识库
    GENERAL = "general"  # 通用常识(仅 allow_general_knowledge=True 时直答)


_GREETINGS = [
    (
        r"^(你好|您好|hi|hello|嗨|早上好|下午好|晚上好)",
        "你好!我是{company}知底助手,可以帮你查询制度、流程和 IT/HR 常见问题。",
    ),
    (r"^(你是谁|您是谁|你叫什么|who are you)", "我是{company}知底助手,回答会基于公司制度文档并标注出处。"),
    (r"^(在吗|在不在|有人吗)", "我在!请直接描述你的问题,例如『年假怎么申请』。"),
    (r"^(谢谢|多谢|感谢|thanks)", "不客气!还有其他问题随时问我。"),
]


def match_chitchat(query: str, company: str) -> str | None:
    """命中寒暄规则则返回固定回复,否则 None。只匹配开头且限制长度,避免吞掉"你好,我想问…"这类真实问题。"""
    q = query.strip()
    if len(q) > 20:  # 长句不当作寒暄,避免"你好,我想问..."被吞掉
        return None
    for pat, resp in _GREETINGS:
        if re.match(pat, q, re.IGNORECASE):
            return resp.format(company=company)
    return None


class BertIntentClassifier:
    """可选插件: 加载微调后的 BERT 二分类模型(0=通用常识, 1=知识库问题)。需要自备企业数据重新训练。"""

    def __init__(self, model_dir: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    def predict(self, query: str) -> Intent:
        """预测意图。max_length=128 截断过长输入,控制推理耗时。"""
        enc = self.tok(query, truncation=True, max_length=128, return_tensors="pt")
        with self._torch.no_grad():
            pred = int(self.model(**enc).logits.argmax(dim=1))
        return Intent.KNOWLEDGE if pred == 1 else Intent.GENERAL


class IntentRouter:
    """意图路由:规则优先,模型可选,默认走知识库。"""

    def __init__(self, company: str, model_dir: str | None = None) -> None:
        self.company = company
        self._bert = None
        if model_dir and os.path.isdir(model_dir):
            try:
                self._bert = BertIntentClassifier(model_dir)
                log.info("已加载意图分类模型: %s", model_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("意图模型加载失败,退化为规则+默认走知识库: %s", e)

    def classify(self, query: str) -> tuple[Intent, str | None]:
        """返回 (意图, 寒暄回复)。仅当意图为 CHITCHAT 时回复非空。"""
        reply = match_chitchat(query, self.company)
        if reply:
            return Intent.CHITCHAT, reply
        if self._bert:
            return self._bert.predict(query), None
        return Intent.KNOWLEDGE, None  # 默认全部查知识库:企业场景宁可拒答,也不让模型自由发挥


STRATEGIES = ("直接检索", "假设问题检索", "子查询检索", "回溯问题检索")
DEFAULT_STRATEGY = "直接检索"


class StrategySelector:
    """检索策略选择器。LLM 不可用时静默回退,保证检索主流程不受影响。"""

    def __init__(self, llm: LLM, enabled: bool = True, short_query_len: int = 12) -> None:
        self.llm, self.enabled, self.short_len = llm, enabled, short_query_len

    def select(self, query: str) -> str:
        """返回白名单内的策略名。"""
        if not self.enabled or len(query.strip()) <= self.short_len:
            return DEFAULT_STRATEGY
        try:
            out = self.llm.complete([{"role": "user", "content": prompts.STRATEGY_PROMPT.format(query=query)}], max_tokens=20)
        except LLMError:
            return DEFAULT_STRATEGY
        for s in STRATEGIES:  # 白名单:只接受已知策略
            if s in out:
                return s
        log.warning("策略选择输出不在白名单内,回退直接检索: %r", out)
        return DEFAULT_STRATEGY
