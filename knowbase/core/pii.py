"""个人敏感信息(PII)脱敏。

背景:
    员工提问里可能夹带手机号、身份证号、银行卡号、邮箱。这些内容一旦被原样发送给外部
    LLM 或写入问答记录,就扩大了数据暴露面,也不利于满足个人信息保护合规要求。

策略:
    - 在"进入系统的第一时间"脱敏:ChatService 对问题脱敏后,检索、LLM 调用、落库用的都是脱敏文本;
    - 只做可确定识别的格式(带校验位/固定位数),宁可漏掉也不误伤普通数字(如工号、金额、制度编号);
    - 替换为带类型的占位符,模型仍能理解"这里有一个手机号",不影响语义。

局限(请在上线评估中确认):
    正则只能识别结构化号码,无法识别姓名、地址等自由文本类 PII;涉密场景请使用私有化模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 18 位居民身份证:6 位地区码 + 8 位出生日期 + 3 位顺序码 + 1 位校验码(可能为 X)。
# 限定出生年份 1900~2099、月份 01~12、日 01~31,可显著减少与普通 18 位数字的误匹配。
_ID_CARD = re.compile(r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9A-Za-z])")
# 中国大陆手机号:1 开头、第二位 3~9、共 11 位;前后不能紧邻数字,避免截取长数字串的一部分。
_MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 银行卡:16~19 位数字,且必须通过 Luhn 校验才视为卡号(普通长数字大概率通不过)。
_BANK_CARD = re.compile(r"(?<!\d)\d{16,19}(?!\d)")


def _luhn_ok(number: str) -> bool:
    """Luhn 校验:银行卡号的通用校验位算法,用来过滤掉并非卡号的长数字串。"""
    total, double = 0, False
    for ch in reversed(number):
        d = int(ch)
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    return total % 10 == 0


@dataclass
class MaskResult:
    """脱敏结果:文本与各类 PII 的命中数量(数量可用于审计/指标,不含原文)。"""

    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def mask_pii(text: str) -> MaskResult:
    """对文本中的结构化 PII 做脱敏。

    处理顺序有意为:身份证 -> 银行卡 -> 手机号 -> 邮箱。身份证与银行卡都是长数字串,
    先处理更具体的身份证,避免被银行卡规则抢先匹配;手机号放在后面,避免吞掉长数字串的子串。

    Args:
        text: 原始文本。

    Returns:
        MaskResult,其中 text 为脱敏后的文本,counts 为 {类型: 数量}。
    """
    counts: dict[str, int] = {}

    def sub(pattern: re.Pattern[str], label: str, s: str, check=None) -> str:
        def repl(m: re.Match[str]) -> str:
            if check is not None and not check(m.group(0)):
                return m.group(0)  # 未通过校验:保持原样,不误伤
            counts[label] = counts.get(label, 0) + 1
            return f"[已脱敏:{label}]"

        return pattern.sub(repl, s)

    out = sub(_ID_CARD, "身份证号", text)
    out = sub(_BANK_CARD, "银行卡号", out, check=_luhn_ok)
    out = sub(_MOBILE, "手机号", out)
    out = sub(_EMAIL, "邮箱", out)
    return MaskResult(out, counts)
