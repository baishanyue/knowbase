"""轻量级运行指标(Prometheus 文本格式),零第三方依赖。

为什么自实现而不引入 prometheus_client:
    本项目只需要少量计数器与耗时直方图,自实现约百行即可;同时避免多进程模式下
    client 库的目录/聚合配置复杂度。若后续指标增多,可平滑替换为 prometheus_client,
    调用方只依赖 `Metrics.inc / observe / render` 三个方法。

注意:
    指标保存在进程内存中,每个 worker 独立统计,由 Prometheus 按实例抓取后聚合;
    进程重启后计数归零(counter 重置是 Prometheus 的标准语义,rate() 可正确处理)。
"""

from __future__ import annotations

import threading
from collections import defaultdict

# 耗时直方图的桶上界(秒)。覆盖 FAQ 毫秒级命中到 LLM 数十秒长尾。
_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

_LabelKey = tuple[tuple[str, str], ...]


def _key(labels: dict[str, str] | None) -> _LabelKey:
    return tuple(sorted((labels or {}).items()))


def _escape(value: str) -> str:
    """转义标签值中的反斜杠、双引号与换行,避免外部可控内容破坏文本格式。"""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _fmt(name: str, key: _LabelKey, extra: tuple[tuple[str, str], ...] = ()) -> str:
    """拼出 `name{k="v",...}`;无标签时只输出 name。"""
    pairs = key + extra
    if not pairs:
        return name
    return name + "{" + ",".join(f'{k}="{_escape(str(v))}"' for k, v in pairs) + "}"


class Metrics:
    """线程安全的计数器与直方图注册表。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[_LabelKey, float]] = defaultdict(lambda: defaultdict(float))
        self._hist: dict[str, dict[_LabelKey, list[float]]] = defaultdict(dict)  # 每个序列: [各桶计数..., sum, count]
        self._help: dict[str, str] = {}

    def describe(self, name: str, help_text: str) -> None:
        """登记指标说明(输出到 # HELP 行)。"""
        self._help[name] = help_text

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        """计数器加值。标签基数需受控:不要把用户 ID、问题文本等高基数内容作为标签。"""
        with self._lock:
            self._counters[name][_key(labels)] += value

    def observe(self, name: str, seconds: float, labels: dict[str, str] | None = None) -> None:
        """向直方图记录一次观测值(秒)。"""
        k = _key(labels)
        with self._lock:
            series = self._hist[name].setdefault(k, [0.0] * (len(_BUCKETS) + 2))
            for i, upper in enumerate(_BUCKETS):
                if seconds <= upper:
                    series[i] += 1
            series[-2] += seconds  # sum
            series[-1] += 1  # count

    def reset(self) -> None:
        """清空全部数据(仅供测试使用)。"""
        with self._lock:
            self._counters.clear()
            self._hist.clear()

    def render(self) -> str:
        """渲染为 Prometheus 文本暴露格式。"""
        lines: list[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} counter")
                for key, val in sorted(series.items()):
                    lines.append(f"{_fmt(name, key)} {val:g}")
            for name, series in sorted(self._hist.items()):
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} histogram")
                for key, vals in sorted(series.items()):
                    for i, upper in enumerate(_BUCKETS):
                        lines.append(f"{_fmt(name + '_bucket', key, (('le', f'{upper:g}'),))} {vals[i]:g}")
                    lines.append(f"{_fmt(name + '_bucket', key, (('le', '+Inf'),))} {vals[-1]:g}")
                    lines.append(f"{_fmt(name + '_sum', key)} {vals[-2]:g}")
                    lines.append(f"{_fmt(name + '_count', key)} {vals[-1]:g}")
        return "\n".join(lines) + "\n"


# 进程级默认注册表:中间件与业务代码直接使用,无需层层传递。
metrics = Metrics()
metrics.describe("knowbase_http_requests_total", "HTTP 请求总数(按方法、路径模板、状态码)")
metrics.describe("knowbase_http_request_seconds", "HTTP 请求耗时(秒)")
metrics.describe("knowbase_chat_total", "问答总数(按路由:faq/rag/refused/chitchat/general/error)")
metrics.describe("knowbase_chat_seconds", "问答端到端耗时(秒)")
metrics.describe("knowbase_pii_masked_total", "被脱敏的 PII 数量(按类型)")
