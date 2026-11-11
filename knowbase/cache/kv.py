"""KV 存储抽象:Redis(生产)或进程内存(开发/单实例)。

用途(仅两类,故接口刻意保持很小):
    - 固定窗口计数:接口限流、登录防爆破;
    - 带过期的标记:登出后令牌(jti)吊销名单。

可用性策略:
    Redis 故障时一律"放行并告警"(fail-open):限流/吊销是辅助防线,不能因为缓存抖动让
    全部用户无法登录或提问。代价是故障期间限流与吊销暂时失效,需由监控及时发现。

多实例提醒:
    MemoryKV 的数据只在当前进程可见,多实例部署必须配置 Redis,否则限流与吊销会各自为政。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

log = logging.getLogger(__name__)


class KV(Protocol):
    """KV 协议:调用方只依赖这四个方法,便于替换实现或注入测试替身。"""

    def incr_window(self, key: str, window_s: int) -> int:
        """固定窗口计数加一并返回当前计数;窗口首次创建时开始计时。"""
        ...

    def set_flag(self, key: str, ttl_s: int) -> None:
        """设置一个带过期时间的标记。"""
        ...

    def has_flag(self, key: str) -> bool:
        """标记是否存在且未过期。"""
        ...

    def ping(self) -> bool:
        """健康检查:后端是否可用。"""
        ...


class MemoryKV:
    """进程内存实现。线程安全,但不跨进程,仅用于开发、测试与单实例。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, tuple[int, float]] = {}  # key -> (计数, 窗口到期时间戳)
        self._flags: dict[str, float] = {}  # key -> 标记到期时间戳

    def incr_window(self, key: str, window_s: int) -> int:
        now = time.time()
        with self._lock:
            count, expires = self._counters.get(key, (0, 0.0))
            if expires <= now:  # 窗口已过期:重新开窗
                count, expires = 0, now + window_s
            count += 1
            self._counters[key] = (count, expires)
            return count

    def set_flag(self, key: str, ttl_s: int) -> None:
        with self._lock:
            self._flags[key] = time.time() + ttl_s

    def has_flag(self, key: str) -> bool:
        with self._lock:
            return self._flags.get(key, 0.0) > time.time()

    def ping(self) -> bool:
        return True


class RedisKV:
    """Redis 实现。所有操作带 2 秒超时,失败时 fail-open 并记录告警。"""

    def __init__(self, url: str) -> None:
        import redis  # 延迟导入:不用 Redis 的环境无需安装该依赖

        self._r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)

    def incr_window(self, key: str, window_s: int) -> int:
        try:
            pipe = self._r.pipeline()
            pipe.incr(key)
            # nx=True:只在键尚无过期时间时设置,保证是"固定窗口"而非每次请求都顺延窗口
            pipe.expire(key, window_s, nx=True)
            return int(pipe.execute()[0])
        except Exception as e:  # noqa: BLE001
            log.warning("redis incr 失败,限流放行: %s", e)
            return 0  # 返回 0 即视为未超限

    def set_flag(self, key: str, ttl_s: int) -> None:
        try:
            self._r.set(key, "1", ex=max(ttl_s, 1))  # Redis 不接受 0/负 TTL
        except Exception as e:  # noqa: BLE001
            log.warning("redis set 失败: %s", e)

    def has_flag(self, key: str) -> bool:
        try:
            return bool(self._r.exists(key))
        except Exception as e:  # noqa: BLE001
            log.warning("redis exists 失败: %s", e)
            return False  # 查不到就当未吊销(fail-open)

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:  # noqa: BLE001
            return False


def make_kv(redis_url: str | None) -> KV:
    """配置了 Redis 地址则用 Redis,否则退化为进程内存。"""
    return RedisKV(redis_url) if redis_url else MemoryKV()
