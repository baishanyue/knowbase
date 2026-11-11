"""数据库引擎与会话管理。

约定:
    - 引擎进程内单例,Session 按"一次请求/一个短阶段"创建、用完即关,不跨请求复用;
    - 流式问答期间不持有 Session(见 ChatService),避免慢速 LLM 占满连接池;
    - SQLite 仅用于本地开发与测试;生产使用 MySQL/PostgreSQL(由配置校验强制)。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from knowbase.db.models import Base


def make_engine(database_url: str) -> Engine:
    """按连接串创建引擎,并为不同数据库设置合适的连接池策略。"""
    if database_url.startswith("sqlite"):
        # check_same_thread=False:FastAPI 的同步接口在线程池中执行,SQLite 默认禁止跨线程使用连接
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in database_url or database_url.endswith("sqlite://"):
            kwargs["poolclass"] = StaticPool  # 内存库每个连接都是独立数据库,必须所有线程共用同一个连接
        return create_engine(database_url, **kwargs)
    return create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,  # 突发流量下的临时连接上限,总连接数 <= 30,需小于数据库 max_connections
        pool_pre_ping=True,  # 取连接前探测,数据库重启/网络闪断后自动重连
        pool_recycle=1800,  # 30 分钟回收,避开数据库/中间件的空闲连接超时
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """创建 Session 工厂。

    expire_on_commit=False:提交后对象属性仍可读取,便于在 `session_scope` 之外直接返回模型;
    autoflush=False:由代码显式 flush,避免查询过程中意外写库。
    """
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def init_db(engine: Engine) -> None:
    """开发/测试用建表(幂等)。生产请引入 Alembic 做版本化迁移,不要依赖启动时自动建表。"""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """事务作用域:正常退出提交,异常回滚并继续抛出,最终一定关闭 Session。"""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
