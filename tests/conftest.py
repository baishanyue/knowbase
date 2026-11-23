"""测试公共夹具:内存数据库 + 假 LLM + 已灌入演示数据的完整应用。

所有测试都在内存 SQLite 与 memory 检索后端上运行,不依赖 Redis、向量库、GPU 或真实 LLM。
"""

import os

os.environ["KNOWBASE_AUTOSTART"] = "0"  # 导入 app.main 时不自动创建默认应用

import pytest
from fastapi.testclient import TestClient

from knowbase.container import build_container
from knowbase.core.config import Settings
from knowbase.main import create_app
from knowbase.rag.llm import LLMError
from scripts.seed_demo import seed

PASSWORD = "Passw0rd!test"


class FakeLLM:
    """可控的假 LLM:记录收到的 prompt,按需失败。回答固定引用 [1],便于断言引用逻辑。"""

    model = "fake-llm"

    def __init__(self) -> None:
        self.fail = False
        self.complete_reply = "直接检索"  # complete() 的固定返回值;测试追问改写时可改成目标问题
        self.calls: list[list[dict]] = []

    def complete(self, messages, temperature=0.1, max_tokens=600):
        self.calls.append(messages)
        if self.fail:
            raise LLMError("boom")
        return self.complete_reply

    def stream(self, messages, temperature=0.1):
        self.calls.append(messages)
        if self.fail:
            raise LLMError("boom")
        yield "根据公司制度,"
        yield "请按流程办理 [1]"


@pytest.fixture()
def llm():
    return FakeLLM()


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        env="test",
        database_url="sqlite://",
        upload_dir=str(tmp_path / "up"),
        password_hash_iterations=1000,
        enable_strategy_llm=False,
        rate_limit_per_minute=1000,
        secret_key="test-secret-test-secret-test-secret-1234",
    )


@pytest.fixture()
def container(settings, llm):
    c = build_container(settings, llm=llm)
    seed(c, PASSWORD)
    return c


@pytest.fixture()
def client(container):
    with TestClient(create_app(container.settings, container)) as cl:
        yield cl


def login(client, username, password=PASSWORD):
    r = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def bob(client):  # IT 部门普通员工
    return login(client, "it_bob")


@pytest.fixture()
def amy(client):  # HR 知识库管理员
    return login(client, "hr_amy")


@pytest.fixture()
def carl(client):  # 财务普通员工
    return login(client, "fin_carl")


@pytest.fixture()
def admin(client):
    return login(client, "admin")
