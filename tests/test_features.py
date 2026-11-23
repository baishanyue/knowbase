"""新增能力的测试:PII 脱敏、运行指标、追问改写、知识缺口导出。"""

from knowbase.core.metrics import Metrics, metrics
from knowbase.core.pii import mask_pii
from knowbase.rag.rewriter import QueryRewriter, looks_like_followup

from .conftest import FakeLLM
from .test_api import ask


# ------------------------------------------------------------------ PII 脱敏
def test_mask_pii_covers_structured_identifiers():
    r = mask_pii("我手机13812345678,邮箱a.b@corp.com,身份证11010519491231002X,卡号6222021234567890128")
    assert "13812345678" not in r.text and "a.b@corp.com" not in r.text
    assert "11010519491231002X" not in r.text and "6222021234567890128" not in r.text
    assert r.counts == {"身份证号": 1, "银行卡号": 1, "手机号": 1, "邮箱": 1}


def test_mask_pii_does_not_touch_ordinary_numbers():
    text = "工号 20231108,报销单 4500 元,制度编号 HR-2024-0001,订单 1234567890123456"  # 最后一个通不过 Luhn
    assert mask_pii(text).text == text


def test_chat_masks_pii_before_persisting_and_calling_llm(client, bob, llm):
    ask(client, bob, "我的手机号13812345678,年假没休完能不能顺延")
    sid = client.get("/api/v1/sessions", headers=bob).json()[0]["id"]
    turns = client.get(f"/api/v1/sessions/{sid}/history", headers=bob).json()
    assert "13812345678" not in turns[0]["question"] and "已脱敏" in turns[0]["question"]
    assert all("13812345678" not in m["content"] for call in llm.calls for m in call)


# ------------------------------------------------------------------ 追问改写
def test_followup_detection():
    assert looks_like_followup("那病假呢")
    assert looks_like_followup("需要审批吗")
    assert not looks_like_followup("员工入职满一年可以休几天年假并如何申请")


def test_rewriter_skips_standalone_and_uses_llm_for_followups():
    llm = FakeLLM()
    llm.complete_reply = "「病假需要什么证明材料」"
    rw = QueryRewriter(llm)
    history = [{"question": "年假怎么申请", "answer": "在 OA 提交申请"}]
    assert rw.rewrite("员工入职满一年可以休几天年假并如何申请", history) == "员工入职满一年可以休几天年假并如何申请"
    assert rw.rewrite("那病假呢", []) == "那病假呢"  # 无历史不改写
    assert rw.rewrite("那病假呢", history) == "病假需要什么证明材料"  # 去掉引号


def test_rewriter_falls_back_to_concatenation():
    llm = FakeLLM()
    llm.fail = True
    history = [{"question": "年假怎么申请", "answer": "在 OA 提交申请"}]
    assert QueryRewriter(llm).rewrite("那病假呢", history) == "年假怎么申请 那病假呢"
    llm.fail, llm.complete_reply = False, "第一行\n第二行"  # 多行输出视为不合格
    assert QueryRewriter(llm).rewrite("那病假呢", history) == "年假怎么申请 那病假呢"


def test_chat_returns_rewritten_query_for_followup(client, bob, llm):
    sid = ask(client, bob, "年假没休完能不能顺延").json()["session_id"]
    llm.complete_reply = "病假需要什么证明"
    r = ask(client, bob, "那病假呢", session_id=sid).json()
    assert r["rewritten_query"] == "病假需要什么证明"


# ------------------------------------------------------------------ 指标
def test_metrics_render_prometheus_format():
    m = Metrics()
    m.inc("demo_total", {"route": 'a"b'})
    m.observe("demo_seconds", 0.3)
    out = m.render()
    assert 'demo_total{route="a\\"b"} 1' in out  # 标签值被转义
    assert 'demo_seconds_bucket{le="0.5"} 1' in out and "demo_seconds_count 1" in out


def test_metrics_endpoint_and_token(client, bob, container):
    metrics.reset()
    ask(client, bob, "年假没休完能不能顺延")
    body = client.get("/metrics").text  # test 环境未配置令牌 -> 开放
    assert 'knowbase_chat_total{route=' in body and "knowbase_http_requests_total" in body
    container.settings.metrics_token = "tok"
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer tok"}).status_code == 200
    container.settings.metrics_token = None
    container.settings.env = "prod"
    assert client.get("/metrics").status_code == 404  # prod 且无令牌:不暴露


# ------------------------------------------------------------------ 知识缺口导出
def test_gaps_export_csv_and_formula_injection_guard(client, bob, amy, admin):
    ask(client, bob, "=HYPERLINK(\"http://x\")公司股权激励方案细则是什么")  # 知识库没有 -> 拒答
    assert client.get("/api/v1/admin/gaps/export", headers=bob).status_code == 403
    r = client.get("/api/v1/admin/gaps/export", headers=amy)
    assert r.status_code == 200 and r.content.startswith("﻿".encode())
    assert "被拒答" in r.text and "'=HYPERLINK" in r.text  # 公式前缀被中和
    actions = [a["action"] for a in client.get("/api/v1/admin/audit", headers=admin).json()]
    assert "gaps.export" in actions
