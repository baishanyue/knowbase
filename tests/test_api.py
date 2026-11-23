"""接口层集成测试:鉴权、问答路由、权限隔离、流式、会话与反馈、管理后台。

每个用例都走真实的 FastAPI 应用(TestClient)与真实的数据库/检索逻辑,只替换 LLM。
"""

import json

from tests.conftest import login


def ask(client, headers, q, **kw):
    return client.post("/api/v1/chat", json={"question": q, **kw}, headers=headers)


# ------------------------------------------------------------------ 鉴权
def test_requires_login(client):
    assert client.post("/api/v1/chat", json={"question": "你好"}).status_code == 401
    assert client.get("/api/v1/auth/me").status_code == 401


def test_login_failure_and_lockout(client):
    for _ in range(10):
        assert client.post("/api/v1/auth/login", json={"username": "it_bob", "password": "bad"}).status_code == 401
    r = client.post("/api/v1/auth/login", json={"username": "it_bob", "password": "bad"})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == 429 and r.headers["X-Request-ID"]


def test_logout_revokes_token(client, bob):
    assert client.get("/api/v1/auth/me", headers=bob).status_code == 200
    assert client.post("/api/v1/auth/logout", headers=bob).status_code == 204
    assert client.get("/api/v1/auth/me", headers=bob).status_code == 401


def test_deactivated_user_loses_access_immediately(client, bob, admin):
    uid = client.get("/api/v1/auth/me", headers=bob).json()["user_id"]
    assert client.patch(f"/api/v1/admin/users/{uid}/active", params={"active": False}, headers=admin).status_code == 200
    assert client.get("/api/v1/auth/me", headers=bob).status_code == 401


# ------------------------------------------------------------------ 问答路由
def test_chitchat_route_no_llm(client, bob, llm):
    r = ask(client, bob, "你好").json()
    assert r["route"] == "chitchat" and "知底助手" in r["answer"] and llm.calls == []


def test_faq_route_hits_before_rag(client, bob, llm):
    r = ask(client, bob, "怎么重置OA密码").json()
    assert r["route"] == "faq" and "自助" in r["answer"]
    assert r["sources"][0]["type"] == "faq" and llm.calls == []


def test_rag_route_returns_answer_with_citations(client, bob, llm):
    r = ask(client, bob, "年假没休完能不能顺延").json()
    assert r["route"] == "rag" and "[1]" in r["answer"]
    src = r["sources"][0]
    assert src["title"] == "员工休假管理制度" and src["cited"] and src["section"].startswith("员工休假管理制度")
    prompt = llm.calls[-1][1]["content"]
    assert "<资料>" in prompt and "3 月 31 日" in prompt  # 资料确实进了提示词


def test_refuses_without_calling_llm_when_kb_has_no_answer(client, bob, llm):
    r = ask(client, bob, "今天北京天气怎么样").json()
    assert r["route"] == "refused" and "知识库中暂无相关信息" in r["answer"] and "8000" in r["answer"]
    assert llm.calls == []  # 不调用 LLM,杜绝凭空编造


def test_restricted_knowledge_not_leaked_to_other_department(client, bob, amy, llm):
    denied = ask(client, bob, "P6 月薪带宽是多少").json()
    assert denied["route"] == "refused" and "25000" not in json.dumps(denied, ensure_ascii=False)
    ok = ask(client, amy, "P6 月薪带宽是多少").json()
    assert ok["route"] == "rag" and ok["sources"][0]["title"] == "薪酬体系与调薪规则"
    # 受限 FAQ 同理
    assert ask(client, bob, "年度调薪什么时候公布").json()["route"] != "faq"
    assert ask(client, amy, "年度调薪什么时候公布").json()["route"] == "faq"


def test_llm_failure_degrades_gracefully(client, bob, llm):
    llm.fail = True
    r = ask(client, bob, "年假没休完能不能顺延")
    assert r.status_code == 502
    with client.stream("POST", "/api/v1/chat/stream", json={"question": "年假没休完能不能顺延"}, headers=bob) as s:
        body = "".join(s.iter_text())
    assert "event: error" in body and "8001" in body  # 用户看到兜底提示和人工渠道
    h = client.get("/api/v1/sessions", headers=bob).json()
    hist = client.get(f"/api/v1/sessions/{h[0]['id']}/history", headers=bob).json()
    assert all(t["route"] == "error" for t in hist)  # 错误轮次被标记,不会污染后续的对话历史


def test_sse_stream_events(client, bob):
    with client.stream("POST", "/api/v1/chat/stream", json={"question": "年假没休完能不能顺延"}, headers=bob) as s:
        assert s.headers["content-type"].startswith("text/event-stream")
        events = [(b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:])) for b in "".join(s.iter_text()).strip().split("\n\n")]
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "end" and "token" in kinds and "sources" in kinds
    assert events[-1][1]["route"] == "rag"


def test_websocket_stream_multi_turn_stays_connected(client, bob):
    token = bob["Authorization"].split()[1]
    with client.websocket_connect(f"/api/v1/chat/ws?token={token}") as ws:
        for q, route in (("你好", "chitchat"), ("年假没休完能不能顺延", "rag")):  # 寒暄后连接不应断开
            ws.send_text(json.dumps({"question": q}))
            while (ev := ws.receive_json())["type"] != "end":
                pass
            assert ev["route"] == route


def test_websocket_rejects_bad_token(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/v1/chat/ws?token=bad") as ws:
        ws.receive_json()


def test_rate_limit(container, client):
    container.settings.rate_limit_per_minute = 3
    h = login(client, "it_bob")
    codes = [ask(client, h, "你好").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_input_validation(client, bob):
    assert client.post("/api/v1/chat", json={"question": ""}, headers=bob).status_code == 422
    assert client.post("/api/v1/chat", json={"question": "x" * 1001}, headers=bob).status_code == 422


# ------------------------------------------------------------------ 会话隔离 / 历史 / 反馈
def test_session_isolation_between_users(client, bob, carl):
    sid = ask(client, bob, "你好").json()["session_id"]
    assert client.get(f"/api/v1/sessions/{sid}/history", headers=carl).status_code == 404
    assert client.delete(f"/api/v1/sessions/{sid}", headers=carl).status_code == 404
    assert ask(client, carl, "你好", session_id=sid).status_code == 404
    assert len(client.get(f"/api/v1/sessions/{sid}/history", headers=bob).json()) == 1


def test_history_feeds_followups_and_can_be_deleted(client, bob, llm):
    sid = ask(client, bob, "年假没休完能不能顺延").json()["session_id"]
    ask(client, bob, "那病假呢需要什么证明", session_id=sid)
    # 取"生成阶段"的调用(system + user 两条消息);追问改写等辅助调用只有一条消息
    gen_call = [m for m in llm.calls if len(m) == 2][-1]
    assert "年假没休完能不能顺延" in gen_call[1]["content"]  # 上一轮进了对话历史
    assert client.delete(f"/api/v1/sessions/{sid}", headers=bob).status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}/history", headers=bob).status_code == 404


def test_feedback_upsert_and_ownership(client, bob, carl):
    r = ask(client, bob, "年假没休完能不能顺延").json()
    body = {"turn_id": r["turn_id"], "rating": -1, "comment": "没说清楚"}
    assert client.post("/api/v1/feedback", json=body, headers=bob).status_code == 204
    assert client.post("/api/v1/feedback", json={**body, "rating": 1}, headers=bob).status_code == 204  # 可修改
    assert client.post("/api/v1/feedback", json=body, headers=carl).status_code == 404  # 不能给别人的回答打分
    assert client.post("/api/v1/feedback", json={**body, "rating": 5}, headers=bob).status_code == 422


# ------------------------------------------------------------------ 管理后台
def test_admin_endpoints_forbidden_for_employees(client, bob):
    for method, url in (
        ("get", "/api/v1/admin/faq"),
        ("get", "/api/v1/admin/documents"),
        ("get", "/api/v1/admin/stats"),
        ("get", "/api/v1/admin/audit"),
        ("post", "/api/v1/admin/users"),
    ):
        assert getattr(client, method)(url, headers=bob).status_code == 403, url


def test_kb_admin_cannot_use_system_admin_endpoints(client, amy):
    assert client.get("/api/v1/admin/audit", headers=amy).status_code == 403
    assert client.get("/api/v1/admin/faq", headers=amy).status_code == 200


def test_faq_crud_takes_effect_immediately(client, bob, amy):
    body = {"question": "食堂几点开饭", "answer": "午餐 11:30-13:30", "category": "admin", "acl_tags": ["public"]}
    fid = client.post("/api/v1/admin/faq", json=body, headers=amy).json()["id"]
    assert ask(client, bob, "食堂几点开饭").json()["route"] == "faq"
    client.put(f"/api/v1/admin/faq/{fid}", json={**body, "status": "disabled"}, headers=amy)
    assert ask(client, bob, "食堂几点开饭").json()["route"] != "faq"
    assert client.post("/api/v1/admin/faq", json={**body, "acl_tags": ['x" or 1==1']}, headers=amy).status_code == 422
    assert client.delete(f"/api/v1/admin/faq/{fid}", headers=amy).status_code == 204


def test_faq_csv_import_reports_bad_rows(client, amy):
    csv_text = (
        "question,answer,category,acl_tags\n班车几点发车,早 8:00,admin,public\n,缺问题,admin,public\n坏权限,答,admin,hack\n"
    )
    r = client.post("/api/v1/admin/faq/import", files={"file": ("f.csv", csv_text.encode(), "text/csv")}, headers=amy).json()
    assert r["imported"] == 1 and len(r["failed"]) == 2


def test_document_upload_end_to_end_via_api(client, amy, bob):
    md = "# 食堂管理办法\n## 用餐时间\n午餐供应时间为 11:30 至 13:30,晚餐 17:30 至 19:00。".encode()
    r = client.post(
        "/api/v1/admin/documents",
        headers=amy,
        data={"title": "食堂管理办法", "category": "admin", "acl_tags": "public"},
        files={"file": ("canteen.md", md, "text/markdown")},
    )
    assert r.status_code == 202
    doc = client.get(f"/api/v1/admin/documents/{r.json()['id']}", headers=amy).json()
    assert doc["status"] == "indexed" and doc["chunk_count"] >= 1  # TestClient 会等后台任务跑完
    ans = ask(client, bob, "晚餐供应时间是什么时候").json()
    assert ans["route"] == "rag" and ans["sources"][0]["title"] == "食堂管理办法"
    # 下线后立即不可检索
    assert client.delete(f"/api/v1/admin/documents/{doc['id']}", headers=amy).status_code == 204
    assert ask(client, bob, "晚餐供应时间是什么时候").json()["route"] == "refused"
    # 非法文件类型
    bad = client.post(
        "/api/v1/admin/documents", headers=amy, data={"title": "x"}, files={"file": ("a.exe", b"MZ", "application/octet-stream")}
    )
    assert bad.status_code == 422


def test_stats_gaps_and_audit(client, bob, amy, admin):
    ask(client, bob, "公司有没有养猫政策")  # 拒答 -> 知识缺口
    ask(client, bob, "公司有没有养猫政策")
    t = ask(client, bob, "年假没休完能不能顺延").json()
    client.post("/api/v1/feedback", json={"turn_id": t["turn_id"], "rating": -1}, headers=bob)
    s = client.get("/api/v1/admin/stats", headers=amy).json()
    assert s["total_questions"] == 3 and s["by_route"]["refused"] == 2 and s["thumbs_down"] == 1
    gaps = client.get("/api/v1/admin/gaps", headers=amy).json()
    assert gaps["refused"][0] == {"question": "公司有没有养猫政策", "count": 2}
    assert gaps["downvoted"][0]["question"] == "年假没休完能不能顺延"
    client.get("/api/v1/admin/audit", headers=bob)  # 越权尝试
    actions = {a["action"] for a in client.get("/api/v1/admin/audit", headers=admin).json()}
    assert {"auth.login", "authz.denied"} <= actions


def test_health_and_error_envelope(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["checks"]["database"]
    err = client.get("/api/v1/admin/faq").json()["error"]
    assert err["code"] == 401 and "request_id" in err
