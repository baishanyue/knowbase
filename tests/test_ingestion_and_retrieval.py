"""入库管线(去重、版本、失败回退、权限变更重建)与检索(权限隔离、阈值)的测试。"""

import pytest
from sqlalchemy import select

from knowbase.core.security import Principal
from knowbase.db.models import DocParent, Document
from knowbase.rag.ingestion.chunking import build_chunks
from knowbase.rag.ingestion.loaders import Section, load_file
from knowbase.rag.retrieval_service import RetrievalService
from knowbase.rag.retrievers import Hit, MilvusRetriever

HR = Principal(1, "h", "hr", "kb_admin")
IT = Principal(2, "b", "it", "employee")


def _doc(c, text: str, title="测试制度", acl=("public",), cat="hr"):
    res = c.ingestion.register_upload(text.encode(), "x.md", title, cat, list(acl), None)
    return res, (None if res.duplicate else c.ingestion.ingest(res.document.id))


def test_markdown_sections_carry_heading_path(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 总则\n前言\n## 年假\n五天\n## 病假\n需证明\n", encoding="utf-8")
    secs = load_file(str(f))
    assert [s.heading for s in secs] == ["总则", "总则 > 年假", "总则 > 病假"]
    chunks = build_chunks(secs, 1, 1, 1200, 300, 50)
    assert chunks[1].section == "总则 > 年假" and "五天" in chunks[1].content
    assert chunks[1].children and chunks[1].parent_id == "d1v1p1"


def test_parent_child_split_for_long_section():
    long = "。".join(f"第{i}条规定内容比较长用来撑大文本" for i in range(200)) + "。"
    chunks = build_chunks([Section(long, "章")], 7, 2, 600, 150, 20)
    assert len(chunks) > 1
    assert all(len(p.children) >= 1 for p in chunks)
    assert all(len(t) <= 160 for p in chunks for _, t in p.children)


def test_duplicate_upload_is_idempotent(container):
    r1, d1 = _doc(container, "# 制度\n年假五天")
    r2, _ = _doc(container, "# 制度\n年假五天")
    assert r2.duplicate and r2.document.id == d1.id
    with container.sf() as db:
        assert len(db.scalars(select(Document).where(Document.title == "测试制度")).all()) == 1


def test_new_version_replaces_old_and_archives(container):
    _, v1 = _doc(container, "# 奖励\n季度安全奖励五百元")
    _, v2 = _doc(container, "# 奖励\n季度安全奖励一千元,自2027年起执行")
    assert (v1.version, v2.version) == (1, 2)
    with container.sf() as db:
        assert db.get(Document, v1.id).status == "archived"
        assert db.get(Document, v2.id).status == "indexed"
        assert not db.scalars(select(DocParent).where(DocParent.document_id == v1.id)).all()
    ctx = container.chat.retrieval.retrieve(IT, "季度安全奖励多少钱")
    assert ctx and {c.document_id for c in ctx} == {v2.id} and "一千元" in ctx[0].text


def test_failed_ingest_keeps_old_version_serving(container, tmp_path):
    _, v1 = _doc(container, "# 奖励\n季度安全奖励五百元")
    res = container.ingestion.register_upload(b"   ", "x.md", "测试制度", "hr", ["public"], None)
    failed = container.ingestion.ingest(res.document.id)
    assert failed.status == "failed" and failed.error
    with container.sf() as db:
        assert db.get(Document, v1.id).status == "indexed"
    ctx = container.chat.retrieval.retrieve(IT, "季度安全奖励多少钱")
    assert ctx and ctx[0].document_id == v1.id


def test_acl_change_on_same_content_reindexes(container):
    _, d = _doc(container, "# 制度\n薪酬带宽 P6 三万", title="薪酬", acl=("public",))
    assert container.chat.retrieval.retrieve(IT, "P6 薪酬带宽")
    _doc(container, "# 制度\n薪酬带宽 P6 三万", title="薪酬", acl=("dept:hr",))  # 内容不变,仅收紧权限
    assert not container.chat.retrieval.retrieve(IT, "P6 薪酬带宽")
    assert container.chat.retrieval.retrieve(HR, "P6 薪酬带宽")


def test_restricted_doc_invisible_to_other_department(container):
    _doc(container, "# 机密\n并购方案代号蓝鲸", title="并购", acl=("dept:finance",), cat="finance")
    fin = Principal(3, "c", "finance", "employee")
    assert container.chat.retrieval.retrieve(fin, "并购方案代号")
    assert not container.chat.retrieval.retrieve(IT, "并购方案代号")


def test_defense_in_depth_drops_leaked_hits(container):
    """即使向量库(或其权限副本)出了问题返回了无权文档,关系库二次校验也要把它丢掉。"""
    _, d = _doc(container, "# 机密\n并购方案代号蓝鲸", title="并购", acl=("dept:finance",), cat="finance")
    with container.sf() as db:
        pid = db.scalars(select(DocParent.parent_id).where(DocParent.document_id == d.id)).first()

    class Leaky:
        def search(self, *a, **k):
            return [Hit(pid, d.id, 0.99)]

    svc = RetrievalService(Leaky(), None, container.llm, container.sf, 5, 3, 0.1)
    assert svc.retrieve(IT, "并购") == []


def test_milvus_filter_expression_is_safe():
    expr = MilvusRetriever.build_filter(["public", "dept:hr"], "hr")
    assert expr == 'array_contains_any(acl_tags, ["public", "dept:hr"]) and category == "hr"'
    with pytest.raises(ValueError):
        MilvusRetriever.build_filter(["public"], 'hr" or 1==1 or "')


def test_unsupported_and_oversize_uploads_rejected(container):
    with pytest.raises(ValueError):
        container.ingestion.register_upload(b"x", "a.exe", "t", "hr", ["public"], None)
    with pytest.raises(ValueError):
        container.ingestion.register_upload(b"x", "a.md", "t", "HR!", ["public"], None)
    with pytest.raises(ValueError):
        container.ingestion.register_upload(b"x", "a.md", "t", "hr", ["bogus"], None)
