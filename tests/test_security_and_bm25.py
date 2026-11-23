"""安全基础能力(密码、JWT、权限标签)与 BM25 置信度的单元测试。"""

import jwt
import pytest

from knowbase.core.security import (
    Principal,
    create_access_token,
    decode_access_token,
    hash_password,
    validate_acl_tags,
    verify_password,
)
from knowbase.db.models import FaqEntry
from knowbase.rag.bm25 import BM25, tokenize
from knowbase.services.faq_service import FaqMatcher


def test_password_hash_roundtrip():
    h = hash_password("s3cret", 1000)
    assert verify_password("s3cret", h)
    assert not verify_password("wrong", h)
    assert not verify_password("s3cret", "garbage")


def test_token_roundtrip_and_expiry():
    p = Principal(1, "u", "hr", "employee")
    tok, _ = create_access_token(p, "k" * 32, 5)
    assert decode_access_token(tok, "k" * 32)["usr"] == "u"
    expired, _ = create_access_token(p, "k" * 32, -1)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired, "k" * 32)
    with pytest.raises(jwt.InvalidSignatureError):
        decode_access_token(tok, "other" * 8)


def test_acl_tag_validation_blocks_injection():
    assert validate_acl_tags(["public", "dept:hr", "public"]) == ["dept:hr", "public"]
    for bad in (['public" or 1==1 or "'], ["dept:HR"], ["admin"], []):
        with pytest.raises(ValueError):
            validate_acl_tags(bad)


def test_principal_access():
    p = Principal(1, "u", "hr", "employee")
    assert p.can_access(["public"]) and p.can_access(["dept:hr", "dept:it"])
    assert not p.can_access(["dept:finance"]) and not p.can_access([])


def test_bm25_confidence_does_not_depend_on_corpus_size():
    """置信度不得依赖语料规模:仅有 1 条 FAQ 时,无关问题的置信度也不能被推高到阈值以上。"""
    one = BM25([tokenize("如何申请年假")])
    r = one.search("今天天气怎么样", top_k=1)
    assert not r or r[0].confidence < 0.3
    assert one.search("如何申请年假", top_k=1)[0].confidence > 0.9


def _faq(db, q, a, tags=("public",), cat="general"):
    e = FaqEntry(question=q, answer=a, acl_tags=list(tags), category=cat)
    db.add(e)
    db.commit()


def test_faq_matcher_threshold_acl_and_margin(container):
    from sqlalchemy import delete

    from knowbase.db.models import FaqEntry as F

    emp = Principal(1, "e", "it", "employee")
    hr = Principal(2, "h", "hr", "employee")
    m = FaqMatcher(threshold=0.6, margin=0.1)
    with container.sf() as db:
        db.execute(delete(F))
        _faq(db, "怎么重置OA密码", "去自助门户", ["public"])
        _faq(db, "年度调薪什么时候公布", "四月", ["dept:hr"])
        assert m.match(db, "怎么重置OA密码", emp).answer == "去自助门户"
        assert m.match(db, "今天中午吃什么", emp) is None  # 无关问题不命中
        assert m.match(db, "年度调薪什么时候公布", emp) is None  # 无权 FAQ 不可命中
        assert m.match(db, "年度调薪什么时候公布", hr).answer == "四月"
        # 运营新增 FAQ 后,索引自动刷新
        _faq(db, "报销发票抬头是什么", "示例科技有限公司", ["public"])
        assert m.match(db, "报销发票抬头是什么", emp).answer == "示例科技有限公司"
