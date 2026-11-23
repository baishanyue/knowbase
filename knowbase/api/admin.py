"""运营/管理后台接口。

分区与权限:
    - FAQ、文档、运营分析:知识库管理员(kb_admin)及以上;
    - 用户、审计:仅系统管理员(admin)。

约束:
    - 所有变更类操作(增删改、上传、导入、停用账号)都写审计日志;
    - 所有 acl_tags 入库前都经过白名单校验,不信任调用方传入的权限标签;
    - 列表接口一律带上限,防止一次拉取全表。
"""

from __future__ import annotations

import csv
import io
from datetime import timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select

from knowbase.api.deps import get_container, require_admin, require_kb_admin
from knowbase.api.schemas import DocumentOut, FaqIn, FaqOut, UserIn, UserOut
from knowbase.container import Container
from knowbase.core.security import Principal, hash_password, validate_acl_tags
from knowbase.db.models import AuditLog, ChatTurn, Department, Document, FaqEntry, Feedback, User, utcnow
from knowbase.db.session import session_scope
from knowbase.services.audit import audit

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _acl(tags: list[str]) -> list[str]:
    """校验权限标签,不合法时转换为 422 响应。"""
    try:
        return validate_acl_tags(tags)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


def _csv_cell(value: object) -> str:
    """防 CSV 公式注入:以 = + - @ 开头的单元格前加单引号,避免 Excel 把用户提问当公式执行。

    导出的内容来自员工提问(不可信输入),导出文件会被运营人员用 Excel 打开,必须处理。
    """
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


# ---------------------------------------------------------------- FAQ
@router.get("/faq", response_model=list[FaqOut])
def list_faq(
    category: str | None = None,
    q: str | None = None,
    limit: int = 100,
    offset: int = 0,
    _: Principal = Depends(require_kb_admin),
    c: Container = Depends(get_container),
):
    """分页查询 FAQ,可按分类与问题关键字过滤(单页最多 500 条)。"""
    with c.sf() as db:
        stmt = select(FaqEntry).order_by(FaqEntry.id.desc()).limit(min(limit, 500)).offset(offset)
        if category:
            stmt = stmt.where(FaqEntry.category == category)
        if q:
            stmt = stmt.where(FaqEntry.question.contains(q))
        return list(db.scalars(stmt))


@router.post("/faq", response_model=FaqOut, status_code=201)
def create_faq(body: FaqIn, p: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """新增一条 FAQ。"""
    with session_scope(c.sf) as db:
        e = FaqEntry(**body.model_dump(exclude={"acl_tags"}), acl_tags=_acl(body.acl_tags), created_by=p.user_id)
        db.add(e)
        db.flush()
        db.refresh(e)
    audit(c, "faq.create", p, {"id": e.id})
    return e


@router.put("/faq/{faq_id}", response_model=FaqOut)
def update_faq(faq_id: int, body: FaqIn, p: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """整体更新一条 FAQ(含权限标签)。"""
    with session_scope(c.sf) as db:
        e = db.get(FaqEntry, faq_id)
        if not e:
            raise HTTPException(404, "FAQ 不存在")
        for k, v in body.model_dump(exclude={"acl_tags"}).items():
            setattr(e, k, v)
        e.acl_tags = _acl(body.acl_tags)
        db.flush()
        db.refresh(e)
    audit(c, "faq.update", p, {"id": faq_id})
    return e


@router.delete("/faq/{faq_id}", status_code=204)
def delete_faq(faq_id: int, p: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """删除一条 FAQ。"""
    with session_scope(c.sf) as db:
        e = db.get(FaqEntry, faq_id)
        if not e:
            raise HTTPException(404, "FAQ 不存在")
        db.delete(e)
    audit(c, "faq.delete", p, {"id": faq_id})


@router.post("/faq/import")
async def import_faq(
    file: UploadFile = File(...), p: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)
):
    """CSV 批量导入。表头: question,answer,category,acl_tags(多个用分号分隔,缺省 public)。

    逐行校验:单行失败不影响其他行,失败行号与原因在响应中返回,便于运营修正后重导。
    """
    raw = (await file.read()).decode("utf-8-sig", errors="replace")  # utf-8-sig 兼容 Excel 导出的 BOM
    ok, bad = 0, []
    with session_scope(c.sf) as db:
        for i, row in enumerate(csv.DictReader(io.StringIO(raw)), 2):  # 行号从 2 起(第 1 行是表头)
            try:
                q, a = (
                    (row.get("question") or row.get("问题") or "").strip(),
                    (row.get("answer") or row.get("答案") or "").strip(),
                )
                cat = (row.get("category") or "general").strip() or "general"
                tags = [t for t in (row.get("acl_tags") or "public").split(";") if t.strip()]
                FaqIn(question=q, answer=a, category=cat, acl_tags=tags)  # 复用接口层的同一套校验
                db.add(FaqEntry(question=q, answer=a, category=cat, acl_tags=validate_acl_tags(tags), created_by=p.user_id))
                ok += 1
            except Exception as e:  # noqa: BLE001 逐行容错,错误信息截断后返回
                bad.append({"line": i, "error": str(e)[:120]})
    audit(c, "faq.import", p, {"ok": ok, "failed": len(bad)})
    return {"imported": ok, "failed": bad}


# ---------------------------------------------------------------- 文档
@router.post("/documents", response_model=DocumentOut, status_code=202)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(..., max_length=200),
    category: str = Form("general"),
    acl_tags: str = Form("public", description="逗号分隔,如 public 或 dept:hr,dept:finance"),
    p: Principal = Depends(require_kb_admin),
    c: Container = Depends(get_container),
):
    """上传文档。

    同步部分只做校验、去重、登记版本并立刻返回 202;耗时的解析/切块/向量化放到后台任务,
    调用方通过 GET /documents/{id} 轮询 status。内容与已有文档完全相同(哈希一致)时视为重复,不重复入库。
    """
    content = await file.read()
    try:
        res = c.ingestion.register_upload(
            content, file.filename or "upload", title.strip(), category, [t for t in acl_tags.split(",") if t.strip()], p.user_id
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if not res.duplicate:
        background.add_task(c.ingestion.ingest, res.document.id)
    audit(
        c,
        "document.upload",
        p,
        {"id": res.document.id, "title": title, "version": res.document.version, "duplicate": res.duplicate},
    )
    return res.document


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(status: str | None = None, _: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """列出最近 200 个文档,可按入库状态过滤。"""
    with c.sf() as db:
        stmt = select(Document).order_by(Document.id.desc()).limit(200)
        if status:
            stmt = stmt.where(Document.status == status)
        return list(db.scalars(stmt))


@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, _: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """查询单个文档及其入库状态。"""
    with c.sf() as db:
        d = db.get(Document, doc_id)
        if not d:
            raise HTTPException(404, "文档不存在")
        return d


@router.post("/documents/{doc_id}/reindex", response_model=DocumentOut)
def reindex_document(doc_id: int, p: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """重建索引:用于入库中断/失败后的重跑,或模型、切块参数调整后的刷新。"""
    d = c.ingestion.reindex(doc_id)
    if d is None:
        raise HTTPException(404, "文档不存在或已归档")
    audit(c, "document.reindex", p, {"id": doc_id})
    return d


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int, p: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """下线文档:同时清理其向量索引,之后不再被检索到。"""
    if not c.ingestion.delete_document(doc_id):
        raise HTTPException(404, "文档不存在")
    audit(c, "document.delete", p, {"id": doc_id})


# ---------------------------------------------------------------- 运营分析
@router.get("/stats")
def stats(days: int = 7, _: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)):
    """近 N 天(1~90)的问答概览:各路由占比、回答率、拒答率、满意度、平均耗时。"""
    since = utcnow() - timedelta(days=max(1, min(days, 90)))
    with c.sf() as db:
        by_route = dict(
            db.execute(select(ChatTurn.route, func.count()).where(ChatTurn.created_at >= since).group_by(ChatTurn.route)).all()
        )
        total = sum(by_route.values())
        fb = dict(
            db.execute(
                select(Feedback.rating, func.count())
                .join(ChatTurn, ChatTurn.id == Feedback.turn_id)
                .where(ChatTurn.created_at >= since)
                .group_by(Feedback.rating)
            ).all()
        )
        avg_ms = db.scalar(select(func.avg(ChatTurn.latency_ms)).where(ChatTurn.created_at >= since)) or 0
    answered = by_route.get("faq", 0) + by_route.get("rag", 0)
    return {
        "days": days,
        "total_questions": total,
        "by_route": by_route,
        "answer_rate": round(answered / total, 3) if total else None,  # 被知识库有效回答的比例
        "refusal_rate": round(by_route.get("refused", 0) / total, 3) if total else None,
        "thumbs_up": fb.get(1, 0),
        "thumbs_down": fb.get(-1, 0),
        "avg_latency_ms": int(avg_ms),
    }


def _gap_rows(c: Container, days: int, limit: int) -> tuple[list, list]:
    """查询知识缺口两类数据:被拒答的问题、被点踩的问题(均按出现次数倒序)。"""
    since = utcnow() - timedelta(days=max(1, min(days, 365)))
    cap = max(1, min(limit, 200))
    with c.sf() as db:
        refused = db.execute(
            select(ChatTurn.question, func.count().label("n"))
            .where(ChatTurn.route == "refused", ChatTurn.created_at >= since)
            .group_by(ChatTurn.question)
            .order_by(func.count().desc())
            .limit(cap)
        ).all()
        downvoted = db.execute(
            select(ChatTurn.question, ChatTurn.route, func.count().label("n"))
            .join(Feedback, Feedback.turn_id == ChatTurn.id)
            .where(Feedback.rating == -1, ChatTurn.created_at >= since)
            .group_by(ChatTurn.question, ChatTurn.route)
            .order_by(func.count().desc())
            .limit(cap)
        ).all()
    return refused, downvoted


@router.get("/gaps")
def knowledge_gaps(
    days: int = 30, limit: int = 50, _: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)
):
    """知识缺口:被拒答次数最多的问题 + 点踩最多的问题。运营据此补文档/补 FAQ,形成闭环。"""
    refused, downvoted = _gap_rows(c, days, limit)
    return {
        "refused": [{"question": q, "count": n} for q, n in refused],
        "downvoted": [{"question": q, "route": r, "count": n} for q, r, n in downvoted],
    }


@router.get("/gaps/export")
def export_gaps(
    days: int = 30, limit: int = 200, p: Principal = Depends(require_kb_admin), c: Container = Depends(get_container)
):
    """导出知识缺口 CSV,供运营在 Excel 中排期补文档。

    带 UTF-8 BOM,Excel 直接打开中文不乱码;单元格做了公式注入防护(见 `_csv_cell`)。
    导出属于批量读取员工提问,同样写审计日志。
    """
    refused, downvoted = _gap_rows(c, days, limit)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["类型", "问题", "路由", "次数"])
    for q, n in refused:
        w.writerow(["被拒答", _csv_cell(q), "refused", n])
    for q, r, n in downvoted:
        w.writerow(["被点踩", _csv_cell(q), r, n])
    audit(c, "gaps.export", p, {"days": days, "rows": len(refused) + len(downvoted)})
    return Response(
        content="﻿" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="knowledge_gaps.csv"'},
    )


# ---------------------------------------------------------------- 用户 / 部门 / 审计(仅系统管理员)
@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserIn, p: Principal = Depends(require_admin), c: Container = Depends(get_container)):
    """创建用户;部门不存在时自动创建。生产环境建议对接企业身份源(SSO)而非手工建账号。"""
    with session_scope(c.sf) as db:
        dept = db.scalars(select(Department).where(Department.code == body.department)).first()
        if not dept:
            dept = Department(code=body.department, name=body.department)
            db.add(dept)
            db.flush()
        if db.scalars(select(User).where(User.username == body.username)).first():
            raise HTTPException(409, "用户名已存在")
        u = User(
            username=body.username,
            display_name=body.display_name,
            role=body.role,
            department_id=dept.id,
            password_hash=hash_password(body.password, c.settings.password_hash_iterations),
        )
        db.add(u)
        db.flush()
        db.refresh(u)
    audit(c, "user.create", p, {"username": body.username, "role": body.role, "department": body.department})
    return u


@router.patch("/users/{user_id}/active", response_model=UserOut)
def set_user_active(user_id: int, active: bool, p: Principal = Depends(require_admin), c: Container = Depends(get_container)):
    """启用/停用账号。停用后已签发的令牌在下次请求时即失效(鉴权会重新校验账号状态)。"""
    with session_scope(c.sf) as db:
        u = db.get(User, user_id)
        if not u:
            raise HTTPException(404, "用户不存在")
        if u.id == p.user_id and not active:
            raise HTTPException(400, "不能停用自己")  # 防止唯一管理员把自己锁在系统外
        u.is_active = active
        db.flush()
        db.refresh(u)
    audit(c, "user.set_active", p, {"user_id": user_id, "active": active})
    return u


@router.get("/audit")
def list_audit(
    action: str | None = None,
    username: str | None = None,
    limit: int = 100,
    _: Principal = Depends(require_admin),
    c: Container = Depends(get_container),
):
    """查询审计日志(最新在前,单次最多 500 条),可按动作与用户名过滤。"""
    with c.sf() as db:
        stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 500))
        if action:
            stmt = stmt.where(AuditLog.action == action)
        if username:
            stmt = stmt.where(AuditLog.username == username)
        return [
            {
                "id": a.id,
                "ts": a.ts.isoformat(),
                "request_id": a.request_id,
                "username": a.username,
                "action": a.action,
                "detail": a.detail,
                "ip": a.ip,
            }
            for a in db.scalars(stmt)
        ]
