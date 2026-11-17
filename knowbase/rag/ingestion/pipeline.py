"""文档入库管线:上传登记(校验/去重/版本) -> 解析 -> 切分 -> 写入索引 -> 切换版本。

关键设计:
    - 内容校验和(SHA-256)相同则幂等,不重复入库;
    - 同标题再次上传则 version+1;新版本入库成功后才归档旧版本并移出索引,
      切换期间旧版本继续服务,员工不会遇到"制度查不到"的空窗;
    - 任一步失败则新文档标记 failed 并记录原因、清理半成品,旧版本不受影响;
    - 磁盘上的文件名用随机 UUID,不使用用户提供的文件名(防路径穿越)。

扩展提示:
    默认用 FastAPI BackgroundTasks 执行 ingest(),部署简单但进程重启会中断进行中的任务
    (文档停在 pending,可调 /reindex 重跑)。大批量或高可靠场景请改为任务队列(Celery/RQ/Arq)。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from knowbase.core.security import validate_acl_tags
from knowbase.db.models import DocParent, Document
from knowbase.rag.ingestion.chunking import build_chunks
from knowbase.rag.ingestion.loaders import SUPPORTED_EXTENSIONS, load_file
from knowbase.rag.retrievers import ChunkRecord, Retriever

log = logging.getLogger(__name__)
_CAT_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")


@dataclass
class RegisterResult:
    """登记结果:文档记录,以及是否为"与现行版本内容完全相同"的重复上传。"""

    document: Document
    duplicate: bool  # True: 内容与现行版本完全相同,未新建版本


class IngestionService:
    """入库服务。对外三个动作:登记上传、入库(可后台执行)、下线/重建。"""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        retriever: Retriever,
        upload_dir: str,
        parent_size: int,
        child_size: int,
        overlap: int,
        max_upload_mb: int,
    ) -> None:
        self._sf, self.retriever, self.upload_dir = session_factory, retriever, upload_dir
        self.parent_size, self.child_size, self.overlap = parent_size, child_size, overlap
        self.max_bytes = max_upload_mb * 1024 * 1024
        os.makedirs(upload_dir, exist_ok=True)

    # ---- 1. 登记 ----
    def register_upload(
        self, content: bytes, filename: str, title: str, category: str, acl_tags: list[str], user_id: int | None
    ) -> RegisterResult:
        """校验并登记一次上传,返回文档记录;不做耗时的解析与向量化。

        Raises:
            ValueError: 扩展名不支持、文件为空/超限、分类或权限标签不合法。
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型 {ext},支持: {', '.join(SUPPORTED_EXTENSIONS)}")
        if not content or len(content) > self.max_bytes:
            raise ValueError(f"文件为空或超过 {self.max_bytes // 1024 // 1024}MB 上限")
        if not _CAT_RE.match(category):
            raise ValueError("category 只能包含小写字母/数字/下划线/连字符,长度<=32")
        acl_tags = validate_acl_tags(acl_tags)
        checksum = hashlib.sha256(content).hexdigest()

        with self._sf() as db:
            latest = db.scalars(
                select(Document).where(Document.title == title, Document.status != "archived").order_by(Document.version.desc())
            ).first()
            if latest and latest.checksum == checksum and latest.status in ("indexed", "pending"):
                # 内容没变,但权限/分类可能变了:只更新元数据;已入库的需同步重建索引,
                # 因为向量库里的 acl_tags 是冗余副本,不同步就会出现"改了权限却仍按旧权限检索"
                if sorted(latest.acl_tags) != acl_tags or latest.category != category:
                    latest.acl_tags, latest.category = acl_tags, category
                    db.commit()
                    if latest.status == "indexed":
                        self._reindex_existing(db, latest)
                return RegisterResult(latest, duplicate=True)
            version = (latest.version + 1) if latest else 1
            safe_name = f"{uuid.uuid4().hex}{ext}"  # 不使用用户文件名落盘,防路径穿越
            path = os.path.join(self.upload_dir, safe_name)
            with open(path, "wb") as f:
                f.write(content)
            doc = Document(
                title=title,
                filename=os.path.basename(filename),
                category=category,
                version=version,
                checksum=checksum,
                acl_tags=acl_tags,
                status="pending",
                storage_path=path,
                uploaded_by=user_id,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            return RegisterResult(doc, duplicate=False)

    # ---- 2. 入库(可在后台任务中执行)----
    def ingest(self, document_id: int) -> Document:
        """解析、切分、写入索引。失败不抛异常而是把文档标记为 failed(便于后台任务安全执行)。"""
        with self._sf() as db:
            doc = db.get(Document, document_id)
            if doc is None:
                raise ValueError("文档不存在")
            try:
                sections = load_file(doc.storage_path)
                parents = build_chunks(sections, doc.id, doc.version, self.parent_size, self.child_size, self.overlap)
                if not parents:
                    raise ValueError("切分后没有有效内容")
                records: list[ChunkRecord] = []
                for p in parents:
                    db.add(
                        DocParent(
                            parent_id=p.parent_id,
                            document_id=doc.id,
                            content=p.content,
                            section=p.section,
                            page=p.page,
                            seq=p.seq,
                        )
                    )
                    records.extend(
                        ChunkRecord(cid, p.parent_id, doc.id, text, doc.category, list(doc.acl_tags), doc.version)
                        for cid, text in p.children
                    )
                db.flush()
                # 顺序很重要:先写向量库,成功后才把状态置为 indexed;中途失败则整体回滚并清理
                self.retriever.index_chunks(records)
                doc.status, doc.chunk_count, doc.error = "indexed", len(records), None
                db.commit()
            except Exception as e:  # noqa: BLE001
                db.rollback()
                self.retriever.delete_document(document_id)  # 清理向量库中可能已写入的半成品
                db.execute(delete(DocParent).where(DocParent.document_id == document_id))
                doc = db.get(Document, document_id)
                doc.status, doc.error = "failed", str(e)[:1000]
                db.commit()
                log.exception("文档入库失败 id=%s", document_id)
                return doc
            # 3. 新版本就绪 -> 归档旧版本
            olds = db.scalars(
                select(Document).where(Document.title == doc.title, Document.id != doc.id, Document.status == "indexed")
            ).all()
            for old in olds:
                self._archive(db, old)
            db.commit()
            log.info("文档入库完成 id=%s title=%s v%s chunks=%s", doc.id, doc.title, doc.version, doc.chunk_count)
            return doc

    # ---- 辅助 ----
    def _archive(self, db: Session, doc: Document) -> None:
        """归档:移出索引、删除父块、状态置为 archived(保留文档记录以便追溯)。"""
        self.retriever.delete_document(doc.id)
        db.execute(delete(DocParent).where(DocParent.document_id == doc.id))
        doc.status, doc.chunk_count = "archived", 0

    def delete_document(self, document_id: int) -> bool:
        """下线文档:归档并删除磁盘上的原文件。文档不存在返回 False。"""
        with self._sf() as db:
            doc = db.get(Document, document_id)
            if not doc:
                return False
            self._archive(db, doc)
            if doc.storage_path and os.path.exists(doc.storage_path):
                os.remove(doc.storage_path)
            db.commit()
            return True

    def _reindex_existing(self, db: Session, doc: Document) -> None:
        """权限/分类变更后重建该文档的索引(向量库里的 acl_tags 是冗余副本,必须同步)。"""
        db.execute(delete(DocParent).where(DocParent.document_id == doc.id))
        self.retriever.delete_document(doc.id)
        doc.status = "pending"
        db.commit()
        self.ingest(doc.id)

    def reindex(self, document_id: int) -> Document | None:
        """重建索引:清理旧块后重新入库。已归档或不存在返回 None。"""
        with self._sf() as db:
            doc = db.get(Document, document_id)
            if not doc or doc.status == "archived":
                return None
            db.execute(delete(DocParent).where(DocParent.document_id == doc.id))
            self.retriever.delete_document(doc.id)
            doc.status = "pending"
            db.commit()
        return self.ingest(document_id)
