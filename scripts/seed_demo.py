"""初始化演示数据: 部门、账号、FAQ、示例制度文档(同步入库)。

用法:  python -m scripts.seed_demo
密码取自环境变量 KNOWBASE_SEED_PASSWORD;未设置则随机生成并仅在标准错误输出一次。
仅用于开发与演示,请勿在生产环境使用演示账号。
"""

from __future__ import annotations

import csv
import json
import os
import secrets
import sys

from sqlalchemy import select

from knowbase.container import Container, build_container
from knowbase.core.config import get_settings
from knowbase.core.security import hash_password
from knowbase.db.models import Department, FaqEntry, User

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS = [  # username, department, role, display
    ("admin", "it", "admin", "系统管理员"),
    ("hr_amy", "hr", "kb_admin", "HR-Amy(知识库管理员)"),
    ("it_bob", "it", "employee", "IT-Bob"),
    ("fin_carl", "finance", "employee", "财务-Carl"),
]
DEPARTMENTS = {"hr": "人力资源部", "it": "信息技术部", "finance": "财务部"}


def seed(c: Container, password: str) -> dict:
    """写入演示用的部门、账号、FAQ 与示例制度文档(同步入库)。幂等:重复执行不会产生重复数据。

    Returns:
        各类数据的写入统计。
    """
    with c.sf() as db:
        for code, name in DEPARTMENTS.items():
            if not db.scalars(select(Department).where(Department.code == code)).first():
                db.add(Department(code=code, name=name))
        db.commit()
        depts = {d.code: d.id for d in db.scalars(select(Department))}
        for username, dept, role, display in USERS:
            if not db.scalars(select(User).where(User.username == username)).first():
                db.add(
                    User(
                        username=username,
                        display_name=display,
                        role=role,
                        department_id=depts[dept],
                        password_hash=hash_password(password, c.settings.password_hash_iterations),
                    )
                )
        db.commit()

        if not db.scalars(select(FaqEntry)).first():
            with open(os.path.join(ROOT, "data", "faq_seed.csv"), encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    db.add(
                        FaqEntry(
                            question=row["question"],
                            answer=row["answer"],
                            category=row["category"],
                            acl_tags=[t for t in row["acl_tags"].split(";") if t],
                        )
                    )
            db.commit()

    docs_dir = os.path.join(ROOT, "data", "sample_docs")
    with open(os.path.join(docs_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    results = []
    for item in manifest:
        with open(os.path.join(docs_dir, item["file"]), "rb") as f:
            res = c.ingestion.register_upload(f.read(), item["file"], item["title"], item["category"], item["acl_tags"], None)
        doc = res.document if res.duplicate else c.ingestion.ingest(res.document.id)
        results.append((doc.title, doc.status, doc.chunk_count))
    return {"documents": results}


def main() -> None:
    """命令行入口:读取密码来源并执行 seed。未提供密码时随机生成,仅向标准错误输出一次,不落盘。"""
    c = build_container(get_settings())
    password = os.getenv("KNOWBASE_SEED_PASSWORD") or secrets.token_urlsafe(9)
    out = seed(c, password)
    print("演示数据已就绪:")
    for t, s, n in out["documents"]:
        print(f"  - {t}: {s} ({n} 个子块)")
    print("账号: " + ", ".join(u[0] for u in USERS))
    if not os.getenv("KNOWBASE_SEED_PASSWORD"):
        print(f"密码(仅显示这一次): {password}", file=sys.stderr)


if __name__ == "__main__":
    main()
