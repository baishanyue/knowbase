"""RAG 质量评估与发布门禁。

用法:
  python -m scripts.evaluate_rag                 # 仅评估检索层(不调 LLM,适合在 CI 中每次提交运行)
  python -m scripts.evaluate_rag --e2e           # 端到端:调用配置的 LLM,检查答案是否包含期望关键词
  python -m scripts.evaluate_rag --min-hit3 0.9  # 自定义门禁阈值,不达标退出码为 1

硬指标:
  1) 检索命中率 hit@1 / hit@3(期望文档是否出现在引用中);
  2) 权限泄露数(必须为 0:无权用户的检索结果中不得出现受限文档);
  3) 拒答准确率(库中没有或无权访问的问题是否正确拒答)。
忠实度、答案相关性等需要评审模型的指标(如 RAGAS)可在发布前作为补充评估。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from knowbase.container import build_container
from knowbase.core.config import get_settings
from knowbase.core.security import Principal
from knowbase.db.models import User
from scripts.seed_demo import ROOT, seed


def load_cases(path: str) -> list[dict]:
    """读取黄金集(JSONL:每行一个用例,含问题、提问人、期望文档/是否应拒答)。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    """运行评估并按门禁阈值返回退出码(0 通过,1 不通过),供 CI 直接使用。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(ROOT, "data", "eval", "golden.jsonl"))
    ap.add_argument("--e2e", action="store_true", help="端到端评估(需要可用的 LLM)")
    ap.add_argument("--min-hit3", type=float, default=0.85)
    ap.add_argument("--min-refusal", type=float, default=0.80)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="knowbase_eval_")
    base = get_settings()
    settings = base.model_copy(
        update={
            "database_url": f"sqlite:///{tmp}/eval.db",
            "upload_dir": f"{tmp}/uploads",
            "password_hash_iterations": 1000,
            "env": "test",
        }
    )

    class _NoLLM:  # 仅评估检索层时不需要 LLM;策略固定为"直接检索"
        model = "none"

        def complete(self, *a, **k):
            return ""

        def stream(self, *a, **k):
            return iter(())

    c = build_container(settings, llm=None if args.e2e else _NoLLM())
    seed(c, "eval-password")

    with c.sf() as db:
        users = {u.username: Principal(u.id, u.username, u.department.code, u.role) for u in db.query(User).all()}

    hit1 = hit3 = n_exp = leaks = n_refuse = refuse_ok = kw_ok = n_kw = 0
    failures: list[str] = []
    for case in load_cases(args.golden):
        p = users[case["user"]]
        ctxs = c.chat.retrieval.retrieve(p, case["question"], "直接检索")
        titles = [x.title for x in ctxs]
        if args.verbose:
            print(f"[{case['id']}] {case['user']}: {case['question']} -> {[(x.title, round(x.score, 2)) for x in ctxs]}")
        if exp := case.get("expect_title"):
            n_exp += 1
            hit1 += bool(titles[:1] == [exp])
            hit3 += exp in titles[:3]
            if exp not in titles[:3]:
                failures.append(f"{case['id']}: 期望 {exp},实际 {titles}")
        for forbid in case.get("forbid_titles", []):
            if forbid in titles:
                leaks += 1
                failures.append(f"{case['id']}: 权限泄露 -> {forbid}")
        if case.get("should_refuse"):
            n_refuse += 1
            ok = not ctxs
            refuse_ok += ok
            if not ok:
                failures.append(f"{case['id']}: 应当拒答,实际检索到 {titles}")
        if args.e2e and case.get("expect_keywords"):
            r = c.chat.answer(p, case["question"])
            n_kw += 1
            good = all(k in r["answer"] for k in case["expect_keywords"])
            kw_ok += good
            if not good:
                failures.append(f"{case['id']}: 答案缺少关键词 {case['expect_keywords']}: {r['answer'][:80]!r}")

    def pct(a: int, b: int) -> float:
        return (a / b) if b else 1.0

    m3, mr = pct(hit3, n_exp), pct(refuse_ok, n_refuse)
    print(f"检索 hit@1 = {pct(hit1, n_exp):.2%}   hit@3 = {m3:.2%}   ({n_exp} 题)")
    print(f"权限泄露   = {leaks}   (必须为 0)")
    print(f"拒答准确率 = {mr:.2%}   ({n_refuse} 题)")
    if args.e2e:
        print(f"答案关键词覆盖 = {pct(kw_ok, n_kw):.2%}   ({n_kw} 题)")
    for f in failures:
        print("  ✗", f)
    passed = leaks == 0 and m3 >= args.min_hit3 and mr >= args.min_refusal
    print("门禁:", "通过 ✅" if passed else "未通过 ❌")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
