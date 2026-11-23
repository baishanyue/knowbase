# 贡献指南

## 本地开发

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt ruff pre-commit
pre-commit install
make test      # 单元与接口测试
make eval      # RAG 质量门禁(检索层)
```

## 代码规范

- Python ≥ 3.11,全量类型注解;使用 `ruff` 做 lint 与格式化(配置见 `pyproject.toml`)。
- 分层约定:`api`(协议与鉴权) → `services`(业务编排) → `rag` / `db` / `cache`(能力与存储)。下层不得依赖上层。
- 注释规范(中文,Google 风格):
  - 每个模块有模块级文档字符串:它解决什么问题、关键设计决策、使用约束;
  - 公共类/函数写文档字符串:用途、参数(Args)、返回(Returns)、异常(Raises);简单到一目了然的私有函数可省略;
  - 行内注释只解释"为什么"(安全考量、边界条件、取舍),不复述代码在做什么;
  - 涉及权限、脱敏、限流、事务边界的代码必须注明安全含义;
  - 禁止用注释保留被废弃的代码;未完成事项写进 Issue,不留 TODO 在代码里。
- 任何涉及权限、检索过滤、入库的改动必须带测试,且 `make eval` 的权限泄露数必须为 0。
- 不得提交密钥、真实员工数据、内部域名或内部文档;示例数据一律使用虚构内容。

## 提交与评审

- 提交信息使用 Conventional Commits(`feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`)。
- 合并前需通过 CI(测试 + 质量门禁),并至少一位 CODEOWNERS 评审。
