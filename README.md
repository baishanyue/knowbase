# 知底 KnowBase · 企业知识库智能问答平台

> **知根知底,有据可查。** 员工用自然语言查询制度、流程、IT/HR 常见问题:答案**只依据公司文档**,
> 每个结论带**出处引用**,查不到就**拒答并转人工**,并按**部门权限**隔离机密文档。

![python](https://img.shields.io/badge/python-3.11+-blue) ![fastapi](https://img.shields.io/badge/FastAPI-009688) ![tests](https://img.shields.io/badge/tests-51%20passed-brightgreen) ![license](https://img.shields.io/badge/license-TBD-lightgrey)

## 为什么做它

把制度文档直接丢给大模型有三个企业级风险:**会编造**、**会越权泄露**、**无法运营**。知底针对性地解决:

| 业务价值 | 实现方式 |
|---|---|
| 减少 HR / IT 重复答疑 | FAQ 高频问题优先命中(毫秒级、零模型成本),未命中再走文档检索问答 |
| 答案可信、可追溯 | 答案标注 `[编号]` 引用,可定位到 文档 · 章节 · 页码 · 版本 |
| 机密制度不外泄 | 文档/FAQ 带部门权限标签,检索层强制过滤 + 取回二次校验 |
| 知识库持续变好 | 拒答与点踩沉淀为"知识缺口",运营据此补文档与 FAQ,可导出 CSV |
| 满足合规审计 | 登录、变更、越权尝试、数据导出均写审计日志;每轮问答留痕;提问入口脱敏个人信息 |
| 多轮对话自然 | 追问("那病假呢")自动补全为完整问题再检索 |
| 可运维 | 请求 ID 贯穿日志、Prometheus `/metrics`、K8s 探针、生产配置强校验 |

## 核心亮点

- **两级路由,成本由低到高**:寒暄规则 → FAQ(自研 BM25 归一化置信度)→ RAG,命中即返回;
- **拒答是一等公民**:相关度低于阈值不带空上下文调用模型,杜绝"硬编";
- **双重权限校验**:检索层按 `acl_tags` 过滤,取父块时按关系库再校验一次,发布门禁要求**权限泄露数 = 0**;
- **文档版本无感切换**:新版本入库成功后才归档旧版本,失败回退不影响线上;
- **可插拔**:LLM / 检索器 / 重排器 / KV 都是协议,测试与 CI 不需要 GPU、向量库或真实模型。

## 技术栈

| 层 | 选型 |
|---|---|
| 服务 | Python 3.11 · FastAPI · Uvicorn · Pydantic v2 · SSE / WebSocket 流式 |
| 数据 | MySQL(业务数据)· SQLAlchemy 2 · Redis(限流、令牌吊销) |
| 检索 | Milvus(稠密 + 稀疏混合检索)· BGE-M3 向量化 · bge-reranker 重排 · jieba + BM25(FAQ) |
| 生成 | 任意 OpenAI 兼容接口(公有云网关 / vLLM / Ollama) |
| 工程 | Docker Compose · GitHub Actions · pytest(51 用例)· ruff · pre-commit(gitleaks) |
| 可观测 | 请求 ID · JSON 日志 · Prometheus `/metrics` · `/health/live` `/health/ready` |

## 架构

```
员工 ─▶ Web 前端 / 企微·钉钉机器人 ─▶ FastAPI(JWT 鉴权 · 限流 · 请求ID · 审计)
                                           │
                               ChatService(问答编排,流式事件)
   ① 寒暄规则 ─▶ ② FAQ 层(BM25 + 部门权限过滤) ──命中──▶ 直接返回
                                           │ 未命中
   ③ 检索策略选择(直接 / HyDE / 子查询 / 回溯)
   ④ 混合检索(Milvus,检索前强制权限过滤)
   ⑤ 取父块(MySQL,二次校验权限与状态) ─▶ ⑥ 重排 ─▶ ⑦ 相关度阈值
                                           │ 有资料                     │ 无可靠资料
   ⑧ LLM 严格依据资料生成 + 引用编号            拒答 + 人工渠道(记为知识缺口)
                                           │
   落库: 问答记录 · 反馈 · 审计日志            运营后台: FAQ · 文档 · 统计 · 缺口 · 用户 · 审计
```

目录结构:

```
knowbase/
  api/        接口层:鉴权依赖、问答、后台、健康检查
  services/   业务编排:问答、FAQ、审计
  rag/        检索能力:意图/策略、检索与重排、提示词、LLM 客户端、入库管线
  db/ cache/  存储:模型与会话、KV(Redis/内存)
  core/       配置、日志、安全(密码/JWT/权限主体)
scripts/      演示数据初始化、RAG 质量评估(发布门禁)
data/         示例制度文档(虚构)、FAQ 种子、评估黄金集
frontend/     演示用单页前端
```

## 快速开始(本地,无需 GPU / 向量库 / 模型文件)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # 配置 KNOWBASE_LLM_BASE_URL / KNOWBASE_LLM_API_KEY / KNOWBASE_LLM_MODEL
make test                     # 单元与接口测试
make eval                     # RAG 质量门禁
KNOWBASE_SEED_PASSWORD='<自行设置>' make demo   # 灌入示例数据并启动: http://127.0.0.1:8080
```

演示账号(密码为你设置的 `KNOWBASE_SEED_PASSWORD`;未设置则随机生成并仅显示一次):

| 账号 | 部门 | 角色 | 用于验证 |
|---|---|---|---|
| `it_bob` | it | 员工 | 询问薪酬带宽 → 拒答(文档仅 HR 可见) |
| `hr_amy` | hr | 知识库管理员 | 同一问题可答;可上传文档、维护 FAQ |
| `fin_carl` | finance | 员工 | 可查财务内部预算审批文档 |
| `admin` | it | 系统管理员 | 用户管理、审计日志 |

`memory` 检索后端为纯 BM25,仅用于开发、演示与 CI;生产请使用 `milvus` 后端。示例数据均为虚构。

## 生产部署

```bash
cp .env.example .env      # 必改:KNOWBASE_SECRET_KEY、各类密码、LLM 配置
# 将 bge-m3 与 bge-reranker-large 模型目录放到 ./models/
docker compose up -d --build
```

`KNOWBASE_ENV=prod` 时启动强校验:密钥长度、非 SQLite、非内存检索、必须配置 LLM;并关闭 `/docs`。

## 权限模型

- 用户属于一个部门,拥有标签 `public` 与 `dept:<部门>`;文档/FAQ 带 `acl_tags`,**有交集才可见**。
- 过滤在检索层强制执行(标签来自服务端身份并经白名单校验);取父块时按关系库再校验一次。
- 修改文档权限自动重建索引;上传新版本后旧版本归档并移出索引;停用账号立即失效。

## 主要接口

完整 OpenAPI 见 `/docs`(非 prod)。

| 接口 | 说明 |
|---|---|
| `POST /api/v1/auth/login` · `logout` · `GET /me` | 登录、吊销、当前用户 |
| `POST /api/v1/chat` | 一次性返回(含 `sources` 引用) |
| `POST /api/v1/chat/stream` · `WS /api/v1/chat/ws` | SSE / WebSocket 流式 |
| `/api/v1/sessions` · `POST /feedback` | 会话与点赞点踩(仅限本人) |
| `/api/v1/admin/faq` · `/faq/import` | FAQ 维护 |
| `/api/v1/admin/documents` | 上传、重建索引、下线 |
| `/api/v1/admin/stats` · `/gaps` | 回答率/拒答率/满意度;知识缺口 |
| `/api/v1/admin/users` · `/audit` | 用户管理、审计(系统管理员) |
| `/api/v1/admin/gaps/export` | 知识缺口 CSV 导出(防公式注入) |
| `/metrics` | Prometheus 指标(prod 需 `KNOWBASE_METRICS_TOKEN`) |
| `/health/live` · `/health/ready` | K8s 探针 |

## 质量保障

- `make test`(51 个用例):鉴权、权限隔离、FAQ 置信度、版本切换、入库失败回退、SSE/WebSocket、限流、审计、脱敏、追问改写、指标、导出等。
- `make eval`:基于 `data/eval/golden.jsonl` 的发布门禁(权限泄露必须为 0;命中率/拒答率不达标则退出码为 1),已接入 CI。接入真实制度后,请补充典型问题并据此标定 `KNOWBASE_MIN_RELEVANCE`。
- `python -m scripts.evaluate_rag --e2e`:接入真实 LLM 做端到端关键词覆盖检查。
- `make lint`:ruff 规范检查。

## 上线前须知

1. **Milvus 检索器、重排器与 `docker-compose.yml`** 需在目标环境验证(自动化测试覆盖除向量库与模型之外的全部逻辑)。请使用 `KNOWBASE_RETRIEVER_BACKEND=milvus` 跑一遍 `make eval` 并标定相关度阈值。
2. **数据库迁移**:当前启动时自动建表;生产请引入 Alembic 做版本化迁移。
3. **文档入库**使用进程内后台任务,重启会中断进行中的任务(文档停在 `pending`,可调用 `/reindex` 重跑);大批量或高可靠场景请接入任务队列。
4. **SSO**:现为账号密码 + JWT;对接企业微信/钉钉/飞书/OIDC 时,新增回调路由并签发同样的 JWT 即可,权限逻辑无需改动。
5. **OCR / 图片 / PPT**:默认未内置,可通过 `register_loader()` 接入(`knowbase/rag/ingestion/loaders.py`)。
6. **意图分类模型**:默认规则 + 一律查知识库;启用 BERT 分类器需用企业自有数据训练并设置 `KNOWBASE_INTENT_MODEL_DIR`。
7. **数据合规**:问答内容会发送给所配置的 LLM。涉密场景使用私有化模型服务,并评估是否需要脱敏。
8. **多实例**:限流与令牌吊销需配置 Redis;`memory` 检索后端不支持多实例。
9. **许可证**:仓库尚未指定 License,请按公司开源/内部代码政策补充。

更多:[SECURITY.md](SECURITY.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md)
