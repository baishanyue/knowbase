.PHONY: install test lint eval demo up

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q

lint:
	ruff check .

# 发布门禁:检索命中率 / 权限泄露(必须0) / 拒答准确率,不达标返回非 0
eval:
	python -m scripts.evaluate_rag --verbose $(ARGS)

# 本地零依赖体验:SQLite + 内存 BM25 检索,不需要向量库、GPU、模型文件。问答生成需要在 .env 配好 LLM。
demo:
	python -m scripts.seed_demo
	uvicorn knowbase.main:app --host 127.0.0.1 --port 8080

up:
	docker compose up -d --build
