FROM python:3.11-slim

ARG INSTALL_ML=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt && \
    if [ "$INSTALL_ML" = "1" ]; then pip install -r requirements-ml.txt; fi

COPY knowbase ./knowbase
COPY scripts ./scripts
COPY frontend ./frontend
COPY data ./data

# 非 root 运行
RUN useradd -r -u 10001 knowbase && mkdir -p /srv/storage && chown -R knowbase /srv
USER knowbase

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health/live').status==200 else 1)"
# 单进程:Milvus 模式下每个进程都会加载一份向量/重排模型,横向扩展请增加副本数而不是 workers
CMD ["uvicorn", "knowbase.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
