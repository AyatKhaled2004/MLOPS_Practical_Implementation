# ==========================================
# STAGE 1: Builder Stage
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./

RUN pip install --no-cache-dir uv && \
    uv pip install --no-cache --system --target=/install \
        fastapi uvicorn[standard] streamlit requests joblib scikit-learn numpy structlog pydantic && \
    find /install -type d -name "__pycache__" -exec rm -rf {} + && \
    find /install -type f -name "*.pyc" -delete && \
    rm -rf /root/.cache /root/.uv

# ==========================================
# STAGE 2: Final Stage
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends tini curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/install" \
    PATH="/install/bin:$PATH"

COPY --from=builder /install /install

COPY --chown=appuser:appuser . .


RUN find /app -type d -name "__pycache__" -exec rm -rf {} + && \
    find /app -type f -name "*.pyc" -delete

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & wait -n"]
