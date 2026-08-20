# Inference image for the RAG QA service.
#
# Two-stage build: dependencies are installed in a builder layer and only the
# resulting virtualenv is copied forward, so the runtime image carries no
# compiler toolchain. CPU-only torch is installed explicitly -- the default
# wheel pulls ~2GB of CUDA libraries that are dead weight in a CPU container.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # Cache HF downloads on a mountable volume so a restart does not re-download.
    HF_HOME=/app/.cache/huggingface \
    OMP_NUM_THREADS=4

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/
COPY app/ ./app/
COPY scripts/ ./scripts/

# Non-root user: the container never needs to write outside /app.
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/artifacts /app/.cache/huggingface \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The service is only useful once the index exists; /health reports index_loaded
# so an orchestrator can tell "process up" from "actually ready".
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=8).status==200 else 1)"

CMD ["uvicorn", "rag_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
