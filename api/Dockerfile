FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# GPU=1 swaps the 5 MB CPU XGBoost for the CUDA 13 build (+~800 MB installed, mostly NCCL); host driver 580+.
# For a 570-579 driver use XGBOOST_GPU_PKG=xgboost-cu12==3.4.1 (CUDA 12.9).
ARG GPU=0
ARG XGBOOST_GPU_PKG=xgboost==3.4.1
RUN if [ "$GPU" = "1" ]; then pip uninstall -y xgboost-cpu && pip install "$XGBOOST_GPU_PKG"; fi
# Optional extra packages, e.g. `API_EXTRA_PIP=catboost docker compose build api`
ARG EXTRA_PIP=""
RUN if [ -n "$EXTRA_PIP" ]; then pip install $EXTRA_PIP; fi

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data \
 && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000
# WEB_CONCURRENCY is read by uvicorn for the worker count.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2} --log-level ${LOG_LEVEL:-info} --timeout-keep-alive 75"]
