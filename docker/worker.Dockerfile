FROM python:3.12-slim AS builder

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYTORCH_SPEC=torch>=2.8,<2.9
ARG PYTORCH_AUDIO_SPEC=torchaudio>=2.8,<2.9
ARG INCLUDE_LTX_PIPELINES=1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY apps/worker/requirements.txt ./requirements.txt
COPY apps/worker/requirements-ltx.txt ./requirements-ltx.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-compile --index-url "${PYTORCH_INDEX_URL}" "${PYTORCH_SPEC}" "${PYTORCH_AUDIO_SPEC}" \
    && pip freeze | grep -E '^(torch|torchaudio|torchvision|nvidia-|triton)==.*' > /tmp/torch-constraints.txt \
    && pip install --no-cache-dir --no-compile -c /tmp/torch-constraints.txt -r requirements.txt \
    && if [ "${INCLUDE_LTX_PIPELINES}" = "1" ]; then pip install --no-cache-dir --no-compile -c /tmp/torch-constraints.txt -r requirements-ltx.txt; fi \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.12/site-packages/torch/include \
        /opt/venv/lib/python3.12/site-packages/torch/test \
        /opt/venv/lib/python3.12/site-packages/torch/share \
    && (find /opt/venv -type f -name "*.so*" -exec strip --strip-unneeded {} + || true)

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY apps/worker ./apps/worker
COPY packages/shared ./packages/shared
ENV PYTHONPATH=/app/apps/worker:/app/packages/shared

ENTRYPOINT ["python", "-m", "scene_worker"]
