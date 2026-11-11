# syntax=docker/dockerfile:1
#
# Scoring server image: `commonlid serve` with the [server] extra.
#
#   docker build -t commonlid-server .
#   docker run -p 8000:8000 \
#     -e COMMONLID_SERVER_TOKEN=... -e HF_TOKEN=... \
#     -v ./server.toml:/config/server.toml:ro \
#     -v commonlid-state:/state -v commonlid-hf:/hf-cache \
#     commonlid-server
#
# See docs/eval_server.md ("Running in Docker").

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

# pycld2 has no wheels for recent Pythons and compiles from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependencies first, so source changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra server

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra server


FROM python:${PYTHON_VERSION}-slim

RUN useradd --create-home --uid 1000 commonlid \
    && mkdir -p /config /state /hf-cache \
    && chown commonlid:commonlid /state /hf-cache

COPY --from=builder /opt/venv /opt/venv
COPY docker/server.toml /config/server.toml

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache

USER commonlid
WORKDIR /home/commonlid

# /state holds the submission counter and an archive with gold labels: keep
# it away from the agent. /hf-cache avoids downloading the datasets again.
VOLUME ["/state", "/hf-cache"]
EXPOSE 8000

# Datasets load before the server answers, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

ENTRYPOINT ["commonlid"]
CMD ["serve", "--config", "/config/server.toml", "--host", "0.0.0.0", "--port", "8000"]
