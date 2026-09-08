# ARM64 is not a preference - AgentCore Runtime requires it (Graviton). The
# build runs on GitHub's ubuntu-24.04-arm runners, so this is native with no
# emulation anywhere.
#
# The build context is the REPOSITORY ROOT, not agent/: the image needs the
# whole `app` package and `data/products.yaml`, because the agent is the
# application, not a shim in front of it.
#     docker build -f agent/Dockerfile .
FROM --platform=linux/arm64 python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, from the lock file, so a change to application code
# does not re-resolve the world.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY scripts ./scripts
COPY data/products.yaml ./data/products.yaml
COPY agent/server.py ./server.py

# The corpus is generated, not committed - every figure in it comes from
# products.yaml. Building it here means the image cannot ship a corpus that
# disagrees with the catalogue it was built from. It is the fallback when no
# Knowledge Base is configured; with one, retrieval goes to Bedrock.
RUN NO_AWS=1 MOCK_LLM=1 uv run --no-dev python scripts/build_corpus.py

# ---------------------------------------------------------------- runtime
FROM --platform=linux/arm64 python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY --from=build /app /app

# Not root. The runtime does not require it and there is no reason to.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin mcb \
    && chown -R mcb:mcb /app
USER mcb

# Contract: host 0.0.0.0, port 8080. Not configurable - the platform routes
# /invocations and /ping to this port and nothing else.
EXPOSE 8080

# One worker. Sessions are isolated by the platform, not by this process, and
# a second worker would only add memory that is billed for the whole session.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
