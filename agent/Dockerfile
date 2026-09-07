# ARM64 is not a preference - AgentCore Runtime requires it (Graviton).
# The build happens on GitHub's ubuntu-24.04-arm runners, so this is a native
# build with no emulation anywhere.
FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Contract: host 0.0.0.0, port 8080. Not configurable - the platform routes
# /invocations and /ping to this port and nothing else.
EXPOSE 8080

# One worker. Sessions are isolated by the platform, not by this process, and
# a second worker would only add memory that is billed for the whole session.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
