.PHONY: install corpus preflight bootstrap run mock test live all

install:
	uv sync --group dev

corpus:
	uv run python scripts/build_corpus.py

preflight:
	uv run python scripts/preflight.py

bootstrap:
	uv run python scripts/bootstrap_aws.py

run:
	uv run uvicorn app.main:app --port 8000

mock:
	MOCK_LLM=1 NO_AWS=1 uv run uvicorn app.main:app --port 8000

test:
	uv run pytest -m "not live"

live:
	uv run pytest -m live -s

all: install corpus test
