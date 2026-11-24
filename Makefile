setup:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/ruff check src tests
	.venv/bin/pytest -v

all:
	.venv/bin/tpshock all

rebuild:
	.venv/bin/tpshock all --skip-fetch

serve:
	.venv/bin/python -m http.server 8000 -d site

.PHONY: setup test all rebuild serve
