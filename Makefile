.PHONY: install test lint play tournament evolve assets

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

play:
	python -m figment.cli play

tournament:
	python -m figment.cli tournament --games 300

evolve:
	python -m figment.cli evolve

assets:
	python -m figment.cli demo
