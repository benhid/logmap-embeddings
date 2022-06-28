all: install version

venv:
	@test -d venv || python3 -m venv .venv

install: venv
	@.venv/bin/activate && python -m pip install -r requirements.txt

version:
	@python -V

clean:
	@rm -rf build dist .eggs *.egg-info
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +

format: clean
	@isort --profile black *.py
	@black *.py

lint:
	@mypy *.py

.PHONY: tests

tests:
	@python -m pytest -s