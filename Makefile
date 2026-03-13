.PHONY: install lint check test test-unit test-int run run-cli data-clean

data-clean:
	rm -rf data/*/
	rm -f data/*

install:
	uv venv --clear
	uv sync
	-git config --global init.defaultBranch main
	-uv run pre-commit install || true

lint:
	uv run ruff check src test --fix

check:
	PYTHONPATH=. uv run ty check src test

test:
	PYTHONPATH=. uv run pytest --cov=src --cov-report=term-missing --tb=short

test-unit:
	PYTHONPATH=. uv run pytest test/unit/ -v --tb=short

test-int:
	PYTHONPATH=. uv run pytest test/integration/ -v --tb=short

run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8001

run-cli:
	uv run python -m src.cli run

docker-build:
	docker build -t video-composer-job-service .

docker-run:
	docker run -p 8001:8001 video-composer-job-service

up:
	docker-compose up -d

up-build:
	docker-compose up -d --build

down:
	docker-compose down
