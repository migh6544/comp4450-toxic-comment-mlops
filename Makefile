.PHONY: install-dev test lint build local-up local-down

install-dev:
	python -m pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest

build:
	docker build -t comp4450-toxic-api ./backend
	docker build -t comp4450-toxic-frontend ./frontend
	docker build -t comp4450-toxic-monitoring ./monitoring

local-up:
	docker compose -f docker-compose.local.yml up --build

local-down:
	docker compose -f docker-compose.local.yml down
