.PHONY: help build build-promax run stop test clean

help:
	@echo "Available targets:"
	@echo "  build          Build Docker image (model-api:v1.0)"
	@echo "  build-promax   Build Docker image for promax (model-api:v1.0.promax)"
	@echo "  run            Run the Docker container"
	@echo "  run-fastapi    Run the FastAPI version in Docker"
	@echo "  stop           Stop the Docker container"
	@echo "  test           Run all tests"
	@echo "  test-flask     Run Flask tests only"
	@echo "  test-fastapi   Run FastAPI tests only"
	@echo "  clean          Remove Docker containers and images"

build:
	docker build -f docker/Dockerfile -t model-api:v1.0 .

build-promax:
	docker build -f docker/Dockerfile.promax -t model-api:v1.0.promax .

run:
	docker run -d --name model-api-container \
		-p 5000:5000 \
		-e MODEL_NAME=resnet50 \
		-e LOG_LEVEL=INFO \
		model-api:v1.0

run-fastapi:
	docker run -d --name model-api-fastapi-container \
		-p 5000:5000 \
		-e MODEL_NAME=resnet50 \
		-e LOG_LEVEL=INFO \
		-e API_FRAMEWORK=fastapi \
		model-api:v1.0.promax

stop:
	docker stop model-api-container || true
	docker rm model-api-container || true
	docker stop model-api-fastapi-container || true
	docker rm model-api-fastapi-container || true

test:
	python -m pytest tests/ -v

test-flask:
	python -m pytest tests/test_app_flask.py -v

test-fastapi:
	python -m pytest tests/test_app_fastapi.py -v

clean:
	docker stop model-api-container || true
	docker rm model-api-container || true
	docker stop model-api-fastapi-container || true
	docker rm model-api-fastapi-container || true
	docker rmi model-api:v1.0 || true
	docker rmi model-api:v1.0.promax || true
