# 简易版Dockerfile
FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir  -r requirements.txt

RUN python -c "import torchvision.models as models; \
             models.resnet50(pretrained=True)"

COPY src/ ./src/
COPY .env.example .env

ENV MODEL_NAME=resnet50 \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=5000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s  --start-interval=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser
CMD ["python", "./src/app.py"]

