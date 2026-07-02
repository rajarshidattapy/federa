FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY federa ./federa
COPY examples ./examples

RUN pip install --upgrade pip \
 && pip install . \
 && pip install torchvision

EXPOSE 8000

CMD ["python", "-m", "examples.mnist_fedavg.server"]
