FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything (including app.py) into the container
COPY . /app

# Install Python deps – NOTE: added python-multipart
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pillow \
    torch \
    torchvision \
    transformers \
    python-multipart

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
