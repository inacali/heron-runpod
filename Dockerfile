# Dockerfile
FROM python:3.11-slim

# Prevents Python from writing .pyc files & enables unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (for Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# torch/torchvision without version pin so pip pulls the right wheel (CPU or CUDA)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pillow \
    torch \
    torchvision \
    transformers

# Copy app code
COPY app.py /app/app.py

# Expose API port
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

