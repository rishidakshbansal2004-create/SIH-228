# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies required for OpenCV, PyTorch, and general builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=user . .

# Set working directory to the API directory so local relative imports resolve properly
WORKDIR $HOME/app/trusted_cv_model_integrity_final

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run FastAPI server via Uvicorn
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
