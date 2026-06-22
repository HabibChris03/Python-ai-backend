# Hugging Face Spaces Docker deployment
# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Allow the non-root user to write model cache files (torch, easyocr, etc.)
ENV HOME=/tmp
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface
ENV TORCH_HOME=/tmp/.cache/torch
ENV EASYOCR_MODULE_PATH=/tmp/.EasyOCR

# Install system dependencies
RUN set -e && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create HF-required non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install requirements
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . /app

# Expose HF Spaces required port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Start application on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
