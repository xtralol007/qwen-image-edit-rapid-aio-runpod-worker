FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_FLASH_ATTENTION=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Install hf_transfer for fast downloads
RUN pip install hf_transfer

# Download transformer weights only (smaller, ~3GB)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V21', cache_dir='/app/hf_cache'); \
"

# Download base model (larger, ~12GB)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen-Image-Edit-2511', cache_dir='/app/hf_cache'); \
"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
