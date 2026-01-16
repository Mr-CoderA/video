# Wan 2.2 (A14B) RunPod Serverless Dockerfile
# Optimized for A100 80GB GPU

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV DIFFUSERS_CACHE=/runpod-volume/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Pre-download models during build (optional - can be done at runtime)
# Uncomment the following lines to bake models into the image (increases image size significantly)
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}
# RUN python -c "from diffusers import WanPipeline, WanImageToVideoPipeline; \
#     WanPipeline.from_pretrained('Wan-AI/Wan2.2-T2V-14B-Diffusers'); \
#     WanImageToVideoPipeline.from_pretrained('Wan-AI/Wan2.2-I2V-14B-480P-Diffusers')"

# Set the entrypoint
CMD ["python", "-u", "src/handler.py"]

