FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip \
    && pip install -U "huggingface_hub[hf_transfer]" \
    && pip install runpod websocket-client Pillow requests

# ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && cd /ComfyUI \
    && pip install -r requirements.txt

# Custom nodes: WanVideoWrapper (sampler/encode) + GGUF model support
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper \
    && cd /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper \
    && pip install -r requirements.txt
RUN git clone https://github.com/city96/ComfyUI-GGUF.git /ComfyUI/custom_nodes/ComfyUI-GGUF \
    && cd /ComfyUI/custom_nodes/ComfyUI-GGUF \
    && pip install -r requirements.txt

# Prepare model directories
RUN mkdir -p /ComfyUI/models/diffusion_models \
             /ComfyUI/models/text_encoders \
             /ComfyUI/models/vae

# Pre-bake the small public Wan VAE at build time (254MB).
# The 14B GGUF (~9.9GB) + UMT5 (~6.7GB) download lazily on first inference
# (see handler.py) to keep the image lean.
RUN wget -q --show-progress \
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
        -O /ComfyUI/models/vae/wan_2.1_vae.safetensors

# Copy worker source
COPY . /worker
RUN mv /worker/loop_api.json /loop_api.json \
    && chmod +x /worker/entrypoint.sh

WORKDIR /

CMD ["/worker/entrypoint.sh"]
