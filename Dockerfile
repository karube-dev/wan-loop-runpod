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
    && pip install -U "huggingface_hub[hf_transfer,hf_xet]" \
    && pip install runpod websocket-client Pillow requests

# ComfyUI (main tracks comfy-kitchen, whose custom-op registration uses PEP 585
# builtins rejected by torch <= 2.7 infer_schema; use cu128 / torch >= 2.8)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && cd /ComfyUI \
    && pip install -r requirements.txt \
    && pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# H3 nodes are native to ComfyUI core (0.30+) - no custom nodes needed.

# Prepare model directories
RUN mkdir -p /ComfyUI/models/diffusion_models \
             /ComfyUI/models/text_encoders \
             /ComfyUI/models/vae \
             /ComfyUI/models/loras

# Pre-bake all models at build time (~40GB). Runtime lazy-download in
# handler.py remains as a fallback if a file is missing.
RUN wget -q --show-progress \
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
        -O /ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    && wget -q --show-progress \
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
        -O /ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    && wget -q --show-progress \
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
        -O /ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors \
    && wget -q --show-progress \
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" \
        -O /ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors \
    && wget -q --show-progress \
        "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
        -O /ComfyUI/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors

# Copy worker source
COPY . /worker
RUN mv /worker/loop_api.json /loop_api.json \
    && chmod +x /worker/entrypoint.sh

WORKDIR /

CMD ["/worker/entrypoint.sh"]
