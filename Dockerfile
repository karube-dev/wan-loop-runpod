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

# Pre-bake all models at build time (datacenter backbone is fast; this removes
# an entire failure class from cold starts). Runtime lazy-download in
# handler.py remains as a fallback if a file is missing.
RUN wget -q --show-progress \
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
        -O /ComfyUI/models/vae/wan_2.1_vae.safetensors \
    && wget -q --show-progress \
        "https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" \
        -O /ComfyUI/models/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors \
    && wget -q --show-progress \
        "https://huggingface.co/DoorZekor/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10/resolve/main/wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf" \
        -O /ComfyUI/models/diffusion_models/wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf

# Copy worker source
COPY . /worker
RUN mv /worker/loop_api.json /loop_api.json \
    && chmod +x /worker/entrypoint.sh

WORKDIR /

CMD ["/worker/entrypoint.sh"]
