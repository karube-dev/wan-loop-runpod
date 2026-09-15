#!/bin/bash
echo "[wan-loop] Container startup..."
echo "[wan-loop] RUNPOD_WEBHOOK_GET_JOB=${RUNPOD_WEBHOOK_GET_JOB:-NOT SET}"

# Start ComfyUI with stdout captured so the handler can report boot failures.
echo "[wan-loop] Starting ComfyUI..."
python /ComfyUI/main.py --listen --disable-auto-launch > /tmp/comfyui.log 2>&1 &
COMFYUI_PID=$!

# Start handler immediately so RunPod sees a healthy worker.
# The handler will download any missing models on the first inference request.
echo "[wan-loop] Starting RunPod handler..."
exec python /worker/handler.py
