# MiniMax H3 Seamless Loop Video — RunPod Serverless

Short seamless-loop video worker built on **MiniMax H3 FL2VA (native ComfyUI
nodes)** with the turbo 8-step LoRA, packaged for RunPod Serverless.

The start image is wired as **both first and last frame**
(`MiniMaxH3ImageToVideo` with `first_frame` + `last_frame`),
so the generated clip returns to its opening frame and plain player repeat
looks like an infinite loop — no post-process crossfade needed.
H3 natively renders **stereo audio** alongside video (describe it in the
prompt's `Audio:` block), so output is post-ready including sound.

```
Local client ── image + motion prompt ──→ RunPod Serverless ── MP4+audio (base64)
                                          MiniMax-H3-FL2VA ── video (loop)
```

---

## Project layout

```
wan-loop-runpod/
├── Dockerfile          # CUDA 12.4 + ComfyUI (H3 native, no custom nodes)
├── handler.py          # RunPod Serverless handler (video in/out)
├── entrypoint.sh       # Starts ComfyUI, then the handler
├── loop_api.json       # ComfyUI workflow (FLF loop I2V)
├── test_loop_endpoint.ps1  # smoke test (in 03_engui_studio root)
├── .runpod/hub.json    # RunPod Hub metadata
└── README.md
```

---

## API

### Request

```json
{
  "input": {
    "image_base64": "data:image/png;base64,...",
    "prompt": "The car steadily drives downhill ... Audio: low engine hum ...",
    "seed": 777,
    "steps": 8,
    "width": 1344,
    "height": 768,
    "length": 124
  }
}
```

`image_path` / `image_url` / `image_base64` all work (exactly one).
124 frames @ 24fps ≈ 5.2s (17k+5 grid). Canvas is 768px short edge.

### Response

```json
{
  "video": "base64-encoded MP4, no data: prefix",
  "kind": "videos",
  "filename": "ComfyUI_00001_.mp4",
  "seed": 777,
  "prompt": "..."
}
```

---

## Models (baked into the image, ~40GB)

| File | Source |
|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (~19.5GB) | `Comfy-Org/MiniMax-H3` (`diffusion_models`) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (~14.6GB) | `Comfy-Org/MiniMax-H3` (`text_encoders`) |
| `minimax_h3_video_vae_fp16.safetensors` (~4.9GB) | `Comfy-Org/MiniMax-H3` (`vae`) |
| `minimax_h3_audio_vae_fp32.safetensors` (~0.6GB) | `Comfy-Org/MiniMax-H3` (`vae`) |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `lightx2v/Minimax-h3-Turbo` (`loras`) |

All downloaded at image build time. The handler's lazy-download
remains as a fallback if a file is missing at runtime.

---

## Build & deploy

### 1. Push to a GitHub repo

```bash
cd wan-loop-runpod
git init
git add .
git commit -m "Wan2.2 seamless-loop serverless worker"
gh repo create wan-loop-runpod --public --source=. --push
```

Pushing to `main` triggers the GHCR build (`.github/workflows/build.yml`).

### 2. Deploy to RunPod Serverless

* **RunPod Console → Serverless → New Endpoint**
* **Container Image**: `ghcr.io/<you>/wan-loop-runpod:<sha>`
* **GPU**: RTX 4090 24GB minimum (H3 int8 + nvfp4 text encoder; A100 80GB for headroom)
* **Container Disk**: ≥100 GB (models ~40GB + overhead)
* **Idle Timeout**: 5–30 s for chained calls (costs while warm)
* **Execution Timeout**: ≥90 min (image pull ~40GB on cold start + generation)
* **Max Workers**: 1–2

### 3. Smoke test

```powershell
powershell -ExecutionPolicy Bypass -File ..\test_loop_endpoint.ps1 -EndpointId <id>
```

---

## Notes

* **Cold start**: image pull (~40GB of baked models) plus 124-frame
  generation with turbo 8 steps (20–60 min on a 4090). Subsequent jobs
  on a warm worker generate immediately.
* **Cost**: while `workersMin=1`, the GPU bills continuously.
  Set `workersMin=0` after testing.
* Turbo LoRA is step-locked: keep `steps` at 8 (4-step LoRA exists but
  targets 768p preview quality).
