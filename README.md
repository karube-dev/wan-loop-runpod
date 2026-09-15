# Wan2.2 Seamless Loop Video — RunPod Serverless

Short seamless-loop video worker built on **Wan2.2 I2V rapid (14B GGUF)** and
**ComfyUI + ComfyUI-WanVideoWrapper**, packaged for RunPod Serverless.

The start image is wired as **both first and last frame**
(`WanVideoImageToVideoEncode` with `end_image` + `fun_or_fl2v_model=false`),
so the generated clip returns to its opening frame and plain player repeat
looks like an infinite loop — no post-process crossfade needed.

```
Local client ── image + motion prompt ──→ RunPod Serverless ── MP4 (base64)
                                          Wan2.2-I2V-rapid ── video (loop)
```

---

## Project layout

```
wan-loop-runpod/
├── Dockerfile          # CUDA 12.4 + ComfyUI + WanVideoWrapper + GGUF
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
    "prompt": "The car steadily drives downhill ... seamless infinite loop.",
    "negative_prompt": "blurry, low quality, ...",
    "seed": 777,
    "steps": 8,
    "cfg": 4.0,
    "width": 832,
    "height": 480,
    "num_frames": 81
  }
}
```

`image_path` / `image_url` / `image_base64` all work (exactly one).
81 frames @ 16fps ≈ 5s. Defaults reproduce the validated local run
(seed 777, 832x480).

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

## Models (baked into the image, ~17GB)

| File | Source |
|---|---|
| `wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf` (~9.9GB) | `DoorZekor/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10` |
| `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` (~6.7GB) | `NSFW-API/NSFW-Wan-UMT5-XXL` |
| `wan_2.1_vae.safetensors` (~254MB) | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (`split_files/vae`) |

All three are downloaded at image build time. The handler's lazy-download
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
* **GPU**: RTX 4090 24GB minimum (14B Q4 + block swap; A100 80GB for headroom)
* **Container Disk**: ≥40 GB (models ~17GB + overhead)
* **Idle Timeout**: 5–30 s for chained calls (costs while warm)
* **Execution Timeout**: ≥40 min (cold start downloads ~17GB on first job)
* **Max Workers**: 1–2

### 3. Smoke test

```powershell
powershell -ExecutionPolicy Bypass -File ..\test_loop_endpoint.ps1 -EndpointId <id>
```

---

## Notes

* **Cold start**: image pull (~17GB of baked models) plus 81-frame
  generation (~10–20 min on a 4090). Subsequent jobs on a warm worker
  generate immediately.
* **Cost**: while `workersMin=1`, the GPU bills continuously.
  Set `workersMin=0` after testing.
* The rapid model is a few-step distillate: keep `steps` at 8
  (the validated value); raising it does not reliably help.
