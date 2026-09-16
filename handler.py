"""
Wan2.2 I2V first-last-frame seamless-loop worker for RunPod Serverless.

Takes a start image (+ optional motion prompt) and generates a short video
clip whose last frame returns to the first frame, so plain player repeat
looks like an infinite loop. Communicates with ComfyUI over HTTP/WebSocket.
The API workflow lives at /loop_api.json inside the container.
"""
import os
import json
import time
import uuid
import base64
import binascii
import logging
import subprocess
import urllib.error
import urllib.request

import runpod
import websocket  # provided by `pip install websocket-client`

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "/loop_api.json")
COMFYUI_INPUT_DIR = os.getenv("COMFYUI_INPUT_DIR", "/ComfyUI/input")
COMFYUI_OUTPUT_DIR = os.getenv("COMFYUI_OUTPUT_DIR", "/ComfyUI/output")
CLIENT_ID = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# I/O helpers (path / url / base64)
# ---------------------------------------------------------------------------
def _save_base64_to_file(data: str, out_path: str) -> str:
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    decoded = base64.b64decode(data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(decoded)
    return out_path


def _download_url_to_file(url: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = subprocess.run(
        ["wget", "-O", out_path, "--no-verbose", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wget failed: {result.stderr}")
    return out_path


def resolve_input(value, dest_dir: str, dest_filename: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected string input, got {type(value).__name__}")

    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.abspath(os.path.join(dest_dir, dest_filename))

    if os.path.isfile(value):
        return value

    if value.startswith("http://") or value.startswith("https://"):
        return _download_url_to_file(value, target)

    try:
        return _save_base64_to_file(value, target)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "Input is not a local file, URL, or valid base64 string"
        ) from exc


# ---------------------------------------------------------------------------
# ComfyUI plumbing
# ---------------------------------------------------------------------------
def queue_prompt(prompt: dict) -> str:
    url = f"http://{SERVER_ADDRESS}:8188/prompt"
    body = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    logger.info("Queuing prompt to %s (payload size: %d bytes)", url, len(body))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read().decode("utf-8")
            data = json.loads(resp_body)
            if data.get("node_errors"):
                raise RuntimeError(f"ComfyUI node_errors: {json.dumps(data['node_errors'])[:2000]}")
            if "prompt_id" not in data:
                raise RuntimeError(f"ComfyUI response missing prompt_id: {resp_body[:1000]}")
            return data["prompt_id"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        logger.error("ComfyUI prompt rejected: HTTP %s - %s", e.code, error_body[:1000])
        raise RuntimeError(
            f"ComfyUI prompt rejected (HTTP {e.code}): {error_body[:1000]}"
        ) from e


def get_history(prompt_id: str) -> dict:
    url = f"http://{SERVER_ADDRESS}:8188/history/{prompt_id}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wait_for_completion(prompt: dict, timeout: int = 2400) -> dict:
    prompt_id = queue_prompt(prompt)
    ws = websocket.WebSocket()
    ws.connect(f"ws://{SERVER_ADDRESS}:8188/ws?clientId={CLIENT_ID}")

    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                ws.settimeout(max(1, int(deadline - time.time())))
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                raise TimeoutError("Timed out waiting for ComfyUI")

            if isinstance(msg, str):
                event = json.loads(msg)
                if event.get("type") == "executing":
                    data = event.get("data") or {}
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        return get_history(prompt_id)
    finally:
        ws.close()

    raise TimeoutError("ComfyUI execution did not finish in time")


def collect_outputs(history_entry: dict) -> list:
    results = []
    for node_id, node_output in (history_entry.get("outputs") or {}).items():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []):
                fullpath = item.get("fullpath") or os.path.join(
                    COMFYUI_OUTPUT_DIR, item.get("subfolder", ""), item.get("filename", "")
                )
                with open(fullpath, "rb") as fh:
                    results.append({
                        "node_id": node_id,
                        "kind": key,
                        "filename": item.get("filename"),
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                        "data_base64": base64.b64encode(fh.read()).decode("utf-8"),
                    })
    return results


# ---------------------------------------------------------------------------
# Workflow wiring (node ids match loop_api.json)
# ---------------------------------------------------------------------------
def load_workflow() -> dict:
    with open(WORKFLOW_PATH, "r") as fh:
        return json.load(fh)


DEFAULT_POSITIVE = (
    "Steady cinematic shot with gentle cyclic motion that ends exactly where it "
    "began, forming a seamless infinite loop. Restrained realistic motion, "
    "photorealistic, high quality. "
    "Audio: subtle ambient night soundscape, low and calm, fading to near-silence "
    "at the loop point."
)


def build_prompt(
    workflow: dict,
    image_path: str,
    prompt_text: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    length: int,
) -> dict:
    prompt = json.loads(json.dumps(workflow))

    prompt["1"]["inputs"]["image"] = os.path.basename(image_path)
    prompt["7"]["inputs"]["prompt"] = prompt_text
    prompt["7"]["inputs"]["width"] = int(width)
    prompt["7"]["inputs"]["height"] = int(height)
    prompt["7"]["inputs"]["length"] = int(length)
    prompt["9"]["inputs"]["noise_seed"] = int(seed)
    prompt["11"]["inputs"]["steps"] = int(steps)

    return prompt


# ---------------------------------------------------------------------------
# Lazy readiness check + model download (called on first inference)
# ---------------------------------------------------------------------------
_COMFYUI_READY = False
_MODEL_DIR = "/ComfyUI/models"

_REQUIRED_MODELS = {
    # H3 FL2VA diffusion (~19.5GB, int8 pruned)
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": {
        "repo": "Comfy-Org/MiniMax-H3",
        "file": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "subfolder": "diffusion_models",
    },
    # Qwen3-VL text encoder (~14.6GB, nvfp4)
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": {
        "repo": "Comfy-Org/MiniMax-H3",
        "file": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "subfolder": "text_encoders",
    },
    # H3 video VAE (~4.9GB)
    "vae/minimax_h3_video_vae_fp16.safetensors": {
        "repo": "Comfy-Org/MiniMax-H3",
        "file": "minimax_h3_video_vae_fp16.safetensors",
        "subfolder": "vae",
    },
    # H3 audio VAE (~0.6GB)
    "vae/minimax_h3_audio_vae_fp32.safetensors": {
        "repo": "Comfy-Org/MiniMax-H3",
        "file": "minimax_h3_audio_vae_fp32.safetensors",
        "subfolder": "vae",
    },
    # Turbo 8-step LoRA
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": {
        "repo": "lightx2v/Minimax-h3-Turbo",
        "file": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    },
}


def _download_model(rel_path: str, info: dict):
    """Download a single model via huggingface_hub. Returns True on success."""
    from huggingface_hub import hf_hub_download
    dest_dir = os.path.join(_MODEL_DIR, os.path.dirname(rel_path))
    dest_file = os.path.join(_MODEL_DIR, rel_path)
    token = os.environ.get("HF_TOKEN")

    if os.path.isfile(dest_file) and os.path.getsize(dest_file) > 0:
        return True

    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading %s from %s...", info["file"], info["repo"])
    try:
        downloaded = hf_hub_download(
            repo_id=info["repo"],
            filename=info["file"],
            subfolder=info.get("subfolder"),
            local_dir_use_symlinks=False,
            local_dir=dest_dir,
            token=token,
        )
        if os.path.isfile(downloaded):
            if os.path.abspath(downloaded) != os.path.abspath(dest_file):
                import shutil
                shutil.copy2(downloaded, dest_file)
            logger.info("Downloaded %s (%d MB)", rel_path, os.path.getsize(dest_file) // (1024 * 1024))
            return True
    except Exception as e:
        logger.error("Failed to download %s: %s", rel_path, e)
    return False


def _comfyui_log_tail(n: int = 40) -> str:
    try:
        with open("/tmp/comfyui.log", "r", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except Exception as e:
        return f"<no comfyui log: {e}>"


def _wait_ready(timeout: int = 2400):
    global _COMFYUI_READY
    if _COMFYUI_READY:
        return

    deadline = time.time() + timeout
    import requests as _requests

    # Wait for ComfyUI HTTP server (short budget: models are baked in the image)
    server_timeout = 600
    server_deadline = time.time() + server_timeout
    logger.info("Waiting for ComfyUI server (timeout=%ds)...", server_timeout)
    server_up = False
    while time.time() < server_deadline:
        try:
            resp = _requests.get(f"http://{SERVER_ADDRESS}:8188/", timeout=5)
            if resp.status_code == 200:
                logger.info("ComfyUI server is up.")
                server_up = True
                break
        except Exception:
            pass
        time.sleep(2)
    if not server_up:
        raise RuntimeError(
            "ComfyUI server did not start within "
            f"{server_timeout}s. Log tail:\n{_comfyui_log_tail()}"
        )

    for rel_path, info in _REQUIRED_MODELS.items():
        full_path = os.path.join(_MODEL_DIR, rel_path)
        if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
            continue
        if time.time() > deadline:
            raise RuntimeError(f"Timeout — model {rel_path} not found after {timeout}s")
        if not _download_model(rel_path, info):
            raise RuntimeError(f"Failed to download {rel_path}")

    _COMFYUI_READY = True
    logger.info("All models are ready.")


# ---------------------------------------------------------------------------
# RunPod entry point
# ---------------------------------------------------------------------------
def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    logger.info("Received job input keys: %s", list(job_input.keys()))

    _wait_ready()

    image_field = next(
        (k for k in ("image_path", "image_url", "image_base64") if k in job_input),
        None,
    )
    if image_field is None:
        raise ValueError(
            "Image is required. Provide one of: image_path, image_url, image_base64"
        )

    tmp_dir = f"/tmp/loop_{uuid.uuid4().hex}"
    os.makedirs(tmp_dir, exist_ok=True)
    image_path = resolve_input(job_input[image_field], tmp_dir, "input.png")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Could not resolve input image at {image_path}")

    prompt_text = job_input.get("prompt", DEFAULT_POSITIVE)
    seed = int(job_input.get("seed", 777))
    steps = int(job_input.get("steps", 8))
    width = int(job_input.get("width", 1344))
    height = int(job_input.get("height", 768))
    length = int(job_input.get("length", 124))  # frames @24fps, 17k+5 grid

    comfy_input_target = os.path.join(COMFYUI_INPUT_DIR, os.path.basename(image_path))
    if os.path.abspath(image_path) != os.path.abspath(comfy_input_target):
        os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
        with open(image_path, "rb") as src, open(comfy_input_target, "wb") as dst:
            dst.write(src.read())
        image_path = comfy_input_target

    workflow = load_workflow()
    prompt = build_prompt(
        workflow,
        image_path=image_path,
        prompt_text=prompt_text,
        seed=seed,
        steps=steps,
        width=width,
        height=height,
        length=length,
    )
    history = wait_for_completion(prompt, timeout=3600)
    prompt_id = list(history.keys())[0]
    outputs = collect_outputs(history[prompt_id])

    if not outputs:
        return {"error": "ComfyUI finished but no video was produced."}

    first = outputs[0]
    return {
        "video": first["data_base64"],
        "kind": first["kind"],
        "filename": first["filename"],
        "seed": seed,
        "prompt": prompt_text,
    }


# ---------------------------------------------------------------------------
# Wait for RUNPOD_WEBHOOK_GET_JOB environment variable
# ---------------------------------------------------------------------------
def _wait_for_runpod_env(timeout: int = 120):
    """Wait for RUNPOD_WEBHOOK_GET_JOB env var to be injected by RunPod Serverless.

    RunPod Serverless injects this env var before starting the container.
    If not present, runpod.serverless.start() thinks it's running locally
    and tries to load test_input.json, which doesn't exist, causing sys.exit(1).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.getenv("RUNPOD_WEBHOOK_GET_JOB"):
            logger.info("RUNPOD_WEBHOOK_GET_JOB is set. Starting serverless worker...")
            return True
        logger.info("Waiting for RUNPOD_WEBHOOK_GET_JOB env var... (%.0fs remaining)", deadline - time.time())
        time.sleep(2)

    logger.warning("RUNPOD_WEBHOOK_GET_JOB not set after %ds. Proceeding anyway...", timeout)
    return False


logger.info("Starting RunPod serverless worker...")
_wait_for_runpod_env()
runpod.serverless.start({"handler": handler})
