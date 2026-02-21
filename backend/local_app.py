"""
FLUX.2 Klein Real-Time Webcam Stylization - Local Backend
==========================================================
Modal-free version of app.py for running locally on a machine with a CUDA GPU.
Identical WebSocket protocol — the frontend connects without any changes.

Usage:
    python backend/local_app.py

Env vars (all optional):
    MODELS_DIR          Path to store/cache model weights  (default: ./models)
    HF_TOKEN            HuggingFace token if the model repo is gated
    USE_TORCH_COMPILE   Set to "1" to enable torch.compile (off by default;
                        can be unstable on Windows)
    HOST                Server bind host  (default: 0.0.0.0)
    PORT                Server bind port  (default: 8000)
"""

import asyncio
import io
import json
import logging
import os
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

log = logging.getLogger("flux2-klein-local")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
LORAS_DIR = os.path.join(MODELS_DIR, "loras")
MODEL_REPO = "black-forest-labs/FLUX.2-klein-4B"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"

LORA_REPOS = {
    "anime-style": "Sawata97/flux2_4b_koni_animestyle",
    "pixel-art": "Limbicnation/pixel-art-lora",
}

# ---------------------------------------------------------------------------
# Weight download
# ---------------------------------------------------------------------------

def download_weights():
    """Download FLUX.2 Klein 4B weights and LoRA adapters if not already cached."""
    from huggingface_hub import snapshot_download

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Base model
    model_dir = os.path.join(MODELS_DIR, "flux2-klein-4b")
    marker = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(marker):
        log.info(f"Downloading {MODEL_REPO} → {model_dir} (this will take a while)...")
        snapshot_download(
            MODEL_REPO,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.onnx*", "*.xml", "*.md", ".gitattributes"],
        )
        log.info("FLUX.2 Klein 4B downloaded.")
    else:
        log.info("FLUX.2 Klein 4B already cached.")

    # LoRA adapters
    os.makedirs(LORAS_DIR, exist_ok=True)
    for lora_id, repo in LORA_REPOS.items():
        lora_dir = os.path.join(LORAS_DIR, lora_id)
        if not os.path.exists(lora_dir) or not os.listdir(lora_dir):
            log.info(f"Downloading LoRA '{lora_id}' from {repo}...")
            snapshot_download(
                repo,
                local_dir=lora_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", ".gitattributes"],
            )
            log.info(f"LoRA '{lora_id}' downloaded.")
        else:
            log.info(f"LoRA '{lora_id}' already cached.")


# ---------------------------------------------------------------------------
# Pipeline server (singleton)
# ---------------------------------------------------------------------------

class FluxServer:
    def __init__(self):
        self.pipe = None
        self.prompt_embeds = None
        self.text_ids = None
        self.current_prompt = "a stylized artistic portrait, high quality, detailed"
        self.current_steps = 4
        self.current_guidance = 1.0
        self.current_seed = 42
        self.current_lora = None
        self.available_loras = {}

    def startup(self):
        """Download weights and initialise the pipeline. Called once at server start."""
        import torch
        download_weights()
        log.info("Initialising FLUX.2 Klein pipeline...")
        self._init_pipeline()
        log.info(f"Pipeline ready. GPU: {torch.cuda.get_device_name()}")
        log.info(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")

    def _init_pipeline(self):
        import torch
        from diffusers import Flux2KleinPipeline

        model_dir = os.path.join(MODELS_DIR, "flux2-klein-4b")

        pipe = Flux2KleinPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
        )
        pipe = pipe.to("cuda")

        self.pipe = pipe

        # Discover available LoRAs
        for lora_id in LORA_REPOS:
            lora_dir = os.path.join(LORAS_DIR, lora_id)
            if os.path.exists(lora_dir):
                self.available_loras[lora_id] = lora_dir
        log.info(f"Available LoRAs: {list(self.available_loras.keys())}")

        # Pre-encode default prompt
        self._encode_prompt()

        if USE_TORCH_COMPILE:
            import torch
            log.info("Compiling transformer with torch.compile (this takes ~60s)...")
            pipe.transformer = torch.compile(pipe.transformer, mode="default")
            log.info("torch.compile applied.")

        # Warmup
        log.info("Warming up pipeline...")
        warmup_img = Image.new("RGB", (384, 384), (128, 128, 128))
        passes = 3 if USE_TORCH_COMPILE else 1
        for i in range(passes):
            self.pipe(
                prompt_embeds=self.prompt_embeds,
                image=[warmup_img],
                num_inference_steps=4,
                guidance_scale=1.0,
                height=384,
                width=384,
                output_type="pil",
            )
            log.info(f"Warmup {i + 1}/{passes} complete")
        log.info("Warmup complete — ready to stream.")

    def _encode_prompt(self):
        import torch
        with torch.no_grad():
            self.prompt_embeds, self.text_ids = self.pipe.encode_prompt(
                prompt=self.current_prompt,
                device="cuda",
                num_images_per_prompt=1,
                max_sequence_length=256,
            )

    def load_lora(self, lora_id: str | None):
        import torch

        if lora_id == self.current_lora:
            return

        if self.current_lora is not None:
            log.info(f"Unloading LoRA '{self.current_lora}'...")
            try:
                self.pipe.unfuse_lora()
                self.pipe.unload_lora_weights()
            except Exception as e:
                log.warning(f"LoRA unload warning: {e}")
            self.current_lora = None

        if lora_id is not None:
            if lora_id not in self.available_loras:
                log.warning(f"LoRA '{lora_id}' not found, ignoring.")
                return
            lora_dir = self.available_loras[lora_id]
            log.info(f"Loading LoRA '{lora_id}' from {lora_dir}...")
            self.pipe.load_lora_weights(lora_dir)
            self.pipe.fuse_lora()
            self.current_lora = lora_id
            log.info(f"LoRA '{lora_id}' loaded and fused.")

        if USE_TORCH_COMPILE:
            torch._dynamo.reset()
            log.info("Compile cache reset after LoRA swap.")

    def update_params(
        self,
        prompt: str = "",
        guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        seed: int = 42,
        lora_id: str | None = None,
        **kwargs,
    ):
        prompt_changed = prompt and prompt != self.current_prompt

        if prompt:
            self.current_prompt = prompt
        self.current_guidance = max(0.0, min(10.0, guidance_scale))
        self.current_steps = max(1, min(8, num_inference_steps))
        self.current_seed = int(seed)

        self.load_lora(lora_id)

        if prompt_changed:
            self._encode_prompt()

    def stylize_frame(self, image: Image.Image) -> Image.Image:
        import torch

        generator = torch.Generator(device="cuda").manual_seed(self.current_seed)

        result = self.pipe(
            prompt_embeds=self.prompt_embeds,
            image=[image],
            num_inference_steps=self.current_steps,
            guidance_scale=self.current_guidance,
            height=384,
            width=384,
            generator=generator,
            output_type="pil",
            max_sequence_length=256,
        ).images[0]

        return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

server = FluxServer()
inference_pool = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup (in a thread so the event loop isn't blocked)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, server.startup)
    yield
    # Shutdown — nothing to clean up


web_app = FastAPI(title="FLUX.2 Klein Realtime (local)", lifespan=lifespan)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/health")
async def health():
    import torch
    mem = torch.cuda.memory_allocated() / 1e9
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(),
        "vram_used_gb": round(mem, 2),
        "model": "flux2-klein-4b-distilled",
        "lora": server.current_lora,
        "torch_compile": USE_TORCH_COMPILE,
    }


@web_app.get("/")
async def root():
    return HTMLResponse(
        "<h1>FLUX.2 Klein Realtime (local)</h1>"
        "<p>WebSocket endpoint: <code>/ws</code></p>"
        "<p><a href='/health'>Health check</a></p>"
    )


@web_app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    loop = asyncio.get_event_loop()
    await websocket.accept()
    await websocket.send_json({
        "status": "ready",
        "worker": socket.gethostname(),
    })

    latest_frame: list[Image.Image | None] = [None]
    frame_ready = asyncio.Event()
    configured = asyncio.Event()
    should_stop = [False]
    frame_count = [0]

    async def receive_loop():
        try:
            while not should_stop[0]:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    log.info("Client disconnected")
                    break

                # JSON config / param update
                if message.get("text") is not None:
                    try:
                        data = json.loads(message["text"])
                    except (json.JSONDecodeError, TypeError):
                        await websocket.send_json({"error": "Invalid JSON"})
                        continue

                    prompt = data.get("prompt", "")
                    guidance_scale = float(data.get("guidance_scale", 1.0))
                    num_inference_steps = int(data.get("num_inference_steps", 4))
                    seed = data.get("seed", 42)
                    lora_id = data.get("lora_id", None)

                    if prompt:
                        try:
                            await loop.run_in_executor(
                                inference_pool,
                                lambda: server.update_params(
                                    prompt=prompt,
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps,
                                    seed=int(seed),
                                    lora_id=lora_id,
                                ),
                            )
                            log.info(f"Params updated: '{prompt[:60]}...' steps={num_inference_steps} lora={lora_id}")
                        except Exception as e:
                            log.error(f"Param update failed: {e}")
                            traceback.print_exc()
                            await websocket.send_json({"status": "error", "error": str(e)})

                    if not configured.is_set():
                        configured.set()
                        await websocket.send_json({"status": "streaming"})

                    continue

                # Raw JPEG frame
                if message.get("bytes") is not None:
                    raw = message["bytes"]
                    if not raw or not configured.is_set():
                        continue
                    try:
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        if img.size != (384, 384):
                            img = img.resize((384, 384), Image.LANCZOS)
                        latest_frame[0] = img
                        frame_ready.set()
                    except Exception as e:
                        log.error(f"Frame decode error: {e}")

        except WebSocketDisconnect:
            log.info("Client disconnected (receive loop)")
        except Exception as e:
            log.error(f"Receive loop error: {e}")
            traceback.print_exc()
        finally:
            should_stop[0] = True
            frame_ready.set()

    async def process_loop():
        try:
            while not should_stop[0]:
                await frame_ready.wait()
                frame_ready.clear()

                if should_stop[0]:
                    break

                image = latest_frame[0]
                if image is None:
                    continue

                try:
                    t0 = time.perf_counter()
                    styled = await loop.run_in_executor(
                        inference_pool,
                        server.stylize_frame,
                        image,
                    )
                    inference_ms = (time.perf_counter() - t0) * 1000

                    out_buf = io.BytesIO()
                    styled.save(out_buf, format="JPEG", quality=90)
                    out_bytes = out_buf.getvalue()

                    await websocket.send_bytes(out_bytes)

                    frame_count[0] += 1
                    if frame_count[0] % 10 == 0:
                        log.info(
                            f"Frame {frame_count[0]}: "
                            f"{inference_ms:.0f}ms inference, "
                            f"{len(out_bytes) // 1024}KB output"
                        )

                except Exception as e:
                    log.error(f"Frame processing error: {e}")
                    traceback.print_exc()

        except Exception as e:
            log.error(f"Process loop error: {e}")
            traceback.print_exc()
        finally:
            should_stop[0] = True

    try:
        await asyncio.gather(receive_loop(), process_loop())
    except Exception as e:
        log.error(f"WebSocket handler error: {e}")
    finally:
        log.info(f"Session ended. Total frames: {frame_count[0]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log.info(f"Starting local FLUX.2 Klein server on {host}:{port}")
    uvicorn.run(web_app, host=host, port=port, log_level="info")
