"""
FLUX.2 Klein Real-Time Webcam Stylization - Modal Deployment
=============================================================
Real-time webcam-to-styled-video using FLUX.2 Klein 4B (distilled, 4-step).
Receives webcam JPEG frames over WebSocket, returns styled JPEG frames.

Uses Flux2KleinPipeline from diffusers with reference-image conditioning
for high-quality artistic output at real-time speeds on A10G.

Deploy:   modal deploy backend/app.py
Serve:    modal serve backend/app.py
"""

import modal
import os

# ---------------------------------------------------------------------------
# Modal infrastructure definitions
# ---------------------------------------------------------------------------

MODELS_DIR = "/models"
LORAS_DIR = "/models/loras"
MODEL_REPO = "black-forest-labs/FLUX.2-klein-4B"

# Available LoRA adapters (id -> HuggingFace repo)
LORA_REPOS = {
    "anime-style": "Sawata97/flux2_4b_koni_animestyle",
    "pixel-art": "Limbicnation/pixel-art-lora",
}

# Persistent volume for caching model weights
volume = modal.Volume.from_name("flux2-klein-weights", create_if_missing=True)

# Container image: CUDA 12.4 + PyTorch 2.6 + diffusers from main (for Flux2 support)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        # PyTorch 2.6 (needed for bfloat16 + Qwen3 support)
        "torch==2.6.0",
        "torchvision==0.21.0",
        # Diffusers from main branch (Flux2 pipelines not in any stable release yet)
        "git+https://github.com/huggingface/diffusers.git",
        # Transformers 4.51+ (needed for Qwen3ForCausalLM)
        "transformers>=4.51.0",
        "accelerate>=0.33.0",
        "peft>=0.15.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.91",
        # FP8 quantization
        "optimum-quanto>=0.2.6",
        # Image processing
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy<2",
        # Server
        "fastapi>=0.112.0",
        "uvicorn>=0.30.0",
        "python-multipart>=0.0.9",
        "pydantic>=2.8.0",
        # HuggingFace
        "huggingface-hub>=0.34.0",
    )
    .env({
        "MODEL_FOLDER": MODELS_DIR,
        "HF_HOME": f"{MODELS_DIR}/.hf_cache",
        # Cache torch.compile artifacts across container restarts
        "TORCHINDUCTOR_CACHE_DIR": f"{MODELS_DIR}/.torch_cache",
    })
)

app = modal.App("flux2-klein-realtime", image=image)


# ---------------------------------------------------------------------------
# Helper: download model weights into the persistent volume
# ---------------------------------------------------------------------------

def download_weights():
    """Download FLUX.2 Klein 4B weights and LoRA adapters if not cached."""
    from huggingface_hub import snapshot_download

    # --- Base model ---
    model_dir = os.path.join(MODELS_DIR, "flux2-klein-4b")
    marker = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(marker):
        print(f"[weights] Downloading {MODEL_REPO}...")
        snapshot_download(
            MODEL_REPO,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "*.bin",
                "*.onnx*",
                "*.xml",
                "*.md",
                ".gitattributes",
            ],
        )
        print("[weights] FLUX.2 Klein 4B downloaded.")
    else:
        print("[weights] FLUX.2 Klein 4B already cached.")

    # --- LoRA adapters ---
    os.makedirs(LORAS_DIR, exist_ok=True)
    for lora_id, repo in LORA_REPOS.items():
        lora_dir = os.path.join(LORAS_DIR, lora_id)
        if not os.path.exists(lora_dir) or not os.listdir(lora_dir):
            print(f"[weights] Downloading LoRA '{lora_id}' from {repo}...")
            snapshot_download(
                repo,
                local_dir=lora_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", ".gitattributes"],
            )
            print(f"[weights] LoRA '{lora_id}' downloaded.")
        else:
            print(f"[weights] LoRA '{lora_id}' already cached.")

    volume.commit()


# ---------------------------------------------------------------------------
# Main server class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L40S",
    memory=65536,
    timeout=3600,
    scaledown_window=120,
    min_containers=0,
    volumes={MODELS_DIR: volume},
)
@modal.concurrent(max_inputs=10)
class Flux2KleinServer:
    """
    Modal class that loads FLUX.2 Klein 4B at startup and serves a WebSocket
    endpoint for real-time webcam stylization.

    Pipeline: FLUX.2 Klein 4B (distilled 4-step) with reference image conditioning.
    The webcam frame is passed as a reference image -- the model generates a new
    image guided by both the text prompt and the reference.
    """

    @modal.enter()
    def startup(self):
        """Called once when the container starts."""
        import torch

        # Step 1: Download weights
        download_weights()

        # Step 2: Initialize pipeline
        print("[startup] Initializing FLUX.2 Klein pipeline...")
        self._init_pipeline()
        print(f"[startup] Pipeline ready. GPU: {torch.cuda.get_device_name()}")
        print(f"[startup] VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")

    def _init_pipeline(self):
        """Initialize the FLUX.2 Klein img2img pipeline."""
        import torch
        from diffusers import Flux2KleinPipeline

        model_dir = os.path.join(MODELS_DIR, "flux2-klein-4b")

        # Load pipeline in bfloat16 (FLUX.2 is designed for bf16)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
        )

        # Move to GPU
        pipe = pipe.to("cuda")

        # Store pipeline and state
        self.pipe = pipe
        self.current_prompt = "a stylized artistic portrait, high quality, detailed"
        self.current_steps = 4
        self.current_guidance = 1.0
        self.current_seed = 42

        # LoRA state
        self.current_lora = None  # Currently loaded LoRA id (None = base model)
        self.available_loras = {}
        for lora_id in LORA_REPOS:
            lora_dir = os.path.join(LORAS_DIR, lora_id)
            if os.path.exists(lora_dir):
                self.available_loras[lora_id] = lora_dir
        print(f"[startup] Available LoRAs: {list(self.available_loras.keys())}")

        # Pre-encode default prompt
        self._encode_prompt()

        # ── torch.compile ──────────────────────────────────────────
        # Using mode="default" to avoid CUDA graph TLS issues in ThreadPoolExecutor.
        # LoRA fuse/unfuse will require torch._dynamo.reset() after each swap,
        # which triggers a one-time retrace (~10-30s) on the next inference call.
        print("[startup] Compiling transformer with torch.compile...")
        pipe.transformer = torch.compile(pipe.transformer, mode="default")
        print("[startup] torch.compile applied.")

        # Warmup: run inference passes to trigger torch.compile tracing
        print("[startup] Warming up pipeline (triggers torch.compile)...")
        from PIL import Image
        warmup_img = Image.new("RGB", (384, 384), (128, 128, 128))
        for i in range(3):
            self.pipe(
                prompt_embeds=self.prompt_embeds,
                image=[warmup_img],
                num_inference_steps=4,
                guidance_scale=1.0,
                height=384,
                width=384,
                output_type="pil",
            )
            print(f"[startup] Warmup {i+1}/3 complete")
        print("[startup] Warmup complete.")

    def _encode_prompt(self):
        """Pre-encode the current prompt into embeddings for faster inference."""
        import torch
        with torch.no_grad():
            self.prompt_embeds, self.text_ids = self.pipe.encode_prompt(
                prompt=self.current_prompt,
                device="cuda",
                num_images_per_prompt=1,
                max_sequence_length=256,
            )

    def load_lora(self, lora_id: str | None):
        """Load, swap, or unload a LoRA adapter.

        Uses fuse_lora() to merge LoRA weights into the base model for
        zero per-frame overhead. After fuse/unfuse, resets the torch.compile
        cache so the next inference call retraces (one-time ~10-30s cost).

        - lora_id=None: unload current LoRA, revert to base model
        - lora_id=<id>: load and fuse the specified LoRA
        - Same as current: no-op
        """
        import torch

        if lora_id == self.current_lora:
            return  # Already loaded (or both None)

        # Unload current LoRA if one is active
        if self.current_lora is not None:
            print(f"[lora] Unloading '{self.current_lora}'...")
            try:
                self.pipe.unfuse_lora()
                self.pipe.unload_lora_weights()
            except Exception as e:
                print(f"[lora] Warning during unload: {e}")
            self.current_lora = None
            print("[lora] Reverted to base model.")

        # Load new LoRA if requested
        if lora_id is not None:
            if lora_id not in self.available_loras:
                print(f"[lora] WARNING: '{lora_id}' not found, ignoring.")
                return

            lora_dir = self.available_loras[lora_id]
            print(f"[lora] Loading '{lora_id}' from {lora_dir}...")
            self.pipe.load_lora_weights(lora_dir)
            self.pipe.fuse_lora()
            self.current_lora = lora_id
            print(f"[lora] '{lora_id}' loaded and fused.")

        # Reset torch.compile cache -- the fused/unfused weights changed
        # the model parameters, so the compiled graph is stale.
        # Next inference call will retrace once, then run fast again.
        torch._dynamo.reset()
        print("[lora] Compile cache reset. Next frame will retrace.")

    def update_params(
        self,
        prompt: str = "",
        guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        seed: int = 42,
        lora_id: str | None = None,
        **kwargs,
    ):
        """Update parameters. Re-encodes prompt if it changed. Swaps LoRA if needed."""
        prompt_changed = prompt and prompt != self.current_prompt

        if prompt:
            self.current_prompt = prompt
        self.current_guidance = max(0.0, min(10.0, guidance_scale))
        self.current_steps = max(1, min(8, num_inference_steps))
        self.current_seed = int(seed)

        # Swap LoRA if the requested one differs from current
        self.load_lora(lora_id)

        if prompt_changed:
            self._encode_prompt()

    def stylize_frame(self, image):
        """Run FLUX.2 Klein with reference image conditioning. Returns PIL Image."""
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

    @modal.asgi_app()
    def web(self):
        """Mount the FastAPI application with WebSocket support."""
        import asyncio
        import io
        import json
        import logging
        import socket
        import time
        import traceback
        from concurrent.futures import ThreadPoolExecutor

        import torch
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse
        from PIL import Image

        log = logging.getLogger("flux2-klein-modal")
        logging.basicConfig(level=logging.INFO)

        web_app = FastAPI(title="FLUX.2 Klein Realtime")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server = self
        inference_pool = ThreadPoolExecutor(max_workers=1)

        @web_app.get("/health")
        async def health():
            mem = torch.cuda.memory_allocated() / 1e9
            return {
                "status": "ok",
                "gpu": torch.cuda.get_device_name(),
                "vram_used_gb": round(mem, 2),
                "model": "flux2-klein-4b-distilled",
                "acceleration": "native bf16",
            }

        @web_app.get("/")
        async def root():
            return HTMLResponse(
                "<h1>FLUX.2 Klein Realtime</h1>"
                "<p>WebSocket endpoint: <code>/ws</code></p>"
                "<p><a href='/health'>Health check</a></p>"
            )

        @web_app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket):
            """
            WebSocket protocol:
            1. Client connects to /ws
            2. Server sends JSON: { "status": "ready" }
            3. Client sends JSON config: { "prompt": "...", "num_inference_steps": 4, ... }
            4. Server sends JSON: { "status": "streaming" }
            5. Client sends raw JPEG bytes (webcam frames)
            6. Server sends back raw JPEG bytes (styled frames)
            7. Client can send JSON text to update params mid-stream
            8. Close connection to stop
            """
            loop = asyncio.get_event_loop()
            await websocket.accept()
            await websocket.send_json({
                "status": "ready",
                "worker": socket.gethostname(),
            })

            # Shared state for frame pipelining
            latest_frame = [None]
            frame_ready = asyncio.Event()
            configured = asyncio.Event()
            should_stop = [False]
            frame_count = [0]

            async def receive_loop():
                """Receive frames/config from client, keep only the latest frame."""
                try:
                    while not should_stop[0]:
                        message = await websocket.receive()

                        # Handle disconnect
                        if message.get("type") == "websocket.disconnect":
                            log.info("Client sent disconnect")
                            break

                        # Handle text messages (JSON config/updates)
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
                                    log.info(f"Params updated: {prompt[:60]}... steps={num_inference_steps} lora={lora_id}")
                                except Exception as e:
                                    log.error(f"Param update failed: {e}")
                                    traceback.print_exc()
                                    await websocket.send_json({
                                        "status": "error",
                                        "error": f"Param update failed: {str(e)}",
                                    })

                            if not configured.is_set():
                                configured.set()
                                await websocket.send_json({"status": "streaming"})

                            continue

                        # Handle binary messages (JPEG frames) -- keep only latest
                        if message.get("bytes") is not None:
                            raw = message["bytes"]
                            if not raw or not configured.is_set():
                                continue

                            try:
                                input_image = Image.open(io.BytesIO(raw)).convert("RGB")
                                # Resize if needed (client sends 512x512)
                                if input_image.size != (384, 384):
                                    input_image = input_image.resize((384, 384), Image.LANCZOS)
                                latest_frame[0] = input_image
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
                """Continuously process the latest frame and send results back."""
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

                            # Encode output as JPEG
                            out_buf = io.BytesIO()
                            styled.save(out_buf, format="JPEG", quality=90)
                            out_bytes = out_buf.getvalue()

                            await websocket.send_bytes(out_bytes)

                            frame_count[0] += 1
                            if frame_count[0] % 30 == 0:
                                log.info(
                                    f"Frame {frame_count[0]}: "
                                    f"{inference_ms:.0f}ms inference, "
                                    f"{len(out_bytes)} bytes out"
                                )

                        except Exception as e:
                            log.error(f"Frame processing error: {e}")
                            traceback.print_exc()

                except Exception as e:
                    log.error(f"Process loop error: {e}")
                    traceback.print_exc()
                finally:
                    should_stop[0] = True

            # Run receive + process loops concurrently
            try:
                await asyncio.gather(receive_loop(), process_loop())
            except Exception as e:
                log.error(f"WebSocket handler error: {e}")
            finally:
                log.info(f"Session ended. Total frames: {frame_count[0]}")

        return web_app
