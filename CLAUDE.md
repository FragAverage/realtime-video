# CLAUDE.md - Project Guidelines for AI Agents

## Project Overview
Real-time webcam stylization app using FLUX.2 Klein 4B (distilled, 4-step) with reference-image conditioning. Monorepo with a Python/Modal backend and Next.js frontend. Webcam frames are sent over WebSocket, stylized by FLUX.2 Klein in real-time, and streamed back.

## Critical Rules
- **NEVER run `npm run dev`, `npm run build`, `npm start`, or `next dev`** — the dev server is already running. Changes are picked up via HMR automatically.
- **NEVER run `modal deploy` or `modal serve`** unless explicitly asked. The backend is deployed and live.
- Only commit when explicitly asked.

## Architecture

### Backend (`backend/`)
- **Runtime:** Python 3.11 inside Modal container (CUDA 12.4, A10G GPU)
- **Framework:** FastAPI + WebSocket
- **Model:** FLUX.2 Klein 4B (distilled) — `black-forest-labs/FLUX.2-klein-4B`
- **Text Encoder:** Qwen3ForCausalLM (extracts hidden states from layers 9, 18, 27)
- **VAE:** AutoencoderKLFlux2 (FLUX.2 native VAE with batch normalization)
- **Pipeline:** `Flux2KleinPipeline` from diffusers (installed from git main)
- **Scheduler:** `FlowMatchEulerDiscreteScheduler`
- **Img2Img approach:** Reference-image conditioning (webcam frame passed as `image=[pil_image]`, model generates stylized output guided by both prompt and reference)
- **dtype:** bfloat16 (FLUX.2 is designed for bf16)
- **Frame processing:** Async producer/consumer pattern — receives frames into latest-frame slot, drops stale frames, overlaps inference with I/O
- **Protocol:** JSON for config, raw JPEG binary for frames
- **Entry point:** `backend/app.py` — Modal `@app.cls` with `@modal.enter()` for pipeline init, `@modal.asgi_app()` for FastAPI
- **Model weights:** Cached in Modal Volume `flux2-klein-weights`
- LSP errors on `app.py` are expected — packages exist only inside the Modal container

### Frontend (`frontend/`)
- **Framework:** Next.js 15.5 (App Router), React 19, TypeScript 5 (strict)
- **Styling:** Tailwind CSS v4 (CSS-based config via `@import "tailwindcss"`, no JS config file)
- **Path alias:** `@/*` → `./src/*`
- **Package manager:** npm (lockfile is `package-lock.json`)

## WebSocket Protocol
1. Client connects to `wss://<host>/ws`
2. Server sends JSON: `{ "status": "ready" }`
3. Client sends JSON config: `{ "prompt": "...", "seed": 42, "guidance_scale": 1.0, "strength": 0.4, "num_inference_steps": 4 }`
4. Server sends JSON: `{ "status": "streaming" }`
5. Client sends raw JPEG bytes (webcam frames, 512x512)
6. Server sends back raw JPEG bytes (styled frames, 512x512)
7. Client can send JSON text to update params mid-stream
8. Close connection to stop

Note: `strength` is sent by the frontend but ignored by the FLUX.2 Klein backend (it uses reference-image conditioning, not traditional img2img strength). `guidance_scale` defaults to 1.0 (ignored for distilled models).

## Code Style — Frontend

### Imports
- Use `@/` path alias for all internal imports
- Group imports: React/Next → external libs → `@/components` → `@/hooks` → `@/lib`
- Named exports for components, hooks, and utilities. Default export only for `page.tsx` and `layout.tsx`

### Components
- All components are client-side (`"use client"` directive at top)
- One component per file in `src/components/`
- Props defined as interface above the component
- Functional components only, no class components
- Hooks in `src/hooks/`, prefixed with `use`

### Styling
- Tailwind utility classes directly on elements, no CSS modules
- Custom CSS properties defined in `globals.css` under `:root` and `@theme`
- Color tokens: `--color-bg`, `--color-surface`, `--color-accent`, `--color-text-primary`, `--color-text-muted`, etc.
- Use `var(--color-*)` in Tailwind arbitrary values
- Custom utility classes: `.glass`, `.glow-cyan`, `.scanlines`, `.mono`, `.pulse-glow`

### TypeScript
- Strict mode enabled
- Shared types in `src/lib/types.ts`
- Key types: `StyleParams`, `ServerMessage`, `StylePreset`
- `DEFAULT_PARAMS` constant exported from `types.ts`

### Naming
- Files: kebab-case
- Components: PascalCase
- Hooks: camelCase with `use` prefix
- Types/Interfaces: PascalCase
- Constants: UPPER_SNAKE_CASE

## File Structure
```
backend/
  app.py                          # Modal deployment (FastAPI + WebSocket + FLUX.2 Klein pipeline)
  requirements.txt                # Reference deps
frontend/
  src/
    app/
      globals.css                 # Tailwind v4 + cyberpunk theme
      layout.tsx                  # Root layout (fonts, dark mode)
      page.tsx                    # Main page (webcam + styled output side by side)
    components/
      video-canvas.tsx            # Canvas renderer + FPS meter (displays styled output)
      prompt-bar.tsx              # Prompt input + mid-stream apply
      style-gallery.tsx           # Preset style buttons (FLUX.2-optimized prompts)
      controls.tsx                # Parameter panel (seed, strength, steps, guidance_scale)
      status-bar.tsx              # Connection/FPS status
    hooks/
      use-websocket.ts            # WebSocket lifecycle + frame streaming
      use-webcam.ts               # getUserMedia + frame capture
    lib/
      types.ts                    # Shared TypeScript types
      msgpack.ts                  # Unused (kept for reference)
```
