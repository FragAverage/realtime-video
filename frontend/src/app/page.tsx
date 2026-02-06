"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Zap, Square, Camera, CameraOff } from "lucide-react";

import { VideoCanvas } from "@/components/video-canvas";
import { PromptBar } from "@/components/prompt-bar";
import { StyleGallery } from "@/components/style-gallery";
import { Controls } from "@/components/controls";
import { StatusBar } from "@/components/status-bar";
import { useWebSocket } from "@/hooks/use-websocket";
import { useWebcam } from "@/hooks/use-webcam";
import type { StyleParams } from "@/lib/types";
import { DEFAULT_PARAMS } from "@/lib/types";

const CAPTURE_FPS = 15;
const CAPTURE_WIDTH = 512;
const CAPTURE_HEIGHT = 512;

export default function Home() {
  // ── State ──────────────────────────────────────────────────────
  const [params, setParams] = useState<StyleParams>({ ...DEFAULT_PARAMS });
  const [playbackFps, setPlaybackFps] = useState(15);

  // ── Hooks ──────────────────────────────────────────────────────
  const ws = useWebSocket();
  const webcam = useWebcam();

  // Webcam capture interval ref
  const webcamIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isStreaming = ws.status === "streaming";

  // ── Param updater ──────────────────────────────────────────────
  const updateParams = useCallback((partial: Partial<StyleParams>) => {
    setParams((prev) => ({ ...prev, ...partial }));
  }, []);

  // ── Start streaming ────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (!params.prompt.trim()) return;

    // Start webcam first and wait for it to be ready
    if (!webcam.isActive) {
      await webcam.start(CAPTURE_WIDTH, CAPTURE_HEIGHT);
      // Small delay to ensure the video element is mounted and stream attached
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    // Connect WebSocket with current params
    ws.connect(params);
  }, [params, webcam, ws]);

  // ── Stop streaming ─────────────────────────────────────────────
  const handleStop = useCallback(() => {
    ws.disconnect();
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current);
      webcamIntervalRef.current = null;
    }
  }, [ws]);

  // ── Webcam frame capture loop ──────────────────────────────────
  // When streaming, capture webcam frames and send to server
  useEffect(() => {
    if (isStreaming && webcam.isActive) {
      const interval = 1000 / CAPTURE_FPS;

      webcamIntervalRef.current = setInterval(() => {
        const frame = webcam.captureFrame(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        if (frame) {
          // Convert base64 to binary and send
          const binary = Uint8Array.from(atob(frame), (c) => c.charCodeAt(0));
          ws.sendFrame(binary.buffer);
        }
      }, interval);

      return () => {
        if (webcamIntervalRef.current) {
          clearInterval(webcamIntervalRef.current);
          webcamIntervalRef.current = null;
        }
      };
    }
  }, [isStreaming, webcam, ws]);

  // ── Style preset selection ─────────────────────────────────────
  const handlePresetSelect = useCallback(
    (prompt: string) => {
      updateParams({ prompt });

      // If already streaming, update prompt on server
      if (isStreaming) {
        ws.sendConfig({ prompt });
      }
    },
    [isStreaming, ws, updateParams]
  );

  // ── Mid-stream prompt apply ────────────────────────────────────
  const handlePromptApply = useCallback(
    (prompt: string) => {
      ws.sendConfig({ prompt });
    },
    [ws]
  );

  // ── Helpers ────────────────────────────────────────────────────
  const hasPrompt = params.prompt.trim().length > 0;

  // ── Render ─────────────────────────────────────────────────────
  return (
    <main className="min-h-screen flex items-center justify-center p-6 md:p-10">
      <div className="w-full max-w-[1100px] flex flex-col gap-5">
        {/* ── Header ──────────────────────────────────────────── */}
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-[var(--color-text-primary)]">
              FLUX.2 Klein{" "}
              <span className="text-[var(--color-accent)]">Realtime</span>
            </h1>
            <p className="text-[12px] text-[var(--color-text-muted)] mt-0.5 mono">
              FLUX.2 Klein 4B Real-Time Webcam Stylization
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Webcam toggle */}
            <button
              onClick={() =>
                webcam.isActive
                  ? webcam.stop()
                  : webcam.start(CAPTURE_WIDTH, CAPTURE_HEIGHT)
              }
              disabled={isStreaming}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[12px] font-medium
                cursor-pointer transition-all duration-200 border
                ${
                  webcam.isActive
                    ? "bg-[var(--color-neon-green)]/10 border-[var(--color-neon-green)]/30 text-[var(--color-neon-green)]"
                    : "bg-[var(--color-surface-soft)] border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-border-bright)]"
                }
                disabled:opacity-50 disabled:cursor-not-allowed
              `}
            >
              {webcam.isActive ? (
                <>
                  <CameraOff className="w-3.5 h-3.5" />
                  Webcam On
                </>
              ) : (
                <>
                  <Camera className="w-3.5 h-3.5" />
                  Enable Webcam
                </>
              )}
            </button>

            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-neon-green)] pulse-glow" />
            <span className="text-[11px] mono text-[var(--color-text-muted)]">
              Modal H100
            </span>
          </div>
        </header>

        {/* ── Video Display Area ──────────────────────────────── */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Webcam input */}
          <div className="glass rounded-2xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] text-[var(--color-text-muted)] font-semibold uppercase tracking-wider">
                Webcam Input
              </span>
              {webcam.isActive && (
                <span className="flex items-center gap-1.5 text-[11px] text-[var(--color-neon-green)]">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-neon-green)] pulse-glow" />
                  Live
                </span>
              )}
            </div>
            {webcam.isActive ? (
              <video
                ref={webcam.videoRef}
                autoPlay
                playsInline
                muted
                className="w-full aspect-square object-cover block rounded-xl bg-black"
              />
            ) : (
              <div className="w-full aspect-square rounded-xl bg-black/50 flex flex-col items-center justify-center gap-3">
                <Camera className="w-8 h-8 text-[var(--color-text-muted)]/30" />
                <p className="text-[13px] text-[var(--color-text-muted)] text-center px-4">
                  Click{" "}
                  <span className="font-semibold text-[var(--color-accent)]">
                    Enable Webcam
                  </span>{" "}
                  to start your camera
                </p>
              </div>
            )}
            {webcam.error && (
              <p className="text-[12px] text-[var(--color-danger)] mt-2">
                {webcam.error}
              </p>
            )}
          </div>

          {/* Styled output */}
          <div className="glass rounded-2xl p-4">
            <div className="text-[11px] text-[var(--color-text-muted)] font-semibold mb-2 uppercase tracking-wider">
              Styled Output
            </div>
            <div className="aspect-square">
              <VideoCanvas
                frameBuffer={ws.frameBuffer}
                playbackFps={playbackFps}
                isPlaying={isStreaming}
                width={CAPTURE_WIDTH}
                height={CAPTURE_HEIGHT}
              />
            </div>
          </div>
        </section>

        {/* ── Status Bar ──────────────────────────────────────── */}
        <StatusBar
          connectionStatus={ws.status}
          frameCount={ws.frameCount}
          playbackFps={playbackFps}
          onPlaybackFpsChange={setPlaybackFps}
          error={ws.error}
        />

        {/* ── Style Gallery ───────────────────────────────────── */}
        <div>
          <p className="text-[12px] text-[var(--color-text-muted)] mb-2">
            Pick a style or write your own prompt below
          </p>
          <StyleGallery
            onSelect={handlePresetSelect}
            activePrompt={params.prompt}
          />
        </div>

        {/* ── Prompt Bar ──────────────────────────────────────── */}
        <PromptBar
          prompt={params.prompt}
          onPromptChange={(prompt) => updateParams({ prompt })}
          onApply={handlePromptApply}
          canApply={isStreaming}
        />

        {/* ── Action Buttons ──────────────────────────────────── */}
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={handleStart}
            disabled={isStreaming || !hasPrompt}
            className="flex items-center gap-2 px-6 py-3 rounded-xl text-[14px] font-bold
              cursor-pointer disabled:cursor-not-allowed disabled:opacity-40
              bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent-strong)]
              text-[#0b1120] transition-all duration-200
              hover:not-disabled:shadow-[0_12px_28px_rgba(37,99,235,0.3)]
              hover:not-disabled:-translate-y-0.5"
          >
            <Zap className="w-4 h-4" />
            Start Streaming
          </button>

          <button
            onClick={handleStop}
            disabled={!isStreaming}
            className="flex items-center gap-2 px-5 py-3 rounded-xl text-[14px] font-semibold
              cursor-pointer disabled:cursor-not-allowed disabled:opacity-40
              bg-[var(--color-surface-soft)] border border-[var(--color-border-bright)]
              text-[var(--color-text-primary)] transition-all duration-200
              hover:not-disabled:shadow-lg hover:not-disabled:-translate-y-0.5"
          >
            <Square className="w-4 h-4" />
            Stop
          </button>

          {/* Hint when no prompt */}
          {!hasPrompt && !isStreaming && (
            <span className="text-[12px] text-[var(--color-text-muted)] italic">
              Select a style preset or enter a prompt to start
            </span>
          )}
        </div>

        {/* ── Advanced Controls ────────────────────────────────── */}
        <Controls
          params={params}
          onParamsChange={updateParams}
          disabled={isStreaming}
        />

        {/* ── Footer ──────────────────────────────────────────── */}
        <footer className="text-center text-[11px] text-[var(--color-text-muted)]/50 pt-4">
          Powered by{" "}
          <a
            href="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-accent)]/60 hover:text-[var(--color-accent)]"
          >
            FLUX.2 Klein 4B
          </a>{" "}
          &middot;{" "}
          <a
            href="https://blackforestlabs.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-accent)]/60 hover:text-[var(--color-accent)]"
          >
            Black Forest Labs
          </a>{" "}
          &middot; Deployed on{" "}
          <a
            href="https://modal.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-accent)]/60 hover:text-[var(--color-accent)]"
          >
            Modal
          </a>
        </footer>
      </div>
    </main>
  );
}
