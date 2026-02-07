"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Zap, Square, Camera, Settings2 } from "lucide-react";

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
const CAPTURE_WIDTH = 384;
const CAPTURE_HEIGHT = 384;

export default function Home() {
  // ── State ──────────────────────────────────────────────────────
  const [params, setParams] = useState<StyleParams>({ ...DEFAULT_PARAMS });
  const [playbackFps, setPlaybackFps] = useState(15);
  const [showControls, setShowControls] = useState(false);

  // ── Hooks ──────────────────────────────────────────────────────
  const ws = useWebSocket();
  const webcam = useWebcam();

  // Webcam capture interval ref
  const webcamIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isStreaming = ws.status === "streaming";
  const isConnecting = ws.status === "connecting" || ws.status === "ready";

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
  useEffect(() => {
    if (isStreaming && webcam.isActive) {
      const interval = 1000 / CAPTURE_FPS;

      webcamIntervalRef.current = setInterval(() => {
        const frame = webcam.captureFrame(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        if (frame) {
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
    (prompt: string, loraId: string | null) => {
      updateParams({ prompt, lora_id: loraId });

      if (isStreaming) {
        ws.sendConfig({ prompt, lora_id: loraId });
      }
    },
    [isStreaming, ws, updateParams]
  );

  // ── Mid-stream prompt apply ────────────────────────────────────
  const handlePromptApply = useCallback(
    (prompt: string) => {
      ws.sendConfig({ prompt, lora_id: params.lora_id });
    },
    [ws, params.lora_id]
  );

  // ── Helpers ────────────────────────────────────────────────────
  const hasPrompt = params.prompt.trim().length > 0;

  // ── Render ─────────────────────────────────────────────────────
  return (
    <main className="h-screen flex flex-col overflow-hidden bg-[var(--color-bg)]">
      {/* ── Video Area (fills available space) ──────────────────── */}
      <div className="flex-1 flex items-center justify-center p-4 pb-0 min-h-0">
        <div className="flex gap-1 h-full max-h-[calc(100vh-220px)] w-full max-w-5xl">
          {/* Webcam input */}
          <div className="flex-1 relative rounded-2xl overflow-hidden bg-zinc-900">
            {webcam.isActive ? (
              <video
                ref={webcam.videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center gap-3">
                <Camera className="w-10 h-10 text-zinc-600" />
                <p className="text-sm text-zinc-500 text-center px-4">
                  Camera will activate when you start streaming
                </p>
              </div>
            )}
            {/* Webcam label */}
            <div className="absolute top-3 left-3">
              <span className="text-[11px] font-medium text-white/60 bg-black/40 backdrop-blur-sm px-2.5 py-1 rounded-full">
                Webcam
              </span>
            </div>
            {webcam.isActive && (
              <div className="absolute top-3 right-3">
                <span className="flex items-center gap-1.5 text-[11px] text-white/80 bg-black/40 backdrop-blur-sm px-2.5 py-1 rounded-full">
                  <span className="w-1.5 h-1.5 rounded-full bg-red-500 pulse-glow" />
                  Live
                </span>
              </div>
            )}
            {webcam.error && (
              <div className="absolute bottom-3 left-3 right-3">
                <p className="text-[12px] text-red-400 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg">
                  {webcam.error}
                </p>
              </div>
            )}
          </div>

          {/* Styled output */}
          <div className="flex-1 relative rounded-2xl overflow-hidden bg-zinc-900">
            <div className="w-full h-full">
              <VideoCanvas
                frameBuffer={ws.frameBuffer}
                playbackFps={playbackFps}
                isPlaying={isStreaming}
                width={CAPTURE_WIDTH}
                height={CAPTURE_HEIGHT}
              />
            </div>
            {/* Output label */}
            <div className="absolute top-3 left-3">
              <span className="text-[11px] font-medium text-white/60 bg-black/40 backdrop-blur-sm px-2.5 py-1 rounded-full">
                Output
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Status indicator (between video and controls) ───────── */}
      <div className="flex justify-center py-2">
        <StatusBar
          connectionStatus={ws.status}
          frameCount={ws.frameCount}
          playbackFps={playbackFps}
          onPlaybackFpsChange={setPlaybackFps}
          error={ws.error}
        />
      </div>

      {/* ── Bottom Controls Dock ────────────────────────────────── */}
      <div className="shrink-0 px-4 pb-4">
        <div className="max-w-3xl mx-auto flex flex-col gap-3">
          {/* Prompt input + action button */}
          <div className="relative">
            <PromptBar
              prompt={params.prompt}
              onPromptChange={(prompt) => updateParams({ prompt })}
              onApply={handlePromptApply}
              canApply={isStreaming}
              onSubmit={handleStart}
              disabled={false}
            />
            {/* Action button inside the prompt bar area */}
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
              {/* Settings toggle */}
              <button
                onClick={() => setShowControls(!showControls)}
                className={`p-2 rounded-lg transition-colors cursor-pointer ${
                  showControls
                    ? "bg-white/10 text-white"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
                }`}
                title="Parameters"
              >
                <Settings2 className="w-4 h-4" />
              </button>

              {/* Start / Stop button */}
              {isStreaming || isConnecting ? (
                <button
                  onClick={handleStop}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-[13px] font-semibold
                    cursor-pointer bg-red-500/90 hover:bg-red-500 text-white transition-all"
                >
                  <Square className="w-3.5 h-3.5" />
                  Stop
                </button>
              ) : (
                <button
                  onClick={handleStart}
                  disabled={!hasPrompt}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-[13px] font-semibold
                    cursor-pointer disabled:cursor-not-allowed disabled:opacity-30
                    bg-white text-black hover:bg-zinc-200 transition-all"
                >
                  <Zap className="w-3.5 h-3.5" />
                  Start
                </button>
              )}
            </div>
          </div>

          {/* Style presets row */}
          <StyleGallery
            onSelect={handlePresetSelect}
            activePrompt={params.prompt}
          />

          {/* Collapsible advanced controls */}
          {showControls && (
            <Controls
              params={params}
              onParamsChange={updateParams}
              disabled={isStreaming}
            />
          )}
        </div>
      </div>
    </main>
  );
}
