"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Zap, Square, Camera, Settings2, CreditCard, Circle, Disc } from "lucide-react";
import { useUser, useClerk, SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";
import { useSearchParams } from "next/navigation";

import { VideoCanvas } from "@/components/video-canvas";
import { PromptBar } from "@/components/prompt-bar";
import { StyleGallery } from "@/components/style-gallery";
import { Controls } from "@/components/controls";
import { StatusBar } from "@/components/status-bar";
import { PricingModal } from "@/components/pricing-modal";
import { useWebSocket } from "@/hooks/use-websocket";
import { useWebcam } from "@/hooks/use-webcam";
import { useUsage } from "@/hooks/use-usage";
import { useRecorder } from "@/hooks/use-recorder";
import type { StyleParams } from "@/lib/types";
import { DEFAULT_PARAMS } from "@/lib/types";
import { formatTime } from "@/lib/plans";

const CAPTURE_FPS = 15;
const CAPTURE_WIDTH = 512;
const CAPTURE_HEIGHT = 512;

export default function Home() {
  // ── State ──────────────────────────────────────────────────────
  const [params, setParams] = useState<StyleParams>({ ...DEFAULT_PARAMS });
  const [playbackFps, setPlaybackFps] = useState(15);
  const [showControls, setShowControls] = useState(false);
  const [showPricing, setShowPricing] = useState(false);
  const [checkoutNotice, setCheckoutNotice] = useState<string | null>(null);

  // ── Auth ────────────────────────────────────────────────────────
  const { isSignedIn, isLoaded: isAuthLoaded } = useUser();
  const clerk = useClerk();

  // ── Hooks ──────────────────────────────────────────────────────
  const ws = useWebSocket();
  const webcam = useWebcam();
  const usage = useUsage();

  // Webcam capture interval ref
  const webcamIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isStreaming = ws.status === "streaming";
  const isConnecting = ws.status === "connecting" || ws.status === "ready";

  // ── Recording ──────────────────────────────────────────────────
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { isRecording, startRecording, stopRecording, takeSnapshot } = useRecorder(canvasRef);

  // Auto-record when streaming starts
  useEffect(() => {
    if (isStreaming && !isRecording) {
      // Short delay to ensure first frames are ready
      const timer = setTimeout(() => {
        startRecording();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isStreaming, isRecording, startRecording]);

  // Stop recording when streaming stops
  useEffect(() => {
    if (!isStreaming && isRecording) {
      stopRecording();
    }
  }, [isStreaming, isRecording, stopRecording]);

  // ── Checkout return handling ────────────────────────────────────
  const searchParams = useSearchParams();
  useEffect(() => {
    const checkoutStatus = searchParams.get("checkout");
    if (checkoutStatus === "success") {
      setCheckoutNotice("Subscription activated! You can now start generating.");
      // Reload user metadata to pick up new plan from webhook
      usage.reloadUsage();
      // Clean up URL
      window.history.replaceState({}, "", window.location.pathname);
      // Auto-dismiss after 5 seconds
      setTimeout(() => setCheckoutNotice(null), 5000);
    } else if (checkoutStatus === "cancel") {
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, [searchParams, usage]);

  // ── Param updater ──────────────────────────────────────────────
  const updateParams = useCallback((partial: Partial<StyleParams>) => {
    setParams((prev) => ({ ...prev, ...partial }));
  }, []);

  // ── Start streaming ────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (!params.prompt.trim()) return;

    // Gate 1: Must be signed in
    if (!isSignedIn) {
      clerk.openSignIn({
        fallbackRedirectUrl: window.location.href,
      });
      return;
    }

    // Gate 2: Must have remaining usage time
    if (!usage.hasTimeRemaining) {
      setShowPricing(true);
      return;
    }

    // Start webcam first and wait for it to be ready
    if (!webcam.isActive) {
      await webcam.start(CAPTURE_WIDTH, CAPTURE_HEIGHT);
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    // Connect WebSocket with current params
    ws.connect(params);
  }, [params, webcam, ws, isSignedIn, clerk, usage]);

  // ── Stop streaming ─────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    ws.disconnect();
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current);
      webcamIntervalRef.current = null;
    }
    await usage.stopTimer();
  }, [ws, usage]);

  // ── Start usage timer when streaming begins ────────────────────
  useEffect(() => {
    if (isStreaming) {
      usage.startTimer();
    }
  }, [isStreaming, usage]);

  // ── Mid-stream usage cutoff ────────────────────────────────────
  useEffect(() => {
    if (isStreaming && !usage.hasTimeRemaining) {
      // Time's up — disconnect and show pricing
      handleStop();
      setShowPricing(true);
    }
  }, [isStreaming, usage.hasTimeRemaining, handleStop]);

  // ── Webcam frame capture loop ──────────────────────────────────
  useEffect(() => {
    if (isStreaming && webcam.isActive) {
      const interval = 1000 / CAPTURE_FPS;

      webcamIntervalRef.current = setInterval(async () => {
        const blob = await webcam.captureFrame(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        if (blob) {
          ws.sendFrame(blob);
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

  // ── Sync usage on tab close ────────────────────────────────────
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (isStreaming) {
        usage.stopTimer();
      }
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [isStreaming, usage]);

  // ── Style preset selection ─────────────────────────────────────
  const handlePresetSelect = useCallback(
    (prompt: string) => {
      updateParams({ prompt });

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

  // ── Manage subscription ────────────────────────────────────────
  const handleManageSubscription = useCallback(async () => {
    try {
      const res = await fetch("/api/portal", { method: "POST" });
      if (res.ok) {
        const { url } = await res.json();
        window.location.href = url;
      }
    } catch (error) {
      console.error("Failed to open portal:", error);
    }
  }, []);

  // ── Helpers ────────────────────────────────────────────────────
  const hasPrompt = params.prompt.trim().length > 0;

  // Usage display
  const remainingTimeDisplay =
    isSignedIn && usage.isLoaded
      ? formatTime(usage.remainingSeconds)
      : null;

  const usageUrgency: "normal" | "warning" | "critical" =
    usage.remainingSeconds <= 10
      ? "critical"
      : usage.remainingSeconds <= 30
        ? "warning"
        : "normal";

  // ── Render ─────────────────────────────────────────────────────
  return (
    <main className="h-screen flex flex-col overflow-hidden bg-[var(--color-bg)]">
      {/* ── Top Bar (auth + branding) ────────────────────────────── */}
      <div className="shrink-0 flex items-center justify-between px-4 pt-3 pb-1">
        <div className="text-[11px] text-zinc-500 mono">FLUX.2 Klein</div>
        <div className="flex items-center gap-2">
          {/* Manage subscription button (signed in with paid plan) */}
          <SignedIn>
            {usage.plan !== "free" && (
              <button
                onClick={handleManageSubscription}
                className="flex items-center gap-1.5 text-[11px] text-zinc-400 hover:text-white transition-colors cursor-pointer"
              >
                <CreditCard className="w-3 h-3" />
                Manage
              </button>
            )}
            {/* Upgrade button (free plan) */}
            {usage.plan === "free" && (
              <button
                onClick={() => setShowPricing(true)}
                className="flex items-center gap-1.5 text-[11px] text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer"
              >
                <Zap className="w-3 h-3" />
                Upgrade
              </button>
            )}
            <UserButton
              appearance={{
                elements: {
                  avatarBox: "w-7 h-7",
                },
              }}
            />
          </SignedIn>
          <SignedOut>
            <SignInButton mode="modal">
              <button className="text-[12px] text-zinc-400 hover:text-white transition-colors cursor-pointer px-3 py-1.5 rounded-lg hover:bg-white/5">
                Sign in
              </button>
            </SignInButton>
          </SignedOut>
        </div>
      </div>

      {/* ── Checkout success notice ──────────────────────────────── */}
      {checkoutNotice && (
        <div className="mx-4 mb-2 px-4 py-2.5 rounded-xl bg-green-500/10 border border-green-500/20 text-green-400 text-[12px] text-center">
          {checkoutNotice}
        </div>
      )}

      {/* ── Video Area (fills available space) ──────────────────── */}
      <div className="flex-1 flex items-center justify-center p-4 pb-0 min-h-0">
        <div className="flex gap-1 h-full max-h-[calc(100vh-260px)] w-full max-w-7xl">
          {/* Webcam input */}
          <div className="flex-1 relative rounded-2xl overflow-hidden bg-zinc-900 aspect-[4/3]">
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
          <div className="flex-1 relative rounded-2xl overflow-hidden bg-zinc-900 aspect-[4/3]">
            <div className="w-full h-full">
              <VideoCanvas
                ref={canvasRef}
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
          remainingTimeDisplay={remainingTimeDisplay}
          usageUrgency={usageUrgency}
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
                className={`p-2 rounded-lg transition-colors cursor-pointer ${showControls
                    ? "bg-white/10 text-white"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
                  }`}
                title="Parameters"
              >
                <Settings2 className="w-4 h-4" />
              </button>

              <div className="w-px h-5 bg-white/10" />

              {/* Snapshot */}
              <button
                onClick={takeSnapshot}
                disabled={!isStreaming}
                className="p-2 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
                title="Take Snapshot"
              >
                <Camera className="w-4 h-4" />
              </button>

              {/* Record (Manual toggle if needed, but mostly auto) */}
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={!isStreaming}
                className={`p-2 rounded-lg transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed ${isRecording
                    ? "text-red-500 hover:text-red-400 bg-red-500/10"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
                  }`}
                title={isRecording ? "Stop Recording" : "Start Recording"}
              >
                {isRecording ? <Disc className="w-4 h-4 animate-pulse" /> : <Circle className="w-4 h-4" />}
              </button>

              <div className="w-px h-5 bg-white/10" />

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
                  disabled={!hasPrompt || !isAuthLoaded}
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

      {/* ── Pricing Modal ─────────────────────────────────────────── */}
      <PricingModal
        isOpen={showPricing}
        onClose={() => setShowPricing(false)}
        currentPlan={usage.plan}
      />
    </main>
  );
}
