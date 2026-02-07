"use client";

import type { ConnectionStatus } from "@/hooks/use-websocket";

interface StatusBarProps {
  /** Current WebSocket connection status */
  connectionStatus: ConnectionStatus;
  /** Number of styled frames received */
  frameCount: number;
  /** Playback FPS setting */
  playbackFps: number;
  /** Called when playback FPS changes */
  onPlaybackFpsChange: (fps: number) => void;
  /** Error message if any */
  error: string | null;
}

const STATUS_CONFIG: Record<
  ConnectionStatus,
  { label: string; dotClass: string }
> = {
  disconnected: { label: "Ready", dotClass: "bg-zinc-500" },
  connecting: {
    label: "Connecting... GPU cold start ~1-2 min",
    dotClass: "bg-yellow-400 pulse-glow",
  },
  ready: { label: "Configuring...", dotClass: "bg-blue-400" },
  streaming: {
    label: "Streaming",
    dotClass: "bg-green-400 pulse-glow",
  },
  error: { label: "Error", dotClass: "bg-red-500" },
};

/**
 * Minimal inline status indicator.
 */
export function StatusBar({
  connectionStatus,
  frameCount,
  playbackFps,
  onPlaybackFpsChange,
  error,
}: StatusBarProps) {
  const { label, dotClass } = STATUS_CONFIG[connectionStatus];

  return (
    <div className="flex items-center gap-4 text-[11px] text-zinc-500">
      {/* Connection status */}
      <div className="flex items-center gap-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
        <span className="mono">
          {label}
          {error && connectionStatus === "error" && (
            <span className="text-red-400 ml-1.5">{error}</span>
          )}
        </span>
      </div>

      {connectionStatus === "streaming" && (
        <>
          <span className="text-zinc-700">|</span>
          <span className="mono">
            {frameCount} frames
          </span>
          <span className="text-zinc-700">|</span>
          <div className="flex items-center gap-1.5">
            <span>Playback</span>
            <input
              type="range"
              min={1}
              max={30}
              value={playbackFps}
              onChange={(e) => onPlaybackFpsChange(Number(e.target.value))}
              className="w-16"
            />
            <span className="mono text-zinc-400 w-4 text-right">
              {playbackFps}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
