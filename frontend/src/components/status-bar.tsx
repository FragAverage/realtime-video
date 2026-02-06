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
  disconnected: { label: "Disconnected", dotClass: "bg-gray-500" },
  connecting: {
    label: "Connecting... (GPU cold start can take 1-2 min)",
    dotClass: "bg-yellow-400 pulse-glow",
  },
  ready: { label: "Configuring...", dotClass: "bg-blue-400" },
  streaming: {
    label: "Streaming",
    dotClass: "bg-[var(--color-neon-green)] pulse-glow",
  },
  error: { label: "Error", dotClass: "bg-[var(--color-danger)]" },
};

/**
 * Status bar showing connection state, frame count, and playback FPS.
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
    <div
      className="flex items-center justify-between gap-4 flex-wrap
      text-[12px] text-[var(--color-text-muted)] px-1"
    >
      {/* Connection status */}
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${dotClass}`} />
        <span className="mono">
          {label}
          {error && connectionStatus === "error" && (
            <span className="text-[var(--color-danger)] ml-2">{error}</span>
          )}
        </span>
      </div>

      {/* Readouts */}
      <div className="flex items-center gap-4">
        <span className="mono">
          Frames{" "}
          <span className="text-[var(--color-text-primary)]">{frameCount}</span>
        </span>

        {/* Playback FPS slider */}
        <div className="flex items-center gap-2">
          <span>Playback</span>
          <input
            type="range"
            min={1}
            max={30}
            value={playbackFps}
            onChange={(e) => onPlaybackFpsChange(Number(e.target.value))}
            className="w-20"
          />
          <span className="mono text-[var(--color-text-primary)] w-5 text-right">
            {playbackFps}
          </span>
        </div>
      </div>
    </div>
  );
}
