"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface VideoCanvasProps {
  /** Ref to the shared frame buffer (array of JPEG Blobs) */
  frameBuffer: React.MutableRefObject<Blob[]>;
  /** Target playback FPS */
  playbackFps: number;
  /** Whether generation is currently active */
  isPlaying: boolean;
  /** Canvas width */
  width?: number;
  /** Canvas height */
  height?: number;
}

/**
 * High-performance canvas renderer for streaming video frames.
 *
 * Uses createImageBitmap for off-main-thread JPEG decoding,
 * and requestAnimationFrame with interval throttling for smooth playback.
 * Includes a real-time FPS meter.
 */
export function VideoCanvas({
  frameBuffer,
  playbackFps,
  isPlaying,
  width = 832,
  height = 480,
}: VideoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentFps, setCurrentFps] = useState(0);
  const [bufferSize, setBufferSize] = useState(0);

  // FPS tracking refs
  const fpsFrameCount = useRef(0);
  const fpsLastTime = useRef(performance.now());
  const lastDrawTime = useRef(0);
  const animFrameId = useRef<number>(0);

  const drawFrame = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const now = performance.now();
    const interval = 1000 / Math.max(1, playbackFps);

    if (now - lastDrawTime.current < interval) {
      if (isPlaying) {
        animFrameId.current = requestAnimationFrame(drawFrame);
      }
      return;
    }

    // Pull next frame from buffer
    if (frameBuffer.current.length > 0) {
      const blob = frameBuffer.current.shift()!;

      try {
        const bitmap = await createImageBitmap(blob);
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        bitmap.close();

        lastDrawTime.current = now;
        fpsFrameCount.current++;

        // Drop old frames if buffer is backing up (keep at most 30)
        const maxBuffer = 30;
        if (frameBuffer.current.length > maxBuffer) {
          const toDrop = frameBuffer.current.length - maxBuffer;
          frameBuffer.current.splice(0, toDrop);
        }
      } catch {
        // Failed to decode frame, skip
      }
    }

    // Update FPS counter every second
    const elapsed = now - fpsLastTime.current;
    if (elapsed >= 1000) {
      setCurrentFps(
        Math.round((fpsFrameCount.current / elapsed) * 1000 * 10) / 10
      );
      fpsFrameCount.current = 0;
      fpsLastTime.current = now;
    }

    // Update buffer size display (throttled)
    setBufferSize(frameBuffer.current.length);

    if (isPlaying) {
      animFrameId.current = requestAnimationFrame(drawFrame);
    }
  }, [frameBuffer, playbackFps, isPlaying]);

  // Start/stop playback loop
  useEffect(() => {
    if (isPlaying) {
      animFrameId.current = requestAnimationFrame(drawFrame);
    }
    return () => {
      if (animFrameId.current) {
        cancelAnimationFrame(animFrameId.current);
      }
    };
  }, [isPlaying, drawFrame]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-auto block rounded-xl bg-black"
      />

      {/* FPS overlay */}
      <div className="absolute top-3 right-3 flex gap-2">
        <span className="mono text-[11px] px-2 py-1 rounded-md bg-black/60 text-[var(--color-neon-green)]">
          {currentFps.toFixed(1)} FPS
        </span>
        {bufferSize > 0 && (
          <span className="mono text-[11px] px-2 py-1 rounded-md bg-black/60 text-[var(--color-text-muted)]">
            BUF {bufferSize}
          </span>
        )}
      </div>

      {/* Scanline overlay */}
      <div className="scanlines absolute inset-0 rounded-xl pointer-events-none" />
    </div>
  );
}
