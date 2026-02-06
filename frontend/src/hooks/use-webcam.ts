"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface UseWebcamReturn {
  /** The MediaStream from getUserMedia */
  stream: MediaStream | null;
  /** Whether the webcam is currently active */
  isActive: boolean;
  /** Start the webcam */
  start: (targetWidth?: number, targetHeight?: number) => Promise<void>;
  /** Stop the webcam and release resources */
  stop: () => void;
  /** Capture a single frame as a base64 JPEG string (no data: prefix) */
  captureFrame: (
    targetWidth: number,
    targetHeight: number,
    quality?: number
  ) => string | null;
  /** Ref to attach to a <video> element for preview */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Error message if getUserMedia fails */
  error: string | null;
}

/**
 * Hook for managing webcam access via getUserMedia.
 * Provides capture-on-demand for sending frames to the backend.
 *
 * The captured frames are center-cropped to the target aspect ratio
 * and scaled to the target resolution before JPEG encoding.
 */
export function useWebcam(): UseWebcamReturn {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Ensure canvas exists
  useEffect(() => {
    if (typeof document !== "undefined" && !canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
  }, []);

  // Attach stream to video element whenever stream changes.
  // This handles the race condition where start() sets the stream
  // before React renders the <video> element (which depends on isActive).
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = useCallback(
    async (targetWidth = 832, targetHeight = 480) => {
      // Stop existing stream
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      try {
        setError(null);
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: targetWidth },
            height: { ideal: targetHeight },
            facingMode: "user",
          },
          audio: false,
        });

        setStream(mediaStream);
        setIsActive(true);

        // Attach to video element for preview
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to access webcam";
        setError(message);
        setIsActive(false);
      }
    },
    [stream]
  );

  const stop = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStream(null);
    setIsActive(false);
  }, [stream]);

  const captureFrame = useCallback(
    (
      targetWidth: number,
      targetHeight: number,
      quality = 0.7
    ): string | null => {
      if (!videoRef.current || !canvasRef.current || !isActive) return null;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      // Snap to nearest multiple of 8 (Krea model requirement)
      const w = Math.round(targetWidth / 8) * 8;
      const h = Math.round(targetHeight / 8) * 8;
      canvas.width = w;
      canvas.height = h;

      const ctx = canvas.getContext("2d");
      if (!ctx) return null;

      // Center-crop the video to the target aspect ratio
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      if (videoWidth === 0 || videoHeight === 0) return null;

      const targetAspect = w / h;
      const videoAspect = videoWidth / videoHeight;

      let sx: number, sy: number, sWidth: number, sHeight: number;

      if (videoAspect > targetAspect) {
        // Video is wider -- crop sides
        sHeight = videoHeight;
        sWidth = videoHeight * targetAspect;
        sx = (videoWidth - sWidth) / 2;
        sy = 0;
      } else {
        // Video is taller -- crop top/bottom
        sWidth = videoWidth;
        sHeight = videoWidth / targetAspect;
        sx = 0;
        sy = (videoHeight - sHeight) / 2;
      }

      ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, w, h);

      // Convert to base64 JPEG (strip the data:image/jpeg;base64, prefix)
      const dataUrl = canvas.toDataURL("image/jpeg", quality);
      return dataUrl.split(",")[1] || null;
    },
    [isActive]
  );

  return {
    stream,
    isActive,
    start,
    stop,
    captureFrame,
    videoRef,
    error,
  };
}
