import { useCallback, useRef, useState } from "react";

interface UseRecorderReturn {
  isRecording: boolean;
  startRecording: () => void;
  stopRecording: () => void;
  takeSnapshot: () => void;
  error: string | null;
}

export function useRecorder(
  canvasRef: React.RefObject<HTMLCanvasElement | null>
): UseRecorderReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      setError("No canvas to record");
      return;
    }

    try {
      // 30 FPS stream
      const stream = canvas.captureStream(30);
      
      // Try widely supported mime types
      let mimeType = "video/webm;codecs=vp9";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
         mimeType = "video/webm;codecs=vp8";
      }
      if (!MediaRecorder.isTypeSupported(mimeType)) {
         mimeType = "video/webm";
      }
      // MP4 fallback for Safari/others if webm fails (though webm is standard for MediaRecorder)
      if (!MediaRecorder.isTypeSupported(mimeType) && MediaRecorder.isTypeSupported("video/mp4")) {
        mimeType = "video/mp4";
      }

      const recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 5000000, // 5 Mbps
      });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const ext = mimeType.includes("mp4") ? "mp4" : "webm";
        a.download = `recording-${new Date().toISOString()}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
        chunksRef.current = [];
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setError(null);
    } catch (err) {
      console.error("Failed to start recording:", err);
      setError("Failed to start recording");
      setIsRecording(false);
    }
  }, [canvasRef]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, []);

  const takeSnapshot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      const dataUrl = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `snapshot-${new Date().toISOString()}.png`;
      a.click();
    } catch (err) {
      console.error("Failed to take snapshot:", err);
      setError("Failed to take snapshot");
    }
  }, [canvasRef]);

  return {
    isRecording,
    startRecording,
    stopRecording,
    takeSnapshot,
    error,
  };
}
