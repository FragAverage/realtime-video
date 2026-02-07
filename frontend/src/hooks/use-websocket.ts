"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { StyleParams, ServerMessage } from "@/lib/types";

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "ready"
  | "streaming"
  | "error";

interface UseWebSocketReturn {
  /** Current connection status */
  status: ConnectionStatus;
  /** Connect to the backend and configure the stream */
  connect: (params: StyleParams) => void;
  /** Disconnect and stop streaming */
  disconnect: () => void;
  /** Send a webcam frame as raw JPEG bytes */
  sendFrame: (jpeg: Blob | ArrayBuffer) => void;
  /** Send a prompt/config update as JSON */
  sendConfig: (params: Partial<StyleParams>) => void;
  /** Buffer of received styled frame Blobs ready for display */
  frameBuffer: React.MutableRefObject<Blob[]>;
  /** Error message if any */
  error: string | null;
  /** Number of styled frames received */
  frameCount: number;
}

/**
 * Hook managing the WebSocket connection to the FLUX.2 Klein backend.
 *
 * Protocol:
 * 1. Client connects to /ws
 * 2. Server sends JSON: { status: "ready" }
 * 3. Client sends JSON config: { prompt, strength, num_inference_steps, ... }
 * 4. Server sends JSON: { status: "streaming" }
 * 5. Client sends raw JPEG bytes (webcam frames)
 * 6. Server sends back raw JPEG bytes (styled frames)
 * 7. Client can send JSON text to update params mid-stream
 * 8. Close to stop
 */
export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [frameCount, setFrameCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const frameBuffer = useRef<Blob[]>([]);
  const frameCountRef = useRef(0);
  const statusRef = useRef<ConnectionStatus>("disconnected");

  // Keep statusRef in sync
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, "component unmount");
        wsRef.current = null;
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close(1000, "client disconnect");
      wsRef.current = null;
    }
    setStatus("disconnected");
  }, []);

  const connect = useCallback(
    (params: StyleParams) => {
      // Close existing connection
      if (wsRef.current) {
        wsRef.current.close(1000, "new connection");
        wsRef.current = null;
      }

      // Reset state
      frameBuffer.current = [];
      frameCountRef.current = 0;
      setFrameCount(0);
      setError(null);
      setStatus("connecting");

      const baseUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
      const url = `${baseUrl}/ws`;

      console.log("[ws] Connecting to:", url);

      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("[ws] Connection opened, waiting for ready...");
      };

      ws.onmessage = (event) => {
        // Text message = JSON status/config response
        if (typeof event.data === "string") {
          console.log("[ws] Text message:", event.data);
          handleJsonMessage(event.data);
          return;
        }

        // Binary message
        const buffer: ArrayBuffer = event.data;

        // Small binary might be JSON (proxy encoding quirk)
        if (buffer.byteLength < 512) {
          try {
            const text = new TextDecoder().decode(buffer);
            if (text.startsWith("{")) {
              console.log("[ws] Binary-as-JSON:", text);
              handleJsonMessage(text);
              return;
            }
          } catch {
            // Not text, treat as frame
          }
        }

        // Styled JPEG frame from server
        if (buffer.byteLength > 0) {
          frameBuffer.current.push(new Blob([buffer], { type: "image/jpeg" }));
          frameCountRef.current += 1;
          if (frameCountRef.current % 5 === 0 || frameCountRef.current < 10) {
            setFrameCount(frameCountRef.current);
          }
        }
      };

      function handleJsonMessage(text: string) {
        try {
          const msg: ServerMessage = JSON.parse(text);

          if (msg.status === "ready") {
            console.log("[ws] Server ready, sending config...");
            setStatus("ready");
            // Send initial config as JSON
            ws.send(JSON.stringify({
              prompt: params.prompt,
              negative_prompt: params.negative_prompt,
              seed: params.seed ?? 42,
              guidance_scale: params.guidance_scale,
              strength: params.strength,
              num_inference_steps: params.num_inference_steps,
              lora_id: params.lora_id ?? null,
            }));
            console.log("[ws] Config sent");
          }

          if (msg.status === "streaming") {
            console.log("[ws] Server streaming, send frames now");
            setStatus("streaming");
          }

          if (msg.status === "error" || msg.error) {
            const errStr = msg.error || "Unknown server error";
            console.error("[ws] Server error:", errStr);
            setError(errStr);
            setStatus("error");
          }
        } catch (e) {
          console.warn("[ws] Failed to parse JSON:", text, e);
        }
      }

      ws.onerror = (event) => {
        console.error("[ws] WebSocket error:", event);
        setError("WebSocket connection error");
        setStatus("error");
      };

      ws.onclose = (event) => {
        console.log(
          "[ws] Connection closed:",
          event.code,
          event.reason,
          "status:",
          statusRef.current
        );
        if (event.code !== 1000 && statusRef.current !== "disconnected") {
          setError(
            `Connection closed: ${event.reason || "code " + event.code}`
          );
          setStatus("error");
        } else if (statusRef.current === "streaming") {
          setStatus("disconnected");
        }
        setFrameCount(frameCountRef.current);
      };
    },
    []
  );

  const sendFrame = useCallback((jpeg: Blob | ArrayBuffer) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(jpeg);
  }, []);

  const sendConfig = useCallback((params: Partial<StyleParams>) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify(params));
  }, []);

  return {
    status,
    connect,
    disconnect,
    sendFrame,
    sendConfig,
    frameBuffer,
    error,
    frameCount,
  };
}
