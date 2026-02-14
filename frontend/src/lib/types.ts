/**
 * Shared TypeScript types for the FLUX.2 Klein real-time stylization app.
 */

import type { PlanId } from "@/lib/plans";

/**
 * User metadata stored in Clerk publicMetadata.
 * Readable on both client and server.
 */
export interface UserUsageMetadata {
  plan: PlanId;
  usageSecondsUsed: number;
  /** ISO date string — start of current billing period (or account creation for free) */
  usagePeriodStart: string;
  chargebeeCustomerId?: string;
  subscriptionId?: string;
  subscriptionStatus?: string;
}

/**
 * Style parameters sent to the backend as JSON when configuring the stream.
 * These control the FLUX.2 Klein pipeline.
 */
export interface StyleParams {
  prompt: string;
  negative_prompt: string;
  seed: number | null;
  guidance_scale: number;
  strength: number;
  num_inference_steps: number;
}

/**
 * Server status messages received over WebSocket.
 */
export interface ServerMessage {
  status?: "ready" | "streaming" | "error";
  worker?: string;
  error?: string;
}

/**
 * Style preset for the gallery.
 */
export interface StylePreset {
  id: string;
  label: string;
  prompt: string;
  color: string; // Tailwind color class for the button accent
}

/**
 * Default style parameters.
 */
export const DEFAULT_PARAMS: StyleParams = {
  prompt: "",
  negative_prompt: "",
  seed: null,
  guidance_scale: 1.0,
  strength: 0.4,
  num_inference_steps: 4,
};
