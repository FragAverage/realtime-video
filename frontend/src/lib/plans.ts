/**
 * Subscription plan definitions and usage limits.
 * Plan IDs and item_price_ids match the Chargebee Product Catalog.
 */

export type PlanId = "free" | "pro" | "ultra";

export interface PlanDefinition {
  id: PlanId;
  label: string;
  /** Price in pence (GBP) */
  priceMonthly: number;
  /** Usage limit in seconds. For free plan, this is lifetime. For paid, monthly. */
  limitSeconds: number;
  /** Chargebee item_price_id, or null for free tier */
  chargebeeItemPriceId: string | null;
  features: string[];
}

export const PLANS: Record<PlanId, PlanDefinition> = {
  free: {
    id: "free",
    label: "Free",
    priceMonthly: 0,
    limitSeconds: 60,
    chargebeeItemPriceId: null,
    features: [
      "1 minute of generation (lifetime)",
      "All style presets",
      "384x384 output",
    ],
  },
  pro: {
    id: "pro",
    label: "Pro",
    priceMonthly: 1999,
    limitSeconds: 600,
    chargebeeItemPriceId: "realtime-pro-GBP-Monthly",
    features: [
      "10 minutes of generation / month",
      "All style presets",
      "384x384 output",
      "Priority GPU access",
    ],
  },
  ultra: {
    id: "ultra",
    label: "Ultra",
    priceMonthly: 3999,
    limitSeconds: 3600,
    chargebeeItemPriceId: "realtime-ultra-GBP-Monthly",
    features: [
      "60 minutes of generation / month",
      "All style presets",
      "384x384 output",
      "Priority GPU access",
      "Early access to new models",
    ],
  },
} as const;

/** Format pence as pound string (e.g., 1999 -> "£19.99") */
export function formatPrice(pence: number): string {
  return `\u00A3${(pence / 100).toFixed(2)}`;
}

/** Format seconds as "M:SS" or "H:MM:SS" */
export function formatTime(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const seconds = s % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}
