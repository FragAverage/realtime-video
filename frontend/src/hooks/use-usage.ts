"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useUser } from "@clerk/nextjs";
import { PLANS, type PlanId } from "@/lib/plans";
import type { UserUsageMetadata } from "@/lib/types";

/** How often (in seconds) to sync usage to the server while streaming */
const SYNC_INTERVAL_SECONDS = 10;

export interface UseUsageReturn {
  /** Current plan ID */
  plan: PlanId;
  /** Total seconds used (server value + local unsent delta) */
  totalSecondsUsed: number;
  /** Seconds remaining before limit is hit */
  remainingSeconds: number;
  /** Whether the user has any remaining time */
  hasTimeRemaining: boolean;
  /** Whether usage data has loaded from Clerk */
  isLoaded: boolean;
  /** Start the streaming timer */
  startTimer: () => void;
  /** Stop the timer and do a final sync */
  stopTimer: () => Promise<void>;
  /** Force reload usage data from Clerk */
  reloadUsage: () => void;
}

export function useUsage(): UseUsageReturn {
  const { user, isLoaded: isUserLoaded } = useUser();

  // Server-confirmed usage
  const [serverSecondsUsed, setServerSecondsUsed] = useState(0);
  const [plan, setPlan] = useState<PlanId>("free");

  // Local timer state
  const [localElapsed, setLocalElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const unsynced = useRef(0); // seconds accumulated since last sync
  const isSyncing = useRef(false);

  // Load initial usage from Clerk metadata
  useEffect(() => {
    if (!isUserLoaded || !user) return;

    const meta = (user.publicMetadata ?? {}) as Partial<UserUsageMetadata>;
    const userPlan = meta.plan ?? "free";
    const usageSeconds = meta.usageSecondsUsed ?? 0;

    setPlan(userPlan);
    setServerSecondsUsed(usageSeconds);
  }, [isUserLoaded, user]);

  const planLimit = PLANS[plan].limitSeconds;
  const totalSecondsUsed = serverSecondsUsed + localElapsed;
  const remainingSeconds = Math.max(0, planLimit - totalSecondsUsed);
  const hasTimeRemaining = remainingSeconds > 0;

  // Sync accumulated seconds to the server
  const syncToServer = useCallback(
    async (seconds: number) => {
      if (seconds <= 0 || isSyncing.current) return;
      isSyncing.current = true;
      try {
        const res = await fetch("/api/usage/sync", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ secondsToAdd: seconds }),
        });
        if (res.ok) {
          const data = await res.json();
          setServerSecondsUsed(data.usageSecondsUsed);
          setPlan(data.plan);
          unsynced.current = 0;
        }
      } catch {
        // Keep unsynced seconds for next attempt
      } finally {
        isSyncing.current = false;
      }
    },
    []
  );

  // Start the streaming timer (call when WebSocket enters "streaming" state)
  const startTimer = useCallback(() => {
    if (timerRef.current) return; // already running
    setLocalElapsed(0);
    unsynced.current = 0;

    timerRef.current = setInterval(() => {
      setLocalElapsed((prev) => prev + 1);
      unsynced.current += 1;

      // Periodic sync
      if (unsynced.current >= SYNC_INTERVAL_SECONDS) {
        const toSync = unsynced.current;
        unsynced.current = 0;
        syncToServer(toSync);
      }
    }, 1000);
  }, [syncToServer]);

  // Stop the timer and flush remaining seconds
  const stopTimer = useCallback(async () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    // Final sync of any unsynced seconds
    if (unsynced.current > 0) {
      await syncToServer(unsynced.current);
    }
    setLocalElapsed(0);
  }, [syncToServer]);

  // Reload usage from Clerk (call after checkout return)
  const reloadUsage = useCallback(() => {
    if (user) {
      user.reload().then(() => {
        const meta = (user.publicMetadata ?? {}) as Partial<UserUsageMetadata>;
        setPlan(meta.plan ?? "free");
        setServerSecondsUsed(meta.usageSecondsUsed ?? 0);
      });
    }
  }, [user]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  return {
    plan,
    totalSecondsUsed,
    remainingSeconds,
    hasTimeRemaining,
    isLoaded: isUserLoaded,
    startTimer,
    stopTimer,
    reloadUsage,
  };
}
