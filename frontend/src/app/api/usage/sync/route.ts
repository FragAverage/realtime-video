import { NextRequest, NextResponse } from "next/server";
import { auth, clerkClient } from "@clerk/nextjs/server";
import { PLANS, type PlanId } from "@/lib/plans";
import type { UserUsageMetadata } from "@/lib/types";

export async function POST(req: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const secondsToAdd = Math.max(0, Math.round(body.secondsToAdd ?? 0));

  if (secondsToAdd === 0) {
    return NextResponse.json({ error: "No seconds to add" }, { status: 400 });
  }

  const client = await clerkClient();
  const user = await client.users.getUser(userId);
  const meta = (user.publicMetadata ?? {}) as Partial<UserUsageMetadata>;

  let plan: PlanId = meta.plan ?? "free";
  let usageSecondsUsed = meta.usageSecondsUsed ?? 0;
  let usagePeriodStart = meta.usagePeriodStart ?? new Date().toISOString();

  // For paid plans, check if we need to reset the monthly counter
  if (plan !== "free") {
    const periodStart = new Date(usagePeriodStart);
    const now = new Date();

    // Reset if the period start is from a previous month
    const monthsSincePeriodStart =
      (now.getFullYear() - periodStart.getFullYear()) * 12 +
      (now.getMonth() - periodStart.getMonth());

    if (monthsSincePeriodStart >= 1) {
      usageSecondsUsed = 0;
      usagePeriodStart = now.toISOString();
    }
  }

  // Add the new seconds
  usageSecondsUsed += secondsToAdd;

  // Compute remaining
  const planLimit = PLANS[plan].limitSeconds;
  const remainingSeconds = Math.max(0, planLimit - usageSecondsUsed);

  // Update Clerk metadata
  await client.users.updateUserMetadata(userId, {
    publicMetadata: {
      ...meta,
      plan,
      usageSecondsUsed,
      usagePeriodStart,
    },
  });

  return NextResponse.json({
    plan,
    usageSecondsUsed,
    remainingSeconds,
    limitSeconds: planLimit,
  });
}
