import { NextRequest, NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import { getChargebee } from "@/lib/chargebee";

/**
 * Creates a Chargebee self-serve portal session for subscription management.
 * Users can update payment methods, change plans, or cancel.
 */
export async function POST(req: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const chargebee = getChargebee();
  const appUrl = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000";

  try {
    const result = await chargebee.portalSession.create({
      customer: {
        id: `clerk_${userId}`,
      },
      redirect_url: appUrl,
    });

    return NextResponse.json({ url: result.portal_session.access_url });
  } catch (error: unknown) {
    console.error("Chargebee portal error:", error);
    const message =
      error instanceof Error ? error.message : "Portal session failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
