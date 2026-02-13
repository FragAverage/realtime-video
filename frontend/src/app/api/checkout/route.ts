import { NextRequest, NextResponse } from "next/server";
import { auth, clerkClient } from "@clerk/nextjs/server";
import { getChargebee } from "@/lib/chargebee";
import { PLANS, type PlanId } from "@/lib/plans";

export async function POST(req: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const planId = body.planId as PlanId;

  const planDef = PLANS[planId];
  if (!planDef || !planDef.chargebeeItemPriceId) {
    return NextResponse.json(
      { error: "Invalid plan" },
      { status: 400 }
    );
  }

  // Get user info from Clerk for pre-filling checkout
  const client = await clerkClient();
  const user = await client.users.getUser(userId);
  const email = user.emailAddresses[0]?.emailAddress;
  const firstName = user.firstName ?? undefined;
  const lastName = user.lastName ?? undefined;

  const chargebee = getChargebee();
  const appUrl = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000";

  try {
    const result = await chargebee.hostedPage.checkoutNewForItems({
      subscription_items: [
        {
          item_price_id: planDef.chargebeeItemPriceId,
          quantity: 1,
        },
      ],
      customer: {
        id: `clerk_${userId}`,
        email,
        first_name: firstName,
        last_name: lastName,
      },
      redirect_url: `${appUrl}?checkout=success`,
      cancel_url: `${appUrl}?checkout=cancel`,
      pass_thru_content: JSON.stringify({
        clerkUserId: userId,
        planId,
      }),
    });

    const hostedPage = result.hosted_page;

    return NextResponse.json({ url: hostedPage.url });
  } catch (error: unknown) {
    console.error("Chargebee checkout error:", error);
    const message =
      error instanceof Error ? error.message : "Checkout failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
