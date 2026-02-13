import { NextRequest, NextResponse } from "next/server";
import { clerkClient } from "@clerk/nextjs/server";
import type { PlanId } from "@/lib/plans";
import type { UserUsageMetadata } from "@/lib/types";

/**
 * Chargebee webhook handler.
 * This route is NOT protected by Clerk middleware (webhooks come from Chargebee).
 */
export async function POST(req: NextRequest) {
  const event = await req.json();
  const eventType = event.event_type as string;
  const content = event.content ?? {};

  console.log(`[Chargebee Webhook] ${eventType}`);

  try {
    switch (eventType) {
      case "subscription_created":
      case "subscription_activated": {
        await handleSubscriptionActive(content);
        break;
      }

      case "subscription_renewed": {
        await handleSubscriptionRenewed(content);
        break;
      }

      case "subscription_cancelled":
      case "subscription_deleted": {
        await handleSubscriptionCancelled(content);
        break;
      }

      case "subscription_changed": {
        await handleSubscriptionChanged(content);
        break;
      }

      default:
        console.log(`[Chargebee Webhook] Unhandled event: ${eventType}`);
    }
  } catch (error) {
    console.error(`[Chargebee Webhook] Error processing ${eventType}:`, error);
    // Still return 200 to prevent retries for processing errors
  }

  return NextResponse.json({ received: true }, { status: 200 });
}

/** Extract Clerk user ID from Chargebee customer or pass_thru_content */
function extractClerkUserId(content: Record<string, unknown>): string | null {
  const subscription = content.subscription as Record<string, unknown> | undefined;
  const customer = content.customer as Record<string, unknown> | undefined;

  // Try pass_thru_content on the hosted page
  const hostedPage = content.hosted_page as Record<string, unknown> | undefined;
  if (hostedPage?.pass_thru_content) {
    try {
      const passThru = JSON.parse(hostedPage.pass_thru_content as string);
      if (passThru.clerkUserId) return passThru.clerkUserId;
    } catch { /* ignore parse errors */ }
  }

  // Try customer ID pattern (clerk_user_xxx)
  const customerId = (customer?.id ?? subscription?.customer_id) as string | undefined;
  if (customerId?.startsWith("clerk_")) {
    return customerId.replace("clerk_", "");
  }

  // Try cf_clerk_user_id custom field
  const cfClerkUserId = customer?.cf_clerk_user_id as string | undefined;
  if (cfClerkUserId) return cfClerkUserId;

  return null;
}

/** Map Chargebee item_price_id to our plan ID */
function itemPriceToPlan(subscription: Record<string, unknown>): PlanId {
  const items = subscription.subscription_items as Array<Record<string, unknown>> | undefined;
  const itemPriceId = items?.[0]?.item_price_id as string | undefined;

  if (itemPriceId?.includes("ultra")) return "ultra";
  if (itemPriceId?.includes("pro")) return "pro";
  return "free";
}

async function handleSubscriptionActive(content: Record<string, unknown>) {
  const clerkUserId = extractClerkUserId(content);
  if (!clerkUserId) {
    console.error("[Chargebee Webhook] Could not extract Clerk user ID");
    return;
  }

  const subscription = content.subscription as Record<string, unknown>;
  const customer = content.customer as Record<string, unknown>;
  const plan = itemPriceToPlan(subscription);

  const client = await clerkClient();
  const user = await client.users.getUser(clerkUserId);
  const existingMeta = (user.publicMetadata ?? {}) as Partial<UserUsageMetadata>;

  await client.users.updateUserMetadata(clerkUserId, {
    publicMetadata: {
      ...existingMeta,
      plan,
      usageSecondsUsed: 0,
      usagePeriodStart: new Date().toISOString(),
      chargebeeCustomerId: customer.id as string,
      subscriptionId: subscription.id as string,
      subscriptionStatus: subscription.status as string,
    },
  });

  console.log(`[Chargebee Webhook] Activated ${plan} plan for user ${clerkUserId}`);
}

async function handleSubscriptionRenewed(content: Record<string, unknown>) {
  const clerkUserId = extractClerkUserId(content);
  if (!clerkUserId) return;

  const subscription = content.subscription as Record<string, unknown>;
  const plan = itemPriceToPlan(subscription);

  const client = await clerkClient();
  const user = await client.users.getUser(clerkUserId);
  const existingMeta = (user.publicMetadata ?? {}) as Partial<UserUsageMetadata>;

  // Reset usage for the new billing period
  await client.users.updateUserMetadata(clerkUserId, {
    publicMetadata: {
      ...existingMeta,
      plan,
      usageSecondsUsed: 0,
      usagePeriodStart: new Date().toISOString(),
      subscriptionStatus: subscription.status as string,
    },
  });

  console.log(`[Chargebee Webhook] Renewed ${plan} plan for user ${clerkUserId}`);
}

async function handleSubscriptionCancelled(content: Record<string, unknown>) {
  const clerkUserId = extractClerkUserId(content);
  if (!clerkUserId) return;

  const subscription = content.subscription as Record<string, unknown>;

  const client = await clerkClient();
  const user = await client.users.getUser(clerkUserId);
  const existingMeta = (user.publicMetadata ?? {}) as Partial<UserUsageMetadata>;

  await client.users.updateUserMetadata(clerkUserId, {
    publicMetadata: {
      ...existingMeta,
      plan: "free",
      subscriptionStatus: subscription.status as string,
    },
  });

  console.log(`[Chargebee Webhook] Cancelled subscription for user ${clerkUserId}`);
}

async function handleSubscriptionChanged(content: Record<string, unknown>) {
  const clerkUserId = extractClerkUserId(content);
  if (!clerkUserId) return;

  const subscription = content.subscription as Record<string, unknown>;
  const plan = itemPriceToPlan(subscription);

  const client = await clerkClient();
  const user = await client.users.getUser(clerkUserId);
  const existingMeta = (user.publicMetadata ?? {}) as Partial<UserUsageMetadata>;

  await client.users.updateUserMetadata(clerkUserId, {
    publicMetadata: {
      ...existingMeta,
      plan,
      subscriptionStatus: subscription.status as string,
    },
  });

  console.log(`[Chargebee Webhook] Changed to ${plan} plan for user ${clerkUserId}`);
}
