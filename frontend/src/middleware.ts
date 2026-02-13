import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

/**
 * Protect API routes that require authentication.
 * The main page is public — auth gating happens at the component level (Start button).
 * Webhook route is excluded since Chargebee calls it directly.
 */
const isProtectedRoute = createRouteMatcher([
  "/api/usage(.*)",
  "/api/checkout(.*)",
  "/api/portal(.*)",
]);

export default clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    // Skip Next.js internals and all static files
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes
    "/(api|trpc)(.*)",
  ],
};
