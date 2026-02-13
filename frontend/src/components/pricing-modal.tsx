"use client";

import { useCallback, useState } from "react";
import { X, Check, Zap, Crown } from "lucide-react";
import { PLANS, formatPrice, type PlanId } from "@/lib/plans";

interface PricingModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentPlan: PlanId;
  /** Called when the user is redirected to checkout */
  onCheckout?: () => void;
}

export function PricingModal({
  isOpen,
  onClose,
  currentPlan,
  onCheckout,
}: PricingModalProps) {
  const [loading, setLoading] = useState<PlanId | null>(null);

  const handleSubscribe = useCallback(
    async (planId: PlanId) => {
      setLoading(planId);
      try {
        const res = await fetch("/api/checkout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ planId }),
        });

        if (!res.ok) {
          const data = await res.json();
          console.error("Checkout error:", data.error);
          return;
        }

        const { url } = await res.json();
        onCheckout?.();
        window.location.href = url;
      } catch (error) {
        console.error("Checkout request failed:", error);
      } finally {
        setLoading(null);
      }
    },
    [onCheckout]
  );

  if (!isOpen) return null;

  const planEntries = [PLANS.pro, PLANS.ultra];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl rounded-2xl border border-white/10 bg-[#111113] p-6 shadow-2xl">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded-lg p-1.5 text-zinc-400 hover:bg-white/5 hover:text-white transition-colors cursor-pointer"
        >
          <X className="h-4 w-4" />
        </button>

        {/* Header */}
        <div className="mb-6 text-center">
          <h2 className="text-xl font-semibold text-white">
            Upgrade Your Plan
          </h2>
          <p className="mt-1.5 text-sm text-zinc-400">
            You&apos;ve used your free generation time. Subscribe to keep
            creating.
          </p>
        </div>

        {/* Plan cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {planEntries.map((plan) => {
            const isCurrentPlan = currentPlan === plan.id;
            const isPopular = plan.id === "ultra";

            return (
              <div
                key={plan.id}
                className={`relative flex flex-col rounded-xl border p-5 transition-all ${
                  isPopular
                    ? "border-white/20 bg-white/[0.04]"
                    : "border-white/8 bg-white/[0.02]"
                }`}
              >
                {isPopular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-white px-3 py-0.5 text-xs font-semibold text-black">
                    Popular
                  </div>
                )}

                {/* Plan header */}
                <div className="mb-4">
                  <div className="flex items-center gap-2">
                    {plan.id === "pro" ? (
                      <Zap className="h-4 w-4 text-cyan-400" />
                    ) : (
                      <Crown className="h-4 w-4 text-amber-400" />
                    )}
                    <h3 className="text-base font-semibold text-white">
                      {plan.label}
                    </h3>
                  </div>
                  <div className="mt-2">
                    <span className="text-2xl font-bold text-white">
                      {formatPrice(plan.priceMonthly)}
                    </span>
                    <span className="text-sm text-zinc-400">/month</span>
                  </div>
                </div>

                {/* Features */}
                <ul className="mb-5 flex-1 space-y-2">
                  {plan.features.map((feature) => (
                    <li
                      key={feature}
                      className="flex items-start gap-2 text-sm text-zinc-300"
                    >
                      <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-green-400" />
                      {feature}
                    </li>
                  ))}
                </ul>

                {/* CTA button */}
                <button
                  onClick={() => handleSubscribe(plan.id)}
                  disabled={isCurrentPlan || loading !== null}
                  className={`w-full rounded-lg px-4 py-2.5 text-sm font-semibold transition-all cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${
                    isPopular
                      ? "bg-white text-black hover:bg-zinc-200"
                      : "bg-zinc-800 text-white hover:bg-zinc-700 border border-white/10"
                  }`}
                >
                  {loading === plan.id
                    ? "Redirecting..."
                    : isCurrentPlan
                      ? "Current Plan"
                      : `Subscribe to ${plan.label}`}
                </button>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <p className="mt-4 text-center text-xs text-zinc-500">
          Secure checkout powered by Chargebee. Cancel anytime.
        </p>
      </div>
    </div>
  );
}
