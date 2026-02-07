"use client";

import { Dice5 } from "lucide-react";
import type { StyleParams } from "@/lib/types";

interface ControlsProps {
  /** Current style parameters */
  params: StyleParams;
  /** Called when any parameter changes */
  onParamsChange: (params: Partial<StyleParams>) => void;
  /** Whether controls should be disabled (during streaming) */
  disabled?: boolean;
}

/**
 * Inline parameter controls panel (no collapsible wrapper - parent controls visibility).
 */
export function Controls({
  params,
  onParamsChange,
  disabled = false,
}: ControlsProps) {
  const randomSeed = () => {
    onParamsChange({ seed: Math.floor(Math.random() * (1 << 24)) });
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-4 flex flex-col gap-4">
      {/* Seed row */}
      <div className="flex items-center gap-3">
        <label className="text-[11px] text-zinc-500 uppercase tracking-wider w-16 shrink-0">
          Seed
        </label>
        <input
          type="number"
          min={0}
          value={params.seed ?? ""}
          placeholder="random"
          onChange={(e) =>
            onParamsChange({
              seed: e.target.value ? Number(e.target.value) : null,
            })
          }
          disabled={disabled}
          className="flex-1 min-w-0 px-3 py-1.5 rounded-lg text-[13px] bg-zinc-800/80
            border border-zinc-700/50 text-white
            outline-none focus:border-zinc-600 transition-colors
            disabled:opacity-40 placeholder:text-zinc-600"
        />
        <button
          onClick={randomSeed}
          disabled={disabled}
          className="p-1.5 rounded-lg bg-zinc-800/80 border border-zinc-700/50
            hover:border-zinc-600 transition-colors cursor-pointer
            disabled:opacity-40 disabled:cursor-not-allowed"
          title="Randomize seed"
        >
          <Dice5 className="w-3.5 h-3.5 text-zinc-400" />
        </button>
      </div>

      {/* Sliders in a row */}
      <div className="grid grid-cols-3 gap-4">
        <SliderField
          label="Strength"
          value={params.strength}
          min={0.2}
          max={1.0}
          step={0.05}
          onChange={(v) => onParamsChange({ strength: v })}
          disabled={disabled}
          format={(v) => v.toFixed(2)}
        />
        <SliderField
          label="Steps"
          value={params.num_inference_steps}
          min={1}
          max={8}
          step={1}
          onChange={(v) => onParamsChange({ num_inference_steps: v })}
          disabled={disabled}
          format={(v) => v.toString()}
        />
        <SliderField
          label="Guidance"
          value={params.guidance_scale}
          min={0.0}
          max={3.0}
          step={0.1}
          onChange={(v) => onParamsChange({ guidance_scale: v })}
          disabled={disabled}
          format={(v) => v.toFixed(1)}
        />
      </div>

      {/* Negative Prompt */}
      <div className="flex items-center gap-3">
        <label className="text-[11px] text-zinc-500 uppercase tracking-wider w-16 shrink-0">
          Neg.
        </label>
        <input
          type="text"
          value={params.negative_prompt}
          onChange={(e) =>
            onParamsChange({ negative_prompt: e.target.value })
          }
          placeholder="blurry, low quality, distorted..."
          disabled={disabled}
          className="flex-1 px-3 py-1.5 rounded-lg text-[13px] bg-zinc-800/80
            border border-zinc-700/50 text-white
            outline-none focus:border-zinc-600 transition-colors
            disabled:opacity-40 placeholder:text-zinc-600"
        />
      </div>
    </div>
  );
}

/* ── Reusable slider field ── */

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  format?: (value: number) => string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-zinc-500 uppercase tracking-wider">
          {label}
        </label>
        <span className="mono text-[12px] text-zinc-400">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full disabled:opacity-40"
      />
    </div>
  );
}
