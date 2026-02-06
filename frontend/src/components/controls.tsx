"use client";

import { ChevronDown, ChevronUp, Dice5 } from "lucide-react";
import { useState } from "react";
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
 * Collapsible parameter controls panel.
 * Contains seed, strength, inference steps, and guidance scale for FLUX.2 Klein.
 */
export function Controls({
  params,
  onParamsChange,
  disabled = false,
}: ControlsProps) {
  const [isOpen, setIsOpen] = useState(false);

  const randomSeed = () => {
    onParamsChange({ seed: Math.floor(Math.random() * (1 << 24)) });
  };

  return (
    <div className="glass rounded-xl overflow-hidden">
      {/* Toggle header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-5 py-3
          cursor-pointer hover:bg-white/[0.02] transition-colors"
      >
        <span className="text-[13px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">
          Parameters
        </span>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-[var(--color-text-muted)]" />
        ) : (
          <ChevronDown className="w-4 h-4 text-[var(--color-text-muted)]" />
        )}
      </button>

      {/* Collapsible body */}
      {isOpen && (
        <div className="px-5 pb-5 flex flex-col gap-5 border-t border-[var(--color-border)]">
          {/* Seed */}
          <div className="flex flex-col gap-1.5 pt-4">
            <label className="text-[12px] text-[var(--color-text-muted)]">
              Seed
            </label>
            <div className="flex gap-2">
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
                className="flex-1 min-w-0 px-3 py-2 rounded-lg text-[14px] bg-[var(--color-surface-soft)]
                  border border-[var(--color-border)] text-[var(--color-text-primary)]
                  outline-none focus:border-[var(--color-accent)]/60 transition-colors
                  disabled:opacity-50 placeholder:text-[var(--color-text-muted)]/40"
              />
              <button
                onClick={randomSeed}
                disabled={disabled}
                className="p-2 rounded-lg bg-[var(--color-surface-soft)] border border-[var(--color-border)]
                  hover:border-[var(--color-border-bright)] transition-colors cursor-pointer
                  disabled:opacity-50 disabled:cursor-not-allowed"
                title="Randomize seed"
              >
                <Dice5 className="w-4 h-4 text-[var(--color-text-muted)]" />
              </button>
            </div>
          </div>

          {/* Sliders */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Strength (img2img denoising strength) */}
            <SliderField
              label="Strength"
              description="How much to transform the input"
              value={params.strength}
              min={0.2}
              max={1.0}
              step={0.05}
              onChange={(v) => onParamsChange({ strength: v })}
              disabled={disabled}
              format={(v) => v.toFixed(2)}
            />

            {/* Inference Steps */}
            <SliderField
              label="Steps"
              description="More steps = better quality, slower"
              value={params.num_inference_steps}
              min={1}
              max={8}
              step={1}
              onChange={(v) => onParamsChange({ num_inference_steps: v })}
              disabled={disabled}
              format={(v) => v.toString()}
            />

            {/* Guidance Scale */}
            <SliderField
              label="Guidance Scale"
              description="1.0 default for Klein distilled"
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
          <div className="flex flex-col gap-1.5">
            <label className="text-[12px] text-[var(--color-text-muted)]">
              Negative Prompt
            </label>
            <input
              type="text"
              value={params.negative_prompt}
              onChange={(e) =>
                onParamsChange({ negative_prompt: e.target.value })
              }
              placeholder="blurry, low quality, distorted..."
              disabled={disabled}
              className="w-full px-3 py-2 rounded-lg text-[14px] bg-[var(--color-surface-soft)]
                border border-[var(--color-border)] text-[var(--color-text-primary)]
                outline-none focus:border-[var(--color-accent)]/60 transition-colors
                disabled:opacity-50 placeholder:text-[var(--color-text-muted)]/40"
            />
          </div>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------- */
/* Reusable slider field                               */
/* -------------------------------------------------- */

function SliderField({
  label,
  description,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  format,
}: {
  label: string;
  description?: string;
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
      <label className="text-[12px] text-[var(--color-text-muted)]">
        {label}
      </label>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className="flex-1 disabled:opacity-50"
        />
        <span className="mono text-[13px] text-[var(--color-text-primary)] w-8 text-right">
          {format ? format(value) : value}
        </span>
      </div>
      {description && (
        <span className="text-[10px] text-[var(--color-text-muted)]/50">
          {description}
        </span>
      )}
    </div>
  );
}
