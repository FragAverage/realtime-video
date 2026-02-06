"use client";

import type { StylePreset } from "@/lib/types";

const PRESETS: StylePreset[] = [
  {
    id: "oil-painting",
    label: "Oil Painting",
    prompt: "oil painting masterpiece, rich impasto texture, visible bold brushstrokes, warm gallery lighting, classical fine art, museum quality",
    color: "border-amber-400/50 hover:border-amber-400 text-amber-300",
  },
  {
    id: "anime",
    label: "Anime",
    prompt: "anime art style, cel-shaded, vivid saturated colors, clean sharp linework, expressive eyes, studio ghibli aesthetic, high detail illustration",
    color: "border-pink-400/50 hover:border-pink-400 text-pink-300",
  },
  {
    id: "cyberpunk",
    label: "Cyberpunk",
    prompt: "cyberpunk digital art, neon magenta and cyan glow, volumetric fog, rain-slicked surfaces, holographic overlays, futuristic dystopian atmosphere, high contrast",
    color: "border-cyan-400/50 hover:border-cyan-400 text-cyan-300",
  },
  {
    id: "watercolor",
    label: "Watercolor",
    prompt: "delicate watercolor painting on textured paper, soft wet edges, transparent color washes, pastel palette, loose artistic brushwork, dreamy atmosphere",
    color: "border-sky-400/50 hover:border-sky-400 text-sky-300",
  },
  {
    id: "pencil-sketch",
    label: "Pencil Sketch",
    prompt: "detailed graphite pencil sketch on cream paper, precise cross-hatching, dramatic light and shadow, fine line work, hyperrealistic drawing technique",
    color: "border-gray-400/50 hover:border-gray-400 text-gray-300",
  },
  {
    id: "pop-art",
    label: "Pop Art",
    prompt: "pop art style, bold flat primary colors, benday halftone dots, thick black outlines, andy warhol roy lichtenstein inspired, screen print aesthetic",
    color: "border-rose-400/50 hover:border-rose-400 text-rose-300",
  },
  {
    id: "pixel-art",
    label: "Pixel Art",
    prompt: "pixel art, retro 16-bit video game sprite aesthetic, limited color palette, crisp pixel edges, nostalgic SNES era, dithering patterns",
    color: "border-emerald-400/50 hover:border-emerald-400 text-emerald-300",
  },
  {
    id: "3d-render",
    label: "3D Render",
    prompt: "3D rendered character, Pixar Disney animation style, smooth subsurface scattering skin, soft volumetric studio lighting, octane render quality, glossy materials",
    color: "border-purple-400/50 hover:border-purple-400 text-purple-300",
  },
];

interface StyleGalleryProps {
  /** Called when a preset is selected */
  onSelect: (prompt: string) => void;
  /** Currently active prompt (to highlight matching preset) */
  activePrompt?: string;
}

/**
 * Horizontal scrollable gallery of style preset buttons.
 * Clicking a preset populates the prompt.
 */
export function StyleGallery({ onSelect, activePrompt }: StyleGalleryProps) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-[11px] uppercase tracking-wider text-[var(--color-text-muted)] font-semibold">
        Style Presets
      </span>
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
        {PRESETS.map((preset) => {
          const isActive = activePrompt === preset.prompt;
          return (
            <button
              key={preset.id}
              onClick={() => onSelect(preset.prompt)}
              className={`
                shrink-0 px-4 py-2 rounded-lg text-[13px] font-medium
                border transition-all duration-200 cursor-pointer
                bg-[var(--color-surface-soft)]
                hover:-translate-y-0.5 hover:shadow-lg
                ${
                  isActive
                    ? `${preset.color} glow-cyan`
                    : `border-[var(--color-border)] text-[var(--color-text-muted)] ${preset.color.split(" ").filter((c: string) => c.startsWith("hover:")).join(" ")}`
                }
              `}
            >
              {preset.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export { PRESETS };
