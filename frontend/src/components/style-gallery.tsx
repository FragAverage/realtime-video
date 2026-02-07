"use client";

import type { StylePreset } from "@/lib/types";

const PRESETS: StylePreset[] = [
  {
    id: "oil-painting",
    label: "Oil Painting",
    prompt: "oil painting masterpiece, rich impasto texture, visible bold brushstrokes, warm gallery lighting, classical fine art, museum quality",
    color: "border-amber-400/50 hover:border-amber-400 text-amber-300",
    lora_id: null,
  },
  {
    id: "anime",
    label: "Anime",
    prompt: "anime_style, cel-shaded, vivid saturated colors, clean sharp linework, expressive eyes, studio ghibli aesthetic, high detail illustration",
    color: "border-pink-400/50 hover:border-pink-400 text-pink-300",
    lora_id: "anime-style",
  },
  {
    id: "clay",
    label: "Clay",
    prompt: "claymation style, stop motion, clay character, realistic materials, depth of field, 3d render, subject made of clay",
    color: "border-orange-400/50 hover:border-orange-400 text-orange-300",
    lora_id: null,
  },
  {
    id: "watercolor",
    label: "Watercolor",
    prompt: "delicate watercolor painting on textured paper, soft wet edges, transparent color washes, pastel palette, loose artistic brushwork, dreamy atmosphere",
    color: "border-sky-400/50 hover:border-sky-400 text-sky-300",
    lora_id: null,
  },
  {
    id: "pencil-sketch",
    label: "Pencil Sketch",
    prompt: "detailed graphite pencil sketch on cream paper, precise cross-hatching, dramatic light and shadow, fine line work, hyperrealistic drawing technique",
    color: "border-gray-400/50 hover:border-gray-400 text-gray-300",
    lora_id: null,
  },
  {
    id: "fortnite",
    label: "Fortnite",
    prompt: "fortnite style, 3d cartoon, stylized, vibrant colors, video game character, unreal engine 5 render, detailed character",
    color: "border-indigo-400/50 hover:border-indigo-400 text-indigo-300",
    lora_id: null,
  },
  {
    id: "pixel-art",
    label: "Pixel Art",
    prompt: "pixel art sprite, retro 16-bit video game sprite aesthetic, limited color palette, crisp pixel edges, nostalgic SNES era, dithering patterns",
    color: "border-emerald-400/50 hover:border-emerald-400 text-emerald-300",
    lora_id: "pixel-art",
  },
  {
    id: "3d-render",
    label: "3D Render",
    prompt: "3D rendered character, Pixar Disney animation style, smooth subsurface scattering skin, soft volumetric studio lighting, octane render quality, glossy materials",
    color: "border-purple-400/50 hover:border-purple-400 text-purple-300",
    lora_id: null,
  },
];

interface StyleGalleryProps {
  /** Called when a preset is selected, with prompt and optional lora_id */
  onSelect: (prompt: string, loraId: string | null) => void;
  /** Currently active prompt (to highlight matching preset) */
  activePrompt?: string;
}

/**
 * Compact horizontal row of style preset pills.
 */
export function StyleGallery({ onSelect, activePrompt }: StyleGalleryProps) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
      {PRESETS.map((preset) => {
        const isActive = activePrompt === preset.prompt;
        return (
          <button
            key={preset.id}
            onClick={() => onSelect(preset.prompt, preset.lora_id)}
            className={`
              shrink-0 px-3.5 py-1.5 rounded-full text-[12px] font-medium
              transition-all duration-150 cursor-pointer
              ${isActive
                ? "bg-white text-black"
                : "bg-zinc-800/80 text-zinc-400 hover:bg-zinc-700/80 hover:text-zinc-300"
              }
            `}
          >
            <span className="flex items-center gap-1.5">
              {preset.label}
              {preset.lora_id && (
                <span className={`text-[9px] font-bold uppercase tracking-wide px-1 py-0.5 rounded ${isActive ? "bg-black/10 text-black/60" : "bg-white/10 text-zinc-500"
                  }`}>
                  LoRA
                </span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export { PRESETS };
