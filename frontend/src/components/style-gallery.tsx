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
    id: "neon-cyberpunk",
    label: "Neon Cyberpunk",
    prompt: "cyberpunk neon portrait, glowing neon edge lighting, deep black shadows, electric blue and hot pink accent lights, rain-slicked reflections, Blade Runner atmosphere, synthwave aesthetic",
    color: "border-cyan-400/50 hover:border-cyan-400 text-cyan-300",
  },
  {
    id: "clay",
    label: "Clay",
    prompt: "claymation style, stop motion, clay character, realistic materials, depth of field, 3d render, subject made of clay",
    color: "border-orange-400/50 hover:border-orange-400 text-orange-300",
  },
  {
    id: "ukiyo-e",
    label: "Ukiyo-e",
    prompt: "ukiyo-e Japanese woodblock print, flat areas of rich color, bold black outlines, flowing organic linework, traditional washi paper texture, Hokusai and Hiroshige inspired composition",
    color: "border-red-400/50 hover:border-red-400 text-red-300",
  },
  {
    id: "low-poly",
    label: "Low Poly",
    prompt: "low poly 3D mesh portrait, geometric faceted triangles, flat-shaded surfaces, clean polygon edges, minimal color palette, abstract crystalline structure, cinema 4D render",
    color: "border-emerald-400/50 hover:border-emerald-400 text-emerald-300",
  },
  {
    id: "fortnite",
    label: "Fortnite",
    prompt: "fortnite style, 3d cartoon, stylized, vibrant colors, video game character, unreal engine 5 render, detailed character",
    color: "border-indigo-400/50 hover:border-indigo-400 text-indigo-300",
  },
  {
    id: "glitch",
    label: "Glitch",
    prompt: "digital glitch art, RGB channel displacement, VHS scan lines and static noise, corrupted data fragments, chromatic aberration, databending distortion, cybernetic aesthetic",
    color: "border-pink-400/50 hover:border-pink-400 text-pink-300",
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
            onClick={() => onSelect(preset.prompt)}
            className={`
              shrink-0 px-3.5 py-1.5 rounded-full text-[12px] font-medium
              transition-all duration-150 cursor-pointer
              ${isActive
                ? "bg-white text-black"
                : "bg-zinc-800/80 text-zinc-400 hover:bg-zinc-700/80 hover:text-zinc-300"
              }
            `}
          >
            {preset.label}
          </button>
        );
      })}
    </div>
  );
}

export { PRESETS };
