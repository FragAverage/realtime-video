"use client";

import { useState, useCallback } from "react";

interface PromptBarProps {
  /** Current prompt text */
  prompt: string;
  /** Called when prompt changes (typing) */
  onPromptChange: (prompt: string) => void;
  /** Called when user wants to apply prompt mid-stream */
  onApply: (prompt: string) => void;
  /** Whether there's an active stream to apply to */
  canApply: boolean;
  /** Called when Enter is pressed (to start streaming) */
  onSubmit?: () => void;
  /** Whether the prompt input is disabled */
  disabled?: boolean;
}

/**
 * Clean prompt input bar. Press Enter to start, Cmd+Enter to apply mid-stream.
 */
export function PromptBar({
  prompt,
  onPromptChange,
  onApply,
  canApply,
  onSubmit,
  disabled = false,
}: PromptBarProps) {
  const [isFocused, setIsFocused] = useState(false);

  const handleApply = useCallback(() => {
    if (prompt.trim() && canApply) {
      onApply(prompt.trim());
    }
  }, [prompt, canApply, onApply]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (canApply) {
          // Mid-stream: apply new prompt
          handleApply();
        } else if (onSubmit) {
          // Not streaming: start
          onSubmit();
        }
      }
    },
    [canApply, handleApply, onSubmit]
  );

  return (
    <div
      className={`rounded-2xl border transition-all duration-200 ${
        isFocused
          ? "border-zinc-600 bg-zinc-900"
          : "border-zinc-800 bg-zinc-900/80"
      }`}
    >
      <textarea
        value={prompt}
        onChange={(e) => onPromptChange(e.target.value)}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        onKeyDown={handleKeyDown}
        placeholder="Describe the style you want..."
        disabled={disabled}
        rows={1}
        className="w-full bg-transparent pl-4 pr-36 py-3 text-[14px] leading-relaxed
          text-white placeholder:text-zinc-600
          resize-none outline-none disabled:opacity-50"
      />
    </div>
  );
}
