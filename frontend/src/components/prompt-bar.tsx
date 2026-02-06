"use client";

import { useState, useCallback } from "react";
import { Send } from "lucide-react";

interface PromptBarProps {
  /** Current prompt text */
  prompt: string;
  /** Called when prompt changes (typing) */
  onPromptChange: (prompt: string) => void;
  /** Called when user wants to apply prompt mid-stream */
  onApply: (prompt: string) => void;
  /** Whether there's an active stream to apply to */
  canApply: boolean;
  /** Whether the prompt input is disabled */
  disabled?: boolean;
}

/**
 * Prompt input bar with "Apply" button for mid-stream prompt updates.
 * Press Cmd+Enter to apply while streaming.
 */
export function PromptBar({
  prompt,
  onPromptChange,
  onApply,
  canApply,
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
      // Cmd/Ctrl+Enter to apply mid-stream
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canApply) {
        e.preventDefault();
        handleApply();
      }
    },
    [canApply, handleApply]
  );

  return (
    <div className="flex flex-col gap-3">
      {/* Prompt textarea */}
      <div
        className={`glass rounded-xl transition-all duration-200 ${
          isFocused ? "glow-cyan-strong border-[var(--color-accent)]/60" : ""
        }`}
      >
        <textarea
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyDown}
          placeholder="Describe the style you want applied to your webcam feed..."
          disabled={disabled}
          rows={2}
          className="w-full bg-transparent px-4 py-3 text-[15px] leading-relaxed
            text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)]/50
            resize-none outline-none disabled:opacity-50"
        />
      </div>

      {/* Apply button (visible when streaming) */}
      {canApply && (
        <div className="flex items-center gap-3">
          <button
            onClick={handleApply}
            disabled={!prompt.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[14px] font-semibold
              transition-all duration-200 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40
              bg-[var(--color-accent-strong)] text-white
              hover:not-disabled:shadow-[0_8px_24px_rgba(37,99,235,0.3)]
              hover:not-disabled:-translate-y-0.5"
          >
            <Send className="w-4 h-4" />
            <span>Apply Style</span>
          </button>
          <span className="text-[11px] text-[var(--color-text-muted)]">
            {"\u2318"}+Enter
          </span>
        </div>
      )}
    </div>
  );
}
