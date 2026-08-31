"use client";

import { TIME_RANGE_OPTIONS } from "@/lib/constants";

export function TimeRangeSelector({ value, onChange, options = TIME_RANGE_OPTIONS, label = "Finestra temporale" }) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap gap-1 rounded-xl border bg-elevated p-1">
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={`min-h-9 flex-1 basis-[45%] rounded-lg px-3 text-xs font-semibold transition-colors ${
              active ? "bg-brand-400/15 text-brand-300 ring-1 ring-inset ring-brand-400/30" : "text-muted hover:text-ink"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
