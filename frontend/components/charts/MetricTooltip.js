"use client";

import { useId, useState } from "react";
import { Info } from "lucide-react";

// Tooltip informativo accessibile anche da tastiera: si apre su hover e su
// focus, si chiude su blur/Escape ed è collegato al trigger con aria-describedby.
export function MetricTooltip({ label, children }) {
  const [visible, setVisible] = useState(false);
  const tooltipId = useId();

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        className="grid h-5 w-5 shrink-0 place-items-center rounded-full text-muted hover:text-ink focus-visible:text-ink"
        aria-label={label ? `Informazioni su ${label}` : "Ulteriori informazioni"}
        aria-describedby={visible ? tooltipId : undefined}
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onFocus={() => setVisible(true)}
        onBlur={() => setVisible(false)}
        onKeyDown={(event) => {
          if (event.key === "Escape") setVisible(false);
        }}
      >
        <Info aria-hidden="true" size={14} />
      </button>
      {visible ? (
        <span
          id={tooltipId}
          role="tooltip"
          className="absolute bottom-full left-1/2 z-20 mb-2 w-56 -translate-x-1/2 rounded-lg border bg-elevated p-3 text-xs leading-5 text-ink shadow-panel"
        >
          {children}
        </span>
      ) : null}
    </span>
  );
}
