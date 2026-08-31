import { RISK_LEVELS } from "@/lib/risk-utils";

export function RiskLegend({ compact = false }) {
  return (
    <div className={compact ? "" : "rounded-2xl border bg-surface p-4 shadow-panel"}>
      <p className="text-xs font-semibold uppercase tracking-[0.1em] text-muted">Legenda del rischio</p>
      <ul className="mt-3 space-y-2">
        {RISK_LEVELS.map((level) => (
          <li key={level.id} className="flex items-center gap-2 text-sm">
            <span className="h-3.5 w-3.5 shrink-0 rounded-sm border border-black/10 dark:border-white/10" style={{ backgroundColor: level.color }} />
            <span>{level.label}</span>
            <span className="ml-auto text-xs text-muted tabular-nums">
              {Math.round(level.min * 100)}–{Math.round(Math.min(level.max, 1) * 100)}%
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11px] leading-4 text-muted">
        Configurazione dimostrativa dell&apos;interfaccia, non una soglia scientifica già validata.
      </p>
    </div>
  );
}
