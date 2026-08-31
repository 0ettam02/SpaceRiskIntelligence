import { formatNumber } from "@/lib/formatters";

const KIND_LABELS = {
  real: "Dato reale del run",
  mock: "Dato mock",
  demo: "Dato dimostrativo",
};

export function KpiCard({ label, value, total, note, icon: Icon, kind }) {
  return (
    <article className="rounded-2xl border bg-surface p-4 shadow-panel transition-transform hover:-translate-y-0.5">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium leading-5 text-muted">{label}</p>
        {Icon ? (
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-brand-400/10 text-brand-300">
            <Icon aria-hidden="true" size={17} />
          </span>
        ) : null}
      </div>
      <p className="mt-5 text-2xl font-bold tracking-tight tabular-nums">
        {formatNumber(value)}
        {total ? <span className="text-base font-medium text-muted"> / {formatNumber(total)}</span> : null}
      </p>
      <p className="mt-1 text-xs text-muted">{note}</p>
      {kind ? <p className="mt-2 text-[10px] uppercase tracking-[0.08em] text-muted/70">{KIND_LABELS[kind] || kind}</p> : null}
    </article>
  );
}
