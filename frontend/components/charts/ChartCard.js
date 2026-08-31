export function ChartCard({ title, description, demo = false, actions, srSummary, children }) {
  return (
    <section className="rounded-2xl border bg-surface p-5 shadow-panel" aria-label={title}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-semibold">{title}</h3>
            {demo ? (
              <span className="rounded-full border border-line bg-elevated px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-muted">
                Dati dimostrativi
              </span>
            ) : null}
          </div>
          {description ? <p className="mt-1 text-xs text-muted">{description}</p> : null}
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      </div>
      {children}
      {srSummary ? <p className="sr-only">{srSummary}</p> : null}
    </section>
  );
}
