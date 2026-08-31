import { StatusBadge } from "@/components/feedback/StatusBadge";

export function PipelineStepper({ steps }) {
  return (
    <ol className="space-y-0">
      {steps.map((step, index) => (
        <li key={step.id} className="relative flex gap-4 pb-8 pl-1 last:pb-0">
          <div className="flex flex-col items-center">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border-2 border-brand-400/40 bg-elevated text-sm font-bold text-brand-300">
              {step.id}
            </span>
            {index < steps.length - 1 ? <span className="mt-1 w-px flex-1 bg-line" aria-hidden="true" /> : null}
          </div>
          <div className="min-w-0 flex-1 rounded-2xl border bg-surface p-4 shadow-panel">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-semibold">{step.title}</h3>
              <StatusBadge status={step.status} />
            </div>
            <dl className="mt-3 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
              <div>
                <dt className="text-xs text-muted">Durata</dt>
                <dd>{step.durationLabel}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Record</dt>
                <dd>{step.records}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Input</dt>
                <dd>{step.input}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Output</dt>
                <dd>{step.output}</dd>
              </div>
            </dl>
            {step.warnings?.length > 0 ? (
              <ul className="mt-3 space-y-1 border-t pt-3 text-xs text-amber-300">
                {step.warnings.map((warning) => (
                  <li key={warning}>⚠ {warning}</li>
                ))}
              </ul>
            ) : null}
          </div>
        </li>
      ))}
    </ol>
  );
}
