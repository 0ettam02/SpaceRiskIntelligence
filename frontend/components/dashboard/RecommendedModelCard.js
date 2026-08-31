import Link from "next/link";
import { ArrowUpRight, Star } from "lucide-react";
import { formatMetric, formatPercent } from "@/lib/formatters";

export function RecommendedModelCard({ model }) {
  return (
    <section className="rounded-2xl border border-brand-400/25 bg-brand-400/[0.05] p-5 shadow-panel" aria-labelledby="recommended-model-title">
      <div className="flex items-center gap-2">
        <span className="grid h-9 w-9 place-items-center rounded-lg bg-brand-400/10 text-brand-300">
          <Star aria-hidden="true" size={16} className="fill-brand-300" />
        </span>
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-brand-300">Modello raccomandato</p>
          <h2 id="recommended-model-title" className="font-semibold">
            {model.model}
          </h2>
        </div>
      </div>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-xs text-muted">Accuracy</dt>
          <dd className="font-semibold tabular-nums">{formatPercent(model.accuracy)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted">Recall</dt>
          <dd className="font-semibold tabular-nums">{formatPercent(model.recall)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted">Precision</dt>
          <dd className="font-semibold tabular-nums">{formatPercent(model.precision)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted">ROC-AUC</dt>
          <dd className="font-semibold tabular-nums">{formatMetric(model.rocAuc)}</dd>
        </div>
      </dl>
      <p className="mt-4 text-sm leading-6 text-muted">{model.note}</p>
      <Link
        href={`/models/${model.slug}`}
        className="mt-4 inline-flex min-h-11 items-center gap-1.5 rounded-lg border bg-surface px-3 text-sm font-semibold text-brand-300 hover:bg-elevated"
      >
        Vedi dettaglio modello
        <ArrowUpRight aria-hidden="true" size={16} />
      </Link>
    </section>
  );
}
