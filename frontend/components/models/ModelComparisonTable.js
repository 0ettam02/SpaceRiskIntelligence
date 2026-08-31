import Link from "next/link";
import { Star } from "lucide-react";
import { MetricTooltip } from "@/components/charts/MetricTooltip";
import { formatMetric, formatPercent } from "@/lib/formatters";

const COLUMNS = [
  { key: "accuracy", label: "Accuracy", tooltip: "Quota di previsioni corrette sul totale. Può essere fuorviante con classi sbilanciate." },
  { key: "balancedAccuracy", label: "Balanced Accuracy", tooltip: "Media fra recall e specificità: più robusta dell'accuracy quando le classi non sono bilanciate." },
  { key: "precision", label: "Precision", tooltip: "Quota di celle segnalate come attive che lo sono realmente." },
  { key: "recall", label: "Recall", tooltip: "Quota di celle realmente attive correttamente identificate dal modello." },
  { key: "f1", label: "F1", tooltip: "Media armonica fra precision e recall." },
  { key: "rocAuc", label: "ROC-AUC", tooltip: "Capacità del modello di separare le due classi su tutte le soglie possibili." },
  { key: "prAuc", label: "PR-AUC", tooltip: "Area sotto la curva Precision-Recall: più informativa della ROC-AUC con classi sbilanciate." },
];

export function ModelComparisonTable({ models }) {
  return (
    <div className="rounded-2xl border bg-surface shadow-panel">
      <div className="hidden overflow-x-auto lg:block">
        <table className="w-full min-w-[860px] border-collapse text-sm">
          <caption className="sr-only">
            Tabella di confronto dei 5 classificatori valutati su fire_next_7d, con Accuracy, Balanced Accuracy, Precision, Recall, F1,
            ROC-AUC, PR-AUC e soglia di decisione.
          </caption>
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted">
              <th scope="col" className="px-4 py-3 font-semibold">
                Modello
              </th>
              {COLUMNS.map((column) => (
                <th key={column.key} scope="col" className="px-3 py-3 text-right font-semibold">
                  <span className="inline-flex items-center justify-end gap-1">
                    {column.label}
                    <MetricTooltip label={column.label}>{column.tooltip}</MetricTooltip>
                  </span>
                </th>
              ))}
              <th scope="col" className="px-3 py-3 text-right font-semibold">
                Soglia
              </th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.slug} className={`border-b last:border-0 ${model.recommended ? "bg-brand-400/[0.06]" : ""}`}>
                <th scope="row" className="px-4 py-3 text-left font-medium">
                  <Link href={`/models/${model.slug}`} className="inline-flex items-center gap-2 hover:text-brand-300">
                    {model.recommended ? <Star aria-hidden="true" size={14} className="fill-brand-300 text-brand-300" /> : null}
                    {model.model}
                    {model.recommended ? <span className="sr-only">(modello raccomandato)</span> : null}
                  </Link>
                </th>
                {COLUMNS.map((column) => (
                  <td key={column.key} className="px-3 py-3 text-right tabular-nums">
                    {formatPercent(model[column.key])}
                  </td>
                ))}
                <td className="px-3 py-3 text-right tabular-nums">{formatMetric(model.threshold, { digits: 2 })}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ul className="divide-y lg:hidden">
        {models.map((model) => (
          <li key={model.slug} className={`p-4 ${model.recommended ? "bg-brand-400/[0.06]" : ""}`}>
            <Link href={`/models/${model.slug}`} className="flex items-center gap-2 font-semibold hover:text-brand-300">
              {model.recommended ? <Star aria-hidden="true" size={14} className="fill-brand-300 text-brand-300" /> : null}
              {model.model}
            </Link>
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              {COLUMNS.map((column) => (
                <div key={column.key} className="flex justify-between gap-2">
                  <dt className="text-muted">{column.label}</dt>
                  <dd className="font-semibold tabular-nums">{formatPercent(model[column.key])}</dd>
                </div>
              ))}
              <div className="flex justify-between gap-2">
                <dt className="text-muted">Soglia</dt>
                <dd className="font-semibold tabular-nums">{formatMetric(model.threshold, { digits: 2 })}</dd>
              </div>
            </dl>
          </li>
        ))}
      </ul>
    </div>
  );
}
