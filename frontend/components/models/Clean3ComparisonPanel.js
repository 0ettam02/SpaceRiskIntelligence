import { formatMetric, formatNumber, formatPercent } from "@/lib/formatters";

const COLUMNS = [
  ["accuracy", "Accuracy", formatPercent],
  ["balancedAccuracy", "Balanced accuracy", formatPercent],
  ["precision", "Precision", formatPercent],
  ["recall", "Recall", formatPercent],
  ["f1", "F1", formatMetric],
  ["rocAuc", "ROC-AUC", formatMetric],
  ["averagePrecision", "PR-AUC", formatMetric],
];

export function Clean3ComparisonPanel({ comparison, forecast }) {
  if (!comparison) return null;
  const rows = [
    ["Random Forest · FIRMS clean3", comparison.firmsOnly],
    ["Random Forest · FIRMS clean3 + ERA5-Land", comparison.firmsEra5],
  ];
  return (
    <section className="space-y-4 rounded-2xl border bg-surface p-4 shadow-panel md:p-6" aria-labelledby="clean3-comparison-title">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-brand-300">Confronto sulla stessa base dati</p>
        <h2 id="clean3-comparison-title" className="mt-1 text-xl font-semibold">Il Random Forest con ERA5-Land</h2>
        <p className="mt-2 text-sm leading-6 text-muted">Entrambi i modelli usano le stesse righe FIRMS italiane, lo stesso split casuale 80/20 e una soglia 0,5. Il secondo aggiunge otto feature ERA5-Land; così il confronto isola l’effetto delle nuove variabili.</p>
      </div>
      <div className="overflow-x-auto rounded-xl border">
        <table className="w-full min-w-[900px] text-sm">
          <caption className="sr-only">Confronto Random Forest FIRMS e FIRMS più ERA5-Land sullo stesso test</caption>
          <thead className="bg-elevated text-xs text-muted"><tr><th scope="col" className="p-3 text-left">Modello</th><th scope="col" className="p-3 text-right">Righe test</th>{COLUMNS.map(([, label]) => <th scope="col" className="p-3 text-right" key={label}>{label}</th>)}</tr></thead>
          <tbody>{rows.map(([label, metric]) => <tr className="border-t" key={label}><th scope="row" className="p-3 text-left font-medium">{label}</th><td className="p-3 text-right tabular-nums">{formatNumber(metric.rows)}</td>{COLUMNS.map(([key, , formatter]) => <td className="p-3 text-right tabular-nums" key={key}>{formatter(metric[key])}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <p className="text-xs leading-5 text-muted">Test: {formatNumber(comparison.firmsEra5.rows)} righe, {formatNumber(comparison.firmsEra5.positiveRows)} positive. Random split non temporale e non geografico: utile per replicare il baseline semplice, ma può sovrastimare le prestazioni su date o località future. Forecast salvato: {forecast?.forecastStart}–{forecast?.forecastEnd}; dati fino al {forecast?.observedThrough}.</p>
    </section>
  );
}
