"use client";

import { useState } from "react";
import { formatDateShort, formatMetric, formatNumber, formatPercent } from "@/lib/formatters";

const LABELS = {
  constant: "Prevalenza del training",
  logistic: "Regressione logistica",
  random_forest_baseline: "Random Forest · configurazione originale",
  random_forest_regularized: "Random Forest · regolarizzata",
  random_forest_firms_only: "Random Forest · solo FIRMS (stesse righe)",
};
const COLUMNS = [
  ["balanced_accuracy", "Balanced accuracy", formatPercent],
  ["precision", "Precision", formatPercent],
  ["recall", "Recall", formatPercent],
  ["f1", "F1", formatMetric],
  ["roc_auc", "ROC-AUC", formatMetric],
  ["average_precision", "Average precision", formatMetric],
  ["false_alerts_per_cell_week", "Falsi allarmi / cella-settimana", formatMetric],
];

export function ResearchResultsPanel({ research }) {
  const [studyId, setStudyId] = useState("era5");
  const [foldId, setFoldId] = useState("latest");
  const [scope, setScope] = useState("seen_cells");
  const studies = research?.studies || [];
  const study = studies.find((item) => item.id === studyId) || studies[0];
  if (!study) {
    return <section className="rounded-2xl border bg-surface p-5"><h2 className="font-semibold">Valutazioni Italia v2 non disponibili</h2><p className="mt-2 text-sm text-muted">I risultati v2 non sono stati caricati. Le metriche della v1 restano consultabili sotto.</p></section>;
  }
  const folds = [...new Set(study.results.map((row) => row.fold))].sort((a, b) => a - b);
  const selectedFold = foldId === "latest" ? folds.at(-1) : Number(foldId);
  const rows = study.results.filter((row) => row.fold === selectedFold);
  const period = rows[0]?.periods.test;
  const support = rows[0]?.test[scope]?.metrics;
  const coverage = research.weatherCoverage;
  const partial = study.scope !== "national_grid";

  return (
    <section className="space-y-5 rounded-2xl border bg-surface p-4 shadow-panel md:p-6" aria-labelledby="research-title">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-brand-300">Italia v2 · risultati sperimentali</p>
        <h2 id="research-title" className="mt-1 text-xl font-semibold">Valutazione dei nuovi modelli</h2>
        <p className="mt-2 text-sm leading-6 text-muted">La mappa usa la Random Forest v2 addestrata con FIRMS + ERA5-Land sulle 3.185 celle con copertura meteo completa. La previsione mostrata è storica (11–17 luglio 2026) e non descrive il rischio odierno. I risultati qui sotto sono valutazioni sperimentali su split temporali e geografici; target e campioni differiscono dalla v1.</p>
        {research.source === "snapshot" ? <p className="mt-2 text-xs text-muted">Modalità demo: risultati reali salvati il {formatDateShort(research.exportedAt)}, non un aggiornamento in tempo reale.</p> : null}
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <label className="text-sm">Dataset
          <select className="mt-1 block min-h-11 w-full rounded-lg border bg-elevated p-2 text-ink" value={study.id} onChange={(event) => { setStudyId(event.target.value); setFoldId("latest"); }}>
            {studies.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
          </select>
        </label>
        <label className="text-sm">Periodo di test
          <select className="mt-1 block min-h-11 w-full rounded-lg border bg-elevated p-2 text-ink" value={foldId} onChange={(event) => setFoldId(event.target.value)}>
            <option value="latest">Ultimo periodo disponibile</option>
            {folds.map((fold) => { const dates = study.results.find((row) => row.fold === fold).periods.test; return <option key={fold} value={fold}>{formatDateShort(dates[0])} – {formatDateShort(dates[1])}</option>; })}
          </select>
        </label>
        <label className="text-sm">Area di valutazione
          <select className="mt-1 block min-h-11 w-full rounded-lg border bg-elevated p-2 text-ink" value={scope} onChange={(event) => setScope(event.target.value)}>
            <option value="seen_cells">Celle note · test nel futuro</option>
            <option value="unseen_spatial_blocks">Aree escluse dall’addestramento</option>
          </select>
        </label>
      </div>

      <dl className="grid grid-cols-2 gap-4 rounded-xl border bg-elevated p-4 md:grid-cols-4">
        {[["Celle del dataset", formatNumber(study.cells)], ["Righe ML complessive", formatNumber(study.rows)], ["Righe del test selezionato", formatNumber(support?.rows)], ["Etichette positive nel test", formatNumber(support?.positive_rows)]].map(([label, value]) => <div key={label}><dt className="text-xs text-muted">{label}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{value}</dd></div>)}
      </dl>
      <div className="rounded-xl border border-brand-400/25 bg-brand-400/[0.05] p-4 text-sm leading-6">
        <p className="font-semibold">{partial ? "ERA5 parziale: queste metriche non rappresentano tutta l’Italia." : "Griglia nazionale: questo esperimento usa soltanto feature FIRMS."}</p>
        {coverage ? <p>Copertura meteo: {formatNumber(coverage.availableCells)} / {formatNumber(coverage.requiredCells)} celle. {coverage.complete ? "Copertura completata." : `Mancano serie ERA5-Land complete per ${formatNumber(coverage.missingCells)} celle.`}</p> : null}
        <p>La v2 è attiva per visualizzare questo snapshot storico. Prima di un uso operativo servono dati aggiornati e una nuova valutazione indipendente. Più dati e controlli rigorosi non garantiscono metriche alte.</p>
      </div>

      <p className="text-sm text-muted">Test: {formatDateShort(period?.[0])} – {formatDateShort(period?.[1])}. Prevalenza positiva: {formatPercent(support?.prevalence)}. Le etichette positive sono finestre temporali e possono riferirsi allo stesso episodio. “—” indica una metrica non stimabile.</p>
      {support?.positive_rows === 0 ? <p role="status" className="text-sm font-semibold">Nessuna etichetta positiva in questo test: ROC-AUC, average precision e recall non sono stimabili.</p> : null}
      <div className="overflow-x-auto rounded-xl border">
        <table className="w-full min-w-[1000px] text-sm">
          <caption className="sr-only">Confronto v2 sullo stesso dataset, periodo e area selezionati</caption>
          <thead className="bg-elevated text-xs text-muted"><tr><th scope="col" className="p-3 text-left">Modello</th>{COLUMNS.map(([key, label]) => <th key={key} scope="col" className="p-3 text-right">{label}</th>)}<th scope="col" className="p-3 text-right">Soglia</th></tr></thead>
          <tbody>{rows.map((row) => { const metric = row.test[scope]?.metrics || {}; return <tr key={row.model} className="border-t"><th scope="row" className="p-3 text-left font-medium">{LABELS[row.model] || row.model}<span className="mt-1 block text-xs font-normal text-muted">{row.featureCount} feature · sperimentale</span></th>{COLUMNS.map(([key, , formatter]) => <td key={key} className="p-3 text-right tabular-nums">{formatter(metric[key])}</td>)}<td className="p-3 text-right tabular-nums">{row.threshold > 1 ? "Nessun allarme" : formatMetric(row.threshold)}</td></tr>; })}</tbody>
        </table>
      </div>
      <p className="text-xs leading-5 text-muted">ROC-AUC e F1 sono espressi tra 0 e 1. L’average precision va confrontata con la prevalenza positiva. {study.thresholdObjective === "f1" ? "La soglia massimizza l’F1 sul solo periodo di validation; le metriche riportate sono calcolate sul test successivo." : `La soglia è scelta sulla validation con budget di ${formatMetric(study.falseAlertBudget, { digits: 1 })} falsi allarmi per cella-settimana; il budget non è garantito sul test.`}</p>
      <details className="rounded-xl border p-4">
        <summary className="cursor-pointer text-sm font-semibold">Errori ed episodi del test selezionato</summary>
        <ul className="mt-3 space-y-3 text-sm">{rows.map((row) => { const { metrics: m, episodes: e } = row.test[scope]; return <li key={row.model}><span className="font-semibold">{LABELS[row.model]}: </span>{formatNumber(m.true_positive)} veri positivi, {formatNumber(m.false_positive)} falsi positivi, {formatNumber(m.false_negative)} falsi negativi. Episodi intercettati: {formatNumber(e.hit_episodes)} / {formatNumber(e.eligible_episodes)}.</li>; })}</ul>
        <p className="mt-3 text-xs text-muted">Gli episodi sono gruppi di rilevamenti nella stessa cella, non incendi certificati. Si contano solo quelli con tutte le sette opportunità di previsione disponibili.</p>
      </details>
      {research.unavailable?.length ? <p role="status" className="text-sm text-muted">Una parte degli esperimenti v2 non è disponibile.</p> : null}
    </section>
  );
}
