import { formatNumber, formatPercent } from "@/lib/formatters";

function Cell({ label, value, total, tone }) {
  const share = total ? value / total : 0;
  const toneStyles =
    tone === "correct"
      ? { borderColor: "rgba(12,163,12,0.35)", background: "rgba(12,163,12,0.08)" }
      : { borderColor: "rgba(208,59,59,0.35)", background: "rgba(208,59,59,0.08)" };
  return (
    <div className="rounded-xl border p-4 text-center" style={toneStyles}>
      <p className="text-xs font-medium text-muted">{label}</p>
      <p className="mt-2 text-2xl font-bold tabular-nums">{formatNumber(value)}</p>
      <p className="mt-1 text-xs text-muted">{formatPercent(share)} del totale</p>
    </div>
  );
}

// Matrice di confusione 2x2, con intestazioni di riga/colonna esplicite così
// il significato di ogni cella non dipende dal colore.
export function ConfusionMatrix({ matrix }) {
  if (!matrix) return null;
  const { truePositive, falsePositive, falseNegative, trueNegative, testRows, source } = matrix;
  const total = testRows || truePositive + falsePositive + falseNegative + trueNegative;

  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-3">
        <div />
        <p className="self-end pb-1 text-center text-xs font-semibold uppercase tracking-wide text-muted">Previsto: attività</p>
        <p className="self-end pb-1 text-center text-xs font-semibold uppercase tracking-wide text-muted">Previsto: nessuna attività</p>

        <p className="flex items-center pr-1 text-xs font-semibold uppercase tracking-wide text-muted [writing-mode:vertical-rl]">
          Reale: attività
        </p>
        <Cell label="Veri positivi" value={truePositive} total={total} tone="correct" />
        <Cell label="Falsi negativi" value={falseNegative} total={total} tone="incorrect" />

        <p className="flex items-center pr-1 text-xs font-semibold uppercase tracking-wide text-muted [writing-mode:vertical-rl]">
          Reale: nessuna attività
        </p>
        <Cell label="Falsi positivi" value={falsePositive} total={total} tone="incorrect" />
        <Cell label="Veri negativi" value={trueNegative} total={total} tone="correct" />
      </div>
      {source ? <p className="mt-3 text-xs text-muted">{source}.</p> : null}
      <table className="sr-only">
        <caption>Matrice di confusione in forma tabellare</caption>
        <thead>
          <tr>
            <th scope="col" />
            <th scope="col">Previsto: attività</th>
            <th scope="col">Previsto: nessuna attività</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Reale: attività</th>
            <td>{truePositive}</td>
            <td>{falseNegative}</td>
          </tr>
          <tr>
            <th scope="row">Reale: nessuna attività</th>
            <td>{falsePositive}</td>
            <td>{trueNegative}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
