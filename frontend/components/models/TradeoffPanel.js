import { formatPercent } from "@/lib/formatters";

export function TradeoffPanel({ recall, specificity, falsePositiveRate }) {
  return (
    <section className="rounded-2xl border bg-surface p-5 shadow-panel" aria-labelledby="tradeoff-title">
      <h3 id="tradeoff-title" className="font-semibold">
        Compromesso fra recall e specificità
      </h3>
      <p className="mt-2 text-sm leading-6 text-muted">
        La Random Forest privilegia il recall rispetto alla specificità: individua la maggior parte delle celle realmente attive, ma a
        fronte di un numero rilevante di falsi positivi.
      </p>
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-xl border bg-elevated p-4">
          <p className="text-xs text-muted">Recall</p>
          <p className="mt-1 text-xl font-bold tabular-nums">{formatPercent(recall)}</p>
          <p className="mt-1 text-xs text-muted">Quota di celle realmente attive individuate</p>
        </div>
        <div className="rounded-xl border bg-elevated p-4">
          <p className="text-xs text-muted">Specificità</p>
          <p className="mt-1 text-xl font-bold tabular-nums">{formatPercent(specificity)}</p>
          <p className="mt-1 text-xs text-muted">Quota di celle realmente inattive correttamente escluse</p>
        </div>
        <div className="rounded-xl border border-amber-400/25 bg-amber-400/[0.06] p-4">
          <p className="text-xs text-amber-300">Falsi positivi sui negativi reali</p>
          <p className="mt-1 text-xl font-bold tabular-nums text-amber-200">{formatPercent(falsePositiveRate)}</p>
          <p className="mt-1 text-xs text-muted">Circa {formatPercent(falsePositiveRate, { digits: 1 })} delle celle realmente inattive viene classificata come attiva</p>
        </div>
      </div>
      <p className="mt-4 text-xs leading-5 text-muted">
        L&apos;Accuracy da sola non descrive questo compromesso: un modello con Accuracy simile ma specificità molto diversa avrebbe un
        comportamento operativo molto diverso.
      </p>
    </section>
  );
}
