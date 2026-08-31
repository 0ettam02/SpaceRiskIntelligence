"use client";

import { useCallback } from "react";
import Link from "next/link";
import { ArrowLeft, GitCompareArrows } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { ChartCard } from "@/components/charts/ChartCard";
import { RocCurveChart } from "@/components/charts/RocCurveChart";
import { PrCurveChart } from "@/components/charts/PrCurveChart";
import { ProbabilityDistributionChart } from "@/components/charts/ProbabilityDistributionChart";
import { ConfusionMatrix } from "@/components/models/ConfusionMatrix";
import { EmptyState } from "@/components/feedback/EmptyState";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { ErrorState } from "@/components/feedback/ErrorState";
import { FEATURE_GROUPS } from "@/lib/constants";
import { formatMetric, formatPercent } from "@/lib/formatters";
import { useAsyncData } from "@/hooks/useAsyncData";
import { getModelDetails } from "@/services/model-service";

const METRIC_ROWS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "balancedAccuracy", label: "Balanced Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "rocAuc", label: "ROC-AUC" },
  { key: "prAuc", label: "PR-AUC" },
];

export default function ModelDetailPage({ params }) {
  const { slug } = params;
  const fetcher = useCallback(() => getModelDetails(slug), [slug]);
  const { data: model, loading, error, reload } = useAsyncData(fetcher, [fetcher]);

  return (
    <div className="space-y-6">
      <Link href="/models" className="inline-flex items-center gap-1.5 text-sm font-medium text-brand-300 hover:underline">
        <ArrowLeft aria-hidden="true" size={16} />
        Torna al confronto dei modelli
      </Link>

      {loading ? <LoadingSkeleton variant="card" rows={5} /> : null}
      {error ? <ErrorState description="Impossibile caricare il dettaglio del modello." onRetry={reload} /> : null}
      {!loading && !error && !model ? <EmptyState title="Modello non trovato" description="Verifica l'indirizzo o torna al confronto dei modelli." /> : null}

      {model ? (
        <>
          <PageHeader
            title={model.model}
            description={`Soglia di decisione: ${formatMetric(model.threshold, { digits: 2 })} · Stato: ${model.status}`}
            badge={
              model.recommended ? (
                <span className="rounded-full border border-brand-400/30 bg-brand-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-brand-300">
                  Modello raccomandato
                </span>
              ) : null
            }
            actions={
              <Link href="/models" className="inline-flex min-h-11 items-center gap-2 rounded-xl border bg-surface px-3 text-sm font-semibold text-brand-300 hover:bg-elevated">
                <GitCompareArrows aria-hidden="true" size={16} />
                Confronta con altri modelli
              </Link>
            }
          />

          <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {METRIC_ROWS.map((row) => (
              <div key={row.key} className="rounded-xl border bg-surface p-4 shadow-panel">
                <p className="text-xs text-muted">{row.label}</p>
                <p className="mt-1 text-xl font-bold tabular-nums">{formatPercent(model[row.key])}</p>
              </div>
            ))}
          </section>

          <div className="grid gap-4 lg:grid-cols-2">
            <ChartCard
              title="Curva ROC"
              demo={!model.curvesAreObserved}
              description={
                model.curvesAreObserved
                  ? "Calcolata dalle predizioni reali sul test set isolato"
                  : "Forma coerente con l'AUC reale del modello, punti non osservati direttamente"
              }
              srSummary={`Curva ROC con AUC ${formatMetric(model.rocAuc)}.`}
            >
              <RocCurveChart data={model.rocCurve} auc={model.rocAuc} />
            </ChartCard>
            <ChartCard
              title="Curva Precision-Recall"
              demo={!model.curvesAreObserved}
              description={
                model.curvesAreObserved
                  ? "Calcolata dalle predizioni reali sul test set isolato"
                  : "Forma coerente con la PR-AUC reale del modello, punti non osservati direttamente"
              }
              srSummary={`Curva Precision-Recall con PR-AUC ${formatMetric(model.prAuc)}.`}
            >
              <PrCurveChart data={model.prCurve} prAuc={model.prAuc} />
            </ChartCard>
          </div>

          <ChartCard
            title="Distribuzione delle probabilità previste"
            demo={!model.curvesAreObserved}
            description={
              model.curvesAreObserved
                ? "Istogramma calcolato sulle predizioni reali del test set, con la soglia di decisione evidenziata"
                : "Istogramma dimostrativo per classe reale, con la soglia di decisione evidenziata"
            }
            srSummary="Istogramma della distribuzione delle probabilità previste, separato per classe reale."
          >
            <ProbabilityDistributionChart histogram={model.probabilityHistogram.histogram} threshold={model.threshold} />
          </ChartCard>

          <ChartCard
            title="Matrice di confusione"
            description={model.confusionMatrix ? "Conteggi reali sul test set isolato" : "Non disponibile per questo modello"}
            srSummary="Matrice di confusione del modello selezionato."
          >
            {model.confusionMatrix ? (
              <ConfusionMatrix matrix={model.confusionMatrix} />
            ) : (
              <EmptyState title="Matrice non disponibile" description="La matrice di confusione dettagliata è disponibile solo per il modello raccomandato in questo run." />
            )}
          </ChartCard>

          <section className="rounded-2xl border bg-surface p-5 shadow-panel">
            <h2 className="font-semibold">Feature utilizzate</h2>
            <p className="mt-1 text-sm text-muted">Le stesse 17 feature sono utilizzate da tutti i modelli confrontati.</p>
            <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {FEATURE_GROUPS.map((group) => (
                <div key={group.id}>
                  <p className="text-xs font-semibold uppercase tracking-[0.1em] text-brand-300">{group.label}</p>
                  <ul className="mt-2 space-y-1 text-sm text-muted">
                    {group.features.map((feature) => (
                      <li key={feature.name}>
                        <code className="text-xs text-ink">{feature.name}</code>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="rounded-2xl border bg-surface p-5 shadow-panel">
              <h2 className="font-semibold">Note metodologiche</h2>
              <ul className="mt-3 list-disc space-y-2 pl-4 text-sm leading-6 text-muted">
                {model.methodologyNotes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </section>
            <section className="rounded-2xl border border-amber-400/25 bg-amber-400/[0.06] p-5 shadow-panel">
              <h2 className="font-semibold">Limiti del modello</h2>
              <ul className="mt-3 list-disc space-y-2 pl-4 text-sm leading-6 text-muted">
                {model.limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            </section>
          </div>
        </>
      ) : null}
    </div>
  );
}
