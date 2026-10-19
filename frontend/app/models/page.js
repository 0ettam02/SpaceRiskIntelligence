"use client";

import { useCallback, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { ChartCard } from "@/components/charts/ChartCard";
import { MetricsBarChart } from "@/components/charts/MetricsBarChart";
import { ModelComparisonTable } from "@/components/models/ModelComparisonTable";
import { ConfusionMatrix } from "@/components/models/ConfusionMatrix";
import { TradeoffPanel } from "@/components/models/TradeoffPanel";
import { RecommendedModelCard } from "@/components/dashboard/RecommendedModelCard";
import { Clean3ComparisonPanel } from "@/components/models/Clean3ComparisonPanel";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { ErrorState } from "@/components/feedback/ErrorState";
import { EmptyState } from "@/components/feedback/EmptyState";
import { useAsyncData } from "@/hooks/useAsyncData";
import { useRefreshListener } from "@/hooks/useRefreshSignal";
import { getModels, getModelDetails } from "@/services/model-service";

const METRIC_OPTIONS = [
  { value: "accuracy", label: "Accuracy" },
  { value: "precision", label: "Precision" },
  { value: "recall", label: "Recall" },
  { value: "f1", label: "F1" },
  { value: "rocAuc", label: "ROC-AUC" },
  { value: "prAuc", label: "PR-AUC" },
];

export default function ModelsPage() {
  const [metric, setMetric] = useState("f1");

  const fetcher = useCallback(() => getModels(), []);
  const { data, loading, error, reload } = useAsyncData(fetcher, [fetcher]);
  useRefreshListener(reload);

  const detailsFetcher = useCallback(() => (data?.recommendedSlug ? getModelDetails(data.recommendedSlug) : Promise.resolve(null)), [data?.recommendedSlug]);
  const { data: recommendedDetails } = useAsyncData(detailsFetcher, [detailsFetcher]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Modelli e valutazioni"
        description="Random Forest FIRMS clean3 + ERA5-Land addestrata sul dataset italiano unito."
      />

      {loading ? (
        <div className="space-y-4">
          <LoadingSkeleton variant="table" rows={5} />
          <LoadingSkeleton variant="chart" />
        </div>
      ) : null}
      {error ? <ErrorState description="Impossibile caricare il confronto dei modelli." onRetry={reload} /> : null}
      {data ? <Clean3ComparisonPanel comparison={data.comparison} forecast={data.models[0]} /> : null}

      {data ? (
        data.models.length === 0 ? (
          <EmptyState title="Modello italiano non disponibile" description="Il backend deve avere il dataset FIRMS + ERA5-Land italiano per addestrare l'algoritmo usato nelle previsioni." />
        ) : (
        <>
          <details className="space-y-5 rounded-2xl border bg-surface p-4">
          <summary className="cursor-pointer font-semibold">Dettaglio del modello attivo · forecast 16–22 luglio 2026</summary>
          <p className="text-sm leading-6 text-muted">Il modello è addestrato sull’80% delle righe con target disponibile e rifittato su tutte le righe etichettate per generare il forecast della data più recente.</p>
          <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_20rem]">
            <ModelComparisonTable models={data.models} />
            {recommendedDetails ? (
              <RecommendedModelCard
                model={{
                  slug: recommendedDetails.slug,
                  model: recommendedDetails.model,
                  accuracy: recommendedDetails.accuracy,
                  recall: recommendedDetails.recall,
                  precision: recommendedDetails.precision,
                  rocAuc: recommendedDetails.rocAuc,
                  note: recommendedDetails.methodologyNotes?.[0],
                }}
              />
            ) : null}
          </div>

          <ChartCard
            title="Metriche del modello"
            description="Metriche sul test casuale 80/20 (seed 42)"
            srSummary="Metrica selezionata del modello FIRMS + ERA5-Land italiano."
            actions={
              <label className="flex items-center gap-2 text-xs text-muted">
                Metrica
                <select
                  value={metric}
                  onChange={(event) => setMetric(event.target.value)}
                  className="min-h-9 rounded-lg border bg-elevated px-2 text-sm text-ink"
                >
                  {METRIC_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            }
          >
            <MetricsBarChart models={data.models} metricKey={metric} />
          </ChartCard>

          <div className="grid gap-4 lg:grid-cols-2">
            <ChartCard
              title="Matrice di confusione — FIRMS + ERA5-Land Italia"
              description="Calcolata sulle predizioni del test casuale 80/20."
              srSummary="Matrice di confusione del modello italiano FIRMS + ERA5-Land."
            >
              {recommendedDetails?.confusionMatrix ? <ConfusionMatrix matrix={recommendedDetails.confusionMatrix} /> : <LoadingSkeleton variant="card" />}
            </ChartCard>
            {recommendedDetails ? (
              <TradeoffPanel
                recall={recommendedDetails.recall}
                specificity={recommendedDetails.specificity}
                falsePositiveRate={recommendedDetails.falsePositiveRate}
              />
            ) : null}
          </div>
          </details>
        </>
        )
      ) : null}
    </div>
  );
}
