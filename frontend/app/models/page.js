"use client";

import { useCallback, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { ChartCard } from "@/components/charts/ChartCard";
import { MetricsBarChart } from "@/components/charts/MetricsBarChart";
import { ModelComparisonTable } from "@/components/models/ModelComparisonTable";
import { ConfusionMatrix } from "@/components/models/ConfusionMatrix";
import { TradeoffPanel } from "@/components/models/TradeoffPanel";
import { RecommendedModelCard } from "@/components/dashboard/RecommendedModelCard";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { ErrorState } from "@/components/feedback/ErrorState";
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
        title="Modelli ML"
        description="Confronto dei classificatori sperimentali valutati su fire_next_7d, su test temporale isolato con embargo di 7 giorni."
      />

      {loading ? (
        <div className="space-y-4">
          <LoadingSkeleton variant="table" rows={5} />
          <LoadingSkeleton variant="chart" />
        </div>
      ) : null}
      {error ? <ErrorState description="Impossibile caricare il confronto dei modelli." onRetry={reload} /> : null}

      {data ? (
        <>
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
            title="Confronto metriche"
            description="Seleziona la metrica da confrontare fra i 5 modelli"
            srSummary="Grafico a barre che confronta una metrica selezionata fra i cinque modelli valutati."
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
              title="Matrice di confusione — Random Forest"
              description="Ricostruita dai conteggi reali del test set isolato (segmento 0)"
              srSummary="Matrice di confusione della Random Forest con veri positivi, falsi negativi, falsi positivi e veri negativi."
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
        </>
      ) : null}
    </div>
  );
}
