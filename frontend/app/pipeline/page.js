"use client";

import { useCallback } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { PipelineStepper } from "@/components/pipeline/PipelineStepper";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { ErrorState } from "@/components/feedback/ErrorState";
import { useAsyncData } from "@/hooks/useAsyncData";
import { useRefreshListener } from "@/hooks/useRefreshSignal";
import { getPipelineStatus } from "@/services/pipeline-service";
import { formatDate } from "@/lib/formatters";

export default function PipelinePage() {
  const fetcher = useCallback(() => getPipelineStatus(), []);
  const { data, loading, error, reload } = useAsyncData(fetcher, [fetcher]);
  useRefreshListener(reload);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Pipeline"
        description="Flusso end-to-end dall'acquisizione NASA FIRMS alle metriche dei modelli, con stato, durata dimostrativa e numero di record per ciascuna fase."
      />

      {loading ? <LoadingSkeleton variant="card" rows={6} /> : null}
      {error ? <ErrorState description="Impossibile caricare lo stato della pipeline." onRetry={reload} /> : null}

      {data ? (
        <>
          <p className="text-sm text-muted">Ultima esecuzione: {formatDate(data.lastRun)}</p>

          <PipelineStepper steps={data.steps} />
        </>
      ) : null}
    </div>
  );
}
