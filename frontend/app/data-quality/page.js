"use client";

import { useCallback } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatusBadge } from "@/components/feedback/StatusBadge";
import { DataQualityCheck } from "@/components/data-quality/DataQualityCheck";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { ErrorState } from "@/components/feedback/ErrorState";
import { useAsyncData } from "@/hooks/useAsyncData";
import { useRefreshListener } from "@/hooks/useRefreshSignal";
import { getDataQuality } from "@/services/data-quality-service";
import { formatDate, formatDays, formatNumber, formatPercent } from "@/lib/formatters";

export default function DataQualityPage() {
  const fetcher = useCallback(() => getDataQuality(), []);
  const { data, loading, error, reload } = useAsyncData(fetcher, [fetcher]);
  useRefreshListener(reload);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Qualità dati"
        description="Copertura temporale, continuità dei segmenti e controlli di qualità sul campione disponibile."
      />

      {loading ? (
        <div className="space-y-4">
          <LoadingSkeleton variant="kpis" />
          <LoadingSkeleton variant="table" rows={6} />
        </div>
      ) : null}
      {error ? <ErrorState description="Impossibile caricare i controlli di qualità." onRetry={reload} /> : null}

      {data ? (
        <>
          <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
            <StatTile label="Giorni osservati" value={formatDays(data.coverage.observedDays)} />
            <StatTile label="Giorni mancanti" value={formatDays(data.coverage.missingDays)} />
            <StatTile label="Copertura" value={formatPercent(data.coverage.observedDays / data.coverage.totalDays)} />
            <StatTile label="Celle campionate" value={formatNumber(data.coverage.sampledCells)} />
            <StatTile label="Segmenti utilizzabili" value={`${data.coverage.usableSegments} / ${data.coverage.totalSegments}`} />
            <StatTile label="Segmenti totali" value={formatNumber(data.coverage.totalSegments)} />
          </section>

          <section className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border bg-surface p-5 shadow-panel">
              <div className="flex items-center justify-between gap-2">
                <h2 className="font-semibold">Stato dei dati grezzi</h2>
                <StatusBadge status={data.rawDataStatus.status} />
              </div>
              <p className="mt-2 text-sm leading-6 text-muted">{data.rawDataStatus.detail}</p>
            </div>
            <div className="rounded-2xl border bg-surface p-5 shadow-panel">
              <div className="flex items-center justify-between gap-2">
                <h2 className="font-semibold">Stato degli artefatti</h2>
                <StatusBadge status={data.artifactsStatus.status} />
              </div>
              <p className="mt-2 text-sm leading-6 text-muted">{data.artifactsStatus.detail}</p>
            </div>
          </section>

          <section>
            <h2 className="mb-3 font-semibold">Controlli di qualità</h2>
            <DataQualityCheck checks={data.checks} />
          </section>

          <section className="rounded-2xl border border-amber-400/25 bg-amber-400/[0.06] p-5 shadow-panel">
            <h2 className="font-semibold">Avvisi</h2>
            <ul className="mt-3 list-disc space-y-2 pl-4 text-sm leading-6 text-muted">
              {data.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </section>

          <section className="rounded-2xl border bg-surface shadow-panel">
            <div className="border-b p-4">
              <h2 className="font-semibold">Segmenti temporali</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse text-sm">
                <caption className="sr-only">Segmenti temporali individuati, con durata ed esito di usabilità per il training.</caption>
                <thead>
                  <tr className="border-b text-left text-xs uppercase tracking-wide text-muted">
                    <th scope="col" className="px-4 py-3">
                      Segmento
                    </th>
                    <th scope="col" className="px-4 py-3">
                      Periodo
                    </th>
                    <th scope="col" className="px-4 py-3 text-right">
                      Giorni
                    </th>
                    <th scope="col" className="px-4 py-3">
                      Esito
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.segments.map((segment) => (
                    <tr key={segment.id} className="border-b last:border-0">
                      <th scope="row" className="px-4 py-3 text-left font-medium">
                        Segmento {segment.id}
                      </th>
                      <td className="px-4 py-3 text-muted">
                        {formatDate(segment.start)} – {formatDate(segment.end)}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums">{segment.days}</td>
                      <td className="px-4 py-3">
                        <StatusBadge status={segment.usableForModel ? "passed" : "not_available"} label={segment.usableForModel ? "Utilizzabile" : "Escluso"} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

function StatTile({ label, value }) {
  return (
    <div className="rounded-xl border bg-surface p-3 text-center shadow-panel">
      <p className="text-[11px] text-muted">{label}</p>
      <p className="mt-1 text-lg font-bold tabular-nums">{value}</p>
    </div>
  );
}
