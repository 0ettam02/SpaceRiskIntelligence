"use client";

import { useCallback, useMemo, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { ChartCard } from "@/components/charts/ChartCard";
import { TimeSeriesChart } from "@/components/charts/TimeSeriesChart";
import { TimeRangeSelector } from "@/components/ui/TimeRangeSelector";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { ErrorState } from "@/components/feedback/ErrorState";
import { EmptyState } from "@/components/feedback/EmptyState";
import { useAsyncData } from "@/hooks/useAsyncData";
import { useRefreshListener } from "@/hooks/useRefreshSignal";
import { getTimeSeries } from "@/services/analysis-service";
import { getDataQuality } from "@/services/data-quality-service";
import { CHART_SERIES, generateDemoForecast } from "@/lib/chart-utils";
import { TIME_RANGE_OPTIONS } from "@/lib/constants";
import { formatDate, formatDateShort, formatDays, formatNumber, formatPercent } from "@/lib/formatters";

const SEGMENT_FILTERS = [
  { value: "all", label: "Tutti" },
  { value: 0, label: "Segmento 0" },
  { value: 4, label: "Segmento 4" },
  { value: 5, label: "Segmento 5" },
];

export default function AnalysisPage() {
  const [windowDays, setWindowDays] = useState(30);
  const [segmentId, setSegmentId] = useState("all");

  const filters = useMemo(() => ({ segmentId, windowDays }), [segmentId, windowDays]);
  const fetcher = useCallback(() => getTimeSeries(filters), [filters]);
  const { data, loading, error, reload } = useAsyncData(fetcher, [fetcher]);
  useRefreshListener(reload);

  const qualityFetcher = useCallback(() => getDataQuality(), []);
  const { data: quality } = useAsyncData(qualityFetcher, [qualityFetcher]);

  const series = useMemo(() => data?.series || [], [data]);
  const forecastSeries = useMemo(() => generateDemoForecast(series), [series]);
  const missingInWindow = useMemo(() => series.filter((point) => point.missing), [series]);
  const usableSegments = useMemo(() => (data?.segments || []).filter((segment) => segment.usableForModel), [data]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analisi temporale"
        description="Andamento dei rilevamenti, dell'intensità FRP e delle celle attive nel tempo, con evidenza dei segmenti temporali e delle finestre di embargo."
      />

      <div className="flex flex-wrap items-center gap-3">
        <TimeRangeSelector value={windowDays} onChange={setWindowDays} options={TIME_RANGE_OPTIONS} />
        <TimeRangeSelector value={segmentId} onChange={setSegmentId} options={SEGMENT_FILTERS} label="Segmento temporale" />
      </div>

      {usableSegments.length ? (
        <p className="text-xs text-muted">
          Un segmento è una finestra di giorni consecutivi senza interruzioni nei rilevamenti:{" "}
          {usableSegments
            .map((segment) => `Segmento ${segment.id} = ${formatDateShort(segment.start)} – ${formatDateShort(segment.end)}`)
            .join(" · ")}
          .
        </p>
      ) : null}

      {loading ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <LoadingSkeleton variant="chart" />
          <LoadingSkeleton variant="chart" />
        </div>
      ) : null}
      {error ? <ErrorState description="Impossibile caricare la serie temporale." onRetry={reload} /> : null}

      {!loading && !error && series.length === 0 ? (
        <EmptyState title="Nessun dato per la selezione corrente" description="Il segmento scelto non produce righe nella finestra selezionata." />
      ) : null}

      {!loading && !error && series.length > 0 ? (
        <>
          {quality ? (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatTile label="Giorni osservati" value={formatDays(quality.coverage.observedDays)} />
              <StatTile label="Giorni mancanti" value={formatDays(quality.coverage.missingDays)} />
              <StatTile label="Copertura" value={formatPercent(quality.coverage.observedDays / quality.coverage.totalDays)} />
              <StatTile label="Segmenti utilizzabili" value={`${quality.coverage.usableSegments} / ${quality.coverage.totalSegments}`} />
            </div>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-2">
            <ChartCard title="Rilevamenti giornalieri" description="Somma dei rilevamenti sul campione disponibile" srSummary="Andamento giornaliero dei rilevamenti.">
              <TimeSeriesChart
                data={series}
                series={[{ key: "detections", label: "Rilevamenti", color: CHART_SERIES.blue.dark }]}
                missingDates={missingInWindow}
              />
            </ChartCard>
            <ChartCard title="Andamento FRP" description="Somma della Fire Radiative Power giornaliera" srSummary="Andamento giornaliero della potenza radiativa aggregata.">
              <TimeSeriesChart
                data={series}
                series={[{ key: "frpSum", label: "FRP (MW)", color: CHART_SERIES.magenta.dark }]}
                valueFormatter={(value) => `${formatNumber(Math.round(value))} MW`}
                missingDates={missingInWindow}
              />
            </ChartCard>
            <ChartCard title="Celle attive" description="Numero di celle del campione con almeno un rilevamento nel giorno" srSummary="Andamento giornaliero del numero di celle attive.">
              <TimeSeriesChart
                data={series}
                series={[{ key: "activeCells", label: "Celle attive", color: CHART_SERIES.aqua.dark }]}
                missingDates={missingInWindow}
              />
            </ChartCard>
            <ChartCard
              title="Osservato vs previsione"
              description="Confronto fra rilevamenti osservati e una previsione dimostrativa a media mobile"
              demo
              srSummary="Confronto fra la serie osservata e una previsione dimostrativa costruita con una media mobile."
            >
              <TimeSeriesChart
                data={forecastSeries}
                kind="line"
                series={[
                  { key: "detections", label: "Osservato", color: CHART_SERIES.blue.dark },
                  { key: "previsto", label: "Previsione (dimostrativa)", color: CHART_SERIES.aqua.dark, dashed: true },
                ]}
              />
            </ChartCard>
          </div>

          {data?.embargoWindows?.length ? (
            <section className="rounded-2xl border bg-surface p-5 shadow-panel">
              <h2 className="font-semibold">Finestre di embargo (7 giorni)</h2>
              <p className="mt-1 text-sm text-muted">
                Ai confini di ciascun segmento utilizzabile viene applicata una finestra di embargo di 7 giorni per separare
                temporalmente training, validation e test ed evitare fughe di informazione.
              </p>
              <ul className="mt-3 space-y-1 text-sm text-muted">
                {data.embargoWindows.map((window) => (
                  <li key={window.segmentId}>• {window.description}</li>
                ))}
              </ul>
            </section>
          ) : null}

          <section className="rounded-2xl border bg-surface shadow-panel">
            <div className="border-b p-4">
              <h2 className="font-semibold">Segmenti temporali</h2>
              <p className="mt-1 text-xs text-muted">Il segmento 5 non produce righe nel dataset ML: non dispone di storia e orizzonte futuro sufficienti.</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-sm">
                <caption className="sr-only">Elenco dei segmenti temporali individuati, con date, durata e utilizzabilità per il training.</caption>
                <thead>
                  <tr className="border-b text-left text-xs uppercase tracking-wide text-muted">
                    <th scope="col" className="px-4 py-3">
                      Segmento
                    </th>
                    <th scope="col" className="px-4 py-3">
                      Inizio
                    </th>
                    <th scope="col" className="px-4 py-3">
                      Fine
                    </th>
                    <th scope="col" className="px-4 py-3 text-right">
                      Giorni
                    </th>
                    <th scope="col" className="px-4 py-3 text-right">
                      Righe ML
                    </th>
                    <th scope="col" className="px-4 py-3">
                      Note
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.segments || []).map((segment) => (
                    <tr key={segment.id} className="border-b last:border-0">
                      <th scope="row" className="px-4 py-3 text-left font-medium">
                        Segmento {segment.id}
                      </th>
                      <td className="px-4 py-3">{formatDate(segment.start)}</td>
                      <td className="px-4 py-3">{formatDate(segment.end)}</td>
                      <td className="px-4 py-3 text-right tabular-nums">{segment.days}</td>
                      <td className="px-4 py-3 text-right tabular-nums">{formatNumber(segment.mlRows || 0)}</td>
                      <td className="px-4 py-3 text-muted">{segment.usableForModel ? "Utilizzabile per il training" : segment.reason}</td>
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
