"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useMemo } from "react";
import { CalendarRange, Database, Grid3X3, Radar, Rows3 } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { KpiCard } from "@/components/dashboard/KpiCard";
import { RecommendedModelCard } from "@/components/dashboard/RecommendedModelCard";
import { PipelineUpdatesList } from "@/components/dashboard/PipelineUpdatesList";
import { ChartCard } from "@/components/charts/ChartCard";
import { TimeSeriesChart } from "@/components/charts/TimeSeriesChart";
import { ClassDistributionChart } from "@/components/charts/ClassDistributionChart";
import { MethodologyAlert } from "@/components/feedback/MethodologyAlert";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { ErrorState } from "@/components/feedback/ErrorState";
import { useAsyncData } from "@/hooks/useAsyncData";
import { useRefreshListener } from "@/hooks/useRefreshSignal";
import { getOverview } from "@/services/overview-service";
import { getMapCells } from "@/services/map-service";
import { CHART_SERIES } from "@/lib/chart-utils";
import { formatDate } from "@/lib/formatters";

const GlobalFireMap = dynamic(() => import("@/components/map/GlobalFireMap").then((mod) => mod.GlobalFireMap), {
  ssr: false,
  loading: () => <LoadingSkeleton variant="map" />,
});

const KPI_ICONS = {
  "raw-detections": Radar,
  "sampled-cells": Grid3X3,
  "panel-rows": Rows3,
  "ml-rows": Database,
  "observed-days": CalendarRange,
};

export default function OverviewPage() {
  const fetcher = useCallback(() => getOverview(), []);
  const { data, loading, error, reload } = useAsyncData(fetcher, [fetcher]);
  useRefreshListener(reload);

  const mapFetcher = useCallback(() => getMapCells({ metric: "probability" }), []);
  const { data: mapData, reload: reloadMap } = useAsyncData(mapFetcher, [mapFetcher]);
  useRefreshListener(reloadMap);
  const previewCells = useMemo(() => (mapData?.cells || []).slice(0, 800), [mapData]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Global Fire Risk Intelligence"
        description="Analisi globale dei rilevamenti satellitari NASA FIRMS e della probabilità sperimentale di attività rilevata nelle celle geografiche nei successivi sette giorni."
        badge={
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-brand-400/30 bg-brand-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-brand-300">
              Research prototype
            </span>
            {data ? <span className="text-xs text-muted">Ultimo dato del run: {formatDate(data.lastRunDate)}</span> : null}
          </div>
        }
      />

      {loading ? <LoadingSkeleton variant="kpis" /> : null}
      {error ? <ErrorState description="Impossibile caricare la panoramica generale." onRetry={reload} /> : null}

      {data ? (
        <>
          <section aria-labelledby="kpi-title">
            <h2 id="kpi-title" className="sr-only">
              Indicatori principali
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
              {data.kpis.map((kpi) => (
                <KpiCard key={kpi.id} label={kpi.label} value={kpi.value} total={kpi.total} note={kpi.note} icon={KPI_ICONS[kpi.id]} kind={kpi.kind} />
              ))}
            </div>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(18rem,0.8fr)]">
            <article className="flex min-h-[26rem] flex-col overflow-hidden rounded-2xl border bg-surface shadow-panel">
              <div className="flex shrink-0 items-center justify-between border-b px-5 py-4">
                <div>
                  <h2 className="font-semibold">Panoramica geospaziale</h2>
                  <p className="mt-1 text-xs text-muted">Anteprima del campione · livello: probabilità a 7 giorni</p>
                </div>
                <Link href="/map" className="min-h-10 rounded-lg border bg-elevated px-3 text-xs font-semibold leading-10 text-brand-300">
                  Apri mappa
                </Link>
              </div>
              <div className="min-h-[21rem] flex-1">
                <GlobalFireMap cells={previewCells} metric="probability" viewMode="cells" compact />
              </div>
            </article>

            <MethodologyAlert items={data.methodologyWarnings} />
          </section>

          <section className="grid gap-4 lg:grid-cols-2">
            <ChartCard
              title="Rilevamenti giornalieri"
              description="Somma dei rilevamenti sul campione di celle disponibile, per data"
              srSummary="Serie storica dei rilevamenti giornalieri sul campione disponibile."
            >
              <TimeSeriesChart
                data={data.dailySeries}
                series={[{ key: "detections", label: "Rilevamenti", color: CHART_SERIES.blue.dark }]}
              />
            </ChartCard>
            <ChartCard
              title="Distribuzione del target"
              description={data.classDistribution.scope}
              srSummary="Distribuzione delle due classi del target fire_next_7d sul set di test disponibile."
            >
              <ClassDistributionChart classes={data.classDistribution.classes} />
            </ChartCard>
          </section>

          <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,0.8fr)]">
            <article className="rounded-2xl border bg-surface p-5 shadow-panel">
              <h2 className="font-semibold">Ultimi aggiornamenti della pipeline</h2>
              <div className="mt-4">
                <PipelineUpdatesList updates={data.pipelineUpdates} />
              </div>
            </article>
            <RecommendedModelCard model={data.recommendedModel} />
          </section>
        </>
      ) : null}
    </div>
  );
}
