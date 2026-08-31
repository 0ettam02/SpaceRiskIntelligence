"use client";

import { X } from "lucide-react";
import { RiskBadge } from "@/components/feedback/RiskBadge";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { EmptyState } from "@/components/feedback/EmptyState";
import { TimeSeriesChart } from "@/components/charts/TimeSeriesChart";
import { CHART_SERIES } from "@/lib/chart-utils";
import { getPredictedClassLabel } from "@/lib/risk-utils";
import { formatCoordinate, formatDate, formatFrp, formatNumber, formatPercent } from "@/lib/formatters";

function Metric({ label, value }) {
  return (
    <div className="rounded-lg border bg-elevated p-3">
      <p className="text-[11px] text-muted">{label}</p>
      <p className="mt-1 text-sm font-semibold tabular-nums">{value}</p>
    </div>
  );
}

function PanelContent({ cell, loading, error, onRetry }) {
  if (loading) return <LoadingSkeleton variant="card" rows={4} />;
  if (error) {
    return (
      <EmptyState
        title="Impossibile caricare la cella"
        description="Riprova a selezionare la cella sulla mappa."
        action={
          onRetry ? (
            <button type="button" onClick={onRetry} className="inline-flex min-h-10 items-center rounded-lg border bg-elevated px-3 text-sm font-semibold">
              Riprova
            </button>
          ) : null
        }
      />
    );
  }
  if (!cell) {
    return <EmptyState title="Nessuna cella selezionata" description="Seleziona un punto sulla mappa per vederne i dettagli." />;
  }

  return (
    <div className="space-y-4 lg:grid lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] lg:gap-x-6 lg:gap-y-0 lg:space-y-0">
      <div className="space-y-4">
        <div>
          <p className="text-xs text-muted">{cell.region}</p>
          <p className="text-lg font-bold tabular-nums">{formatCoordinate(cell.lat, cell.lon)}</p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <RiskBadge level={cell.riskLevel} />
            <span className="text-xs text-muted">
              Classe prevista: <span className="font-semibold text-ink">{getPredictedClassLabel(cell.predictedClass)}</span>
            </span>
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-muted">Probabilità 7 giorni</p>
          <p className="mt-1 text-2xl font-bold tabular-nums text-brand-300">{formatPercent(cell.probability)}</p>
        </div>

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-2">
          <Metric label="Rilevamenti (3gg)" value={formatNumber(cell.detections3d)} />
          <Metric label="Rilevamenti (7gg)" value={formatNumber(cell.detections7d)} />
          <Metric label="Rilevamenti (14gg)" value={formatNumber(cell.detections14d)} />
          <Metric label="Rilevamenti (30gg)" value={formatNumber(cell.detections30d)} />
          <Metric label="Giorni attivi (7gg)" value={formatNumber(cell.activeDaysLast7)} />
          <Metric label="Giorni attivi (30gg)" value={formatNumber(cell.activeDaysLast30)} />
          <Metric label="FRP aggregato" value={formatFrp(cell.frpSum)} />
          <Metric label="Ultimo rilevamento" value={formatDate(cell.lastDetectionDate)} />
        </div>

        <div>
          <p className="text-xs text-muted">
            Modello: <span className="font-medium text-ink">{cell.model}</span> · Soglia: <span className="font-medium text-ink">{cell.threshold}</span>
          </p>
          {cell.referenceDate ? (
            <p className="mt-1 text-xs text-muted">
              Previsione calcolata al: <span className="font-medium text-ink">{formatDate(cell.referenceDate)}</span> (ultima data con
              feature complete per questa cella, non necessariamente oggi).
            </p>
          ) : null}
        </div>
      </div>

      <div className="mt-4 space-y-4 lg:mt-0">
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-muted">Storico rilevamenti (30 giorni)</p>
          <TimeSeriesChart
            data={cell.historicalSeries}
            series={[{ key: "detections", label: "Rilevamenti", color: CHART_SERIES.blue.dark }]}
            height={160}
          />
        </div>

        <p className="rounded-lg border border-amber-400/20 bg-amber-400/[0.06] p-3 text-xs leading-5 text-muted">
          La probabilità stimata descrive la possibilità di un ulteriore rilevamento satellitare nella cella, non il numero certo di
          incendi futuri né la loro estensione fisica.
        </p>
      </div>
    </div>
  );
}

export function CellDetailsPanel({ cell, loading, error, onClose, onRetry }) {
  const hasSelection = Boolean(cell) || loading || error;

  return (
    <>
      <div className="hidden shrink-0 flex-col rounded-2xl border bg-surface shadow-panel lg:flex">
        <div className="flex items-center justify-between border-b p-4">
          <p className="font-semibold">Dettaglio cella</p>
          {hasSelection ? (
            <button type="button" onClick={onClose} aria-label="Chiudi dettaglio cella" className="grid h-9 w-9 place-items-center rounded-lg border bg-elevated">
              <X aria-hidden="true" size={16} />
            </button>
          ) : null}
        </div>
        <div className="p-4">
          <PanelContent cell={cell} loading={loading} error={error} onRetry={onRetry} />
        </div>
      </div>

      {hasSelection ? (
        <div className="fixed inset-x-0 bottom-0 z-40 lg:hidden">
          <div className="max-h-[75vh] overflow-y-auto rounded-t-2xl border-t bg-surface p-4 shadow-panel">
            <div className="mb-3 flex items-center justify-between">
              <span className="mx-auto h-1.5 w-12 rounded-full bg-line" aria-hidden="true" />
            </div>
            <div className="mb-3 flex items-center justify-between">
              <p className="font-semibold">Dettaglio cella</p>
              <button type="button" onClick={onClose} aria-label="Chiudi dettaglio cella" className="grid h-9 w-9 place-items-center rounded-lg border bg-elevated">
                <X aria-hidden="true" size={16} />
              </button>
            </div>
            <PanelContent cell={cell} loading={loading} error={error} onRetry={onRetry} />
          </div>
        </div>
      ) : null}
    </>
  );
}
