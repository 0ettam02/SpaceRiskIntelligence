"use client";

import dynamic from "next/dynamic";
import { useCallback, useMemo, useState } from "react";
import { LocateFixed, RotateCcw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { FilterPanel } from "@/components/ui/FilterPanel";
import { TimeRangeSelector } from "@/components/ui/TimeRangeSelector";
import { RiskLegend } from "@/components/map/RiskLegend";
import { CellDetailsPanel } from "@/components/map/CellDetailsPanel";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";
import { EmptyState } from "@/components/feedback/EmptyState";
import { ErrorState } from "@/components/feedback/ErrorState";
import { useDisclosure } from "@/hooks/useDisclosure";
import { useAsyncData } from "@/hooks/useAsyncData";
import { useRefreshListener } from "@/hooks/useRefreshSignal";
import { getMapCells, getCellDetails } from "@/services/map-service";
import { findNearestCell } from "@/data/mock-map";
import { MAP_METRICS } from "@/lib/constants";
import { RISK_LEVELS } from "@/lib/risk-utils";

const GlobalFireMap = dynamic(() => import("@/components/map/GlobalFireMap").then((mod) => mod.GlobalFireMap), {
  ssr: false,
  loading: () => <LoadingSkeleton variant="map" />,
});

const VIEW_MODES = [
  { value: "cluster", label: "Cluster" },
  { value: "cells", label: "Celle 0,1°" },
];

const INTERVAL_OPTIONS = [
  { value: 3, label: "3 giorni" },
  { value: 7, label: "7 giorni" },
  { value: 14, label: "14 giorni" },
  { value: 30, label: "30 giorni" },
];

export default function MapPage() {
  const [metric, setMetric] = useState("probability");
  const [viewMode, setViewMode] = useState("cells");
  const [intervalDays, setIntervalDays] = useState(7);
  const [riskLevel, setRiskLevel] = useState("all");
  const [minLastDetectionDate, setMinLastDetectionDate] = useState("");
  const [selectedCellId, setSelectedCellId] = useState(null);
  const [command, setCommand] = useState(null);
  const [searchInput, setSearchInput] = useState({ lat: "", lon: "" });
  const [searchMessage, setSearchMessage] = useState("");
  const filtersDrawer = useDisclosure(false);

  const filters = useMemo(() => ({ riskLevel, minLastDetectionDate, metric }), [riskLevel, minLastDetectionDate, metric]);
  const cellsFetcher = useCallback(() => getMapCells(filters), [filters]);
  const { data: cellsData, loading: cellsLoading, error: cellsError, reload: reloadCells } = useAsyncData(cellsFetcher, [cellsFetcher]);

  const cellDetailsFetcher = useCallback(() => (selectedCellId ? getCellDetails(selectedCellId) : Promise.resolve(null)), [selectedCellId]);
  const { data: cellDetails, loading: cellLoading, error: cellError, reload: reloadCell } = useAsyncData(cellDetailsFetcher, [cellDetailsFetcher]);

  const cells = cellsData?.cells || [];
  useRefreshListener(reloadCells);

  const handleSearch = (event) => {
    event.preventDefault();
    const lat = Number(searchInput.lat);
    const lon = Number(searchInput.lon);
    if (Number.isNaN(lat) || Number.isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      setSearchMessage("Inserisci coordinate valide (latitudine -90/90, longitudine -180/180).");
      return;
    }
    const nearest = findNearestCell(cells, lat, lon);
    if (!nearest) {
      setSearchMessage("Nessuna cella campionata entro un raggio ragionevole da queste coordinate.");
      setCommand({ action: "flyTo", lat, lon, nonce: Date.now() });
      return;
    }
    setSearchMessage("");
    setSelectedCellId(nearest.id);
    setCommand({ action: "flyTo", lat: nearest.lat, lon: nearest.lon, nonce: Date.now() });
  };

  const handleReset = () => {
    setCommand({ action: "reset", nonce: Date.now() });
    setSelectedCellId(null);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex h-[calc(100vh-8rem)] min-h-[36rem] flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <PageHeader
          title="Mappa globale"
          description="Esplora le celle geografiche campionate: cluster di densità, probabilità stimata a 7 giorni, intensità FRP e giorni attivi recenti."
        />
        <button
          type="button"
          onClick={filtersDrawer.open}
          className="inline-flex min-h-11 items-center gap-2 rounded-xl border bg-surface px-3 text-sm font-semibold lg:hidden"
        >
          Filtri
        </button>
      </div>

      <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <div className="min-h-0 lg:overflow-y-auto lg:pr-1">
        <FilterPanel
          title="Filtri mappa"
          isOpen={filtersDrawer.isOpen}
          onClose={filtersDrawer.close}
          onReset={() => {
            setRiskLevel("all");
            setMinLastDetectionDate("");
            setIntervalDays(7);
          }}
        >
          <div>
            <p className="mb-2 text-xs font-semibold text-muted">Metrica visualizzata</p>
            <div className="space-y-1">
              {MAP_METRICS.map((option) => (
                <label key={option.value} className="flex min-h-9 items-center gap-2 rounded-lg px-2 text-sm hover:bg-elevated">
                  <input
                    type="radio"
                    name="map-metric"
                    value={option.value}
                    checked={metric === option.value}
                    onChange={() => setMetric(option.value)}
                    className="h-4 w-4 accent-brand-500"
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </div>

          <div>
            <p className="mb-2 text-xs font-semibold text-muted">Visualizzazione</p>
            <TimeRangeSelector value={viewMode} onChange={setViewMode} options={VIEW_MODES} label="Modalità di visualizzazione" />
          </div>

          <div>
            <p className="mb-2 text-xs font-semibold text-muted">Intervallo temporale metrica</p>
            <TimeRangeSelector value={intervalDays} onChange={setIntervalDays} options={INTERVAL_OPTIONS} label="Intervallo temporale" />
          </div>

          <div>
            <label htmlFor="risk-filter" className="mb-2 block text-xs font-semibold text-muted">
              Livello di rischio
            </label>
            <select
              id="risk-filter"
              value={riskLevel}
              onChange={(event) => setRiskLevel(event.target.value)}
              className="min-h-10 w-full rounded-lg border bg-elevated px-2 text-sm"
            >
              <option value="all">Tutti i livelli</option>
              {RISK_LEVELS.map((level) => (
                <option key={level.id} value={level.id}>
                  {level.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="date-filter" className="mb-2 block text-xs font-semibold text-muted">
              Ultimo rilevamento dopo il
            </label>
            <input
              id="date-filter"
              type="date"
              value={minLastDetectionDate}
              onChange={(event) => setMinLastDetectionDate(event.target.value)}
              className="min-h-10 w-full rounded-lg border bg-elevated px-2 text-sm"
            />
          </div>

          <form onSubmit={handleSearch}>
            <p className="mb-2 text-xs font-semibold text-muted">Ricerca per coordinate</p>
            <div className="grid grid-cols-2 gap-2">
              <label className="sr-only" htmlFor="search-lat">
                Latitudine
              </label>
              <input
                id="search-lat"
                type="number"
                step="0.1"
                placeholder="Lat"
                value={searchInput.lat}
                onChange={(event) => setSearchInput((prev) => ({ ...prev, lat: event.target.value }))}
                className="min-h-10 w-full rounded-lg border bg-elevated px-2 text-sm"
              />
              <label className="sr-only" htmlFor="search-lon">
                Longitudine
              </label>
              <input
                id="search-lon"
                type="number"
                step="0.1"
                placeholder="Lon"
                value={searchInput.lon}
                onChange={(event) => setSearchInput((prev) => ({ ...prev, lon: event.target.value }))}
                className="min-h-10 w-full rounded-lg border bg-elevated px-2 text-sm"
              />
            </div>
            <button type="submit" className="mt-2 inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-lg border bg-elevated text-sm font-semibold">
              <LocateFixed aria-hidden="true" size={16} />
              Vai alle coordinate
            </button>
            {searchMessage ? (
              <p role="status" className="mt-2 text-xs text-amber-300">
                {searchMessage}
              </p>
            ) : null}
          </form>

          <button
            type="button"
            onClick={handleReset}
            className="inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-lg border bg-elevated text-sm font-semibold text-muted hover:text-ink"
          >
            <RotateCcw aria-hidden="true" size={16} />
            Reimposta visualizzazione
          </button>

          <RiskLegend compact />
        </FilterPanel>
        </div>

        <div className="relative min-h-[24rem] overflow-hidden rounded-2xl border bg-surface shadow-panel">
          {cellsLoading ? (
            <div className="absolute inset-0">
              <LoadingSkeleton variant="map" />
            </div>
          ) : cellsError ? (
            <div className="absolute inset-0 flex items-center justify-center p-6">
              <ErrorState title="Impossibile caricare le celle" description="Riprova ad aggiornare i filtri o la pagina." onRetry={reloadCells} />
            </div>
          ) : cells.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center p-6">
              <EmptyState title="Nessuna cella corrisponde ai filtri" description="Prova ad ampliare i filtri selezionati." />
            </div>
          ) : (
            <GlobalFireMap
              cells={cells}
              metric={metric}
              viewMode={viewMode}
              intervalDays={intervalDays}
              selectedCellId={selectedCellId}
              onCellSelect={setSelectedCellId}
              command={command}
            />
          )}
          <div className="pointer-events-none absolute bottom-3 left-3 max-w-[13rem] lg:hidden">
            <div className="pointer-events-auto">
              <RiskLegend />
            </div>
          </div>
          <p className="sr-only" role="status">
            {cells.length} celle mostrate sulla mappa con la metrica {MAP_METRICS.find((m) => m.value === metric)?.label}.
          </p>
        </div>
      </div>
      </div>

      <CellDetailsPanel
        cell={cellDetails}
        loading={Boolean(selectedCellId) && cellLoading}
        error={Boolean(selectedCellId) && Boolean(cellError)}
        onClose={() => setSelectedCellId(null)}
        onRetry={reloadCell}
      />
    </div>
  );
}
