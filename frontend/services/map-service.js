import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import { MOCK_MAP_CELLS, findCellById } from "@/data/mock-map";
import { getRiskLevel } from "@/lib/risk-utils";

function matchesFilters(cell, filters) {
  if (!filters) return true;
  if (filters.riskLevel && filters.riskLevel !== "all" && cell.riskLevel !== filters.riskLevel) {
    return false;
  }
  if (filters.minLastDetectionDate && cell.lastDetectionDate < filters.minLastDetectionDate) {
    return false;
  }
  if (filters.metric === "probability" && typeof filters.minProbability === "number") {
    return cell.probability >= filters.minProbability;
  }
  return true;
}

/**
 * Contratto JSON di getMapCells(filters):
 * filters: { riskLevel?: string, minLastDetectionDate?: string, metric?: string }
 * → { cells: Array<CellSummary>, total: number }
 */
export async function getMapCells(filters = {}) {
  if (isMockMode) {
    const cells = MOCK_MAP_CELLS.filter((cell) => matchesFilters(cell, filters));
    return withSimulatedLatency({ cells, total: cells.length });
  }
  const query = new URLSearchParams(filters).toString();
  return fetchFromApi(`/map/cells?${query}`);
}

/**
 * Contratto JSON di getCellDetails(cellId):
 * → CellDetails | null, dove CellDetails estende CellSummary con
 *   historicalSeries: Array<{ date, detections }>.
 */
export async function getCellDetails(cellId) {
  if (isMockMode) {
    const cell = findCellById(cellId);
    if (!cell) return withSimulatedLatency(null);
    return withSimulatedLatency({ ...cell, risk: getRiskLevel(cell.probability) });
  }
  return fetchFromApi(`/map/cells/${cellId}`);
}
