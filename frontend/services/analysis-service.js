import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import { ITALY_DAILY_SERIES } from "@/data/mock-italy-series";

/**
 * Contratto JSON di getTimeSeries(filters):
 * filters: { segmentId?: number | "all", windowDays?: 7 | 14 | 30 | 90 }
 * → {
 *     series: Array<{ date, segmentId, detections, frpSum, activeCells, cellsSampled, missing }>,
 *     // Serie aggregata del dataset italiano FIRMS + ERA5-Land.
 *     segments: Array<SegmentSummary>,
 *     embargoWindows: Array<{ segmentId, description }>,
 *   }
 */
export async function getTimeSeries(filters = {}) {
  if (isMockMode) {
    const { windowDays } = filters;
    let series = ITALY_DAILY_SERIES;
    if (windowDays) {
      series = series.slice(-windowDays);
    }
    return withSimulatedLatency({
      series,
      segments: [{
        id: 0, label: "Campione italiano", start: "2025-05-26", end: "2026-07-10", days: 411,
        usableForModel: true, reason: "Dataset FIRMS + ERA5-Land italiano",
        rowsInSample: 7809, totalDetections: 75, totalFrp: 0, mlRows: 7809,
      }],
      embargoWindows: [{ segmentId: "italy-temporal-split", description: "Split temporale italiano con embargo di 7 giorni fra training, validation e test del modello FIRMS + ERA5-Land." }],
    });
  }
  const query = new URLSearchParams(filters).toString();
  return fetchFromApi(`/analysis/time-series?${query}`);
}
