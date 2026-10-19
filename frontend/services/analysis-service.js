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
        id: 0, label: "FIRMS clean3 + ERA5-Land", start: "2026-06-18", end: "2026-07-15", days: 28,
        usableForModel: true, reason: "Dataset FIRMS + ERA5-Land italiano",
        rowsInSample: 331, totalDetections: 733, totalFrp: 0, mlRows: 6951,
      }],
      embargoWindows: [{ segmentId: "italy-temporal-split", description: "Holdout casuale riproducibile 80/20; non isola celle o date." }],
    });
  }
  const query = new URLSearchParams(filters).toString();
  return fetchFromApi(`/analysis/time-series?${query}`);
}
