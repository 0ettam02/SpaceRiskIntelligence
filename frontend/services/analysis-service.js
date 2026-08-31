import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import { SEGMENT_SUMMARIES, EMBARGO_WINDOWS, getSegmentSeries } from "@/data/mock-analysis";

/**
 * Contratto JSON di getTimeSeries(filters):
 * filters: { segmentId?: number | "all", windowDays?: 7 | 14 | 30 | 90 }
 * → {
 *     series: Array<{ date, segmentId, detections, frpSum, activeCells, cellsSampled, missing }>,
 *     // Con segmentId "all" i giorni senza dati compaiono con i campi numerici a
 *     // null e missing:true, così i grafici mostrano l'interruzione reale invece
 *     // di collegare segmenti distanti mesi con una linea continua.
 *     segments: Array<SegmentSummary>,
 *     embargoWindows: Array<{ segmentId, description }>,
 *   }
 */
export async function getTimeSeries(filters = {}) {
  if (isMockMode) {
    const { segmentId = "all", windowDays } = filters;
    let series = getSegmentSeries(segmentId);
    if (windowDays) {
      series = series.slice(-windowDays);
    }
    return withSimulatedLatency({
      series,
      segments: SEGMENT_SUMMARIES,
      embargoWindows: EMBARGO_WINDOWS,
    });
  }
  const query = new URLSearchParams(filters).toString();
  return fetchFromApi(`/analysis/time-series?${query}`);
}
