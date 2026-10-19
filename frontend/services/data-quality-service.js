import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import {
  COVERAGE_SUMMARY,
  RAW_DATA_STATUS,
  ARTIFACTS_STATUS,
  QUALITY_CHECKS,
  DATA_QUALITY_WARNINGS,
} from "@/data/mock-data-quality";

/**
 * Contratto JSON di getDataQuality():
 * → {
 *     coverage: { observedDays, totalDays, missingDays, sampledCells, usableSegments, totalSegments },
 *     rawDataStatus: { status, label, detail },
 *     artifactsStatus: { status, label, detail },
 *     checks: Array<{ id, label, status, detail }>,
 *     warnings: string[],
 *     segments: Array<SegmentSummary>,
 *   }
 */
export async function getDataQuality() {
  if (isMockMode) {
    return withSimulatedLatency({
      coverage: COVERAGE_SUMMARY,
      rawDataStatus: RAW_DATA_STATUS,
      artifactsStatus: ARTIFACTS_STATUS,
      checks: QUALITY_CHECKS,
      warnings: DATA_QUALITY_WARNINGS,
      segments: [{ id: 0, label: "FIRMS clean3 + ERA5-Land", start: "2026-06-18", end: "2026-07-15", days: 28, usableForModel: true, reason: "Dataset FIRMS clean3 filtrato sulla griglia italiana e unito al meteo giornaliero" }],
    });
  }
  return fetchFromApi("/data-quality");
}
