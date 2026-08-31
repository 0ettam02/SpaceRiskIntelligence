import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import {
  COVERAGE_SUMMARY,
  RAW_DATA_STATUS,
  ARTIFACTS_STATUS,
  QUALITY_CHECKS,
  DATA_QUALITY_WARNINGS,
} from "@/data/mock-data-quality";
import { TIME_SEGMENTS } from "@/lib/constants";

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
      segments: TIME_SEGMENTS,
    });
  }
  return fetchFromApi("/data-quality");
}
