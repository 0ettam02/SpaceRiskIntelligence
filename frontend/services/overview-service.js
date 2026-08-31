import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import {
  OVERVIEW_KPIS,
  TARGET_CLASS_DISTRIBUTION,
  METHODOLOGY_WARNINGS,
  PIPELINE_UPDATES,
  RECOMMENDED_MODEL_SUMMARY,
  LAST_RUN_DATE,
} from "@/data/mock-overview";
import { CALENDAR_SERIES } from "@/data/mock-analysis";

/**
 * Contratto JSON di getOverview():
 * {
 *   kpis: Array<{ id, label, value, total?, note, kind }>,
 *   lastRunDate: string (ISO date),
 *   dailySeries: Array<{ date, detections, frpSum, activeCells, missing }>, // un punto per ogni giorno del run, null nei giorni senza dati
 *   classDistribution: { scope, classes: Array<{ id, label, value }> },
 *   methodologyWarnings: string[],
 *   pipelineUpdates: Array<{ date, title, description }>,
 *   recommendedModel: { slug, model, accuracy, recall, precision, rocAuc, threshold, note },
 * }
 */
export async function getOverview() {
  if (isMockMode) {
    return withSimulatedLatency({
      kpis: OVERVIEW_KPIS,
      lastRunDate: LAST_RUN_DATE,
      dailySeries: CALENDAR_SERIES,
      classDistribution: TARGET_CLASS_DISTRIBUTION,
      methodologyWarnings: METHODOLOGY_WARNINGS,
      pipelineUpdates: PIPELINE_UPDATES,
      recommendedModel: RECOMMENDED_MODEL_SUMMARY,
    });
  }
  return fetchFromApi("/overview");
}
