import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import { PIPELINE_STEPS, PIPELINE_LAST_RUN } from "@/data/mock-pipeline";

/**
 * Contratto JSON di getPipelineStatus():
 * → {
 *     steps: Array<{ id, title, status, durationLabel, records, input, output, warnings: string[] }>,
 *     lastRun: string (ISO date),
 *   }
 */
export async function getPipelineStatus() {
  if (isMockMode) {
    return withSimulatedLatency({
      steps: PIPELINE_STEPS,
      lastRun: PIPELINE_LAST_RUN,
    });
  }
  return fetchFromApi("/pipeline/status");
}
