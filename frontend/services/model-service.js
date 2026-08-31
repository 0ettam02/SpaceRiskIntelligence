import { isMockMode, fetchFromApi, withSimulatedLatency } from "@/services/api-client";
import {
  MODELS,
  RECOMMENDED_MODEL_SLUG,
  RANDOM_FOREST_CONFUSION_MATRIX,
  MODEL_FEATURES,
  MODEL_METHODOLOGY_NOTES,
  MODEL_LIMITATIONS,
} from "@/data/mock-models";
import { generateRocCurve, generatePrCurve, generateProbabilityHistogram } from "@/lib/chart-utils";

/**
 * Contratto JSON di getModels():
 * → { models: Array<ModelMetrics>, recommendedSlug: string }
 */
export async function getModels() {
  if (isMockMode) {
    return withSimulatedLatency({ models: MODELS, recommendedSlug: RECOMMENDED_MODEL_SLUG });
  }
  return fetchFromApi("/models");
}

/**
 * Contratto JSON di getModelDetails(slug):
 * → ModelMetrics & {
 *     features: string[],
 *     methodologyNotes: string[],
 *     limitations: string[],
 *     confusionMatrix: object | null,
 *     rocCurve: Array<{fpr, tpr}>,
 *     prCurve: Array<{recall, precision}>,
 *     probabilityHistogram: object,
 *     curvesAreObserved: boolean, // false in modalità mock (curve sintetiche), true quando il backend
 *                                 // le calcola dalle predizioni reali sul test set
 *   } | null
 */
export async function getModelDetails(slug) {
  if (isMockMode) {
    const model = MODELS.find((item) => item.slug === slug);
    if (!model) return withSimulatedLatency(null);
    return withSimulatedLatency({
      ...model,
      features: MODEL_FEATURES,
      methodologyNotes: MODEL_METHODOLOGY_NOTES[slug] || [],
      limitations: MODEL_LIMITATIONS,
      confusionMatrix: slug === RECOMMENDED_MODEL_SLUG ? RANDOM_FOREST_CONFUSION_MATRIX : null,
      rocCurve: generateRocCurve(model.rocAuc),
      prCurve: generatePrCurve(model.prAuc, model.precision),
      probabilityHistogram: generateProbabilityHistogram(model.threshold),
      curvesAreObserved: false,
    });
  }
  return fetchFromApi(`/models/${slug}`);
}
