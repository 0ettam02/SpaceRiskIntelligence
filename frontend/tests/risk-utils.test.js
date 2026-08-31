import { describe, expect, it } from "vitest";
import { getPredictedClassLabel, getRiskLevel, getStatusMeta } from "@/lib/risk-utils";

describe("risk classification", () => {
  it("classifies low probabilities as bassa", () => {
    expect(getRiskLevel(0.05).id).toBe("bassa");
  });

  it("classifies mid-range probabilities as moderata", () => {
    expect(getRiskLevel(0.3).id).toBe("moderata");
  });

  it("classifies high probabilities as elevata", () => {
    expect(getRiskLevel(0.6).id).toBe("elevata");
  });

  it("classifies very high probabilities as molto elevata", () => {
    expect(getRiskLevel(0.9).id).toBe("molto-elevata");
  });

  it("handles boundary values consistently with the documented thresholds", () => {
    expect(getRiskLevel(0.25).id).toBe("moderata");
    expect(getRiskLevel(0.75).id).toBe("molto-elevata");
  });

  it("returns null for missing probabilities", () => {
    expect(getRiskLevel(undefined)).toBeNull();
  });

  it("maps predicted class codes to Italian labels", () => {
    expect(getPredictedClassLabel(1)).toBe("Attività prevista");
    expect(getPredictedClassLabel(0)).toBe("Nessuna attività prevista");
  });

  it("falls back to a default status when unknown", () => {
    expect(getStatusMeta("unknown-status").label).toBe("Non disponibile");
  });
});
