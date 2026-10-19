import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelComparisonTable } from "@/components/models/ModelComparisonTable";
import { MODELS, RECOMMENDED_MODEL_SLUG } from "@/data/mock-models";

const SELECTED_MODEL = MODELS.filter((model) => model.slug === RECOMMENDED_MODEL_SLUG);

describe("ModelComparisonTable", () => {
  it("renders one row per model with its Accuracy value", () => {
    render(<ModelComparisonTable models={SELECTED_MODEL} />);
    expect(screen.getAllByText("Random Forest — FIRMS + ERA5 (Italia)").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Italia · clean3 FIRMS + ERA5-Land · 331 celle").length).toBeGreaterThan(0);
    expect(screen.getAllByText("78,2%").length).toBeGreaterThan(0);
  });

  it("marks the recommended model", () => {
    render(<ModelComparisonTable models={SELECTED_MODEL} />);
    expect(screen.getAllByText("(algoritmo usato per le previsioni)").length).toBeGreaterThan(0);
  });
});
