import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelComparisonTable } from "@/components/models/ModelComparisonTable";
import { MODELS } from "@/data/mock-models";

describe("ModelComparisonTable", () => {
  it("renders one row per model with its Accuracy value", () => {
    render(<ModelComparisonTable models={MODELS} />);
    expect(screen.getAllByText("Random Forest").length).toBeGreaterThan(0);
    expect(screen.getAllByText("SVM RBF approssimata").length).toBeGreaterThan(0);
    expect(screen.getAllByText("73,1%").length).toBeGreaterThan(0);
  });

  it("marks the recommended model", () => {
    render(<ModelComparisonTable models={MODELS} />);
    expect(screen.getAllByText("(modello raccomandato)").length).toBeGreaterThan(0);
  });
});
