import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import OverviewPage from "@/app/page";

vi.mock("@/components/map/GlobalFireMap", () => ({
  GlobalFireMap: () => <div data-testid="mock-map" />,
}));

vi.mock("@/services/overview-service", () => ({
  getOverview: () =>
    Promise.resolve({
      kpis: [{ id: "raw-detections", label: "Rilevamenti analizzati", value: 16255053, note: "Stima righe grezze valide", kind: "real" }],
      lastRunDate: "2026-07-17",
      dailySeries: [{ date: "2026-07-01", detections: 10, frpSum: 100, activeCells: 5 }],
      classDistribution: { scope: "Test", classes: [{ id: "negative", label: "Negativo", value: 10 }, { id: "positive", label: "Positivo", value: 20 }] },
      methodologyWarnings: ["Avvertenza di test."],
      pipelineUpdates: [{ date: "2026-07-17", title: "Aggiornamento di test", description: "Descrizione di test." }],
      recommendedModel: { slug: "random-forest", model: "Random Forest", accuracy: 0.731, recall: 0.903, precision: 0.715, rocAuc: 0.816, threshold: 0.38, note: "Nota di test." },
    }),
}));

describe("OverviewPage", () => {
  it("renders the headline, the research prototype badge and the KPI cards once data resolves", async () => {
    render(<OverviewPage />);

    expect(await screen.findByRole("heading", { name: /global fire risk intelligence/i })).toBeInTheDocument();
    expect(screen.getByText(/research prototype/i)).toBeInTheDocument();
    expect(await screen.findByText("Rilevamenti analizzati")).toBeInTheDocument();
    expect(screen.getByText("16.255.053")).toBeInTheDocument();
  });

  it("renders the methodology warnings and the recommended model", async () => {
    render(<OverviewPage />);
    expect(await screen.findByText("Avvertenza di test.")).toBeInTheDocument();
    expect(screen.getByText("Random Forest")).toBeInTheDocument();
  });
});
