import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ResearchResultsPanel } from "@/components/models/ResearchResultsPanel";
import snapshot from "@/data/research-results.json";

 describe("ResearchResultsPanel", () => {
  it("shows the current ARCO paired evaluation and its partial coverage", () => {
    render(<ResearchResultsPanel research={snapshot} />);
    expect(screen.getByText(/Modalità demo: risultati reali salvati/)).toBeInTheDocument();
    expect(screen.getByText(/ERA5 parziale: queste metriche/)).toBeInTheDocument();
    expect(screen.getByText(/3\.185\s*\/\s*3\.475 celle/)).toBeInTheDocument();
    expect(screen.getByText(/290 celle/)).toBeInTheDocument();
    expect(screen.getByText("0,228")).toBeInTheDocument();
    expect(screen.getByText(/La soglia massimizza l’F1 sul solo periodo di validation/)).toBeInTheDocument();
    expect(screen.getByText(/La mappa usa la Random Forest v2/)).toBeInTheDocument();
  });

  it("switches dataset without mixing national FIRMS with partial ERA5 scores", () => {
    render(<ResearchResultsPanel research={snapshot} />);
    fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "firms" } });
    expect(screen.getByText("3.475")).toBeInTheDocument();
    expect(screen.getByText(/questo esperimento usa soltanto feature FIRMS/)).toBeInTheDocument();
    expect(screen.getByText("0,643")).toBeInTheDocument();
    expect(screen.queryByText("Random Forest · solo FIRMS (stesse righe)")).not.toBeInTheDocument();
  });

  it("reports undefined metrics when a spatial test has no positives", () => {
    const noPositiveSnapshot = structuredClone(snapshot);
    const era5Fold = noPositiveSnapshot.studies.find((study) => study.id === "era5").results
      .find((row) => row.fold === 0 && row.model === "random_forest_regularized");
    era5Fold.test.unseen_spatial_blocks.metrics.positive_rows = 0;
    era5Fold.test.unseen_spatial_blocks.metrics.roc_auc = null;
    era5Fold.test.unseen_spatial_blocks.metrics.average_precision = null;
    era5Fold.test.unseen_spatial_blocks.metrics.recall = null;
    render(<ResearchResultsPanel research={noPositiveSnapshot} />);
    fireEvent.change(screen.getByLabelText("Periodo di test"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Area di valutazione"), { target: { value: "unseen_spatial_blocks" } });
    expect(screen.getByRole("status")).toHaveTextContent("Nessuna etichetta positiva");
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("shows an explicit unavailable state instead of fabricated research metrics", () => {
    render(<ResearchResultsPanel research={{ studies: [] }} />);
    expect(screen.getByText("Valutazioni Italia v2 non disponibili")).toBeInTheDocument();
  });
});
