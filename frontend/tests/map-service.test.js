import { describe, expect, it } from "vitest";
import { getMapCells, getCellDetails } from "@/services/map-service";
import { MOCK_MAP_CELLS } from "@/data/mock-map";

describe("map-service filters", () => {
  it("returns every sampled cell without filters", async () => {
    const result = await getMapCells();
    expect(result.total).toBe(MOCK_MAP_CELLS.length);
  });

  it("filters cells by qualitative risk level", async () => {
    const result = await getMapCells({ riskLevel: "molto-elevata" });
    expect(result.cells.length).toBeGreaterThan(0);
    expect(result.cells.every((cell) => cell.riskLevel === "molto-elevata")).toBe(true);
  });

  it("filters out cells whose last detection predates the requested date", async () => {
    const result = await getMapCells({ minLastDetectionDate: "2026-07-15" });
    expect(result.cells.every((cell) => cell.lastDetectionDate >= "2026-07-15")).toBe(true);
  });

  it("resolves cell details for a known id and null for an unknown one", async () => {
    const knownId = MOCK_MAP_CELLS[0].id;
    const details = await getCellDetails(knownId);
    expect(details).not.toBeNull();
    expect(details.id).toBe(knownId);

    const missing = await getCellDetails("does-not-exist");
    expect(missing).toBeNull();
  });
});
