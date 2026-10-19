import { describe, expect, it } from "vitest";
import { getMapCells, getCellDetails } from "@/services/map-service";
import { MOCK_MAP_CELLS } from "@/data/mock-map";

describe("map-service filters", () => {
  it("returns every sampled cell without filters", async () => {
    const result = await getMapCells();
    expect(result.total).toBe(MOCK_MAP_CELLS.length);
    expect(result.modelSlug).toBe("random-forest-italia-era5");
    expect(result.cells.every((cell) => cell.lat >= 35 && cell.lat <= 48 && cell.lon >= 6 && cell.lon <= 20)).toBe(true);
    expect(result.cells.length).toBe(331);
    expect(result.cells.every((cell) => cell.model === "Random Forest FIRMS clean3 + ERA5-Land")).toBe(true);
    expect(result.forecast.forecastStart).toBe("2026-07-16");
  });

  it("filters cells by qualitative risk level", async () => {
    const result = await getMapCells({ riskLevel: "bassa" });
    expect(result.cells.length).toBeGreaterThan(0);
    expect(result.cells.every((cell) => cell.riskLevel === "bassa")).toBe(true);
    expect(result.cells.length).toBeLessThan(331);
  });

  it("filters out cells whose last detection predates the requested date", async () => {
    const result = await getMapCells({ minLastDetectionDate: "2026-07-10" });
    expect(result.cells.every((cell) => cell.lastDetectionDate && cell.lastDetectionDate >= "2026-07-10")).toBe(true);
    expect(result.cells.length).toBeGreaterThan(0);
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
