import { describe, expect, it } from "vitest";
import { formatCoordinate, formatDate, formatFrp, formatNumber, formatPercent } from "@/lib/formatters";

describe("formatters", () => {
  it("formats large numbers with Italian grouping", () => {
    expect(formatNumber(16255053)).toBe("16.255.053");
  });

  it("returns a placeholder for missing numbers", () => {
    expect(formatNumber(undefined)).toBe("—");
    expect(formatNumber(null)).toBe("—");
  });

  it("formats a probability as an Italian percentage", () => {
    expect(formatPercent(0.731)).toBe("73,1%");
    expect(formatPercent(0.5, { digits: 0 })).toBe("50%");
  });

  it("formats dates in long Italian form", () => {
    expect(formatDate("2026-07-17")).toContain("2026");
    expect(formatDate("2026-07-17")).toContain("luglio");
  });

  it("formats coordinates with hemisphere labels", () => {
    expect(formatCoordinate(-15.9, 25.6)).toBe("15.90°S, 25.60°E");
    expect(formatCoordinate(34.2, -118.5)).toBe("34.20°N, 118.50°O");
  });

  it("formats FRP values with the MW unit", () => {
    expect(formatFrp(2791)).toBe("2.791 MW");
  });
});
