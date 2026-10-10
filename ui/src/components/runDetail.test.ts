import { describe, expect, it } from "vitest";
import { fmtMetric } from "../format";
import { cellShade } from "./RunDetail";

describe("fmtMetric", () => {
  it("always shows three decimals", () => {
    expect(fmtMetric(0.8579)).toBe("0.858");
    expect(fmtMetric(0.9)).toBe("0.900");
    expect(fmtMetric(1)).toBe("1.000");
    expect(fmtMetric(undefined)).toBe("—");
    expect(fmtMetric(NaN)).toBe("—");
  });
});

describe("cellShade", () => {
  it("shades the diagonal in the accent colour and errors in the error colour", () => {
    expect(cellShade(0.79, true).background).toContain("--series-1) 79%");
    expect(cellShade(0.2, false).background).toContain("--bad) 16%");
    expect(cellShade(0, false).background).toContain("--bad) 0%");
  });
  it("uses white text only on a strong diagonal", () => {
    expect(cellShade(0.8, true).color).toBe("#ffffff");
    expect(cellShade(0.8, false).color).toBe("var(--text-primary)");
    expect(cellShade(0.3, true).color).toBe("var(--text-primary)");
  });
});
