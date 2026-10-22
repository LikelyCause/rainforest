import { describe, expect, it } from "vitest";
import { fmtMetric } from "../format";
import { cellShade, classTabs, headlineTiles, regressionTabs, regressionTiles } from "./RunDetail";
import type { RegressionEvaluation } from "../types";

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

describe("headlineTiles", () => {
  const m = {
    accuracy: 0.8,
    precision: 0.7,
    recall: 0.6,
    f1: 0.65,
    specificity: 0.9,
    precision_macro: 0.5,
    recall_macro: 0.4,
    f1_macro: 0.45,
  };
  const value = (rows: ReturnType<typeof headlineTiles>, label: string) => rows.find((r) => r[0] === label)?.[1];
  it("shows the positive class for binary targets", () => {
    const rows = headlineTiles(m, true);
    expect(rows.map((r) => r[0])).toEqual([
      "Accuracy",
      "Bal. acc.",
      "ROC AUC",
      "PR AUC",
      "Log loss",
      "Precision",
      "Recall",
      "F1",
      "Specificity",
    ]);
    expect([value(rows, "Precision"), value(rows, "Recall"), value(rows, "F1"), value(rows, "Specificity")]).toEqual([
      0.7, 0.6, 0.65, 0.9,
    ]);
  });
  it("shows macro averages for multiclass targets, and no specificity", () => {
    const rows = headlineTiles(m, false);
    expect([value(rows, "Precision"), value(rows, "Recall"), value(rows, "F1"), value(rows, "Specificity")]).toEqual([
      0.5,
      0.4,
      0.45,
      undefined,
    ]);
  });
});

describe("regressionTiles", () => {
  it("leads with R² and explains a missing MAPE", () => {
    const rows = regressionTiles({ rmse: 2, mae: 1, r2: 0.9 });
    expect(rows.map(([l]) => l)).toEqual(["R²", "RMSE", "MAE", "MAPE"]);
    expect(rows[3][1]).toBeUndefined();
    expect(rows[3][2]).toMatch(/0/);
  });
});

describe("regressionTabs", () => {
  it("offers ranges only when the run scored them", () => {
    const ev: RegressionEvaluation = {
      kind: "regression",
      n: 1,
      scatter: { actual: [], predicted: [] },
      residuals: { edges: [], counts: [] },
    };
    expect(regressionTabs(ev).map((t) => t.key)).toEqual(["scatter", "errors"]);
    expect(
      regressionTabs({
        ...ev,
        bands: {
          labels: ["< 1", "≥ 1"],
          edges: [1],
          confusion_matrix: [
            [1, 0],
            [0, 1],
          ],
          per_class: [],
        },
      }).map((t) => t.key),
    ).toEqual(["scatter", "errors", "bands"]);
    expect(regressionTabs(ev, true).map((t) => t.key)).toEqual(["scatter", "errors", "features"]);
  });
});

describe("classTabs", () => {
  it("offers the matrix, the ROC curves and the per-class table", () => {
    expect(classTabs().map((t) => t.key)).toEqual(["matrix", "roc", "classes"]);
  });
  it("adds Threshold and Top features when the run has them", () => {
    expect(classTabs(true, true).map((t) => t.key)).toEqual(["matrix", "roc", "classes", "threshold", "features"]);
  });
});
