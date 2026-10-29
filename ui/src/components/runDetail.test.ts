import { describe, expect, it } from "vitest";
import { fmtMetric } from "../format";
import {
  bestF1Note,
  cellShade,
  chosenThreshold,
  classTabs,
  headlineTiles,
  regressionTabs,
  regressionTiles,
  scoresNote,
} from "./RunDetail";
import type { F1Threshold, ModelRun, RegressionEvaluation } from "../types";

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
  it("says a binary target's precision, recall and F1 are at 0.5", () => {
    const hint = (rows: ReturnType<typeof headlineTiles>, label: string) => rows.find((r) => r[0] === label)?.[2];
    for (const label of ["Precision", "Recall", "F1"]) {
      expect(hint(headlineTiles(m, true), label)).toBe("Positive class, at 0.5");
      expect(hint(headlineTiles(m, false), label)).toBe("Macro average over classes");
    }
  });
});

describe("chosenThreshold", () => {
  const f1: F1Threshold = { threshold: 0.31, valid_f1: 0.62, test_f1: 0.58, note: null };
  const run = (config: ModelRun["config"], report: Partial<ModelRun["report"]> | null) =>
    ({ config, report }) as Pick<ModelRun, "config" | "report">;
  it("is a one-vs-rest run's stored F1 choice", () => {
    expect(chosenThreshold(run({ positive_class: "V01" }, { f1_threshold: f1 }))).toEqual(f1);
  });
  it("is null for any other run, or one without a stored choice", () => {
    expect(chosenThreshold(run({}, { f1_threshold: f1 }))).toBeNull();
    expect(chosenThreshold(run({ positive_class: "V01" }, {}))).toBeNull();
    expect(chosenThreshold(run({ positive_class: "V01" }, null))).toBeNull();
  });
});

describe("scoresNote", () => {
  it("says class targets use the most likely class when the run has no chosen threshold", () => {
    expect(scoresNote(null)).toBe("Class targets use the most likely class (0.5 threshold for binary targets)");
  });
  it("says the scores are at 0.5, and gives the run's own threshold and its F1 on each split", () => {
    expect(scoresNote({ threshold: 0.31, valid_f1: 0.6234, test_f1: 0.58, note: null })).toBe(
      "Precision, recall and F1 here are at 0.5. This run's threshold is 0.31, the highest F1 on validation: " +
        "F1 0.623 on validation, 0.580 on test.",
    );
  });
  it("leaves out the test F1 when the export has no test rows", () => {
    expect(scoresNote({ threshold: 0.31, valid_f1: 0.6234, test_f1: null, note: null })).toBe(
      "Precision, recall and F1 here are at 0.5. This run's threshold is 0.31, the highest F1 on validation: " +
        "F1 0.623 on validation.",
    );
  });
  it("calls the threshold a fallback and adds the note when no threshold had a positive F1", () => {
    const note = "No threshold gives a positive F1 on validation; 0.5 is used";
    expect(scoresNote({ threshold: 0.5, valid_f1: 0, test_f1: 0, note })).toBe(
      "Precision, recall and F1 here are at 0.5. This run's threshold is 0.50 (fallback): " +
        `F1 0.000 on validation, 0.000 on test. ${note}.`,
    );
  });
});

describe("bestF1Note", () => {
  const tail = " Scores here are for this split; pick a threshold on validation, then check it on test.";
  it("sets the best F1 against 0.5 when the run has no chosen threshold", () => {
    expect(bestF1Note({ threshold: 0.3, f1: 0.6 }, null)).toBe(
      `Best F1 is 0.600 at 0.3 (the scores above use 0.5).${tail}`,
    );
    expect(bestF1Note({ threshold: 0.5, f1: 0.6 }, null)).toBe(`Best F1 is 0.600 at 0.5.${tail}`);
  });
  it("sets it against the run's chosen threshold when it has one", () => {
    expect(bestF1Note({ threshold: 0.3, f1: 0.6 }, 0.28)).toBe(
      `Best F1 is 0.600 at 0.3 (the chosen threshold is 0.28).${tail}`,
    );
    expect(bestF1Note({ threshold: 0.28, f1: 0.6 }, 0.28)).toBe(
      `Best F1 is 0.600 at 0.28, the chosen threshold.${tail}`,
    );
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
