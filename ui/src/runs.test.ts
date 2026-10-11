import { describe, expect, it } from "vitest";
import { bestRuns, metricFor, pickExport, runColumns, spreadFor, starTitle, whyFailed } from "./runs";
import type { ModelRun } from "./types";

const run = (id: number, valid: Record<string, Record<string, number>>, test: Record<string, Record<string, number>> = valid) =>
  ({ id, metrics: { valid, test } }) as unknown as ModelRun;

describe("metricFor", () => {
  it("reads binary AUC, multiclass OvR AUC and macro F1", () => {
    const r = run(1, { a: { roc_auc: 0.8, f1_macro: 0.6 }, b: { roc_auc_ovr_macro: 0.7 } });
    expect(metricFor(r, "valid", "a", "primary")).toBe(0.8);
    expect(metricFor(r, "valid", "b", "primary")).toBe(0.7);
    expect(metricFor(r, "test", "a", "secondary")).toBe(0.6);
    expect(metricFor(r, "test", "missing", "primary")).toBeUndefined();
  });
});

describe("PR AUC", () => {
  it("is a column for class targets only, read as pr_auc or its macro and starred like the others", () => {
    expect(runColumns("binary").map((c) => c.label)).toEqual(["AUC", "PR AUC", "F1 (macro)"]);
    expect(runColumns("regression").map((c) => c.metric)).toEqual(["primary", "secondary"]);
    expect(metricFor(run(1, { a: { roc_auc: 0.6, pr_auc: 0.18 } }), "test", "a", "pr")).toBe(0.18);
    expect(metricFor(run(1, { b: { roc_auc_ovr_macro: 0.6, pr_auc_macro: 0.3 } }), "test", "b", "pr")).toBe(0.3);
    expect(metricFor(run(1, { p: { r2: 0.8, rmse: 3 } }), "test", "p", "pr")).toBeUndefined();
    expect(bestRuns([run(1, { a: { pr_auc: 0.1 } }), run(2, { a: { pr_auc: 0.2 } })], ["a"])).toEqual({ "a:pr": 2 });
  });
});

describe("bestRuns", () => {
  it("stars the best validation score, not the best test score", () => {
    const runs = [
      run(1, { a: { roc_auc: 0.90 } }, { a: { roc_auc: 0.80 } }),
      run(2, { a: { roc_auc: 0.85 } }, { a: { roc_auc: 0.95 } }),
    ];
    expect(bestRuns(runs, ["a"])).toEqual({ "a:primary": 1 });
  });

  it("needs two scored runs and skips runs still training", () => {
    const pending = { id: 3, metrics: null } as unknown as ModelRun;
    expect(bestRuns([run(1, { a: { roc_auc: 0.9 } }), pending], ["a"])).toEqual({});
    expect(bestRuns([run(1, { a: { f1_macro: 0.5 } }), run(2, { a: { f1_macro: 0.7 } }), pending], ["a"]))
      .toEqual({ "a:secondary": 2 });
  });
});

describe("number targets", () => {
  const reg = (id: number, r2: number, rmse: number) => run(id, { p: { r2, rmse } });
  it("reads R² and RMSE", () => {
    expect(metricFor(reg(1, 0.8, 3), "test", "p", "primary")).toBe(0.8);
    expect(metricFor(reg(1, 0.8, 3), "test", "p", "secondary")).toBe(3);
  });
  it("stars the highest R² and the lowest RMSE", () => {
    const best = bestRuns([reg(1, 0.8, 3), reg(2, 0.7, 2)], ["p"]);
    expect(best["p:primary"]).toBe(1);
    expect(best["p:secondary"]).toBe(2);
  });
});

describe("pickExport", () => {
  it("keeps the chosen export, else takes the newest", () => {
    const xs = [{ id: 3 }, { id: 9 }, { id: 5 }];
    expect(pickExport(xs, 5)?.id).toBe(5);
    expect(pickExport(xs, null)?.id).toBe(9);
    expect(pickExport(xs, 42)?.id).toBe(9);
    expect(pickExport([], null)).toBeNull();
  });
});

describe("starTitle", () => {
  const r = run(1, { a: { roc_auc: 0.9 } }, { a: { roc_auc: 0.8 } });
  it("on the validation split, the star is simply the best number", () => {
    expect(starTitle(r, "valid", "a", "primary")).toBe("Best on the validation split");
  });
  it("on the test split, it says where the star was picked and the score there", () => {
    expect(starTitle(r, "test", "a", "primary")).toMatch(/validation split \(0\.900\).*Validation/);
  });
});

describe("whyFailed", () => {
  it("explains a restart", () => {
    expect(whyFailed("Interrupted by an API restart.")).toBe(
      "The API restarted while it was training (an update or a restart of the server), which stops running jobs.");
  });
  it("explains running out of memory", () => {
    expect(whyFailed("MemoryError: Unable to allocate 3.1 GiB")).toMatch(/memory/i);
  });
  it("passes a run's own message through, first line only", () => {
    expect(whyFailed("ValueError: Target 'y' has a single value in the training split; there is nothing to predict.\n\nTraceback ..."))
      .toBe("Target 'y' has a single value in the training split; there is nothing to predict.");
    expect(whyFailed(null)).toBe("The run failed without a message.");
  });
});

describe("cross-validation", () => {
  const cvRun = {
    id: 9,
    metrics: { valid: { a: { roc_auc: 0.8 }, n: { r2: 0.7 } }, cv: { a: { roc_auc: 0.82, f1_macro: 0.6 }, n: { r2: 0.71, rmse: 3 } } },
    report: { cross_validation: { folds: 3, stratified_on: "a", note: null, targets: {
      a: { roc_auc: { mean: 0.82, std: 0.02, folds: [0.8, 0.82, 0.84] }, f1_macro: { mean: 0.6, std: 0.05, folds: [0.55, null, 0.65] } },
      n: { r2: { mean: 0.71, std: 0.01, folds: [0.7, 0.71, 0.72] }, rmse: { mean: 3, std: 0.2, folds: [2.8, 3, 3.2] } },
    } } },
  } as unknown as ModelRun;
  it("reads the CV mean like a split, and its spread for the same metric", () => {
    expect(metricFor(cvRun, "cv", "a", "primary")).toBe(0.82);
    expect(spreadFor(cvRun, "a", "primary")).toBe(0.02);
    expect(spreadFor(cvRun, "a", "secondary")).toBe(0.05);
    expect(spreadFor(cvRun, "n", "secondary")).toBe(0.2);
    expect(spreadFor(run(1, { a: { roc_auc: 0.9 } }), "a", "primary")).toBeUndefined();
  });
  it("says the star is still picked on validation, without the test-split sentence", () => {
    const t = starTitle(cvRun, "cv", "a", "primary");
    expect(t).toContain("validation split (0.800)");
    expect(t).not.toContain("test split");
  });
});
