import { describe, expect, it } from "vitest";
import { bestRuns, metricFor, pickExport } from "./runs";
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
