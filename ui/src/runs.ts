import type { ModelRun } from "./types";

export type RunMetric = "auc" | "f1";
export type Split = "valid" | "test";

export function metricFor(r: ModelRun, split: Split, target: string, m: RunMetric): number | undefined {
  const x = r.metrics?.[split]?.[target];
  if (!x) return undefined;
  const v = m === "auc" ? x.roc_auc ?? x.roc_auc_ovr_macro : x.f1_macro;
  return typeof v === "number" ? v : undefined;
}

/**
 * Which run to star per target and metric (keys `target:metric`, values run ids), among runs of one export.
 * Runs are ranked on the validation split, so picking the best never peeks at the test split the table shows.
 * A star needs at least two runs with a validation score.
 */
export function bestRuns(runs: ModelRun[], targets: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const t of targets) {
    for (const m of ["auc", "f1"] as const) {
      const scored = runs.flatMap((r) => {
        const v = metricFor(r, "valid", t, m);
        return v === undefined ? [] : [{ id: r.id, v }];
      });
      if (scored.length > 1) out[`${t}:${m}`] = scored.reduce((a, b) => (b.v > a.v ? b : a)).id;
    }
  }
  return out;
}
