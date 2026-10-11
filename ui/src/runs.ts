import { fmtMetric } from "./format";
import type { ModelRun } from "./types";

/** The two numbers the runs table shows per target: AUC and macro F1 for classes, R² and RMSE for a number. */
export type RunMetric = "primary" | "secondary";
export type Split = "valid" | "test";

const isNumber = (x: Record<string, number> | undefined) => !!x && ("r2" in x || "rmse" in x);

export function metricFor(r: ModelRun, split: Split, target: string, m: RunMetric): number | undefined {
  const x = r.metrics?.[split]?.[target];
  if (!x) return undefined;
  const v = isNumber(x) ? (m === "primary" ? x.r2 : x.rmse) : m === "primary" ? x.roc_auc ?? x.roc_auc_ovr_macro : x.f1_macro;
  return typeof v === "number" ? v : undefined;
}

/** RMSE is the one metric here where lower is better. */
const lowerIsBetter = (r: ModelRun, target: string, m: RunMetric) => m === "secondary" && isNumber(r.metrics?.valid?.[target]);

/**
 * Which run to star per target and metric (keys `target:metric`, values run ids), among runs of one export.
 * Runs are ranked on the validation split, so picking the best never peeks at the test split the table shows.
 * A star needs at least two runs with a validation score.
 */
export function bestRuns(runs: ModelRun[], targets: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const t of targets) {
    for (const m of ["primary", "secondary"] as const) {
      const scored = runs.flatMap((r) => {
        const v = metricFor(r, "valid", t, m);
        return v === undefined ? [] : [{ id: r.id, v, low: lowerIsBetter(r, t, m) }];
      });
      if (scored.length > 1) out[`${t}:${m}`] = scored.reduce((a, b) => ((b.low ? b.v < a.v : b.v > a.v) ? b : a)).id;
    }
  }
  return out;
}

/** Column headings for a target's two numbers. */
export function columnLabels(kind: string | undefined): [string, string] {
  return kind === "regression" ? ["R²", "RMSE"] : ["AUC", "F1 (macro)"];
}

/** The export the Train panel works on: the chosen one while it's finished, else the newest finished one. */
export function pickExport<E extends { id: number }>(done: E[], chosen: number | null): E | null {
  return done.find((e) => e.id === chosen) ?? done.reduce<E | null>((a, b) => (a && a.id > b.id ? a : b), null);
}

/** The ★'s tooltip. The star is always picked on the validation split, so on the test split it can sit on a lower
 * number than another run's; the tooltip says why. */
export function starTitle(r: ModelRun, split: Split, target: string, m: RunMetric): string {
  if (split === "valid") return "Best on the validation split";
  return `Best on the validation split (${fmtMetric(metricFor(r, "valid", target, m))}). The ★ is picked there so the `
    + "test split stays an honest check; switch to Validation to see the scores it was picked on.";
}
