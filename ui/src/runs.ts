import { fmtMetric } from "./format";
import type { ModelOption, ModelRun } from "./types";

type Named = Pick<ModelOption, "label" | "description" | "regressor_label" | "regressor_description" | "mixed_label">;

/** A model's name for targets of these kinds, as the API's `registry.label_for`: the linear model is "Ridge
 * regression" when every target is a number and "Logistic and ridge regression" for a chain of both. */
export function modelLabel(m: Named, kinds: (string | undefined)[]): string {
  if (!m.regressor_label || !kinds.includes("regression")) return m.label;
  if (kinds.every((k) => k === "regression")) return m.regressor_label;
  return m.mixed_label ?? `${m.label} and ${m.regressor_label.toLowerCase()}`;
}

/** The model's description for targets of these kinds: its regressor's when every target is a number. */
export function modelDescription(m: Named, kinds: (string | undefined)[]): string {
  return kinds.length && kinds.every((k) => k === "regression") && m.regressor_description ? m.regressor_description : m.description;
}

/** A run's model name: the one it was trained under, else the catalog's for the targets it trained on. */
export function runLabel(r: ModelRun, option: Named | undefined): string {
  if (r.report?.model_label) return r.report.model_label;
  return option ? modelLabel(option, Object.values(r.report?.target_kinds ?? {})) : r.model;
}

/** The two numbers the runs table shows per target: AUC and macro F1 for classes, R² and RMSE for a number. */
export type RunMetric = "primary" | "pr" | "secondary";
export type Split = "valid" | "test" | "cv";

const isNumber = (x: Record<string, unknown> | undefined) => !!x && ("r2" in x || "rmse" in x);

/** Which metric a column shows for a target's metrics: AUC (binary, else one-vs-rest) and macro F1, or R² and RMSE. */
function keyFor(x: Record<string, unknown>, m: RunMetric): string {
  if (isNumber(x)) return m === "primary" ? "r2" : m === "pr" ? "" : "rmse";
  if (m === "pr") return "pr_auc" in x ? "pr_auc" : "pr_auc_macro";
  return m === "primary" ? ("roc_auc" in x ? "roc_auc" : "roc_auc_ovr_macro") : "f1_macro";
}

export function metricFor(r: ModelRun, split: Split, target: string, m: RunMetric): number | undefined {
  const x = r.metrics?.[split]?.[target];
  if (!x) return undefined;
  const v = x[keyFor(x, m)];
  return typeof v === "number" ? v : undefined;
}

/** The spread (standard deviation over folds) of the cross-validation score `metricFor(r, "cv", …)` shows. */
export function spreadFor(r: ModelRun, target: string, m: RunMetric): number | undefined {
  const x = r.report?.cross_validation?.targets?.[target];
  return x ? x[keyFor(x, m)]?.std : undefined;
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
    for (const m of ["primary", "pr", "secondary"] as const) {
      const scored = runs.flatMap((r) => {
        const v = metricFor(r, "valid", t, m);
        return v === undefined ? [] : [{ id: r.id, v, low: lowerIsBetter(r, t, m) }];
      });
      // A tie goes to the earliest run, as the API's own "best on validation" does, so both name the same run.
      if (scored.length > 1) out[`${t}:${m}`] = scored.reduce((a, b) => ((b.low ? b.v < a.v : b.v > a.v) || (b.v === a.v && b.id < a.id) ? b : a)).id;
    }
  }
  return out;
}

/** A target's columns in the runs table; PR AUC is the score to compare when the positive class is rare. */
export function runColumns(kind: string | undefined): { metric: RunMetric; label: string }[] {
  return kind === "regression" ? [{ metric: "primary", label: "R²" }, { metric: "secondary", label: "RMSE" }]
    : [{ metric: "primary", label: "AUC" }, { metric: "pr", label: "PR AUC" }, { metric: "secondary", label: "F1 (macro)" }];
}

/** The export the Train panel works on: the chosen one while it's finished, else the newest finished one. */
export function pickExport<E extends { id: number }>(done: E[], chosen: number | null): E | null {
  return done.find((e) => e.id === chosen) ?? done.reduce<E | null>((a, b) => (a && a.id > b.id ? a : b), null);
}

/** The ★'s tooltip. The star is always picked on the validation split, so on the test split it can sit on a lower
 * number than another run's; the tooltip says why. */
export function starTitle(r: ModelRun, split: Split, target: string, m: RunMetric): string {
  if (split === "valid") return "Best on the validation split";
  if (split === "cv") return `Best on the validation split (${fmtMetric(metricFor(r, "valid", target, m))}); the ★ is picked there whichever scores are shown.`;
  return `Best on the validation split (${fmtMetric(metricFor(r, "valid", target, m))}). The ★ is picked there so the `
    + "test split stays an honest check; switch to Validation to see the scores it was picked on.";
}

/** Why a run failed, in a sentence: the API's own message (its first line, without the exception type), or a plain
 * explanation for the failures that don't come from the run's settings. */
export function whyFailed(error: string | null | undefined): string {
  if (!error) return "The run failed without a message.";
  if (error.startsWith("Interrupted by an API restart")) {
    return "The API restarted while it was training (an update or a restart of the server), which stops running jobs.";
  }
  if (/MemoryError|Unable to allocate|Killed/.test(error)) {
    return "The server ran out of memory training it. Try fewer trees or leaves, fewer chain folds, cross-validation off, or a smaller export.";
  }
  return error.split("\n")[0].replace(/^[A-Za-z]*(Error|Exception): /, "");
}

export const isActive = (s: string) => s === "pending" || s === "running";

/** The toast for a batch of runs that has finished: how many finished or failed, and the best validation score on
 * the first target (the score the ★ is picked on). */
export function batchSummary(runs: ModelRun[], target: string | undefined, kind: string | undefined): string {
  const done = runs.filter((r) => r.status === "done");
  const failed = runs.length - done.length;
  const head = done.length
    ? `${done.length} run${done.length === 1 ? "" : "s"} finished${failed ? `, ${failed} failed` : ""}`
    : `${failed} run${failed === 1 ? "" : "s"} failed`;
  if (!target || !done.length) return head;
  const scores = done.map((r) => metricFor(r, "valid", target, "primary")).filter((v): v is number => v !== undefined);
  if (!scores.length) return head;
  return `${head} · best ${runColumns(kind)[0].label} ${fmtMetric(Math.max(...scores))}`;
}
