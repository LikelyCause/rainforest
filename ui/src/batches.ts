import { settingProblem } from "./settings";
import { parseSweep } from "./sweep";
import type { Batch, ClassCount, ModelSetting, Range, SettingValue, Skip, Space } from "./types";

export type { ClassCount } from "./types";

/** One axis of the search as the dialog holds it, typed text and all, before it is read into the space. */
export interface AxisDraft {
  name: string;
  mode: "list" | "range";
  values: string[];
  low: string;
  high: string;
  log: boolean;
  integer: boolean;
}

/** The classes a one-vs-rest batch trains and, for every other class, why not: the API's `batches.plan_classes`,
 * so the preview names the same skips the batch will. `validRows` 0 means the validation split is empty. */
export function planClasses(
  classes: ClassCount[],
  validRows: number | null,
  wanted: string[] | null,
  minPos: number,
): { trained: string[]; skipped: Skip[] } {
  const trained: string[] = [];
  const skipped: Skip[] = [];
  for (const c of classes) {
    const reason =
      wanted !== null && !wanted.includes(c.class)
        ? "not selected"
        : c.train < minPos
          ? `fewer than ${minPos} positives in training`
          : validRows === 0
            ? "no validation rows"
            : c.valid === 0
              ? "no validation positives"
              : null;
    if (reason) skipped.push({ class: c.class, reason });
    else trained.push(c.class);
  }
  return { trained, skipped };
}

// Python's string order (by code point), not the locale's, so ties fall as the API's sort leaves them.
const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/** A one-vs-rest search's default classes: the most frequent, the median and the rarest that would train. */
export function searchClassesDefault(classes: ClassCount[], validRows: number | null, minPos: number): string[] {
  const { trained } = planClasses(classes, validRows, null, minPos);
  const train = new Map(classes.map((c) => [c.class, c.train]));
  const sorted = [...trained].sort((a, b) => train.get(b)! - train.get(a)! || byCodePoint(a, b));
  if (!sorted.length) return [];
  return [...new Set([sorted[0], sorted[Math.floor((sorted.length - 1) / 2)], sorted[sorted.length - 1]])];
}

const isNumberSetting = (s: ModelSetting) => s.kind === "int" || s.kind === "float" || s.kind === "int_or_none";

/** A typed range for a number setting, held to the API's range rules and to the setting's own limits. A whole-number
 * setting's range is always drawn as whole numbers. */
export function parseRange(
  s: ModelSetting,
  low: string,
  high: string,
  log: boolean,
  integer: boolean,
): { range: Range | null; error: string | null } {
  const fail = (error: string) => ({ range: null, error });
  if (!isNumberSetting(s)) return fail(`${s.label} takes a list of values, not a range`);
  const lo = Number(low),
    hi = Number(high);
  if (!low.trim() || !high.trim() || !Number.isFinite(lo) || !Number.isFinite(hi))
    return fail("Give a low and a high number");
  if (lo > hi) return fail("The low is above the high");
  if (log && lo <= 0) return fail("A log range needs a low above 0");
  const whole = integer || s.kind !== "float";
  if (whole && Math.ceil(lo) > Math.floor(hi)) return fail("The range holds no whole number");
  // Only the limits matter here; a whole-number setting's ends needn't be whole, the draws are.
  const limits = { ...s, kind: "float" as const };
  const end = ([name, v]: [string, number]) => {
    const p = settingProblem(limits, v);
    return p ? `${name} ${p}` : null;
  };
  const problem = end(["Low", lo]) ?? end(["High", hi]);
  return problem ? fail(problem) : { range: { low: lo, high: hi, log, integer: whole }, error: null };
}

/** The dialog's axes as the API's space, or the first problem, named by its setting. */
export function buildSpace(axes: AxisDraft[], catalog: ModelSetting[]): { space: Space; error: string | null } {
  const space: Space = {};
  if (!axes.length) return { space, error: "Choose a setting to search over" };
  for (const a of axes) {
    const s = catalog.find((x) => x.name === a.name);
    if (!s) return { space, error: `${a.name}: not a setting of this model` };
    if (a.mode === "range") {
      const { range, error } = parseRange(s, a.low, a.high, a.log, a.integer);
      if (error) return { space, error: `${s.label}: ${error}` };
      space[a.name] = range!;
    } else {
      const { values, error } = parseSweep(s, a.values.join(", "));
      if (error) return { space, error: `${s.label}: ${error}` };
      space[a.name] = values;
    }
  }
  return { space, error: null };
}

const isRange = (v: SettingValue[] | Range): v is Range => !Array.isArray(v);

/** How many distinct combinations the space holds: a list's values, a whole-number range's integers; any other range
 * is unbounded. A random search can't draw more distinct ones than this. */
export function distinctCombinations(space: Space): number {
  return Object.values(space).reduce<number>(
    (n, v) => n * (!isRange(v) ? v.length : v.integer ? Math.floor(v.high) - Math.ceil(v.low) + 1 : Infinity),
    1,
  );
}

/** How many runs the batch starts: a run per trained class, or per combination (times the classes of a one-vs-rest
 * search). The error is what the API would refuse the sampler for. */
export function runCount(p: {
  kind: "one_vs_rest" | "search";
  classes?: number;
  space?: Space;
  sampler?: { kind: "grid" | "random"; n?: number };
}): { runs: number; error: string | null } {
  if (p.kind === "one_vs_rest") return { runs: p.classes ?? 0, error: null };
  const space = p.space ?? {};
  const per = p.classes ?? 1;
  if (p.sampler?.kind === "random") {
    const n = p.sampler.n ?? 0;
    if (!Number.isInteger(n) || n < 1) return { runs: 0, error: "Draw at least one combination" };
    const most = distinctCombinations(space);
    if (n > most)
      return { runs: n * per, error: `The settings hold only ${most} distinct combinations; draw at most ${most}` };
    return { runs: n * per, error: null };
  }
  if (Object.values(space).some(isRange))
    return { runs: 0, error: "A grid takes lists of values; use random sampling for a range" };
  return { runs: Object.values(space).reduce((n, v) => n * (v as SettingValue[]).length, 1) * per, error: null };
}

const grouped = new Intl.NumberFormat("en-US");

/** Above the cap, the API's own refusal (routers/batches.py `_cap`), so the dialog and a 422 read the same. */
export function capProblem(runs: number, max: number): string | null {
  return runs > max
    ? `This batch would start ${grouped.format(runs)} runs; at most ${grouped.format(max)} may run in one (BATCH_MAX_RUNS)`
    : null;
}

const NUMBER_METRICS = ["r2", "rmse", "mae", "mape", "band_accuracy", "band_f1_macro"];
const BINARY_METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "specificity"];
const MULTICLASS_METRICS = ["f1_macro", "pr_auc_macro", "roc_auc_ovr_macro"];
const SHARED_CLASS_METRICS = [
  "f1_macro",
  "accuracy",
  "balanced_accuracy",
  "log_loss",
  "precision_macro",
  "recall_macro",
  "f1_weighted",
  "precision_weighted",
  "recall_weighted",
];

export const METRIC_NAME: Record<string, string> = {
  pr_auc: "PR AUC",
  roc_auc: "ROC AUC",
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
  specificity: "Specificity",
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  log_loss: "Log loss",
  f1_macro: "F1 (macro)",
  precision_macro: "Precision (macro)",
  recall_macro: "Recall (macro)",
  f1_weighted: "F1 (weighted)",
  precision_weighted: "Precision (weighted)",
  recall_weighted: "Recall (weighted)",
  pr_auc_macro: "PR AUC (macro)",
  roc_auc_ovr_macro: "ROC AUC (one-vs-rest, macro)",
  r2: "R²",
  rmse: "RMSE",
  mae: "MAE",
  mape: "MAPE",
  band_accuracy: "Band accuracy",
  band_f1_macro: "Band F1 (macro)",
};
export const LOWER_IS_BETTER = new Set(["log_loss", "rmse", "mae", "mape"]);

/** The validation metric a search ranks on by default: the API's `batches.default_ranking`. */
export function defaultRanking(kind: string | undefined): string {
  return kind === "multiclass" ? "f1_macro" : kind === "regression" ? "r2" : "pr_auc";
}

/** The metrics the API lets a search of this target kind rank on (`batches.ranking_problem`), the default first. */
export function rankingChoices(kind: string | undefined): string[] {
  if (kind === "regression") return NUMBER_METRICS;
  const own = kind === "multiclass" ? MULTICLASS_METRICS : BINARY_METRICS;
  return [...new Set([defaultRanking(kind), ...own, ...SHARED_CLASS_METRICS])];
}

// The settings that weight classes (the API's registry.CLASS_WEIGHTING); balance sets one itself.
const CLASS_WEIGHTING = ["class_weight", "scale_pos_weight", "is_unbalance"];

/** The class-weight settings that balance would contradict: changed from their defaults, or searched over. */
export function weightingClash(params: Record<string, SettingValue>, searched: string[]): string[] {
  return CLASS_WEIGHTING.filter((k) => k in params || searched.includes(k));
}

/** "12 of 40 done · 1 failed · 27 waiting", from a batch's runs by status, and the ended share for its bar. */
export function batchProgress(b: Pick<Batch, "counts">): { ended: number; total: number; text: string } {
  const c = b.counts;
  const total = c.pending + c.running + c.done + c.failed + c.cancelled;
  const n = (v: number) => grouped.format(v);
  const parts = [
    `${n(c.done)} of ${n(total)} done`,
    c.failed ? `${n(c.failed)} failed` : null,
    c.cancelled ? `${n(c.cancelled)} cancelled` : null,
    c.running ? `${n(c.running)} training` : null,
    c.pending ? `${n(c.pending)} waiting` : null,
  ];
  return { ended: c.done + c.failed + c.cancelled, total, text: parts.filter(Boolean).join(" · ") };
}
