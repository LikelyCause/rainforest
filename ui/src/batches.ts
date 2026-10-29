import { fmtInt, fmtMetric } from "./format";
import { settingProblem, settingText } from "./settings";
import { parseSweep } from "./sweep";
import type {
  Batch,
  BatchDetail,
  BatchKind,
  BatchRequest,
  BatchRow,
  Calibration,
  ClassCount,
  Device,
  LeaderboardEntry,
  ModelSetting,
  Range,
  SettingValue,
  Skip,
  Space,
} from "./types";

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

/** The classes a one-vs-rest batch trains and, for every other class, why not. The batch plan's `reason` is the
 * server's rule; only unticking, which needs instant feedback, is decided here. */
export function classPlan(classes: ClassCount[], ticked: string[] | null): { trained: string[]; skipped: Skip[] } {
  const trained: string[] = [];
  const skipped: Skip[] = [];
  for (const c of classes) {
    const reason = ticked && !ticked.includes(c.class) ? "not selected" : c.reason;
    if (reason) skipped.push({ class: c.class, reason });
    else trained.push(c.class);
  }
  return { trained, skipped };
}

export const KIND_LABEL: Record<BatchKind, string> = { one_vs_rest: "Per class", search: "Search" };

/** The tag on a run of a batch: its kind and number ("Search #3"), with the batch's name for a tooltip. Until the
 * experiment's batches have loaded, only the number. */
export function batchTag(
  batches: Pick<Batch, "id" | "name" | "kind">[] | undefined,
  id: number,
): { label: string; name: string } {
  const b = batches?.find((x) => x.id === id);
  return b ? { label: `${KIND_LABEL[b.kind]} #${id}`, name: b.name } : { label: `Batch #${id}`, name: `Batch #${id}` };
}

/** The dialog's choices as POST /api/exports/{id}/batches takes them. A one-vs-rest batch or search carries its
 * target and balance; `ticked` (null: every class) is a one-vs-rest batch's classes, `searchClasses` a search's. */
export function batchRequest(f: {
  kind: BatchKind;
  name: string;
  model: string;
  device: Device;
  params: Record<string, SettingValue>;
  seed: number;
  cv: number;
  cvFolds: number;
  ovr: boolean;
  target: string | null;
  balance: boolean;
  ticked: string[] | null;
  space?: Space;
  sampler?: { kind: "grid" | "random"; n: number; seed: number };
  ranking?: string;
  searchClasses?: string[];
}): BatchRequest {
  return {
    kind: f.kind,
    name: f.name.trim() || null,
    model: f.model,
    device: f.device,
    params: f.params,
    seed: f.seed,
    cv: f.cv,
    cv_folds: f.cvFolds,
    ...(f.ovr ? { target: f.target, balance: f.balance } : {}),
    // Unticked classes are left out, so the batch lists them as not selected.
    ...(f.kind === "one_vs_rest" ? { classes: f.ticked } : {}),
    ...(f.kind === "search"
      ? {
          space: f.space ?? {},
          sampler: f.sampler?.kind === "random" ? { ...f.sampler } : { kind: "grid" as const },
          ranking: f.ranking,
          ...(f.ovr ? { search_classes: f.searchClasses ?? [] } : {}),
        }
      : {}),
  };
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

// ---- the batch page ------------------------------------------------------------------------------------------------

type Split = "valid" | "test";
export type OvrSortKey =
  "class" | `positives.${Split | "train"}` | `${Split}.${"pr_auc" | "roc_auc" | "f1"}` | "threshold" | "weak";

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

function sortValue(r: BatchRow, key: OvrSortKey): number | string | null {
  if (key === "class") return r.positive_class;
  if (key === "threshold") return r.threshold ?? null;
  if (key === "weak") return r.weak == null ? null : Number(r.weak);
  const [group, field] = key.split(".");
  if (group === "positives") return r.positives?.[field as "train"] ?? null;
  return r[group as Split]?.[field as "pr_auc"] ?? null;
}

/** A one-vs-rest batch's rows sorted on a column; rows without a value (still waiting, failed) stay last either way,
 * and ties keep the batch's order. Without a key, the batch's order. */
export function sortOvrRows(rows: BatchRow[], key: OvrSortKey | null, dir: "asc" | "desc"): BatchRow[] {
  if (!key) return [...rows];
  const sign = dir === "asc" ? 1 : -1;
  return rows
    .map((r, i) => ({ r, i, v: sortValue(r, key) }))
    .sort((a, b) => {
      if (a.v === null || b.v === null) return a.v === b.v ? a.i - b.i : a.v === null ? 1 : -1;
      const c = typeof a.v === "string" ? collator.compare(a.v, String(b.v)) : a.v - (b.v as number);
      return c ? sign * c : a.i - b.i;
    })
    .map((x) => x.r);
}

/** Why the server flagged a row weak, in the limits it used. Only its flag decides; the numbers only explain it. */
export function weakReasons(r: BatchRow, rule: BatchDetail["weak_rule"]): string[] {
  if (r.weak !== true) return [];
  const pr = r.valid?.pr_auc;
  const positives = r.positives?.test;
  const out: string[] = [];
  if (pr != null && pr < rule.pr_auc)
    out.push(`Validation PR AUC ${fmtMetric(pr)} is below ${rule.pr_auc} (OVR_WEAK_PR_AUC)`);
  if (positives != null && positives < rule.test_positives)
    out.push(`${fmtInt(positives)} positives in test, fewer than ${rule.test_positives}`);
  return out.length
    ? out
    : [`Validation PR AUC below ${rule.pr_auc} or fewer than ${rule.test_positives} positives in test`];
}

/** A one-vs-rest run's threshold in words. A note means no threshold had a positive F1, so 0.5 stands in. */
export const describeThreshold = (threshold: number, note: string | null | undefined) =>
  note ? `${threshold.toFixed(2)} (fallback)` : `${threshold.toFixed(2)}, the highest F1 on validation`;

/** The Brier pair worth showing: the calibrator was fit on validation, so its pair there is in-sample and only the
 * test pair is honest. Validation only when the export has no test rows. */
export function brierPair(c: Calibration | null | undefined): { split: Split; before: number; after: number } | null {
  if (!c) return null;
  return c.test
    ? { split: "test", before: c.test.brier_before, after: c.test.brier_after }
    : { split: "valid", before: c.brier_before, after: c.brier_after };
}

/** The Train panel's values for a one-vs-rest batch with a search row's settings: the batch's own settings, the
 * row's searched values over them, and its seed and chain folds. */
export function oneVsRestValues(
  spec: Pick<Batch["spec"], "params" | "seed" | "cv">,
  settings: Record<string, SettingValue>,
): Record<string, SettingValue> {
  return { ...spec.params, ...settings, seed: spec.seed, cv: spec.cv };
}

const labelOf = (catalog: ModelSetting[] | null | undefined, name: string) =>
  catalog?.find((s) => s.name === name)?.label ?? name;

/** A search's space, a line per setting: its values, or its range and how it was drawn. */
export function describeSpace(space: Space, catalog: ModelSetting[] | null | undefined): string[] {
  return Object.entries(space).map(([name, v]) => {
    const s = catalog?.find((x) => x.name === name);
    if (!isRange(v)) return `${labelOf(catalog, name)} ${v.map((x) => settingText(s, x)).join(", ")}`;
    const how = [v.log ? "log scale" : null, v.integer ? "whole numbers" : null].filter(Boolean);
    return `${labelOf(catalog, name)} from ${settingText(s, v.low)} to ${settingText(s, v.high)}${how.length ? `, ${how.join(", ")}` : ""}`;
  });
}

export function describeSampler(sampler: Batch["spec"]["sampler"], space: Space): string {
  if (!sampler) return "";
  if (sampler.kind === "random") return `Random · ${fmtInt(sampler.n)} draws · seed ${sampler.seed}`;
  const n = Object.values(space).reduce((k, v) => k * (isRange(v) ? 1 : v.length), 1);
  return `Grid · ${fmtInt(n)} combination${n === 1 ? "" : "s"}`;
}

/** The searched settings whose values differ between combinations, which are the ones worth plotting. */
export function variedSettings(entries: Pick<LeaderboardEntry, "settings">[], order: string[]): string[] {
  return order.filter((k) => new Set(entries.map((e) => JSON.stringify(e.settings[k] ?? null))).size > 1);
}

export interface ChartAxis {
  name: string;
  label: string;
  numeric: boolean;
  log: boolean;
  /** A non-numeric axis's values, as shown, in order. */
  categories: string[];
}
export interface ChartPoint {
  x: number | string;
  y: number | string | null;
  value: number;
  rank: number;
  settings: Record<string, SettingValue>;
}

function chartAxis(
  name: string,
  entries: Pick<LeaderboardEntry, "settings">[],
  space: Space,
  catalog: ModelSetting[] | null | undefined,
): ChartAxis {
  const s = catalog?.find((x) => x.name === name);
  const values = entries.map((e) => e.settings[name]);
  const axis = space[name];
  // A "no limit" value is a word, not a point on a number line.
  const numeric =
    values.every((v) => typeof v === "number" && Number.isFinite(v)) &&
    !(s?.kind === "int_or_none" && values.includes(s.none_value as SettingValue));
  const logScale = axis && isRange(axis) ? axis.log : (s?.log ?? false);
  const log = numeric && logScale && values.every((v) => (v as number) > 0);
  let categories: string[] = [];
  if (!numeric) {
    const seen = [...new Set([...(axis && !isRange(axis) ? axis : []), ...values])];
    const options = s?.options?.map((o) => o.value) ?? [];
    if (options.length) seen.sort((a, b) => options.indexOf(a) - options.indexOf(b));
    categories = [...new Set(seen.map((v) => settingText(s, v)))];
  }
  return { name, label: labelOf(catalog, name), numeric, log, categories };
}

/** A search's ranked combinations as points of a 1-D (metric by one setting) or 2-D (two settings, coloured by the
 * metric) chart. Numbers sit on a value axis, logarithmic when the setting was searched on one; anything else is a
 * category in the catalogue's order. */
export function searchChartData(
  entries: LeaderboardEntry[],
  x: string,
  y: string | null,
  space: Space,
  catalog: ModelSetting[] | null | undefined,
): { x: ChartAxis; y: ChartAxis | null; points: ChartPoint[] } {
  const ax = chartAxis(x, entries, space, catalog);
  const ay = y ? chartAxis(y, entries, space, catalog) : null;
  const at = (a: ChartAxis, v: SettingValue) =>
    a.numeric
      ? (v as number)
      : settingText(
          catalog?.find((s) => s.name === a.name),
          v,
        );
  const points = entries.flatMap((e) =>
    e.value === null || e.rank === null
      ? []
      : [
          {
            x: at(ax, e.settings[x]),
            y: ay ? at(ay, e.settings[ay.name]) : null,
            value: e.value,
            rank: e.rank,
            settings: e.settings,
          },
        ],
  );
  return { x: ax, y: ay, points };
}
