import type {
  BatchDetail,
  DeploymentField,
  DeploymentModel,
  DeploymentRequest,
  FieldWarning,
  ModelRun,
  ThresholdRule,
} from "./types";

/** A threshold rule as the deploy dialog holds it: `value` is the typed minimum precision or fixed threshold. */
export interface RuleDraft {
  kind: ThresholdRule["kind"];
  value: string;
}
export const F1_RULE: RuleDraft = { kind: "f1", value: "" };

/** A run the deploy dialog would deploy: `binary` for a one-vs-rest run, which takes a threshold. */
export interface DeployModel {
  runId: number;
  label: string;
  binary: boolean;
}

const asNumber = (s: string) => (s.trim() === "" ? NaN : Number(s));

/** The rule as the API takes it, or why the API would refuse its number. */
export function parseRule(d: RuleDraft): { rule: ThresholdRule; error: null } | { rule: null; error: string } {
  if (d.kind === "f1") return { rule: { kind: "f1" }, error: null };
  const v = asNumber(d.value);
  if (d.kind === "precision")
    return v > 0 && v <= 1
      ? { rule: { kind: "precision", min: v }, error: null }
      : { rule: null, error: "Precision must be above 0 and at most 1" };
  return v >= 0 && v <= 1
    ? { rule: { kind: "fixed", value: v }, error: null }
    : { rule: null, error: "A fixed threshold must be from 0 to 1" };
}

/** The dialog's choices as POST /api/experiments/{id}/deployments takes them. `runIds` null deploys the batch's
 * finished runs. Rules go only to one-vs-rest models, and the F1 rule for all is the API's default, so it is left
 * out. */
export function deployRequest(f: {
  name: string;
  batchId: number | null;
  runIds: number[] | null;
  models: DeployModel[];
  rule: RuleDraft;
  overrides: Record<number, RuleDraft>;
}): { request: DeploymentRequest; error: null } | { request: null; error: string } {
  if (f.runIds === null && f.batchId === null) return { request: null, error: "Choose the runs to deploy" };
  const request: DeploymentRequest = { name: f.name.trim() || null };
  if (f.batchId !== null) request.batch_id = f.batchId;
  if (f.runIds !== null) request.run_ids = f.runIds;
  const binary = f.models.filter((m) => m.binary);
  if (!binary.length) return { request, error: null };
  if (f.rule.kind !== "f1") {
    const all = parseRule(f.rule);
    if (all.error !== null) return { request: null, error: all.error };
    request.threshold_rule = all.rule;
  }
  const thresholds: Record<number, ThresholdRule> = {};
  for (const m of binary) {
    const d = f.overrides[m.runId];
    if (!d) continue;
    const own = parseRule(d);
    if (own.error !== null) return { request: null, error: `${m.label}: ${own.error}` };
    thresholds[m.runId] = own.rule;
  }
  if (Object.keys(thresholds).length) request.thresholds = thresholds;
  return { request, error: null };
}

/** What Deploy… on a batch page deploys: the selected rows, else a finished one-vs-rest batch's finished runs
 * (`runIds` null), and why it can't when it can't. */
export function batchDeployment(
  b: Pick<BatchDetail, "kind" | "status" | "rows">,
  selected: number[],
): { runIds: number[] | null; models: DeployModel[]; blocker: string | null } {
  const model = (r: BatchDetail["rows"][number]): DeployModel => ({
    runId: r.run_id,
    label: r.positive_class ?? r.label ?? `Run ${r.run_id}`,
    binary: r.positive_class !== null,
  });
  if (selected.length) {
    const rows = b.rows.filter((r) => selected.includes(r.run_id));
    return { runIds: rows.map((r) => r.run_id), models: rows.map(model), blocker: null };
  }
  const none = { runIds: null, models: [] };
  if (b.kind === "search") return { ...none, blocker: "Select the rows to deploy" };
  if (b.status === "running")
    return { ...none, blocker: "The batch is still running: deploy it once it ends, or select its finished runs" };
  const done = b.rows.filter((r) => r.status === "done");
  if (!done.length) return { ...none, blocker: "None of the batch's runs finished" };
  return { runIds: null, models: done.map(model), blocker: null };
}

/** A run that can be deployed on its own: finished, with one target (a chain answers for several). */
export function runDeployable(run: Pick<ModelRun, "status" | "report" | "config">): boolean {
  if (run.status !== "done") return false;
  return (run.report?.order ?? run.config?.order ?? []).length === 1;
}

/** How the deploy dialog names a run: a one-vs-rest run by its class, as the deployment keys its scores. */
export function runDeployLabel(run: Pick<ModelRun, "id" | "config" | "report">): string {
  if (run.config?.positive_class) return run.config.positive_class;
  const name = run.report?.model_label ?? `Run ${run.id}`;
  const target = (run.report?.order ?? run.config?.order ?? [])[0];
  if (!target) return name;
  return run.report?.target_kinds?.[target] === "regression"
    ? `${name}, predicting ${target} as a number`
    : `${name}, predicting ${target}'s class`;
}

/** The page's `curl` example: two records of the deployment's fields, from the dataset's rows where there are any,
 * else null for each. The body is single-quoted for the shell, so an apostrophe in a value is closed and reopened. */
export function curlExample(
  origin: string,
  deploymentId: number,
  fields: DeploymentField[],
  example: Record<string, unknown>[] | null,
): string {
  const record = (i: number) =>
    JSON.stringify(Object.fromEntries(fields.map((f) => [f.name, example?.[i]?.[f.name] ?? null])));
  const quote = (s: string) => s.replaceAll("'", `'\\''`);
  return [
    `curl -X POST ${origin}/api/deployments/${deploymentId}/predict \\`,
    '  -H "Authorization: Bearer rf_…" \\',
    '  -H "Content-Type: application/json" \\',
    `  -d '{"records": [`,
    `    ${quote(record(0))},`,
    `    ${quote(record(1))}`,
    "  ]}'",
  ].join("\n");
}

export const MAX_SCORE_COLUMNS = 10;

/** The score columns a test's results table shows: all of them up to MAX_SCORE_COLUMNS, past that none until
 * Show all scores (each row's top class and its score are shown either way). */
export function scoreColumns(labels: string[], showAll: boolean): { shown: string[]; hidden: number; toggle: boolean } {
  if (labels.length <= MAX_SCORE_COLUMNS) return { shown: labels, hidden: 0, toggle: false };
  return showAll ? { shown: labels, hidden: 0, toggle: true } : { shown: [], hidden: labels.length, toggle: true };
}

/** How a rule chose a threshold, in words; `fallback` when the API noted it fell back to the F1 choice. */
export function ruleText(rule: ThresholdRule, fallback = false): string {
  if (rule.kind === "fixed") return "fixed";
  if (rule.kind === "precision")
    return fallback
      ? `the F1 choice: no threshold reaches precision ${rule.min} on validation`
      : `the lowest with precision at least ${rule.min} on validation`;
  return fallback ? "0.5: no threshold gives a positive F1 on validation" : "best F1 on validation";
}

/** A deployed model's threshold and the rule that chose it; a multiclass model predicts its top class, a number run
 * a number. */
export function modelThreshold(m: DeploymentModel): { value: string; rule: string | null; fallback: boolean } {
  if (m.kind === "multiclass") return { value: "Top class", rule: null, fallback: false };
  if (m.kind === "regression") return { value: "A number", rule: null, fallback: false };
  const fallback = Boolean(m.threshold_note);
  return {
    value: m.threshold === null ? "—" : m.threshold.toFixed(2),
    rule: m.threshold_rule ? ruleText(m.threshold_rule, fallback) : null,
    fallback,
  };
}

/** A field is flagged when more than half of the values the file gave for it didn't convert. */
export const moreThanHalf = (w: { failed: number; values: number }) => w.failed > w.values / 2;

/** A test's per-field conversion warnings, the flagged fields first, then by the share of values that failed. */
export function fieldWarnings(
  warnings: Record<string, FieldWarning> | null,
): { field: string; failed: number; values: number; share: number; flagged: boolean }[] {
  return Object.entries(warnings ?? {})
    .map(([field, w]) => ({
      field,
      failed: w.failed,
      values: w.values,
      share: w.values ? w.failed / w.values : 1,
      flagged: w.flagged ?? moreThanHalf(w),
    }))
    .sort((a, b) => Number(b.flagged) - Number(a.flagged) || b.share - a.share || a.field.localeCompare(b.field));
}
