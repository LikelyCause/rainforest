import { toInput } from "./decision";
import { baselineLabel, fmtInt } from "./format";
import { SETTING_LABEL } from "./recipes";
import type { Decision, DecisionInput, Experiment, ExportRows, FieldResponse, Phase, Role } from "./types";

export type ExperimentFilter = "all" | "mine" | "team" | "archived";

interface Viewer { id: number; role: Role }

/** Archived experiments show only under "archived"; the rest under "all", and split by whose they are. */
export function filterExperiments(list: Experiment[], filter: ExperimentFilter, me: Viewer): Experiment[] {
  if (filter === "archived") return list.filter((x) => x.archived_at);
  const live = list.filter((x) => !x.archived_at);
  if (filter === "mine") return live.filter((x) => x.owner === me.id || x.assessor_id === me.id);
  if (filter === "team") return live.filter((x) => x.assessor_id === null);
  return live;
}

export function sourceLabel(x: Pick<Experiment, "assessor_id" | "recipe_family" | "recipe_version" | "recipe_name" | "owner_name">, me?: Viewer): string {
  if (x.assessor_id === null) return baselineLabel(x.recipe_family, x.recipe_version, false, x.recipe_name);
  const whose = me && x.assessor_id === me.id ? "My" : `${x.owner_name ?? "An assessor"}'s`;
  return `${whose} assessment (${baselineLabel(x.recipe_family, x.recipe_version, true, x.recipe_name)})`;
}

export function privacyNote(x: Pick<Experiment, "private" | "assessor_id" | "recipe_family" | "recipe_version" | "recipe_name">): string | null {
  const v = baselineLabel(x.recipe_family, x.recipe_version, false, x.recipe_name);
  if (x.private) return `Only you can see this until ${v} reaches team review.`;
  if (x.assessor_id !== null) return `Visible to the team since ${v} reached review.`;
  return null;
}

/** Renaming, notes, archiving: the owner (for a personal one, its assessor) or a lead. */
export const canEdit = (x: Pick<Experiment, "owner" | "assessor_id">, me: Viewer) =>
  me.role !== "viewer" && (me.role === "lead" || me.id === x.owner || me.id === x.assessor_id);

/** Why the viewer can't add exports or runs, or null when they can. */
export function addBlocked(x: Pick<Experiment, "archived_at" | "assessor_id" | "owner_name">, me: Viewer): string | null {
  if (me.role === "viewer") return "Viewers can look but not change anything.";
  if (x.archived_at) return "Archived. Unarchive it to add exports and runs.";
  if (x.assessor_id !== null && x.assessor_id !== me.id) {
    return `Only ${x.owner_name ?? "its assessor"} adds to this experiment; it is built from their assessment.`;
  }
  return null;
}

export interface SourceOption { value: string; label: string }

const FAMILY_ORDER = ["tree", "linear", "neural"];

/** What the viewer can build a new experiment from: their own assessment of the baseline being worked (when they
 * assess it), then every finalized baseline version, by model type and newest first. */
export function sourceOptions(
  versions: { id: number; family: string; version: number; phase: Phase; name?: string | null }[],
  team: { recipe: { id: number; family: string; version: number; phase: Phase; name?: string | null }; me: unknown } | undefined,
  me: Viewer,
): SourceOption[] {
  const out: SourceOption[] = [];
  if (team?.me && team.recipe.phase !== "finalized") {
    out.push({ value: `assessment:${team.recipe.id}:${me.id}`,
      label: `My assessment (${baselineLabel(team.recipe.family, team.recipe.version, true, team.recipe.name)})` });
  }
  const finalized = versions.filter((v) => v.phase === "finalized")
    .sort((a, b) => Number(b.id === team?.recipe.id) - Number(a.id === team?.recipe.id)
      || FAMILY_ORDER.indexOf(a.family) - FAMILY_ORDER.indexOf(b.family) || b.version - a.version);
  for (const v of finalized) {
    out.push({ value: `recipe:${v.id}`, label: `${baselineLabel(v.family, v.version, false, v.name)} · finalized` });
  }
  return out;
}

/** The experiment's runs in the MLflow UI: the dataset's MLflow experiment, filtered to this experiment's tag. */
export function mlflowExperimentUrl(mlflowExperimentId: string | null | undefined, experimentId: number): string | null {
  if (!mlflowExperimentId) return null;
  const filter = `tags.\`rainforest.experiment_id\` = '${experimentId}'`;
  return `/mlflow/#/experiments/${mlflowExperimentId}?searchFilter=${encodeURIComponent(filter)}`;
}

export function metricLabel(metric: string): string {
  return metric === "r2" ? "R²" : metric.startsWith("roc_auc") ? "AUC" : metric;
}

type BaselineField = NonNullable<FieldResponse["baseline"]>;
const COMPARED = ["action", "semantic_type", "target_task", "target_bins", "imputation", "impute_value",
  "add_missing_indicator", "transform", "scaling", "encoding", "params"] as const;

const baselineInput = (base: BaselineField, family: string): DecisionInput =>
  toInput({ ...base, ...(base.settings ?? {}) } as unknown as Decision, family);

/** What a draft in an experiment changes from its baseline, as setting labels (notes don't count). */
export function baselineDiff(draft: DecisionInput, base: FieldResponse["baseline"]): string[] {
  if (!base) return [];
  const b = baselineInput(base, draft.family ?? "");
  return COMPARED.filter((k) => JSON.stringify(draft[k] ?? null) !== JSON.stringify(b[k] ?? null))
    .map((k) => SETTING_LABEL[k] ?? k);
}

/** The draft put back to its baseline's decision and settings; the experiment's own note stays. */
export function revertedInput(draft: DecisionInput, base: BaselineField): DecisionInput {
  return { ...baselineInput(base, draft.family ?? ""), notes: draft.notes, reviewed: true };
}

export interface RowRules {
  dedupe: boolean;
  exclude: { field: string; values: string[] }[];
  first_per: { key: string; order: string | null } | null;
  group_by: string | null;
}
export const NO_RULES: RowRules = { dedupe: false, exclude: [], first_per: null, group_by: null };

/** The row rules of the dataset's newest export that set any, so the next export starts from them. A failed export's
 * rules may be why it failed, so those don't count. */
export function lastRowRules(exports: { id: number; status?: string; config: Record<string, any> | null }[]): { from: number; rules: RowRules } | null {
  for (const e of [...exports].sort((a, b) => b.id - a.id)) {
    if (e.status === "failed") continue;
    const c = e.config ?? {};
    const rules: RowRules = { dedupe: Boolean(c.dedupe), exclude: c.exclude ?? [], first_per: c.first_per ?? null, group_by: c.group_by ?? null };
    if (rules.dedupe || rules.exclude.length || rules.first_per || rules.group_by) return { from: e.id, rules };
  }
  return null;
}

/** Rows per split as the export cuts `n` rows: the held-out share rounds up, then test's part of it does (scikit-learn's
 * shuffle split). A grouped split moves whole groups, so its sizes only come close to these. */
export function splitCounts(n: number, valid: number, test: number): { train: number; valid: number; test: number } {
  const holdout = valid + test;
  if (holdout <= 0 || n <= 0) return { train: Math.max(n, 0), valid: 0, test: 0 };
  const rest = Math.ceil(holdout * n);
  const t = valid <= 0 ? rest : test <= 0 ? 0 : Math.ceil((test / holdout) * rest);
  return { train: n - rest, valid: rest - t, test: t };
}

/** The split counts the panel shows: the server's exact ones (the export's own split, groups included) when it has
 * them, else an estimate from the rows to split, marked `about` for a grouped split, which moves whole groups. */
export function splitPreview(r: ExportRows | undefined, fileRows: number | null, valid: number, test: number, grouped: boolean):
  { counts: { train: number; valid: number; test: number } | null; about: boolean } {
  if (r?.split) return { counts: r.split, about: false };
  const n = r?.split_rows ?? fileRows;
  return n === null ? { counts: null, about: false } : { counts: splitCounts(n, valid, test), about: grouped };
}

/** "69,990 rows after the row rules (1,246 left out)", or the file's count when the rules leave every row. Kept rows
 * without a target value are said too, since the split leaves them out. */
export function rowsSentence(r: ExportRows): string {
  const left = r.rows - r.kept;
  const head = left > 0 ? `${fmtInt(r.kept)} rows after the row rules (${fmtInt(left)} left out)` : `${fmtInt(r.rows)} rows`;
  const missing = r.missing_target > 0
    ? `; ${fmtInt(r.missing_target)} of them ${r.missing_target === 1 ? "has" : "have"} no target value and ${r.missing_target === 1 ? "stays" : "stay"} out of the split` : "";
  return head + missing;
}

/** An experiment page's places, as `?tab=`; anything else opens Runs. */
export type ExperimentTab = "runs" | "train" | "exports" | "adjustments" | "notes";
const TABS: ExperimentTab[] = ["runs", "train", "exports", "adjustments", "notes"];
export const experimentTab = (v: string | null): ExperimentTab => (TABS.includes(v as ExperimentTab) ? (v as ExperimentTab) : "runs");

/** Where an experiment's fields come from, as its header chip says it: a frozen baseline, or an assessment as it stands. */
export function sourceChip(x: Pick<Experiment, "assessor_id" | "recipe_family" | "recipe_version" | "recipe_name" | "owner_name">, me?: Viewer):
  { lens: "frozen" | "private"; text: string; title: string } {
  if (x.assessor_id === null) {
    return { lens: "frozen", text: `From ${baselineLabel(x.recipe_family, x.recipe_version, false, x.recipe_name)} · as finalized`,
      title: "Exports use the version's frozen snapshot; later versions don't change it." };
  }
  const whose = me && x.assessor_id === me.id ? "my" : `${x.owner_name ?? "an assessor"}'s`;
  return { lens: "private", text: `From ${whose} assessment of ${baselineLabel(x.recipe_family, x.recipe_version, true, x.recipe_name)} · as it stands`,
    title: "Each export uses the assessment as it stands when it is made; later edits don't change an export already made." };
}

/** The finalized-baseline card on Experiments: shown right after finalizing (`?finalized=`), or while no live
 * experiment is built on the dataset's finalized baseline yet. */
export function finalizedPrompt(
  list: Pick<Experiment, "recipe_id" | "archived_at">[] | undefined,
  baseline: { id: number; phase: Phase | null } | null,
  param: string | null,
): number | null {
  if (param && Number(param)) return Number(param);
  if (!list || !baseline || baseline.phase !== "finalized") return null;
  return list.some((x) => !x.archived_at && x.recipe_id === baseline.id) ? null : baseline.id;
}
