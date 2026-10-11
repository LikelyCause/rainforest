import { toInput } from "./decision";
import { baselineLabel } from "./format";
import { SETTING_LABEL } from "./recipes";
import type { Decision, DecisionInput, Experiment, FieldResponse, Phase, Role } from "./types";

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
    .sort((a, b) => FAMILY_ORDER.indexOf(a.family) - FAMILY_ORDER.indexOf(b.family) || b.version - a.version);
  for (const v of finalized) {
    out.push({ value: `recipe:${v.id}`, label: `${baselineLabel(v.family, v.version, false, v.name)} (finalized)` });
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
