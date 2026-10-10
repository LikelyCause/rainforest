import type { Experiment, Phase, Role } from "./types";

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

export function sourceLabel(x: Pick<Experiment, "assessor_id" | "recipe_version" | "owner_name">, me?: Viewer): string {
  if (x.assessor_id === null) return `Recipe v${x.recipe_version}`;
  const whose = me && x.assessor_id === me.id ? "My" : `${x.owner_name ?? "An assessor"}'s`;
  return `${whose} assessment (v${x.recipe_version})`;
}

export function privacyNote(x: Pick<Experiment, "private" | "assessor_id" | "recipe_version">): string | null {
  if (x.private) return `Only you can see this until recipe v${x.recipe_version} reaches team review.`;
  if (x.assessor_id !== null) return `Visible to the team since recipe v${x.recipe_version} reached review.`;
  return null;
}

/** Renaming, notes, archiving: the owner (for a personal one, its assessor) or a lead. */
export const canEdit = (x: Pick<Experiment, "owner" | "assessor_id">, me: Viewer) =>
  me.role === "lead" || me.id === x.owner || me.id === x.assessor_id;

/** Why the viewer can't add exports or runs, or null when they can. */
export function addBlocked(x: Pick<Experiment, "archived_at" | "assessor_id" | "owner_name">, me: Viewer): string | null {
  if (x.archived_at) return "Archived. Unarchive it to add exports and runs.";
  if (x.assessor_id !== null && x.assessor_id !== me.id) {
    return `Only ${x.owner_name ?? "its assessor"} adds to this experiment; it is built from their assessment.`;
  }
  return null;
}

export interface SourceOption { value: string; label: string }

/** What the viewer can build a new experiment from: their own assessment of the version being worked (when they
 * assess it), then every finalized version, newest first. */
export function sourceOptions(
  versions: { id: number; version: number; phase: Phase }[],
  team: { recipe: { id: number; version: number; phase: Phase }; me: unknown } | undefined,
  me: Viewer,
): SourceOption[] {
  const out: SourceOption[] = [];
  if (team?.me && team.recipe.phase !== "finalized") {
    out.push({ value: `assessment:${team.recipe.id}:${me.id}`, label: `My assessment (v${team.recipe.version})` });
  }
  for (const v of [...versions].filter((v) => v.phase === "finalized").sort((a, b) => b.version - a.version)) {
    out.push({ value: `recipe:${v.id}`, label: `Recipe v${v.version} (finalized)` });
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
  return metric.startsWith("roc_auc") ? "AUC" : metric;
}
