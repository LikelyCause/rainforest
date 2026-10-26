import { can } from "./can";
import type { Dataset, FamilyKey, FieldListItem, Role } from "./types";

export const FAMILIES: FamilyKey[] = ["tree", "linear", "neural"];

/** What the Baseline tab says in place of the setup form, or null when the viewer gets the form. `locked`: a
 * refresh holds the dataset, and the targets offered would be the fields it is about to replace. */
export function setupNote(me: { role: Role } | undefined, locked: boolean): string | null {
  if (can(me, "startBaseline", locked)) return null;
  if (me?.role === "lead")
    return "A refresh is running. Set up the baseline once it finishes, so its targets come from the new pull's fields.";
  return "A lead sets up this dataset's first baseline: its model type, targets and who assesses it. Until then you can browse the fields and Explore.";
}
const TARGET_TYPES = new Set(["binary", "categorical", "numeric_discrete", "numeric_continuous"]);
const LABEL_TYPES = new Set(["binary", "categorical"]);
const MAX_TARGET_LEVELS = 100;

export interface StartBaselineForm {
  family: FamilyKey;
  start: "scratch" | "baseline" | "recipe";
  source_recipe_id: number | null;
  targets: string[];
  assessors: number[];
  /** The baseline's name, for a model type that has no baseline yet. */
  name: string;
}

/** Why a field can't be a target (the API's own rule), or null when it can. */
export function eligibleTarget(f: Pick<FieldListItem, "semantic_type" | "n_unique">): string | null {
  if (!TARGET_TYPES.has(f.semantic_type)) return `${f.semantic_type.replace("_", " ")} fields can't be targets`;
  if (LABEL_TYPES.has(f.semantic_type) && f.n_unique !== null && f.n_unique > MAX_TARGET_LEVELS) {
    return `more than ${MAX_TARGET_LEVELS} distinct values`;
  }
  return null;
}

/** The start dialog's first answer: the first model type without a baseline (Trees on a new dataset), populated
 * from the most recently finalized baseline when there is one, the current targets, and the viewer assessing. */
export function startBaselineDefaults(
  ds: Pick<Dataset, "baseline" | "baselines">,
  me: { id: number },
  fields: Pick<FieldListItem, "name" | "action">[],
): StartBaselineForm {
  const taken = new Set(ds.baseline?.setup ? [] : ds.baselines.map((b) => b.family));
  const family: FamilyKey = FAMILIES.find((f) => !taken.has(f)) ?? "tree";
  const finalized = ds.baselines.filter((b) => b.latest_finalized_id !== null);
  const source = finalized.length ? Math.max(...finalized.map((b) => b.latest_finalized_id as number)) : null;
  return {
    family,
    start: source === null ? "scratch" : "baseline",
    source_recipe_id: source,
    targets: fields.filter((f) => f.action === "target").map((f) => f.name),
    assessors: [me.id],
    name: "",
  };
}
