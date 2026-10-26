import type { DatasetStatus, PullSummary, Role } from "./types";

/** What someone might do in the app. Viewers do none of it but change their own picture; the API refuses the same. */
export const ACTIONS = [
  "upload",
  "editDataset",
  "decide",
  "comment",
  "createExperiment",
  "export",
  "train",
  "recompute",
  "startBaseline",
  "renameBaseline",
  "assign",
  "finalize",
  "deleteDataset",
  "manageUsers",
  "join",
  "changeOwnPicture",
  "querySql",
  "chooseModel",
] as const;
export type Action = (typeof ACTIONS)[number];

const LEAD_ONLY = new Set<Action>([
  "startBaseline",
  "renameBaseline",
  "assign",
  "finalize",
  "deleteDataset",
  "manageUsers",
  "join",
  "chooseModel",
]);

/** Actions that stay open while a refresh holds the dataset: they don't write to it. */
const OPEN_WHILE_LOCKED = new Set<Action>(["upload", "manageUsers", "join", "changeOwnPicture", "querySql"]);

/** `locked`: a refresh is running on the dataset the action is for, and the API refuses its writes. */
export function can(me: { role: Role } | null | undefined, action: Action, locked = false): boolean {
  if (!me) return false;
  if (locked && !OPEN_WHILE_LOCKED.has(action)) return false;
  if (me.role === "viewer") return action === "changeOwnPicture";
  return !LEAD_ONLY.has(action) || me.role === "lead";
}

/** Why a dataset's own controls on the Datasets list are off, as their tooltip: a refresh holds it. Null when not. */
export function lockedTitle(ds: { status: DatasetStatus }): string | null {
  return ds.status === "refreshing" ? "A refresh is running; try again once it finishes" : null;
}

/** Why Refresh… is off, as its tooltip; null when it can run: a lead, a connected dataset that is ready, no pull
 * running. A dataset whose connect failed never becomes ready (a refresh only replaces data it has), so waiting
 * would not help. */
export function refreshBlocker(
  me: { role: Role } | null | undefined,
  ds: { source: object | null; status: DatasetStatus; last_pull: Pick<PullSummary, "status"> | null },
): string | null {
  if (me?.role !== "lead") return "Only a lead can refresh a dataset";
  if (ds.source === null) return "This dataset isn't connected to a source";
  if (ds.status === "refreshing" || ds.last_pull?.status === "running") return "A refresh is already running";
  if (ds.status === "failed")
    return "The connect failed, so there is nothing to refresh: delete this dataset and connect it again";
  if (ds.status !== "ready") return "Wait until the dataset is ready";
  return null;
}
