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

/** A lead may refresh a connected dataset that is ready and has no pull running. */
export function canRefresh(
  me: { role: Role } | null | undefined,
  ds: { source: object | null; status: DatasetStatus; last_pull: Pick<PullSummary, "status"> | null },
): boolean {
  return me?.role === "lead" && ds.source !== null && ds.status === "ready" && ds.last_pull?.status !== "running";
}
