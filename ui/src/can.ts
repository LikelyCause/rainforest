import type { Role } from "./types";

/** What someone might do in the app. Viewers do none of it but change their own picture; the API refuses the same. */
export const ACTIONS = [
  "upload", "editDataset", "decide", "comment", "createExperiment", "export", "train", "recompute",
  "startBaseline", "assign", "finalize", "deleteDataset", "manageUsers", "join", "changeOwnPicture",
] as const;
export type Action = (typeof ACTIONS)[number];

const LEAD_ONLY = new Set<Action>(["startBaseline", "assign", "finalize", "deleteDataset", "manageUsers", "join"]);

export function can(me: { role: Role } | null | undefined, action: Action): boolean {
  if (!me) return false;
  if (me.role === "viewer") return action === "changeOwnPicture";
  return !LEAD_ONLY.has(action) || me.role === "lead";
}
