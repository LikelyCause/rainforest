import type { Dataset, Me, Phase, QueueItem, TeamState } from "./types";

export const PHASE_LABEL: Record<Phase, string> = { assessing: "Assessing", review: "Team review", finalized: "Finalized" };

/** What the viewer works on in Review and Bulk edit, and whether they can change it. */
export interface Access {
  /** "mine": the viewer's own blind assessment; "team": the team decision. */
  source: "mine" | "team";
  canEdit: boolean;
  /** Why editing is off (or what is being edited), for a banner. */
  note: string;
}

export function accessOf(ds: Pick<Dataset, "phase" | "recipe_version">, me: Pick<Me, "id" | "role">, team: Pick<TeamState, "me"> | undefined): Access {
  const v = `recipe v${ds.recipe_version ?? 1}`;
  if (ds.phase === "assessing") {
    if (team?.me) {
      return team.me.submitted
        ? { source: "mine", canEdit: false, note: "You submitted your assessment. Unsubmit it on the Team page to make changes." }
        : { source: "mine", canEdit: true, note: "Your assessment. Nobody else sees your choices until team review." };
    }
    return { source: "team", canEdit: false, note: `The ${v} is being assessed. You can browse the fields and their suggestions; only assigned assessors make choices.` };
  }
  if (ds.phase === "review") {
    return me.role === "lead"
      ? { source: "team", canEdit: true, note: "Team review: changes here set the team decision." }
      : { source: "team", canEdit: false, note: "Team review: a lead sets the team decision. You can comment on the Team page." };
  }
  if (ds.phase === "finalized") return { source: "team", canEdit: false, note: `The ${v} is finalized; its decisions are frozen.` };
  return { source: "team", canEdit: true, note: "" };
}

/** "2 keep · 1 drop", the spread of choices on a field in the review queue. */
export function spread(item: Pick<QueueItem, "assessments">, typeLabel: (t: string) => string = (t) => t): string {
  const counts = new Map<string, number>();
  const types = new Set(item.assessments.map((a) => a.semantic_type));
  for (const a of item.assessments) {
    const k = types.size > 1 ? `${a.action} as ${typeLabel(a.semantic_type)}` : a.action;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  return [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([k, n]) => `${n} ${k}`).join(" · ");
}

/** Which assessors' choices differ from the others on each part of the decision, for highlighting. */
export function differing<T extends Record<string, unknown>>(rows: T[], keys: (keyof T)[]): Set<keyof T> {
  const out = new Set<keyof T>();
  for (const k of keys) {
    if (new Set(rows.map((r) => JSON.stringify(r[k] ?? null))).size > 1) out.add(k);
  }
  return out;
}

/** The question to ask before finalizing, or null when nothing is left open. */
export function finalizeQuestion(open: number, version: number): string | null {
  if (open === 0) return null;
  return `${open} disagreement${open === 1 ? " is" : "s are"} still open. Finalize recipe v${version} anyway, keeping ` +
    `${open === 1 ? "its" : "their"} current team decision${open === 1 ? "" : "s"}?`;
}

/** Next open item after `current` in queue order, wrapping; null when none is open. */
export function nextOpen(queue: Pick<QueueItem, "field_id" | "resolved">[], current: number | null): number | null {
  const at = queue.findIndex((q) => q.field_id === current);
  const order = [...queue.slice(at + 1), ...queue.slice(0, at + 1)];
  return order.find((q) => !q.resolved && q.field_id !== current)?.field_id ?? null;
}
