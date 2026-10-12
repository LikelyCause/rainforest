import { bandsText } from "./decision";
import type { Dataset, Me, Phase, QueueItem, TeamState } from "./types";

export const PHASE_LABEL: Record<Phase, string> = { assessing: "Assessing", review: "Team review", finalized: "Finalized" };

/** What the viewer works on in Review and Bulk edit, and whether they can change it. */
export interface Access {
  /** "mine": the viewer's own blind assessment; "team": the team decision; "experiment": a team experiment's copy
   * of its baseline; "baseline": a finalized baseline. */
  source: "mine" | "team" | "experiment" | "baseline";
  canEdit: boolean;
  /** Why editing is off (or what is being edited), for a banner. */
  note: string;
}

export function accessOf(ds: Pick<Dataset, "phase"> & { baseline: (Pick<NonNullable<Dataset["baseline"]>, "label"> & { setup?: boolean }) | null },
  me: Pick<Me, "id" | "role">, team: Pick<TeamState, "me"> | undefined): Access {
  const v = ds.baseline?.label ?? "baseline";
  if (me.role === "viewer") {
    return { source: "team", canEdit: false, note: ds.phase === "assessing"
      ? `${v} is being assessed; as a Viewer you see its decisions from team review on. Shown: the latest finalized baseline, or the suggestions.`
      : "You're a Viewer: you can look at everything the team shares but not change it." };
  }
  if (ds.baseline?.setup) {
    return { source: "team", canEdit: false, note: "This dataset's first baseline isn't set up yet; it is read-only until a lead sets it up on the Baseline tab." };
  }
  if (ds.phase === "assessing") {
    if (team?.me) {
      return team.me.submitted
        ? { source: "mine", canEdit: false, note: "You submitted your assessment. Unsubmit it on the Baseline tab to make changes." }
        : { source: "mine", canEdit: true, note: "Your assessment. Nobody else sees your choices until team review." };
    }
    return { source: "team", canEdit: false, note: `${v} is being assessed. You can browse the fields and their suggestions; only assigned assessors make choices.` };
  }
  if (ds.phase === "review") {
    return me.role === "lead"
      ? { source: "team", canEdit: true, note: "Team review: changes here set the team decision." }
      : { source: "team", canEdit: false, note: "Team review: a lead sets the team decision. You can comment in the team review queue." };
  }
  if (ds.phase === "finalized") return { source: "team", canEdit: false, note: `${v} is finalized; its decisions are frozen. A lead can start a new version on the Baseline tab.` };
  return { source: "team", canEdit: true, note: "" };
}

/** "2 keep · 1 drop", the spread of choices on a field in the review queue. */
export function spread(item: Pick<QueueItem, "assessments">, typeLabel: (t: string) => string = (t) => t): string {
  const counts = new Map<string, number>();
  const types = new Set(item.assessments.map((a) => a.semantic_type));
  const tasks = new Set(item.assessments.map((a) => bandsText(a.target_task, a.target_bins)));
  for (const a of item.assessments) {
    const task = bandsText(a.target_task, a.target_bins);
    const k = types.size > 1 ? `${a.action} as ${typeLabel(a.semantic_type)}`
      : tasks.size > 1 && task ? `${a.action} as ${task}` : a.action;
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
export function finalizeQuestion(open: number, label: string): string | null {
  if (open === 0) return null;
  return `${open} disagreement${open === 1 ? " is" : "s are"} still open. Finalize ${label} anyway, keeping ` +
    `${open === 1 ? "its" : "their"} current team decision${open === 1 ? "" : "s"}?`;
}

/** Next open item after `current` in queue order, wrapping; null when none is open. */
export function nextOpen(queue: Pick<QueueItem, "field_id" | "resolved">[], current: number | null): number | null {
  const at = queue.findIndex((q) => q.field_id === current);
  const order = [...queue.slice(at + 1), ...queue.slice(0, at + 1)];
  return order.find((q) => !q.resolved && q.field_id !== current)?.field_id ?? null;
}

/** Assessors grouped by the value they chose, most common first (ties keep the assessors' order). */
export interface ValueGroup { value: string | null; who: string[]; at: number[] }
export function valueGroups(values: (string | null | undefined)[], names: string[]): ValueGroup[] {
  const groups: ValueGroup[] = [];
  values.forEach((v, i) => {
    const value = v ?? null;
    const g = groups.find((x) => x.value === value);
    if (g) { g.who.push(names[i]); g.at.push(i); } else groups.push({ value, who: [names[i]], at: [i] });
  });
  return groups.sort((a, b) => b.who.length - a.who.length);
}

/** The assessors whose value is the odd one out: those outside a strict majority. None when there's no majority. */
export function oddOnes(values: (string | null | undefined)[]): Set<number> {
  const groups = valueGroups(values, values.map(String));
  if (groups.length < 2 || groups[0].who.length === groups[1].who.length) return new Set();
  return new Set(groups.slice(1).flatMap((g) => g.at));
}

/** Names short enough for a vote line: the full name, or for a long one the first name and an initial ("Priya R.").
 * Wherever that leaves two people looking the same, both keep their full names. */
export function shortNames(names: string[]): string[] {
  const short = names.map((n) => {
    const full = n.trim().replace(/\s+/g, " ");
    const parts = full.split(" ");
    return full.length <= 14 || parts.length < 2 ? full : `${parts[0]} ${parts[parts.length - 1][0]}.`;
  });
  return short.map((s, i) => (short.some((t, j) => j !== i && t === s && names[j].trim() !== names[i].trim()) ? names[i].trim() : s));
}

/** How the assessors voted on one part of the decision, for the team decision panel: "All 3 chose keep", or
 * "2 Maya, Dana · 1 Priya: number", naming the value wherever it isn't the team decision's current one. */
export function voteLine(values: (string | null | undefined)[], names: string[], current?: string | null): string | null {
  const groups = valueGroups(values, shortNames(names)).filter((g) => g.value !== null);
  if (groups.length === 0) return null;
  if (groups.length === 1) {
    const n = values.length;
    if (groups[0].who.length < n) return `${groups[0].who.join(", ")} chose ${groups[0].value}`;
    return `${n === 1 ? "The assessor" : n === 2 ? "Both" : `All ${n}`} chose ${groups[0].value}`;
  }
  return groups.map((g) => `${g.who.length} ${g.who.join(", ")}${g.value === current ? "" : `: ${g.value}`}`).join(" · ");
}

/** The values a row is compared on: only those of the assessors in `among` when it is given (e.g. a setting only
 * those who kept the field chose), else everyone's. */
export function comparedValues(r: { values: (string | null | undefined)[]; among?: number[] }): (string | null | undefined)[] {
  return r.among ? r.among.map((i) => r.values[i]) : r.values;
}

/** Splits the compared settings into the ones where the assessors differ and the ones they all share. A row limited
 * to `among` differs only when those assessors differ; with nobody in `among` it is left out of both. */
export function splitAgreement<T extends { values: (string | null | undefined)[]; among?: number[] }>(rows: T[]): { differ: T[]; agree: T[] } {
  const differ: T[] = [];
  const agree: T[] = [];
  for (const r of rows) {
    const vs = comparedValues(r);
    if (vs.length === 0) continue;
    (new Set(vs.map((v) => v ?? null)).size > 1 ? differ : agree).push(r);
  }
  return { differ, agree };
}

/** "Submitted. 2 of 3 are in. Team review opens when Priya Raman submits (21/37)." */
export function submittedText(assessors: { name: string; submitted_at: string | null; reviewed: number; total: number }[]): string {
  const n = assessors.length;
  const waiting = assessors.filter((a) => !a.submitted_at);
  const k = n - waiting.length;
  if (waiting.length === 0) return `${k} of ${n} are in. Team review is opening.`;
  const who = waiting.length <= 2
    ? waiting.map((a) => `${a.name} (${a.reviewed}/${a.total})`).join(" and ")
    : `${waiting.length} more people`;
  return `${k} of ${n} ${k === 1 ? "is" : "are"} in. Team review opens when ${who} submit${waiting.length === 1 ? "s" : ""}.`;
}
