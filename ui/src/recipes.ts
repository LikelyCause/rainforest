import { baselineLabel, fmtInt, fmtPct } from "./format";
import { attentionOf } from "./signals";
import { bandsText, taskLabel } from "./decision";
import type {
  DiffChange,
  FieldListItem,
  RecipeVersion,
  ReopenReason,
  ReopenedField,
  ReuseSummary,
  StartSummary,
  TargetBins,
  TargetTask,
} from "./types";

const typeName = (t: string | undefined) => (t ?? "?").replace("_", " ");

export const REASON_LABEL: Record<ReopenReason, string> = {
  new: "New field",
  type_changed: "Type changed",
  drift: "Drifted",
  target: "Target no longer valid",
  settings: "Settings no longer valid",
};

/** One line on why a field was reopened, with the numbers behind it. */
export function reasonDetail(r: ReopenedField): string {
  if (r.reason === "type_changed") return `${typeName(r.before)} → ${typeName(r.after)}`;
  if (r.reason === "drift") {
    const parts: string[] = [];
    if (r.drift?.null_pct) {
      const [a, b] = r.drift.null_pct;
      parts.push(
        `missing ${fmtPct(a)} → ${fmtPct(b)} (${b >= a ? "+" : "−"}${(Math.abs(b - a) * 100).toFixed(1)} pts)`,
      );
    }
    if (r.drift?.n_unique) {
      const [a, b] = r.drift.n_unique;
      parts.push(`distinct values ${fmtInt(a)} → ${fmtInt(b)}${a ? ` (×${(b / a).toFixed(2)})` : ""}`);
    }
    return parts.join(" · ");
  }
  if (r.reason === "new") return "not in the recipe";
  return r.problem ?? "";
}

export interface ReuseGroup {
  reason: ReopenReason;
  label: string;
  fields: { name: string; detail: string }[];
}

/** The reuse summary as the page shows it: reopened fields grouped by reason (in a fixed order), the rest as lists. */
export function reuseGroups(
  s: Pick<ReuseSummary, "carried" | "reopened" | "missing"> &
    Partial<Pick<ReuseSummary, "targets" | "targets_not_carried">>,
): {
  carried: string[];
  reopened: ReuseGroup[];
  missing: string[];
  targets: string[];
  targetsNotCarried: { name: string; problem: string }[];
  needsAssessing: boolean;
} {
  const order: ReopenReason[] = ["new", "type_changed", "drift", "target", "settings"];
  const reopened = order
    .map((reason) => ({
      reason,
      label: REASON_LABEL[reason],
      fields: s.reopened.filter((r) => r.reason === reason).map((r) => ({ name: r.name, detail: reasonDetail(r) })),
    }))
    .filter((g) => g.fields.length > 0);
  return {
    carried: s.carried,
    reopened,
    missing: s.missing,
    targets: s.targets ?? [],
    targetsNotCarried: s.targets_not_carried ?? [],
    needsAssessing: s.reopened.length > 0,
  };
}

export const SETTING_LABEL: Record<string, string> = {
  action: "Decision",
  semantic_type: "Type",
  notes: "Notes",
  imputation: "Missing values",
  impute_value: "Fill value",
  add_missing_indicator: "Missing flag",
  transform: "Transform",
  scaling: "Scaling",
  encoding: "Encoding",
  params: "Encoder parameters",
  target_task: "Predict as",
  target_bins: "Bands",
};

export function fmtValue(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    return entries.length ? entries.map(([k, x]) => `${k}=${String(x)}`).join(", ") : "—";
  }
  return typeName(String(v));
}

/** "Trees · Missing values: none → mean", or "Decision: keep → drop" for the shared decision. */
export function fmtChange(c: DiffChange, familyLabel: (f: string) => string = (f) => f): string {
  const label = SETTING_LABEL[c.key] ?? c.key;
  const show =
    c.key === "target_task"
      ? (v: unknown) => (v ? taskLabel(v as TargetTask) : "—")
      : c.key === "target_bins"
        ? (v: unknown) => (v ? bandsText("bins", v as TargetBins)!.replace("Bands · ", "") : "—")
        : fmtValue;
  return `${c.family ? `${familyLabel(c.family)} · ` : ""}${label}: ${show(c.before)} → ${show(c.after)}`;
}

export interface PickFilter {
  search: string;
  types: Set<string>;
  attention: boolean;
}

/** Fields offered for reopening, filtered like the Review list (name, type, needs attention). */
export function pickable(fields: FieldListItem[], f: PickFilter): FieldListItem[] {
  const q = f.search.trim().toLowerCase();
  return fields.filter(
    (x) =>
      (!q || x.name.toLowerCase().includes(q)) &&
      (f.types.size === 0 || f.types.has(x.semantic_type)) &&
      (!f.attention || (attentionOf(x) !== null && attentionOf(x) !== "edited")),
  );
}

/** The label of the button that starts the next version. */
export function startLabel(next: number, reopen: number): string {
  return reopen === 0
    ? `Start v${next} and edit directly`
    : `Start v${next}, reopening ${reopen} field${reopen === 1 ? "" : "s"}`;
}

/** What a new version could not carry over from the one before, one sentence each; empty when it carried everything. */
export function startNotes(s: StartSummary | null | undefined): string[] {
  if (!s) return [];
  const out: string[] = [];
  if (s.missing.length) out.push(`No longer in the data: ${s.missing.join(", ")}.`);
  if (s.targets_not_carried.length)
    out.push(
      `Not carried over as targets: ${s.targets_not_carried.map((t) => `${t.name} (${t.problem})`).join(", ")}.`,
    );
  return out;
}

/** What a version was built from: a reused baseline of another dataset, the version or baseline before, or the suggestions.
 * `inline`: it follows "from" mid-sentence, so it starts in lower case. */
export function builtFrom(
  v: Pick<RecipeVersion, "reuse" | "parent_id" | "parent_version" | "parent_family" | "parent_name">,
  inline = false,
): string {
  if (v.reuse) {
    const s = v.reuse.source;
    return `${inline ? "reused" : "Reused"} ${s.family ? baselineLabel(s.family, s.version, false, s.name) : `v${s.version}`} of ${s.dataset_name ?? `dataset #${s.dataset_id}`}${v.parent_id === null ? " (since deleted)" : ""}`;
  }
  if (v.parent_version && v.parent_family)
    return baselineLabel(v.parent_family, v.parent_version, false, v.parent_name);
  return "the suggestions";
}
