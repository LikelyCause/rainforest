import type { DerivedField, DerivedInput } from "./types";

type Group = { label: string; values?: string[]; ranges?: [number, number][] };

/** One group per line, "Label: 390-459, 785, V45": a numeric a-b is a range, anything else an exact value. */
export function parseGroups(text: string): { groups: Group[]; error: string | null } {
  const groups: Group[] = [];
  for (const [i, raw] of text.split("\n").entries()) {
    const line = raw.trim();
    if (!line) continue;
    const at = line.lastIndexOf(":");
    if (at <= 0) return { groups, error: `Line ${i + 1}: write it as "Label: values"` };
    const g: Group = { label: line.slice(0, at).trim() };
    for (const tok of line.slice(at + 1).split(",").map((t) => t.trim()).filter(Boolean)) {
      const m = tok.match(/^(-?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(-?\d+(?:\.\d+)?)$/);
      if (m) (g.ranges ??= []).push([Number(m[1]), Number(m[2])]);
      else (g.values ??= []).push(tok);
    }
    if (!g.values && !g.ranges) return { groups, error: `Line ${i + 1}: ${g.label} has no values` };
    groups.push(g);
  }
  return { groups, error: groups.length ? null : "Add at least one group" };
}

export const groupsText = (groups: Group[]) =>
  groups.map((g) => `${g.label}: ${[...(g.ranges ?? []).map(([a, b]) => `${a}-${b}`), ...(g.values ?? [])].join(", ")}`).join("\n");

export function describeDerived(f: DerivedField | DerivedInput): string {
  if (f.kind === "formula") return f.spec.expr;
  if (f.kind === "count") return `count of ${f.spec.values.join(" / ")} across ${f.spec.fields.length} field${f.spec.fields.length === 1 ? "" : "s"}`;
  return `${f.spec.source} in ${f.spec.groups.length} group${f.spec.groups.length === 1 ? "" : "s"}, else ${f.spec.other || "Other"}`;
}
