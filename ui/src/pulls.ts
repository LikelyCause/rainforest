import { fmtInt } from "./format";
import type { Pull, PullDrift } from "./types";

const count = (n: number, one: string, many = `${one}s`) => `${fmtInt(n)} ${n === 1 ? one : many}`;

/** The fields a pull reopened for one model type, each once, in the order the pull lists them. */
export function reopenedNames(drift: PullDrift | null | undefined, family: string): string[] {
  const names = drift?.[family]?.reopened.map((r) => r.name) ?? [];
  return [...new Set(names)];
}

/** The Baselines page with the New version form open for `family` and these fields chosen. */
export function reopenPath(datasetId: number, family: string, names: string[]): string {
  const q = new URLSearchParams({ new: family });
  if (names.length) q.set("reopen", names.join(","));
  return `/d/${datasetId}/baselines?${q}`;
}

/** The names in a `reopen` query value. */
export function parseReopen(value: string | null): string[] {
  return [...new Set((value ?? "").split(",").filter(Boolean))];
}

export function removedDuplicates(pull: Pull): number {
  return (pull.per_source ?? []).reduce((n, s) => n + (s.removed_duplicates ?? 0), 0);
}

/** "Pull #12 · 750 rows · 2 collections · 50 duplicates removed", or just the outcome when it didn't finish. */
export function pullHeadline(pull: Pull): string {
  const head = `Pull #${pull.id}`;
  if (pull.status !== "done") return `${head} ${pull.status}`;
  const parts = [head];
  if (pull.rows != null) parts.push(count(pull.rows, "row"));
  if (pull.per_source?.length) parts.push(count(pull.per_source.length, "collection"));
  const dupes = removedDuplicates(pull);
  if (dupes > 0) parts.push(`${count(dupes, "duplicate")} removed`);
  return parts.join(" · ");
}

/** What flattening did to the documents, one readable line per kind of note. */
export function flattenLines(notes: Pull["flatten_notes"]): string[] {
  if (!notes) return [];
  const lines: string[] = [];
  if (notes.arrays?.length) lines.push(`Arrays kept as JSON text: ${notes.arrays.join(", ")}`);
  if (notes.mixed?.length) lines.push(`Mixed types read as text: ${notes.mixed.join(", ")}`);
  const renamed = Object.entries(notes.renamed ?? {});
  if (renamed.length) lines.push(`Renamed: ${renamed.map(([from, to]) => `${from} → ${to}`).join(", ")}`);
  if (notes.compared_on?.length) lines.push(`Duplicates compared on: ${notes.compared_on.join(", ")}`);
  return lines;
}
