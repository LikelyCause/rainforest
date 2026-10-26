import { familyShort, fmtAgo, fmtInt } from "./format";
import type { BaselineSummary, Dataset, Pull, PullDrift, PullSummary } from "./types";

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

/** A version is being assessed or is in team review, so no new one can start until it is finalized. */
export const versionInProgress = (ds: Pick<Dataset, "phase">) => ds.phase === "assessing" || ds.phase === "review";

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

type PullTimes = Pick<PullSummary, "status" | "started_at" | "finished_at">;

/** "pulled 3 minutes ago", "last pull failed 3 minutes ago" or "pulling…"; null before any pull. */
export function pullState(last: PullTimes | null): string | null {
  if (!last) return null;
  if (last.status === "running") return "pulling…";
  if (last.status === "failed") return `last pull failed ${fmtAgo(last.finished_at ?? last.started_at)}`;
  return last.finished_at ? `pulled ${fmtAgo(last.finished_at)}` : null;
}

/** Who started pull `id`. A list fetched before that pull began doesn't have it, and its newest entry is someone
 * else's pull, so that gives null rather than a name. */
export function pullStarter(pulls: Pull[] | undefined, id: number | undefined): string | null {
  return pulls?.find((p) => p.id === id)?.created_by_name ?? null;
}

const andList = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);

/** What to say about the finalized baselines a finished pull wasn't compared with: a connect pull is never compared,
 * and a refresh compares only the baselines finalized when it ran. Null when it covers every one of them. */
export function uncomparedNote(
  pull: Pick<Pull, "drift">,
  baselines: Pick<BaselineSummary, "family" | "latest_finalized_id">[],
): string | null {
  const finalized = baselines.filter((b) => b.latest_finalized_id !== null).map((b) => b.family);
  const missing = finalized.filter((f) => !pull.drift?.[f]);
  if (finalized.length === 0)
    return Object.keys(pull.drift ?? {}).length ? null : "No finalized baseline to compare with yet.";
  if (missing.length === 0) return null;
  const names = andList(missing.map(familyShort));
  return `This pull predates the finalized ${names} baseline${missing.length === 1 ? "" : "s"}; refresh to compare.`;
}

/** Whether the Team page shows the latest pull rather than a field: when the URL asks for it, or, with no field
 * asked for, when that pull finished after the version was finalized (it is what a lead acts on next). */
export function showsPull(
  asked: { pull: boolean; field: boolean },
  ds: Pick<Dataset, "source" | "last_pull">,
  finalizedAt: string | null,
): boolean {
  const last = ds.last_pull;
  if (!ds.source || !last) return false;
  if (asked.pull) return true;
  if (asked.field || last.status === "running" || !last.finished_at || !finalizedAt) return false;
  return new Date(last.finished_at).getTime() > new Date(finalizedAt).getTime();
}
