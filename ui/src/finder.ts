import { LEAK_AT } from "./signals";
import type { FieldFilter } from "./types";

/** Rows the Bulk edit table shows; the bulk actions themselves apply to every match on the server. */
export const QUERY_LIMIT = 500;

export interface Preset { key: string; label: string; filters: FieldFilter[]; sort?: string }
export const PRESETS: Preset[] = [
  { key: "unreviewed", label: "Unreviewed", filters: [{ column: "reviewed", op: "=", value: "false" }] },
  { key: "missing", label: "Missing > 20%", filters: [{ column: "null_pct", op: ">", value: 0.2 }], sort: "-null_pct" },
  { key: "skewed", label: "Skewed numerics", filters: [{ column: "skewness", op: ">", value: 2 }], sort: "-skewness" },
  { key: "target", label: "Target-associated", filters: [{ column: "max_target_assoc", op: ">=", value: 0.3 }, { column: "is_target", op: "=", value: "false" }], sort: "-max_target_assoc" },
  { key: "highcard", label: "High-card categoricals", filters: [{ column: "inferred_type", op: "in", value: ["categorical"] }, { column: "n_unique", op: ">", value: 50 }], sort: "-n_unique" },
  { key: "text", label: "Text", filters: [{ column: "inferred_type", op: "in", value: ["text"] }] },
  { key: "leaks", label: "Possible leaks", filters: [{ column: "max_target_assoc", op: ">=", value: LEAK_AT }, { column: "is_target", op: "=", value: "false" }], sort: "-max_target_assoc" },
  { key: "edited", label: "Edited by you", filters: [{ column: "decision_source", op: "in", value: ["user"] }] },
];

export interface FinderState { tab: "fields" | "sql"; preset: string | null; modified: boolean; filters: FieldFilter[]; sort: string }

/** Finder state from the URL: explicit filters (`f`) win over a preset; an unreadable `f` falls back to it. */
export function finderState(params: URLSearchParams): FinderState {
  const preset = params.get("preset");
  const raw = params.get("f");
  let filters: FieldFilter[] | null = null;
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) filters = parsed;
    } catch { /* fall back to the preset */ }
  }
  const named = PRESETS.find((p) => p.key === preset);
  return {
    tab: params.get("tab") === "sql" ? "sql" : "fields",
    preset: preset ?? (raw ? null : "unreviewed"),
    modified: Boolean(raw),
    filters: filters ?? (named ?? PRESETS[0]).filters,
    sort: params.get("sort") ?? named?.sort ?? "ordinal",
  };
}

/** Filters that are ready to send: a value is set (or the operator needs none). */
export const completeFilters = (filters: FieldFilter[]) => filters.filter((f) =>
  f.op === "is_null" || f.op === "not_null"
  || (f.value !== undefined && f.value !== "" && !(Array.isArray(f.value) && !f.value.length)));

/**
 * What a bulk action will touch. Past QUERY_LIMIT matches the table shows only the first rows, but the server
 * resolves the same filters, so the action covers every match except the rows you unticked.
 */
export function selectionLabel(shown: number, excluded: number, truncated: boolean): string {
  if (!truncated) return `${shown - excluded} selected`;
  return `all matches${excluded ? ` except ${excluded}` : ""} (more than ${QUERY_LIMIT})`;
}
