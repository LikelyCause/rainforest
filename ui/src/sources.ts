import { fmtInt } from "./format";
import type { CollectionSpec, DatasetSource, FlattenKind, PreviewResult } from "./types";

const KIND_LABEL: Partial<Record<FlattenKind, string>> = {
  mixed: "mixed types, read as text",
  json: "array, stored as JSON text",
  boolean: "yes/no",
};

/** What the pull does with a column of this kind, in the words the preview shows next to it. */
export function flattenKindLabel(kind: FlattenKind): string {
  return KIND_LABEL[kind] ?? kind;
}

/** The estimate against the limit; an estimate exactly at the limit still fits. */
export function budgetMessage(estimated: number, max: number): string {
  return estimated > max
    ? `${fmtInt(estimated)} cells exceeds the limit of ${fmtInt(max)}`
    : `${fmtInt(estimated)} of ${fmtInt(max)} cells`;
}

/** Why `text` can't be a filter or projection, or null when it is blank or a JSON object. */
export function parseJsonDocument(text: string): string | null {
  if (!text.trim()) return null;
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return "Not valid JSON";
  }
  return value !== null && typeof value === "object" && !Array.isArray(value) ? null : "Must be a JSON object";
}

/** "vehicles · curated, mining": the connection and what was taken from it. */
export function sourceSummary(source: Pick<DatasetSource, "connection" | "collections">): string {
  return `${source.connection} · ${source.collections.map((c) => c.label).join(", ")}`;
}

/** What the Connect dialog asks of one chosen collection. */
export interface CollectionPick {
  label: string;
  filter: string;
  projection: string;
}
export const BLANK_PICK: CollectionPick = { label: "", filter: "", projection: "" };

/** Everything the Connect dialog holds; `name` and `dedupe` are null until the user changes the default. */
export interface ConnectDraft {
  connection: string;
  database: string;
  typedDatabase: string;
  /** In the order ticked: the order collections are pulled and duplicates are kept in. */
  chosen: string[];
  picks: Record<string, CollectionPick>;
  name: string | null;
  addSource: boolean;
  dedupe: boolean | null;
  matchOn: string;
}

export type ConnectEdit =
  | { type: "connection"; value: string }
  | { type: "database"; value: string }
  | { type: "typedDatabase"; value: string }
  | { type: "toggle"; name: string }
  | { type: "moveUp"; index: number }
  | { type: "pick"; name: string; patch: Partial<CollectionPick> }
  | { type: "name"; value: string }
  | { type: "addSource"; value: boolean }
  | { type: "dedupe"; value: boolean }
  | { type: "matchOn"; value: string };

export const newDraft = (connection: string): ConnectDraft => ({
  connection,
  database: "",
  typedDatabase: "",
  chosen: [],
  picks: {},
  name: null,
  addSource: false,
  dedupe: null,
  matchOn: "",
});

/** The draft after one edit. Picks and options belong to the collections of one database, so choosing another
 * database or connection starts them over. */
export function editDraft(d: ConnectDraft, e: ConnectEdit): ConnectDraft {
  switch (e.type) {
    case "connection":
      return newDraft(e.value);
    case "database":
      return e.value === d.database
        ? d
        : { ...newDraft(d.connection), typedDatabase: d.typedDatabase, database: e.value };
    case "typedDatabase":
      return { ...d, typedDatabase: e.value };
    case "toggle":
      return {
        ...d,
        chosen: d.chosen.includes(e.name) ? d.chosen.filter((c) => c !== e.name) : [...d.chosen, e.name],
      };
    case "moveUp": {
      if (e.index < 1 || e.index >= d.chosen.length) return d;
      const chosen = [...d.chosen];
      [chosen[e.index - 1], chosen[e.index]] = [chosen[e.index], chosen[e.index - 1]];
      return { ...d, chosen };
    }
    case "pick":
      return { ...d, picks: { ...d.picks, [e.name]: { ...(d.picks[e.name] ?? BLANK_PICK), ...e.patch } } };
    case "name":
      return { ...d, name: e.value };
    case "addSource":
      return { ...d, addSource: e.value };
    case "dedupe":
      return { ...d, dedupe: e.value };
    case "matchOn":
      return { ...d, matchOn: e.value };
  }
}

/** What the preview and the pull send for a collection: blank filters and projections are left out. */
export function specOf(name: string, pick: CollectionPick, withLabel: boolean): CollectionSpec {
  const spec: CollectionSpec = { name };
  if (withLabel && pick.label.trim()) spec.label = pick.label.trim();
  if (pick.filter.trim()) spec.filter = pick.filter.trim();
  if (pick.projection.trim()) spec.projection = pick.projection.trim();
  return spec;
}

export const labelOf = (name: string, pick: CollectionPick | undefined) => pick?.label.trim() || name;

/** The body of a preview of the draft's collections; labels aren't part of it. */
export function previewRequest(d: ConnectDraft): { database: string; collections: CollectionSpec[] } {
  return { database: d.database, collections: d.chosen.map((n) => specOf(n, d.picks[n] ?? BLANK_PICK, false)) };
}

/** Equal for two drafts exactly when their previews would be the same request. */
export const previewKey = (d: ConnectDraft) => JSON.stringify([d.connection, previewRequest(d)]);

/** The columns typed into Match on, trimmed, each once. */
export function matchColumns(text: string): string[] {
  return [
    ...new Set(
      text
        .split(",")
        .map((c) => c.trim())
        .filter(Boolean),
    ),
  ];
}

/** The names that aren't columns of the preview; none before a preview, since there is nothing to check against. */
export function unknownColumns(names: string[], preview: PreviewResult | null): string[] {
  if (!preview) return [];
  const known = new Set(preview.columns.map((c) => c.name));
  return names.filter((n) => !known.has(n));
}

/** "vin, model_year": the columns duplicates will be compared on, null before a preview. The rule the preview
 * endpoint applies: `matchOn` once the preview has every column in it (what Connect then sends), else the preview's
 * default. Worked out here so typing in Match on doesn't need another preview. */
export function comparedOnText(preview: PreviewResult | null, matchOn: string[]): string | null {
  if (!preview?.compared_on) return null;
  const columns = matchOn.length > 0 && unknownColumns(matchOn, preview).length === 0 ? matchOn : preview.compared_on;
  return columns.join(", ") || "no shared columns, so nothing is left out";
}

/** Why Connect can't go ahead yet, in the order the dialog asks for things; null when it can. */
export function connectBlocker(s: {
  chosen: number;
  invalid: boolean;
  preview: PreviewResult | null;
  name: string;
  dedupe: boolean;
  unknownMatch: string[];
}): string | null {
  if (s.chosen === 0) return "Choose at least one collection.";
  if (s.invalid) return "Fix the marked fields first.";
  if (!s.preview) return "Preview the collections first.";
  if (!s.preview.budget.fits) return "Too large to pull: narrow a filter or add a projection.";
  if (!s.name.trim()) return "Name the dataset.";
  if (s.dedupe && s.unknownMatch.length > 0)
    return `Match on names columns the preview doesn't have: ${s.unknownMatch.join(", ")}.`;
  return null;
}
