import { fmtInt } from "./format";
import type { DatasetSource, FlattenKind } from "./types";

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
