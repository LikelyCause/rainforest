import type { Decision, DecisionInput, Dim, OptionCatalog, Suggestion, TargetBins, TargetTask } from "./types";

export const NUMERIC = new Set(["numeric_continuous", "numeric_discrete"]);
const LABELLED = new Set(["categorical", "binary"]);
const kindOf = (t: string) => (NUMERIC.has(t) ? "numeric" : LABELLED.has(t) ? "labelled" : t);
const isNumber = (v: string | null | undefined) => v != null && v.trim() !== "" && Number.isFinite(Number(v));

export function toInput(d: Decision | Suggestion, family: string): DecisionInput {
  return {
    family,
    action: d.action,
    semantic_type: d.semantic_type,
    imputation: d.imputation,
    impute_value: d.impute_value ?? null,
    add_missing_indicator: d.add_missing_indicator ?? false,
    transform: d.transform,
    scaling: d.scaling,
    encoding: d.encoding,
    params: d.params ?? {},
    target_task: "target_task" in d ? d.target_task ?? null : null,
    target_bins: "target_bins" in d ? d.target_bins ?? null : null,
    notes: "notes" in d ? d.notes : null,
    reviewed: "reviewed" in d ? d.reviewed : false,
  };
}

/**
 * Re-type a draft: the suggestion's settings when retyping to the suggested type, otherwise keep what still
 * applies and fall back to the first valid option. A fill value survives a retype within numerics (or labels);
 * from another kind, constant filling carries over only with a fill that reads as a number.
 */
export function retype(d: DecisionInput, type: string, catalog: OptionCatalog, sugg: Suggestion): DecisionInput {
  const allowed = catalog.allowed[d.family][type];
  const handlesNan = catalog.families.find((f) => f.value === d.family)?.handles_nan ?? true;
  const sameKind = kindOf(d.semantic_type) === kindOf(type);
  const fillCarries = sameKind || (NUMERIC.has(type) && isNumber(d.impute_value));
  const suggested = sugg.semantic_type === type;
  const pick = (dim: Dim) => {
    if (suggested) return sugg[dim];
    const carried = dim !== "imputation" || d.imputation !== "constant" || fillCarries;
    if (allowed[dim].includes(d[dim]) && carried) return d[dim];
    if (dim === "imputation" && NUMERIC.has(type)) return handlesNan ? "none" : "median";
    return allowed[dim][0];
  };
  const transform = pick("transform");
  const encoding = pick("encoding");
  const imputation = pick("imputation");
  const nonFeature = catalog.non_feature_types.includes(type);
  const fill = suggested ? sugg.impute_value ?? null
    : LABELLED.has(type) ? (sameKind && d.impute_value ? d.impute_value : "__missing__")
    : type === "text" ? (sameKind ? d.impute_value ?? "" : "")
    : imputation === "constant" ? (fillCarries && isNumber(d.impute_value) ? d.impute_value : "0")
    : null;
  // A task the new type can't take goes, so the API settles the type's default.
  const keepTask = d.target_task !== null && (d.target_task === "classes" ? type !== "numeric_continuous" : NUMERIC.has(type));
  return {
    ...d,
    semantic_type: type,
    target_task: keepTask ? d.target_task : null,
    target_bins: keepTask ? d.target_bins : null,
    action: d.action === "target" ? "target" : nonFeature ? "drop" : d.action,
    imputation,
    transform,
    scaling: pick("scaling"),
    encoding,
    impute_value: fill,
    params: suggested ? sugg.params : { ...(catalog.default_params[transform] ?? {}), ...(catalog.default_params[encoding] ?? {}) },
    add_missing_indicator: NUMERIC.has(type) ? d.add_missing_indicator : false,
  };
}

/** Choosing another transform or encoding starts it from its default parameters; the other one's stay. */
export function withOption(d: DecisionInput, dim: "transform" | "encoding", value: string, catalog: OptionCatalog): DecisionInput {
  const other = dim === "transform" ? d.encoding : d.transform;
  const names = new Set((catalog.settings[other] ?? []).map((s) => s.name));
  const kept = Object.fromEntries(Object.entries(d.params ?? {}).filter(([k]) => names.has(k)));
  const params = { ...kept, ...(catalog.default_params[value] ?? {}) };
  return dim === "transform" ? { ...d, transform: value, params } : { ...d, encoding: value, params };
}

const MAX_CLASSES = 100;

/** How a target of this type is predicted unless someone chooses otherwise; null when it can't be a target. Mirrors
 * the API's semantics.default_task. */
export function defaultTask(type: string, nUnique: number | null): TargetTask | null {
  if (type === "binary" || type === "categorical") return "classes";
  if (type === "numeric_continuous") return "regression";
  if (type === "numeric_discrete") return (nUnique ?? 0) <= MAX_CLASSES ? "classes" : "regression";
  return null;
}

export const taskLabel = (t: TargetTask) => ({ classes: "Classes", regression: "Number", bins: "Bands" })[t];

/** "Bands · 4 equal-sized", "Bands · cut at 10, 25", "Number"; null for no task. */
export function bandsText(task: TargetTask | null | undefined, bins: TargetBins | null | undefined): string | null {
  if (!task) return null;
  if (task !== "bins" || !bins) return taskLabel(task);
  if ("positive" in bins) return `One vs rest · ${bins.positive.join(" / ")}`;
  return `Bands · ${"k" in bins ? `${bins.k} equal-sized` : `cut at ${bins.edges.join(", ")}`}`;
}

/** Cut points typed as "10, 25, 80": the numbers, or what's wrong with them (the API's rule). */
export function parseEdges(text: string): number[] | string {
  const parts = text.split(",").map((s) => s.trim()).filter(Boolean);
  const nums = parts.map(Number);
  if (!parts.length || nums.some((n) => !Number.isFinite(n))) return "Enter numbers separated by commas";
  if (nums.length > 19) return "At most 19 cut points";
  if (nums.some((n, i) => i > 0 && n <= nums[i - 1])) return "Cut points must be ascending, each different";
  return nums;
}

/** A typed band count: a whole number from 2 to 20, else null (the API's rule). */
export function bandCount(text: string): number | null {
  const k = Number(text);
  return text.trim() !== "" && Number.isInteger(k) && k >= 2 && k <= 20 ? k : null;
}
