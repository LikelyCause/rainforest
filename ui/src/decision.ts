import type { Decision, DecisionInput, Dim, OptionCatalog, Suggestion } from "./types";

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
  const encoding = pick("encoding");
  const imputation = pick("imputation");
  const nonFeature = catalog.non_feature_types.includes(type);
  const fill = suggested ? sugg.impute_value ?? null
    : LABELLED.has(type) ? (sameKind && d.impute_value ? d.impute_value : "__missing__")
    : type === "text" ? (sameKind ? d.impute_value ?? "" : "")
    : imputation === "constant" ? (fillCarries && isNumber(d.impute_value) ? d.impute_value : "0")
    : null;
  return {
    ...d,
    semantic_type: type,
    action: d.action === "target" ? "target" : nonFeature ? "drop" : d.action,
    imputation,
    transform: pick("transform"),
    scaling: pick("scaling"),
    encoding,
    impute_value: fill,
    params: suggested ? sugg.params : { ...(catalog.default_params[encoding] ?? {}) },
    add_missing_indicator: NUMERIC.has(type) ? d.add_missing_indicator : false,
  };
}

export interface ParamRule { integer: boolean; min: number; max: number }

/** Why typed text isn't a valid encoder parameter, or null; mirrors the API's check. */
export function paramProblem(text: string, rule: ParamRule | undefined): string | null {
  const n = Number(text);
  if (text.trim() === "" || !Number.isFinite(n)) return "Enter a number";
  if (!rule) return null;
  if (rule.integer && !Number.isInteger(n)) return "Enter a whole number";
  if (n < rule.min || n > rule.max) return `Between ${rule.min} and ${rule.max}`;
  return null;
}

/** Switching encodings starts from the new encoding's default parameters; the old ones don't apply. */
export function withEncoding(d: DecisionInput, encoding: string, catalog: OptionCatalog): DecisionInput {
  return { ...d, encoding, params: { ...(catalog.default_params[encoding] ?? {}) } };
}
