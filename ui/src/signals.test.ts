import { describe, expect, it } from "vitest";
import { attentionOf, fieldSignals, LEAK_AT, missingTone } from "./signals";
import type { FieldDetail, FieldListItem, TargetAssoc } from "./types";

const field = (patch: Partial<FieldDetail> = {}): FieldDetail => ({
  id: 1, dataset_id: 1, ordinal: 0, name: "income", db_column: "income", physical_dtype: "Float64",
  inferred_type: "numeric_continuous", inference_confidence: 0.92, inference_reasons: [], is_target: false,
  row_count: 1000, null_count: 0, null_pct: 0, n_unique: 900, unique_pct: 0.9, mean: 1, std: 1, min_value: 0,
  max_value: 10, median: 1, skewness: 0.3, kurtosis: 0, zeros_pct: 0, outlier_pct: 0.01, top_value: null,
  top_value_pct: null, avg_length: null, max_target_assoc: null, leakage_flag: false, active_family: "tree",
  stats: {}, charts: {}, suggestion: {} as FieldDetail["suggestion"], ...patch,
});

const listItem = (patch: Partial<FieldListItem> = {}): FieldListItem => ({
  id: 1, ordinal: 0, name: "f", inferred_type: "numeric_continuous", inference_confidence: 0.9, is_target: false,
  null_pct: 0, n_unique: 50, max_target_assoc: 0.1, leakage: false, action: "keep", semantic_type: "numeric_continuous",
  reviewed: false, source: "suggested", ...patch,
});

const assoc = (name: string, strength: number): TargetAssoc => ({ field_id: 9, name, method: "spearman", value: strength, strength });
const byKey = (signals: ReturnType<typeof fieldSignals>) => Object.fromEntries(signals.map((s) => [s.key, s]));

describe("missingTone", () => {
  it("tones only past the thresholds", () => {
    expect(missingTone(null)).toBeUndefined();
    expect(missingTone(0.19)).toBeUndefined();
    expect(missingTone(0.2)).toBe("warn");
    expect(missingTone(0.5)).toBe("bad");
  });
});

describe("fieldSignals", () => {
  it("names the strongest target and flags leakage", () => {
    const s = byKey(fieldSignals(field(), [assoc("a", 0.4), assoc("b", LEAK_AT)], "numeric_continuous"));
    expect(s.target.label).toBe("Leakage");
    expect(s.target.tone).toBe("bad");
    expect(s.target.value).toContain("→ b");
  });

  it("is amber for a strong but not leaky association, plain below", () => {
    expect(byKey(fieldSignals(field(), [assoc("a", 0.6)], "numeric_continuous")).target.tone).toBe("warn");
    expect(byKey(fieldSignals(field(), [assoc("a", 0.2)], "numeric_continuous")).target.tone).toBeUndefined();
  });

  it("skips the target signal for targets and fields without associations", () => {
    expect(byKey(fieldSignals(field({ is_target: true }), [assoc("a", 0.99)], "binary")).target).toBeUndefined();
    expect(byKey(fieldSignals(field(), [], "numeric_continuous")).target).toBeUndefined();
  });

  it("warns that stats describe the inferred type after a retype", () => {
    const s = byKey(fieldSignals(field(), [], "categorical"));
    expect(s.retyped.tone).toBe("warn");
    expect(s.retyped.value).toBe("numeric continuous");
  });

  it("reports values read as missing, with examples", () => {
    const s = byKey(fieldSignals(field({ stats: { unparseable_count: 3, inf_count: 2, unparseable_examples: ["n/a?"] } }), [], "numeric_continuous"));
    expect(s.coerced.value).toBe("5");
    expect(s.coerced.rule).toContain("n/a?");
  });

  it("calls an all-unique leveled field ID-like and tones skew and confidence", () => {
    const cat = field({ inferred_type: "categorical", n_unique: 1000, unique_pct: 1, row_count: 1000 });
    expect(byKey(fieldSignals(cat, [], "categorical")).levels.label).toBe("ID-like");
    const skewed = byKey(fieldSignals(field({ skewness: 3, inference_confidence: 0.5 }), [], "numeric_continuous"));
    expect(skewed.skew.tone).toBe("warn");
    expect(skewed.confidence.tone).toBe("warn");
  });
});

describe("attentionOf", () => {
  it("ranks leak over missing over confidence over unreviewed edits", () => {
    expect(attentionOf(listItem({ leakage: true, null_pct: 0.9 }))).toBe("leak");
    expect(attentionOf(listItem({ max_target_assoc: LEAK_AT }))).toBe("leak");
    expect(attentionOf(listItem({ null_pct: 0.5, inference_confidence: 0.1 }))).toBe("missing");
    expect(attentionOf(listItem({ n_unique: 1 }))).toBe("missing");
    expect(attentionOf(listItem({ inference_confidence: 0.6, source: "user" }))).toBe("confidence");
    expect(attentionOf(listItem({ source: "user" }))).toBe("edited");
    expect(attentionOf(listItem({ source: "user", reviewed: true }))).toBeNull();
  });

  it("never flags a target", () => {
    expect(attentionOf(listItem({ action: "target", leakage: true }))).toBeNull();
  });
});
