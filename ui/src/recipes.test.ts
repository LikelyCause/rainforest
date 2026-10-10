import { describe, expect, it } from "vitest";
import { fmtChange, fmtValue, pickable, reasonDetail, reuseGroups, startLabel } from "./recipes";
import type { FieldListItem, ReuseSummary } from "./types";

const summary: ReuseSummary = {
  source: { recipe_id: 4, dataset_id: 2, dataset_name: "Churn", version: 3 },
  carried: ["c", "y"],
  reopened: [
    { name: "x", reason: "type_changed", before: "numeric_continuous", after: "categorical" },
    { name: "z", reason: "drift", drift: { null_pct: [0, 0.335], n_unique: [100, 250] } },
    { name: "w2", reason: "new" },
    { name: "n", reason: "new" },
  ],
  missing: ["w"],
  targets: ["y"],
  targets_not_carried: [],
};

describe("reuseGroups", () => {
  it("groups reopened fields by reason in a fixed order, keeping field order within each", () => {
    const g = reuseGroups(summary);
    expect(g.reopened.map((x) => [x.reason, x.fields.map((f) => f.name)])).toEqual([
      ["new", ["w2", "n"]], ["type_changed", ["x"]], ["drift", ["z"]],
    ]);
    expect(g.carried).toEqual(["c", "y"]);
    expect(g.missing).toEqual(["w"]);
    expect(g.needsAssessing).toBe(true);
  });

  it("says when nothing needs assessing", () => {
    expect(reuseGroups({ ...summary, reopened: [] }).needsAssessing).toBe(false);
  });
});

describe("reasonDetail", () => {
  it("shows the numbers behind drift", () => {
    expect(reasonDetail(summary.reopened[1])).toBe("missing 0.0% → 33.5% (+33.5 pts) · distinct values 100 → 250 (×2.50)");
    expect(reasonDetail({ name: "q", reason: "drift", drift: { null_pct: [0.4, 0.2] } })).toBe("missing 40.0% → 20.0% (−20.0 pts)");
  });

  it("names the type change and passes problems through", () => {
    expect(reasonDetail(summary.reopened[0])).toBe("numeric continuous → categorical");
    expect(reasonDetail({ name: "k", reason: "target", problem: "110 distinct values" })).toBe("110 distinct values");
  });
});

describe("fmtChange", () => {
  it("formats decision and per-family setting changes", () => {
    expect(fmtChange({ key: "action", family: null, before: "keep", after: "drop" })).toBe("Decision: keep → drop");
    expect(fmtChange({ key: "imputation", family: "tree", before: "none", after: "mean" }, () => "Trees"))
      .toBe("Trees · Missing values: none → mean");
    expect(fmtChange({ key: "semantic_type", family: null, before: "numeric_continuous", after: "numeric_discrete" }))
      .toBe("Type: numeric continuous → numeric discrete");
  });

  it("formats empty, boolean and object values", () => {
    expect(fmtValue(null)).toBe("—");
    expect(fmtValue(true)).toBe("yes");
    expect(fmtValue({})).toBe("—");
    expect(fmtValue({ max_categories: 30 })).toBe("max_categories=30");
  });
});

const field = (name: string, extra: Partial<FieldListItem> = {}): FieldListItem => ({
  id: name.length, ordinal: 0, name, inferred_type: "numeric_continuous", inference_confidence: 0.9, is_target: false,
  null_pct: 0, n_unique: 10, max_target_assoc: 0.1, leakage: false, action: "keep", semantic_type: "numeric_continuous",
  reviewed: true, source: "suggested", ...extra,
});

describe("pickable", () => {
  const fields = [field("age"), field("city", { semantic_type: "categorical" }), field("gappy", { null_pct: 0.8 })];
  it("filters by name, type and attention", () => {
    const none = { search: "", types: new Set<string>(), attention: false };
    expect(pickable(fields, none)).toHaveLength(3);
    expect(pickable(fields, { ...none, search: "CI" }).map((f) => f.name)).toEqual(["city"]);
    expect(pickable(fields, { ...none, types: new Set(["categorical"]) }).map((f) => f.name)).toEqual(["city"]);
    expect(pickable(fields, { ...none, attention: true }).map((f) => f.name)).toEqual(["gappy"]);
  });
});

describe("startLabel", () => {
  it("says whether the version is assessed or edited directly", () => {
    expect(startLabel(2, 0)).toBe("Start v2 and edit directly");
    expect(startLabel(3, 1)).toBe("Start v3, reopening 1 field");
    expect(startLabel(3, 4)).toBe("Start v3, reopening 4 fields");
  });
});
