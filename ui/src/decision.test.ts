import { describe, expect, it } from "vitest";
import { bandsText, defaultTask, parseEdges, retype, taskLabel, withOption } from "./decision";
import type { DecisionInput, OptionCatalog, Suggestion } from "./types";

const NUM = { imputation: ["none", "mean", "median", "constant"], transform: ["none"], scaling: ["none"], encoding: ["none"] };
const NUM_LINEAR = { imputation: ["mean", "median", "constant"], transform: ["none", "log1p"], scaling: ["none", "standard"], encoding: ["none"] };
const CAT = { imputation: ["constant", "most_frequent"], transform: ["none"], scaling: ["none"], encoding: ["ordinal", "onehot", "frequency"] };
const catalog = {
  allowed: {
    tree: { numeric_continuous: NUM, numeric_discrete: NUM, categorical: CAT, binary: { ...CAT, encoding: ["binary"] },
      identifier: { imputation: ["none"], transform: ["none"], scaling: ["none"], encoding: ["none"] } },
    linear: { numeric_continuous: NUM_LINEAR, numeric_discrete: NUM_LINEAR, categorical: { ...CAT, encoding: ["onehot", "frequency"] } },
  },
  families: [{ value: "tree", handles_nan: true }, { value: "linear", handles_nan: false }],
  non_feature_types: ["identifier", "constant"],
  default_params: { onehot: { max_categories: 30, min_frequency: 0.01 }, ordinal: { min_frequency: 0.001 } },
} as unknown as OptionCatalog;

const input = (patch: Partial<DecisionInput>): DecisionInput => ({
  family: "tree", action: "keep", semantic_type: "numeric_continuous", imputation: "none", impute_value: null,
  add_missing_indicator: false, transform: "none", scaling: "none", encoding: "none", params: {}, notes: null,
  reviewed: false, ...patch,
});
const sugg = (patch: Partial<Suggestion>): Suggestion => ({
  action: "keep", semantic_type: "categorical", imputation: "constant", impute_value: "__missing__",
  add_missing_indicator: false, transform: "none", scaling: "none", encoding: "ordinal", params: { min_frequency: 0.001 },
  reasons: [], ...patch,
});

describe("retype", () => {
  it("keeps your fill value between numeric types", () => {
    const d = input({ imputation: "constant", impute_value: "7.5" });
    const out = retype(d, "numeric_discrete", catalog, sugg({}));
    expect(out).toMatchObject({ semantic_type: "numeric_discrete", imputation: "constant", impute_value: "7.5" });
  });

  it("falls back to a valid fill when a numeric constant was blank", () => {
    const out = retype(input({ imputation: "constant", impute_value: "" }), "numeric_discrete", catalog, sugg({}));
    expect(out.impute_value).toBe("0");
  });

  it("drops a label fill when a categorical becomes numeric", () => {
    const d = input({ family: "linear", semantic_type: "categorical", imputation: "constant", impute_value: "__missing__", encoding: "onehot" });
    const out = retype(d, "numeric_continuous", catalog, sugg({}));
    expect(out).toMatchObject({ imputation: "median", impute_value: null, encoding: "none" });
    expect(retype({ ...d, family: "tree" }, "numeric_continuous", catalog, sugg({})).imputation).toBe("none");
  });

  it("keeps a numeric fill that reads as a number when coming from a label type", () => {
    const d = input({ semantic_type: "categorical", imputation: "constant", impute_value: "3", encoding: "ordinal" });
    expect(retype(d, "numeric_continuous", catalog, sugg({}))).toMatchObject({ imputation: "constant", impute_value: "3" });
  });

  it("gives labels a missing label and resets encoder params", () => {
    const out = retype(input({}), "categorical", { ...catalog, allowed: catalog.allowed } as OptionCatalog,
      sugg({ semantic_type: "numeric_continuous" }));
    expect(out).toMatchObject({ imputation: "constant", impute_value: "__missing__", encoding: "ordinal", params: { min_frequency: 0.001 } });
  });

  it("takes the suggestion's settings when retyping back to the suggested type", () => {
    const s = sugg({ encoding: "onehot", params: { max_categories: 12, min_frequency: 0.02 } });
    const out = retype(input({ imputation: "constant", impute_value: "5" }), "categorical", catalog, s);
    expect(out).toMatchObject({ encoding: "onehot", params: s.params, impute_value: "__missing__" });
  });

  it("drops non-feature types and never changes a target", () => {
    expect(retype(input({}), "identifier", catalog, sugg({})).action).toBe("drop");
    expect(retype(input({ action: "target" }), "identifier", catalog, sugg({})).action).toBe("target");
  });

  it("starts a kept transform and the encoding from their default params", () => {
    const linear = { ...catalog.allowed.linear, numeric_discrete: { ...NUM_LINEAR, transform: ["none", "log1p", "power"] } };
    const cat = { ...withSettings, allowed: { ...catalog.allowed, linear } } as unknown as OptionCatalog;
    const d = input({ family: "linear", imputation: "median", transform: "power", params: { k: 9 } });
    expect(retype(d, "numeric_discrete", cat, sugg({})).params).toEqual({ k: 5 });
  });
});

const withSettings = {
  ...catalog,
  settings: {
    onehot: [{ name: "max_categories" }, { name: "min_frequency" }],
    ordinal: [{ name: "min_frequency" }],
    power: [{ name: "k" }],
  },
  default_params: { ...catalog.default_params, power: { k: 5 } },
} as unknown as OptionCatalog;

describe("withOption", () => {
  it("starts a new encoding from its default params", () => {
    const d = input({ semantic_type: "categorical", encoding: "onehot", params: { max_categories: 5, min_frequency: 0.2 } });
    expect(withOption(d, "encoding", "ordinal", withSettings).params).toEqual({ min_frequency: 0.001 });
    expect(withOption(d, "encoding", "frequency", withSettings).params).toEqual({});
  });

  it("starts a new transform from its defaults and keeps the encoding's", () => {
    const d = input({ transform: "none", encoding: "onehot", params: { max_categories: 5, left_over: 1 } });
    const out = withOption(d, "transform", "power", withSettings);
    expect(out.transform).toBe("power");
    expect(out.params).toEqual({ max_categories: 5, k: 5 });
    expect(withOption(input({ transform: "power", params: { k: 9 } }), "transform", "none", withSettings).params).toEqual({});
  });
});

describe("target tasks", () => {
  it("defaults like the API", () => {
    expect(defaultTask("binary", 2)).toBe("classes");
    expect(defaultTask("numeric_continuous", 30)).toBe("regression");
    expect(defaultTask("numeric_discrete", 100)).toBe("classes");
    expect(defaultTask("numeric_discrete", 101)).toBe("regression");
    expect(defaultTask("text", 5)).toBeNull();
  });
  it("labels", () => expect(taskLabel("regression")).toBe("Number"));
  it("describes bands", () => {
    expect(bandsText("bins", { k: 4 })).toBe("Bands · 4 equal-sized");
    expect(bandsText("bins", { edges: [10, 25] })).toBe("Bands · cut at 10, 25");
    expect(bandsText("regression", null)).toBe("Number");
    expect(bandsText(null, null)).toBeNull();
  });
  it("parses cut points", () => {
    expect(parseEdges("10, 25.5,80")).toEqual([10, 25.5, 80]);
    expect(parseEdges("5, 1")).toMatch(/ascending/);
    expect(parseEdges("a")).toMatch(/number/);
  });
});

describe("band count", () => {
  it("accepts whole numbers from 2 to 20 only", async () => {
    const { bandCount } = await import("./decision");
    expect(bandCount("12")).toBe(12);
    expect(bandCount("1")).toBeNull();
    expect(bandCount("21")).toBeNull();
    expect(bandCount("2.5")).toBeNull();
  });
});
