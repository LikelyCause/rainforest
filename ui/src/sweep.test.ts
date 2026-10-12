import { describe, expect, it } from "vitest";
import { addSweepValues, combinations, parseSweep, readSweepValue, SWEEP_MAX } from "./sweep";
import type { ModelSetting } from "./types";

const base = { help: "", group: "trees", step: null, log: false, options: null, applies: "both", when: null, none_value: null, none_label: null } as const;
const leaves: ModelSetting = { ...base, name: "num_leaves", label: "Leaves", kind: "int", default: 31, min: 2, max: 4096 };
const lr: ModelSetting = { ...base, name: "learning_rate", label: "Rate", kind: "float", default: 0.1, min: 0.001, max: 1 };
const depth: ModelSetting = { ...base, name: "max_depth", label: "Depth", kind: "int_or_none", default: -1, min: 1, max: 64, none_value: -1, none_label: "no limit" };
const boost: ModelSetting = { ...base, name: "boosting_type", label: "Boosting", kind: "choice", default: "gbdt", min: null, max: null,
  options: [{ value: "gbdt", label: "GBDT" }, { value: "dart", label: "DART" }] };

describe("parseSweep", () => {
  it("reads numbers, dropping repeats", () => {
    expect(parseSweep(leaves, "15, 31, 63, 31")).toEqual({ values: [15, 31, 63], error: null });
    expect(parseSweep(lr, "0.01 0.05,0.1")).toEqual({ values: [0.01, 0.05, 0.1], error: null });
  });
  it("names the value that breaks the setting's rule", () => {
    expect(parseSweep(leaves, "15, 1").error).toMatch(/1:.*from 2/);
    expect(parseSweep(leaves, "15, 2.5").error).toMatch(/whole number/);
    expect(parseSweep(leaves, "abc").error).toMatch(/number/);
  });
  it("takes the none label and choice labels", () => {
    expect(parseSweep(depth, "4, no limit").values).toEqual([4, -1]);
    expect(parseSweep(boost, "dart, GBDT").values).toEqual(["dart", "gbdt"]);
  });
  it("wants two or more values", () => {
    expect(parseSweep(leaves, "15").error).toMatch(/two/);
  });
});

describe("combinations", () => {
  it("crosses every axis over the base values", () => {
    const out = combinations({ seed: 1 }, [{ name: "a", values: [1, 2] }, { name: "b", values: ["x", "y"] }]);
    expect(out).toEqual([{ seed: 1, a: 1, b: "x" }, { seed: 1, a: 1, b: "y" }, { seed: 1, a: 2, b: "x" }, { seed: 1, a: 2, b: "y" }]);
  });
  it("is capped", () => {
    expect(SWEEP_MAX).toBe(50);
  });
});

describe("value lists", () => {
  it("adds typed values one per entry, skipping repeats", () => {
    expect(addSweepValues(leaves, ["15"], "31 63, 15")).toEqual(["15", "31", "63"]);
    expect(addSweepValues(depth, ["4"], "no limit, 8")).toEqual(["4", "no limit", "8"]);
    expect(addSweepValues(boost, ["gbdt"], "GBDT, DART")).toEqual(["gbdt", "DART"]);
  });
  it("reads each value with its problem", () => {
    expect(readSweepValue(leaves, "1").problem).toMatch(/from 2/);
    expect(readSweepValue(depth, "no limit")).toEqual({ value: -1, problem: null });
  });
  it("joins back into what parseSweep reads", () => {
    const list = addSweepValues(depth, [], "4, no limit");
    expect(parseSweep(depth, list.join(", ")).values).toEqual([4, -1]);
  });
});
