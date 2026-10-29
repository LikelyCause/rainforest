import { describe, expect, it } from "vitest";
import {
  buildSpace,
  capProblem,
  distinctCombinations,
  defaultRanking,
  parseRange,
  planClasses,
  rankingChoices,
  runCount,
  searchClassesDefault,
  weightingClash,
  type ClassCount,
} from "./batches";
import type { ModelSetting } from "./types";

// The same counts as api/tests/test_batches_planning.py, so both sides are held to one rule.
const counts = (o: Record<string, [number, number, number]>): ClassCount[] =>
  Object.entries(o).map(([c, [train, valid, test]]) => ({ class: c, train, valid, test }));

describe("planClasses", () => {
  it("trains every class above the minimum", () => {
    expect(planClasses(counts({ A: [50, 10, 10], B: [10, 3, 2] }), null, null, 10)).toEqual({
      trained: ["A", "B"],
      skipped: [],
    });
  });

  it("skips too few training positives with the reason", () => {
    expect(planClasses(counts({ A: [50, 10, 10], B: [9, 3, 2] }), null, null, 10)).toEqual({
      trained: ["A"],
      skipped: [{ class: "B", reason: "fewer than 10 positives in training" }],
    });
  });

  it("skips a class with no validation positives, and every class when the validation split is empty", () => {
    expect(planClasses(counts({ A: [50, 10, 10], B: [40, 0, 5] }), null, null, 10).skipped).toEqual([
      { class: "B", reason: "no validation positives" },
    ]);
    expect(planClasses(counts({ A: [50, 0, 10] }), 0, null, 10)).toEqual({
      trained: [],
      skipped: [{ class: "A", reason: "no validation rows" }],
    });
  });

  it("gives each class one reason, the training minimum before the validation split", () => {
    expect(planClasses(counts({ A: [2, 0, 0] }), 0, null, 10).skipped).toEqual([
      { class: "A", reason: "fewer than 10 positives in training" },
    ]);
  });

  it("lists unticked classes as not selected, ahead of any other reason", () => {
    expect(planClasses(counts({ A: [50, 10, 10], B: [9, 3, 2], C: [60, 12, 12] }), null, ["B", "C"], 10)).toEqual({
      trained: ["C"],
      skipped: [
        { class: "A", reason: "not selected" },
        { class: "B", reason: "fewer than 10 positives in training" },
      ],
    });
    expect(planClasses(counts({ A: [2, 0, 0], B: [60, 12, 12] }), null, ["B"], 10).skipped).toEqual([
      { class: "A", reason: "not selected" },
    ]);
  });

  it("uses the configured minimum", () => {
    expect(planClasses(counts({ A: [50, 10, 10] }), null, null, 200).skipped).toEqual([
      { class: "A", reason: "fewer than 200 positives in training" },
    ]);
  });
});

describe("searchClassesDefault", () => {
  it("is the most frequent, the median and the rarest above the minimum", () => {
    const c = counts({
      V01: [5000, 900, 900],
      V02: [1200, 200, 200],
      V03: [800, 100, 100],
      V04: [300, 50, 50],
      V05: [40, 8, 8],
      V06: [12, 3, 3],
      V07: [4, 1, 1],
    });
    expect(searchClassesDefault(c, null, 10)).toEqual(["V01", "V03", "V06"]);
  });

  it("is distinct when few classes qualify, and skips classes that could not train", () => {
    expect(searchClassesDefault(counts({ A: [50, 10, 10], B: [20, 5, 5] }), null, 10)).toEqual(["A", "B"]);
    expect(searchClassesDefault(counts({ A: [50, 10, 10], B: [2, 1, 1] }), null, 10)).toEqual(["A"]);
    expect(searchClassesDefault(counts({ A: [5, 1, 1] }), null, 10)).toEqual([]);
    expect(searchClassesDefault(counts({ A: [500, 90, 90], B: [400, 0, 80], C: [30, 6, 6] }), null, 10)).toEqual([
      "A",
      "C",
    ]);
  });

  it("breaks a tie in frequency by name, as the API sorts", () => {
    expect(searchClassesDefault(counts({ b: [50, 10, 10], a: [50, 10, 10] }), null, 10)).toEqual(["a", "b"]);
  });
});

const base = {
  help: "",
  group: "trees",
  step: null,
  log: false,
  options: null,
  applies: "both",
  when: null,
  none_value: null,
  none_label: null,
} as const;
const leaves: ModelSetting = {
  ...base,
  name: "num_leaves",
  label: "Leaves",
  kind: "int",
  default: 31,
  min: 2,
  max: 4096,
};
const lr: ModelSetting = {
  ...base,
  name: "learning_rate",
  label: "Rate",
  kind: "float",
  default: 0.1,
  min: 0.001,
  max: 1,
  log: true,
};
const boosting: ModelSetting = {
  ...base,
  name: "boosting_type",
  label: "Boosting",
  kind: "choice",
  default: "gbdt",
  min: null,
  max: null,
  options: [
    { value: "gbdt", label: "GBDT" },
    { value: "dart", label: "DART" },
  ],
};
const catalog = [leaves, lr, boosting];

describe("parseRange", () => {
  it("reads a range with its scale, and an int setting's range is always whole numbers", () => {
    expect(parseRange(lr, "0.01", "0.3", true, false)).toEqual({
      range: { low: 0.01, high: 0.3, log: true, integer: false },
      error: null,
    });
    expect(parseRange(leaves, "8", "256", false, false).range).toEqual({
      low: 8,
      high: 256,
      log: false,
      integer: true,
    });
    expect(parseRange({ ...lr, max: 10 }, "0.5", "3", false, true).range?.integer).toBe(true);
  });

  it("refuses what the API refuses, naming the problem", () => {
    expect(parseRange(lr, "", "0.3", false, false).error).toBe("Give a low and a high number");
    expect(parseRange(lr, "abc", "0.3", false, false).error).toBe("Give a low and a high number");
    expect(parseRange(lr, "0.5", "0.1", false, false).error).toBe("The low is above the high");
    expect(parseRange({ ...lr, min: null }, "0", "0.3", true, false).error).toBe("A log range needs a low above 0");
    expect(parseRange(lr, "0.2", "0.4", false, true).error).toBe("The range holds no whole number");
  });

  it("keeps both ends inside the setting's own limits", () => {
    expect(parseRange(leaves, "1", "64", false, false).error).toBe("Low must be from 2 to 4096");
    expect(parseRange(lr, "0.01", "2", false, false).error).toBe("High must be from 0.001 to 1");
  });

  it("is only for number settings", () => {
    expect(parseRange(boosting, "1", "2", false, false).error).toBe("Boosting takes a list of values, not a range");
  });
});

describe("buildSpace", () => {
  it("turns each axis into a list of values or a range", () => {
    const out = buildSpace(
      [
        { name: "num_leaves", mode: "list", values: ["15", "31", "63"], low: "", high: "", log: false, integer: true },
        { name: "learning_rate", mode: "range", values: [], low: "0.01", high: "0.3", log: true, integer: false },
        {
          name: "boosting_type",
          mode: "list",
          values: ["GBDT", "dart"],
          low: "",
          high: "",
          log: false,
          integer: false,
        },
      ],
      catalog,
    );
    expect(out).toEqual({
      space: {
        num_leaves: [15, 31, 63],
        learning_rate: { low: 0.01, high: 0.3, log: true, integer: false },
        boosting_type: ["gbdt", "dart"],
      },
      error: null,
    });
  });

  it("names the setting whose values are wrong, and needs at least one setting", () => {
    expect(buildSpace([], catalog).error).toBe("Choose a setting to search over");
    expect(
      buildSpace(
        [{ name: "num_leaves", mode: "list", values: ["15", "1"], low: "", high: "", log: false, integer: true }],
        catalog,
      ).error,
    ).toBe("Leaves: 1: must be from 2 to 4096");
    expect(
      buildSpace(
        [{ name: "learning_rate", mode: "range", values: [], low: "0.3", high: "0.1", log: false, integer: false }],
        catalog,
      ).error,
    ).toBe("Rate: The low is above the high");
  });
});

describe("distinctCombinations", () => {
  it("counts what a space can hold: lists and whole-number ranges are finite, other ranges are not", () => {
    expect(distinctCombinations({ a: [1, 2], b: ["x", "y", "z"] })).toBe(6);
    expect(distinctCombinations({ a: [1, 2], d: { low: 2, high: 4, log: false, integer: true } })).toBe(6);
    expect(distinctCombinations({ d: { low: 1.5, high: 3.2, log: true, integer: true } })).toBe(2);
    expect(distinctCombinations({ a: [1, 2], r: { low: 0, high: 1, log: false, integer: false } })).toBe(Infinity);
  });
});

describe("runCount", () => {
  const space = { a: [1, 2, 3], b: ["x", "y"] };
  it("is a run per trained class for one-vs-rest", () => {
    expect(runCount({ kind: "one_vs_rest", classes: 7 })).toEqual({ runs: 7, error: null });
  });

  it("is a grid's combinations, times the classes of a one-vs-rest search", () => {
    expect(runCount({ kind: "search", space, sampler: { kind: "grid" } })).toEqual({ runs: 6, error: null });
    expect(runCount({ kind: "search", space, sampler: { kind: "grid" }, classes: 3 })).toEqual({
      runs: 18,
      error: null,
    });
  });

  it("is n draws, times the classes, for random", () => {
    const range = { r: { low: 0, high: 1, log: false, integer: false } };
    expect(runCount({ kind: "search", space: range, sampler: { kind: "random", n: 20 }, classes: 3 }).runs).toBe(60);
  });

  it("refuses a range in a grid and more draws than the space holds", () => {
    const range = { r: { low: 0, high: 1, log: false, integer: false } };
    expect(runCount({ kind: "search", space: range, sampler: { kind: "grid" } }).error).toBe(
      "A grid takes lists of values; use random sampling for a range",
    );
    expect(runCount({ kind: "search", space, sampler: { kind: "random", n: 7 } }).error).toBe(
      "The settings hold only 6 distinct combinations; draw at most 6",
    );
    expect(runCount({ kind: "search", space, sampler: { kind: "random", n: 0 } }).error).toBe(
      "Draw at least one combination",
    );
  });
});

describe("capProblem", () => {
  it("says the count against the cap only above it, worded as the API's 422", () => {
    expect(capProblem(500, 500)).toBeNull();
    expect(capProblem(1200, 500)).toBe(
      "This batch would start 1,200 runs; at most 500 may run in one (BATCH_MAX_RUNS)",
    );
  });
});

describe("ranking", () => {
  it("defaults to PR-AUC for binary or one-vs-rest, macro-F1 for several classes, R² for a number", () => {
    expect(defaultRanking("binary")).toBe("pr_auc");
    expect(defaultRanking("multiclass")).toBe("f1_macro");
    expect(defaultRanking("regression")).toBe("r2");
    expect(defaultRanking(undefined)).toBe("pr_auc");
  });

  it("offers the metrics the API accepts for the target's kind, the default first", () => {
    const binary = rankingChoices("binary");
    expect(binary[0]).toBe("pr_auc");
    expect(binary).toContain("roc_auc");
    expect(binary).not.toContain("pr_auc_macro");
    const multi = rankingChoices("multiclass");
    expect(multi[0]).toBe("f1_macro");
    expect(multi).toContain("pr_auc_macro");
    expect(multi).not.toContain("pr_auc");
    expect(rankingChoices("regression")).toEqual(["r2", "rmse", "mae", "mape", "band_accuracy", "band_f1_macro"]);
  });
});

describe("weightingClash", () => {
  it("names the class-weight settings that balance would contradict, set or searched", () => {
    expect(weightingClash({ num_leaves: 63 }, [])).toEqual([]);
    expect(weightingClash({ scale_pos_weight: 3 }, [])).toEqual(["scale_pos_weight"]);
    expect(weightingClash({}, ["is_unbalance", "num_leaves"])).toEqual(["is_unbalance"]);
  });
});
