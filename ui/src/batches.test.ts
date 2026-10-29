import { describe, expect, it } from "vitest";
import {
  batchProgress,
  batchRequest,
  batchTag,
  buildSpace,
  capProblem,
  classPlan,
  distinctCombinations,
  defaultRanking,
  parseRange,
  rankingChoices,
  runCount,
  weightingClash,
} from "./batches";
import type { Batch, ClassCount, ModelSetting } from "./types";

// Each class as GET …/batch-plan gives it: the server's own reason for skipping it, or null.
const plan: ClassCount[] = [
  { class: "A", train: 50, valid: 10, test: 10, reason: null },
  { class: "B", train: 9, valid: 3, test: 2, reason: "fewer than 10 positives in training" },
  { class: "C", train: 60, valid: 12, test: 12, reason: null },
];

describe("classPlan", () => {
  it("trains what the server would, and skips with the server's reason", () => {
    expect(classPlan(plan, null)).toEqual({
      trained: ["A", "C"],
      skipped: [{ class: "B", reason: "fewer than 10 positives in training" }],
    });
  });

  it("lists every unticked class as not selected, ahead of the server's reason", () => {
    expect(classPlan(plan, ["C"])).toEqual({
      trained: ["C"],
      skipped: [
        { class: "A", reason: "not selected" },
        { class: "B", reason: "not selected" },
      ],
    });
  });

  it("keeps the server's reason for a ticked class that can't train", () => {
    expect(classPlan(plan, ["B", "C"]).skipped).toEqual([
      { class: "A", reason: "not selected" },
      { class: "B", reason: "fewer than 10 positives in training" },
    ]);
    expect(classPlan(plan, []).trained).toEqual([]);
  });
});

describe("batchTag", () => {
  it("names a run's batch, or its number until the batches have loaded", () => {
    const batches = [{ id: 4, name: "LightGBM · one model per class of vehicle" }] as Batch[];
    expect(batchTag(batches, 4)).toBe("LightGBM · one model per class of vehicle");
    expect(batchTag(undefined, 4)).toBe("Batch #4");
    expect(batchTag(batches, 9)).toBe("Batch #9");
  });
});

describe("batchRequest", () => {
  const common = {
    name: "  ",
    model: "lightgbm",
    device: "cpu" as const,
    params: { num_leaves: 63 },
    seed: 42,
    cv: 5,
    cvFolds: 0,
    target: "vehicle",
    balance: true,
  };
  const search = {
    space: { learning_rate: { low: 0.01, high: 0.3, log: true, integer: false } },
    sampler: { kind: "random" as const, n: 20, seed: 3 },
    ranking: "pr_auc",
    searchClasses: ["V01", "V04"],
  };

  it("sends a one-vs-rest batch's target, balance and classes, null for all of them", () => {
    expect(batchRequest({ ...common, kind: "one_vs_rest", ovr: true, ticked: null })).toEqual({
      kind: "one_vs_rest",
      name: null,
      model: "lightgbm",
      device: "cpu",
      params: { num_leaves: 63 },
      seed: 42,
      cv: 5,
      cv_folds: 0,
      target: "vehicle",
      balance: true,
      classes: null,
    });
    expect(batchRequest({ ...common, kind: "one_vs_rest", ovr: true, ticked: ["V01"], name: " Mine " })).toMatchObject({
      name: "Mine",
      classes: ["V01"],
    });
  });

  it("sends a whole-target search without a target, balance or classes", () => {
    const body = batchRequest({ ...common, ...search, kind: "search", ovr: false, ticked: null });
    expect(body).toMatchObject({ kind: "search", space: search.space, ranking: "pr_auc" });
    expect(body.sampler).toEqual({ kind: "random", n: 20, seed: 3 });
    for (const k of ["target", "balance", "classes", "search_classes"]) expect(body).not.toHaveProperty(k);
  });

  it("sends a one-vs-rest search's target, balance and search classes, but no classes", () => {
    const body = batchRequest({ ...common, ...search, kind: "search", ovr: true, ticked: ["X"] });
    expect(body).toMatchObject({ target: "vehicle", balance: true, search_classes: ["V01", "V04"] });
    expect(body).not.toHaveProperty("classes");
  });

  it("sends a grid sampler as just its kind", () => {
    const body = batchRequest({
      ...common,
      ...search,
      sampler: { kind: "grid", n: 20, seed: 3 },
      kind: "search",
      ovr: false,
      ticked: null,
    });
    expect(body.sampler).toEqual({ kind: "grid" });
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

describe("batchProgress", () => {
  const batch = (counts: Partial<Batch["counts"]>) =>
    ({ counts: { pending: 0, running: 0, done: 0, failed: 0, cancelled: 0, ...counts } }) as Batch;
  it("counts the ended runs over all of them and says what the rest are doing", () => {
    expect(batchProgress(batch({ pending: 1200, running: 1, done: 10, failed: 2 }))).toEqual({
      ended: 12,
      total: 1213,
      text: "10 of 1,213 done · 2 failed · 1 training · 1,200 waiting",
    });
    expect(batchProgress(batch({ done: 3, cancelled: 4 })).text).toBe("3 of 7 done · 4 cancelled");
  });
});
