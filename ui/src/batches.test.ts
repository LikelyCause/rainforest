import { describe, expect, it } from "vitest";
import {
  batchProgress,
  batchRequest,
  batchTag,
  brierPair,
  buildSpace,
  capProblem,
  classPlan,
  distinctCombinations,
  defaultRanking,
  describeSampler,
  describeSpace,
  describeThreshold,
  oneVsRestValues,
  parseRange,
  rankingChoices,
  runCount,
  searchChartData,
  sortOvrRows,
  variedSettings,
  weakReasons,
  weightingClash,
} from "./batches";
import type { Batch, BatchRow, ClassCount, LeaderboardEntry, ModelSetting } from "./types";

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
  it("labels a run's batch by its kind and number, keeping its name for the tooltip", () => {
    const batches = [
      { id: 4, kind: "one_vs_rest", name: "LightGBM · one model per class of vehicle" },
      { id: 3, kind: "search", name: "LightGBM · search over learning_rate" },
    ] as Batch[];
    expect(batchTag(batches, 4)).toEqual({ label: "Per class #4", name: "LightGBM · one model per class of vehicle" });
    expect(batchTag(batches, 3)).toEqual({ label: "Search #3", name: "LightGBM · search over learning_rate" });
  });
  it("says only its number until the batches have loaded", () => {
    expect(batchTag(undefined, 4)).toEqual({ label: "Batch #4", name: "Batch #4" });
    expect(batchTag([], 9)).toEqual({ label: "Batch #9", name: "Batch #9" });
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

const row = (o: Partial<BatchRow>): BatchRow => ({
  run_id: 1,
  status: "done",
  status_message: null,
  error: null,
  label: null,
  positive_class: "A",
  retried: false,
  fit_seconds: 1,
  finished_at: null,
  ...o,
});
const scores = (pr_auc: number | null) => ({ pr_auc, roc_auc: null, f1: null });

describe("sortOvrRows", () => {
  const rows = [
    row({ run_id: 1, positive_class: "V10", valid: scores(0.7), positives: { train: 50, valid: 9, test: 40 } }),
    row({ run_id: 2, positive_class: "V2", valid: scores(0.9), positives: { train: 900, valid: 90, test: 300 } }),
    row({ run_id: 3, positive_class: "V3", status: "pending", valid: scores(null), positives: null }),
    row({ run_id: 4, positive_class: "V1", valid: scores(0.4), positives: { train: 20, valid: 4, test: 12 } }),
  ];
  const ids = (r: BatchRow[]) => r.map((x) => x.run_id);

  it("keeps the batch's own order when not sorted", () => {
    expect(ids(sortOvrRows(rows, null, "desc"))).toEqual([1, 2, 3, 4]);
  });
  it("sorts a metric either way, with the rows that lack it last both times", () => {
    expect(ids(sortOvrRows(rows, "valid.pr_auc", "desc"))).toEqual([2, 1, 4, 3]);
    expect(ids(sortOvrRows(rows, "valid.pr_auc", "asc"))).toEqual([4, 1, 2, 3]);
  });
  it("sorts positives by split", () => {
    expect(ids(sortOvrRows(rows, "positives.test", "asc"))).toEqual([4, 1, 2, 3]);
  });
  it("sorts classes as people read numbered names", () => {
    expect(ids(sortOvrRows(rows, "class", "asc"))).toEqual([4, 2, 3, 1]);
  });
  it("puts weak rows first when sorted on weak", () => {
    const flagged = [row({ run_id: 1, weak: false }), row({ run_id: 2, weak: true }), row({ run_id: 3, weak: null })];
    expect(ids(sortOvrRows(flagged, "weak", "desc"))).toEqual([2, 1, 3]);
  });
  it("leaves its input alone", () => {
    sortOvrRows(rows, "valid.pr_auc", "desc");
    expect(ids(rows)).toEqual([1, 2, 3, 4]);
  });
});

describe("weakReasons", () => {
  const rule = { pr_auc: 0.6, test_positives: 30 };
  it("names each limit a weak row is under, with the limit the server used", () => {
    const r = row({ weak: true, valid: scores(0.42), positives: { train: 50, valid: 9, test: 12 } });
    expect(weakReasons(r, rule)).toEqual([
      "Validation PR AUC 0.420 is below 0.6 (OVR_WEAK_PR_AUC)",
      "12 positives in test, fewer than 30",
    ]);
  });
  it("gives no reasons for a row the server didn't flag, whatever its numbers", () => {
    expect(weakReasons(row({ weak: false, valid: scores(0.42) }), rule)).toEqual([]);
    expect(weakReasons(row({ weak: null }), rule)).toEqual([]);
  });
  it("still says why when the numbers on the row don't show it", () => {
    const r = row({ weak: true, valid: scores(0.9), positives: { train: 1, valid: 1, test: 99 } });
    expect(weakReasons(r, rule)).toEqual(["Validation PR AUC below 0.6 or fewer than 30 positives in test"]);
  });
});

describe("brierPair", () => {
  const valid = {
    method: "sigmoid" as const,
    valid_positives: 40,
    valid_negatives: 400,
    brier_before: 0.2,
    brier_after: 0.1,
  };
  it("shows the test pair, since the validation pair is where the calibrator was fit", () => {
    expect(brierPair({ ...valid, test: { brier_before: 0.21, brier_after: 0.12 } })).toEqual({
      split: "test",
      before: 0.21,
      after: 0.12,
    });
  });
  it("falls back to the validation pair when the export has no test rows", () => {
    expect(brierPair(valid)).toEqual({ split: "valid", before: 0.2, after: 0.1 });
    expect(brierPair(null)).toBeNull();
  });
});

describe("oneVsRestValues", () => {
  it("takes the batch's settings, the row's searched values over them, and the run fields", () => {
    const spec = { params: { n_estimators: 200, learning_rate: 0.1 }, seed: 7, cv: 0 } as unknown as Batch["spec"];
    expect(oneVsRestValues(spec, { learning_rate: 0.0312, num_leaves: 63 })).toEqual({
      n_estimators: 200,
      learning_rate: 0.0312,
      num_leaves: 63,
      seed: 7,
      cv: 0,
    });
  });
});

describe("describeSpace and describeSampler", () => {
  const space = {
    learning_rate: { low: 0.0123456, high: 0.3, log: true, integer: false },
    num_leaves: [15, 31, 63],
    boosting_type: ["gbdt", "dart"],
  };
  it("names each axis with its values or range, numbers shortened as settings are", () => {
    expect(describeSpace(space, catalog)).toEqual([
      "Rate from 0.01235 to 0.3, log scale",
      "Leaves 15, 31, 63",
      "Boosting GBDT, DART",
    ]);
    expect(describeSpace({ max_depth: { low: 2, high: 9, log: false, integer: true } }, null)).toEqual([
      "max_depth from 2 to 9, whole numbers",
    ]);
  });
  it("says how the combinations were chosen", () => {
    const grid = { num_leaves: [15, 31], boosting_type: ["gbdt", "dart"] };
    expect(describeSampler({ kind: "grid", n: null, seed: 0 }, grid)).toBe("Grid · 4 combinations");
    expect(describeSampler({ kind: "random", n: 20, seed: 3 }, space)).toBe("Random · 20 draws · seed 3");
  });
});

const entry = (rank: number | null, value: number | null, settings: LeaderboardEntry["settings"]) => ({
  rank,
  value,
  settings,
  run_ids: [rank ?? 0],
  classes: {},
  cv_std: null,
  fit_seconds: 1,
});

describe("variedSettings", () => {
  it("lists the searched settings that take more than one value, in the space's order", () => {
    const entries = [
      entry(1, 0.9, { learning_rate: 0.1, num_leaves: 31, boosting_type: "gbdt" }),
      entry(2, 0.8, { learning_rate: 0.2, num_leaves: 31, boosting_type: "dart" }),
    ];
    expect(variedSettings(entries, ["num_leaves", "learning_rate", "boosting_type"])).toEqual([
      "learning_rate",
      "boosting_type",
    ]);
  });
});

describe("searchChartData", () => {
  const entries = [
    entry(2, 0.8, { learning_rate: 0.2, boosting_type: "dart", num_leaves: 31 }),
    entry(1, 0.9, { learning_rate: 0.05, boosting_type: "gbdt", num_leaves: 63 }),
    entry(null, null, { learning_rate: 0.1, boosting_type: "gbdt", num_leaves: 15 }),
  ];
  const space = {
    learning_rate: { low: 0.01, high: 0.3, log: true, integer: false },
    boosting_type: ["gbdt", "dart"],
    num_leaves: [15, 31, 63],
  };
  it("plots the ranked combinations against one setting, on a log axis when it is searched on one", () => {
    const d = searchChartData(entries, "learning_rate", null, space, catalog);
    expect(d.x).toEqual({ name: "learning_rate", label: "Rate", numeric: true, log: true, categories: [] });
    expect(d.y).toBeNull();
    expect(d.points).toEqual([
      { x: 0.2, y: null, value: 0.8, rank: 2, settings: entries[0].settings },
      { x: 0.05, y: null, value: 0.9, rank: 1, settings: entries[1].settings },
    ]);
  });
  it("puts a choice on a category axis in the catalogue's order, by label", () => {
    const d = searchChartData(entries, "boosting_type", "num_leaves", space, catalog);
    expect(d.x).toMatchObject({ numeric: false, log: false, categories: ["GBDT", "DART"] });
    expect(d.y).toMatchObject({ name: "num_leaves", numeric: true, log: false });
    expect(d.points.map((p) => [p.x, p.y])).toEqual([
      ["DART", 31],
      ["GBDT", 63],
    ]);
  });
  it("keeps a log axis off when a value isn't above 0", () => {
    const d = searchChartData([entry(1, 0.9, { learning_rate: 0 })], "learning_rate", null, space, catalog);
    expect(d.x.log).toBe(false);
  });
});

describe("describeThreshold", () => {
  it("calls a one-vs-rest run's threshold the highest F1 on validation", () => {
    expect(describeThreshold(0.31, null)).toBe("0.31, the highest F1 on validation");
    expect(describeThreshold(0.3, null)).toBe("0.30, the highest F1 on validation");
  });
  it("calls it a fallback when the run has a note, since no threshold had a positive F1", () => {
    expect(describeThreshold(0.5, "No threshold gives a positive F1 on validation; 0.5 is used")).toBe(
      "0.50 (fallback)",
    );
  });
});
