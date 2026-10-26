import { describe, expect, it } from "vitest";
import {
  lastRowRules,
  rowsSentence,
  splitCounts,
  splitPreview,
  addBlocked,
  baselineDiff,
  metricLabel,
  canEdit,
  experimentTab,
  filterExperiments,
  finalizedPrompt,
  mlflowExperimentUrl,
  mlflowHome,
  privacyNote,
  revertedInput,
  sourceChip,
  sourceLabel,
  sourceOptions,
} from "./experiments";
import type { Experiment } from "./types";

const me = { id: 2, role: "member" as const };
const lead = { id: 1, role: "lead" as const };
const x = (id: number, over: Partial<Experiment> = {}): Experiment => ({
  id,
  dataset_id: 1,
  name: `X${id}`,
  owner: null,
  owner_name: null,
  recipe_id: 10,
  recipe_version: 1,
  recipe_family: "tree" as const,
  recipe_name: null,
  is_default: false,
  recipe_phase: "finalized",
  assessor_id: null,
  source: "recipe:10",
  notes: null,
  created_at: "",
  archived_at: null,
  private: false,
  ...over,
});

describe("filterExperiments", () => {
  const list = [
    x(1),
    x(2, { owner: 2, assessor_id: 2, private: true }),
    x(3, { owner: 2 }),
    x(4, { owner: 3, assessor_id: 3 }),
    x(5, { owner: 2, archived_at: "2026-10-01" }),
  ];
  const ids = (f: Parameters<typeof filterExperiments>[1]) => filterExperiments(list, f, me).map((e) => e.id);

  it("keeps archived experiments out of every view but their own", () => {
    expect(ids("all")).toEqual([1, 2, 3, 4]);
    expect(ids("archived")).toEqual([5]);
  });

  it("mine is what I own or built from my assessment; team is recipe-sourced", () => {
    expect(ids("mine")).toEqual([2, 3]);
    expect(ids("team")).toEqual([1, 3]);
  });
});

describe("labels", () => {
  it("names the source from the viewer's side", () => {
    expect(sourceLabel(x(1, { recipe_version: 3 }))).toBe("Trees baseline v3");
    expect(sourceLabel(x(2, { assessor_id: 2, owner_name: "Ana" }), me)).toBe("My assessment (Trees v1)");
    expect(sourceLabel(x(2, { assessor_id: 2, owner_name: "Ana" }), lead)).toBe("Ana's assessment (Trees v1)");
  });

  it("explains privacy only for personal experiments", () => {
    expect(privacyNote(x(1))).toBeNull();
    expect(privacyNote(x(2, { assessor_id: 2, private: true, recipe_version: 2 }))).toMatch(
      /until Trees baseline v2 reaches team review/,
    );
    expect(privacyNote(x(2, { assessor_id: 2 }))).toMatch(/Visible to the team/);
  });
});

describe("who may do what", () => {
  it("lets the owner, the assessor or a lead edit", () => {
    expect(canEdit(x(1, { owner: 2 }), me)).toBe(true);
    expect(canEdit(x(1, { assessor_id: 2 }), me)).toBe(true);
    expect(canEdit(x(1, { owner: 3 }), me)).toBe(false);
    expect(canEdit(x(1), lead)).toBe(true);
  });

  it("lets anyone add to a team experiment, only the assessor to a personal one, nobody to an archived one", () => {
    expect(addBlocked(x(1), me)).toBeNull();
    expect(addBlocked(x(1, { assessor_id: 2 }), me)).toBeNull();
    expect(addBlocked(x(1, { assessor_id: 3, owner_name: "Bo" }), lead)).toMatch(/Only Bo/);
    expect(addBlocked(x(1, { archived_at: "t" }), me)).toMatch(/Archived/);
  });

  it("lets nobody change an experiment or add to it while a refresh holds its dataset", () => {
    expect(canEdit(x(1), lead, true)).toBe(false);
    expect(canEdit(x(1, { owner: 2 }), me, true)).toBe(false);
    expect(addBlocked(x(1), lead, true)).toBe(
      "A refresh is running on this dataset: exports and runs wait until it finishes.",
    );
    expect(addBlocked(x(1), me, false)).toBeNull();
  });
});

describe("sourceOptions", () => {
  const versions = [
    { id: 10, family: "tree", version: 1, phase: "finalized" as const },
    { id: 12, family: "tree", version: 3, phase: "assessing" as const },
    { id: 11, family: "tree", version: 2, phase: "finalized" as const },
    { id: 13, family: "linear", version: 1, phase: "finalized" as const },
  ];
  const team = (phase: "assessing" | "review" | "finalized", assessor: boolean) => ({
    recipe: { id: 12, family: "tree", version: 3, phase },
    me: assessor ? { submitted: false } : null,
  });

  it("offers my assessment first while I assess, then finalized versions newest first", () => {
    expect(sourceOptions(versions, team("assessing", true), me)).toEqual([
      { value: "assessment:12:2", label: "My assessment (Trees v3)" },
      { value: "recipe:11", label: "Trees baseline v2 · finalized" },
      { value: "recipe:10", label: "Trees baseline v1 · finalized" },
      { value: "recipe:13", label: "Linear baseline v1 · finalized" },
    ]);
  });

  it("offers only finalized versions to someone who doesn't assess", () => {
    expect(sourceOptions(versions, team("assessing", false), me).map((o) => o.value)).toEqual([
      "recipe:11",
      "recipe:10",
      "recipe:13",
    ]);
    expect(sourceOptions([], undefined, me)).toEqual([]);
  });

  it("puts the dataset's current baseline first once it is finalized", () => {
    const current = { recipe: { id: 13, family: "linear", version: 1, phase: "finalized" as const }, me: null };
    expect(sourceOptions(versions, current, me).map((o) => o.value)).toEqual(["recipe:13", "recipe:11", "recipe:10"]);
  });
});

describe("mlflowHome", () => {
  it("ends in a slash, since nginx serves /mlflow/ only", () => {
    expect(mlflowHome("/mlflow")).toBe("/mlflow/");
    expect(mlflowHome("https://mlflow.example.com")).toBe("https://mlflow.example.com/");
    expect(mlflowHome(null)).toBeNull();
  });
});

describe("mlflowExperimentUrl", () => {
  it("filters the dataset's MLflow experiment to this experiment's runs", () => {
    expect(mlflowExperimentUrl("/mlflow", null, 4)).toBeNull();
    const url = mlflowExperimentUrl("/mlflow", "7", 4)!;
    expect(url.startsWith("/mlflow/#/experiments/7?searchFilter=")).toBe(true);
    expect(decodeURIComponent(url.split("=").slice(1).join("="))).toBe("tags.`rainforest.experiment_id` = '4'");
  });

  it("builds the experiment's MLflow link on the browse base", () => {
    expect(mlflowExperimentUrl("/mlflow", "9", 4)).toBe(
      "/mlflow/#/experiments/9?searchFilter=" + encodeURIComponent("tags.`rainforest.experiment_id` = '4'"),
    );
    expect(mlflowExperimentUrl("https://mlflow.example.com", "9", 4)).toMatch(
      /^https:\/\/mlflow\.example\.com\/#\/experiments\/9/,
    );
    expect(mlflowExperimentUrl(null, "9", 4)).toBeNull();
  });
});

describe("baseline comparison", () => {
  const base = {
    action: "keep" as const,
    semantic_type: "numeric_continuous",
    notes: null,
    settings: {
      imputation: "none",
      impute_value: null,
      add_missing_indicator: false,
      transform: "none",
      scaling: "none",
      encoding: "none",
      params: {},
    },
  };
  const draft = {
    family: "tree",
    action: "drop" as const,
    semantic_type: "numeric_continuous",
    imputation: "median",
    impute_value: null,
    add_missing_indicator: false,
    transform: "none",
    scaling: "none",
    encoding: "none",
    params: {},
    notes: "mine",
    reviewed: true,
  };
  it("lists what an experiment changed, by label", () => {
    expect(baselineDiff(draft, base)).toEqual(["Decision", "Missing values"]);
    expect(baselineDiff(draft, null)).toEqual([]);
  });
  it("reverts a draft to the baseline's decision and settings, keeping the note", () => {
    const r = revertedInput(draft, base);
    expect(r).toMatchObject({ action: "keep", imputation: "none", notes: "mine" });
    expect(baselineDiff(r, base)).toEqual([]);
  });
  it("compares and restores how a target is predicted", () => {
    const tbase = { ...base, action: "target" as const, target_task: "bins" as const, target_bins: { k: 4 } };
    const tdraft = { ...draft, action: "target" as const, target_task: "regression" as const, target_bins: null };
    expect(baselineDiff(tdraft, tbase)).toEqual(["Predict as", "Bands", "Missing values"]);
    expect(revertedInput(tdraft, tbase)).toMatchObject({ target_task: "bins", target_bins: { k: 4 } });
  });
});

describe("metricLabel", () => {
  it("names AUC and R²", () => {
    expect(metricLabel("roc_auc_ovr_macro")).toBe("AUC");
    expect(metricLabel("r2")).toBe("R²");
  });
});

describe("lastRowRules", () => {
  const ex = (id: number, config: Record<string, unknown>) => ({ id, config });
  it("takes the newest export that set row rules, whatever experiment it was in", () => {
    const rules = {
      dedupe: true,
      exclude: [{ field: "d", values: ["11"] }],
      first_per: { key: "p", order: "e" },
      group_by: "p",
    };
    expect(lastRowRules([ex(3, { seed: 42 }), ex(2, rules), ex(1, { dedupe: true })])).toEqual({ from: 2, rules });
  });
  it("fills what an older export didn't record, and is null when none set rules", () => {
    expect(lastRowRules([ex(1, { group_by: "p" })])).toEqual({
      from: 1,
      rules: { dedupe: false, exclude: [], first_per: null, group_by: "p" },
    });
    expect(lastRowRules([ex(1, { dedupe: false, exclude: [] })])).toBeNull();
    expect(lastRowRules([])).toBeNull();
  });
  it("skips failed exports, whose rules may be why they failed", () => {
    expect(
      lastRowRules([
        { id: 3, status: "failed", config: { group_by: "x" } },
        { id: 2, status: "done", config: { dedupe: true } },
      ]),
    ).toEqual({ from: 2, rules: { dedupe: true, exclude: [], first_per: null, group_by: null } });
  });
});

describe("experimentTab", () => {
  it("opens Runs for a missing or unknown tab", () => {
    expect(experimentTab(null)).toBe("runs");
    expect(experimentTab("bogus")).toBe("runs");
    expect(experimentTab("train")).toBe("train");
  });
});

describe("sourceChip", () => {
  it("says a team experiment uses the finalized snapshot and a personal one the assessment as it stands", () => {
    expect(sourceChip(x(1), me)).toMatchObject({ lens: "frozen", text: "From Trees baseline v1 · as finalized" });
    expect(sourceChip(x(2, { assessor_id: 2, owner_name: "Me" }), me).text).toBe(
      "From my assessment of Trees v1 · as it stands",
    );
    expect(sourceChip(x(3, { assessor_id: 3, owner_name: "Dana" }), me)).toMatchObject({
      lens: "private",
      text: "From Dana's assessment of Trees v1 · as it stands",
    });
  });
});

describe("finalizedPrompt", () => {
  const base = { id: 10, phase: "finalized" as const };
  it("follows ?finalized= after finalizing", () => {
    expect(finalizedPrompt([x(1)], base, "12")).toBe(12);
  });
  it("offers the finalized baseline while nothing live is built on it", () => {
    expect(finalizedPrompt([], base, null)).toBe(10);
    expect(finalizedPrompt([x(1, { archived_at: "2026-01-01" })], base, null)).toBe(10);
    expect(finalizedPrompt([x(1)], base, null)).toBeNull();
    expect(finalizedPrompt([], { id: 10, phase: "assessing" }, null)).toBeNull();
  });
});

describe("split preview", () => {
  it("cuts rows as the export does: the held-out share rounds up, then test's part of it", () => {
    expect(splitCounts(69_990, 0.15, 0.15)).toEqual({ train: 48_993, valid: 10_498, test: 10_499 });
    expect(splitCounts(490, 0.2, 0.15)).toEqual({ train: 318, valid: 98, test: 74 });
    expect(splitCounts(100, 0.2, 0)).toEqual({ train: 80, valid: 20, test: 0 });
    expect(splitCounts(100, 0, 0.1)).toEqual({ train: 90, valid: 0, test: 10 });
    expect(splitCounts(100, 0, 0)).toEqual({ train: 100, valid: 0, test: 0 });
  });
  it("says how many rows the row rules leave, and the kept rows without a target", () => {
    const base = {
      rows: 71_236,
      kept: 71_236,
      removed: {},
      missing_target: 0,
      split_rows: 71_236,
      groups: null,
      split: null,
      split_error: null,
    };
    expect(rowsSentence(base)).toBe("71,236 rows");
    expect(rowsSentence({ ...base, kept: 69_990, split_rows: 69_990 })).toBe(
      "69,990 rows after the row rules (1,246 left out)",
    );
    expect(rowsSentence({ ...base, missing_target: 1 })).toBe(
      "71,236 rows; 1 of them has no target value and stays out of the split",
    );
  });
});

describe("splitPreview", () => {
  const r = {
    rows: 1030,
    kept: 991,
    removed: {},
    missing_target: 0,
    split_rows: 991,
    groups: 13,
    split: null,
    split_error: null,
  };
  it("shows the server's exact split, groups included", () => {
    expect(splitPreview({ ...r, split: { train: 843, valid: 88, test: 60 } }, 1030, 0.15, 0.15, true)).toEqual({
      counts: { train: 843, valid: 88, test: 60 },
      about: false,
    });
  });
  it("estimates from the rows to split without it, as about for a grouped split", () => {
    expect(splitPreview(r, 1030, 0.15, 0.15, true)).toEqual({
      counts: { train: 693, valid: 149, test: 149 },
      about: true,
    });
    expect(splitPreview(r, 1030, 0.15, 0.15, false).about).toBe(false);
    expect(splitPreview(undefined, 100, 0.2, 0, false)).toEqual({
      counts: { train: 80, valid: 20, test: 0 },
      about: false,
    });
    expect(splitPreview(undefined, null, 0.2, 0, false)).toEqual({ counts: null, about: false });
  });
});
