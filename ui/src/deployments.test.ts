import { describe, expect, it } from "vitest";
import {
  batchDeployment,
  curlExample,
  deployRequest,
  fieldWarnings,
  moreThanHalf,
  ruleText,
  runDeployable,
  runDeployLabel,
  scoreColumns,
  modelThreshold,
  type DeployModel,
} from "./deployments";
import type { BatchDetail, BatchRow, DeploymentModel, ModelRun } from "./types";

const ovr: DeployModel[] = [
  { runId: 1, label: "V01", binary: true },
  { runId: 2, label: "V02", binary: true },
];
const f1 = { kind: "f1" as const, value: "" };

describe("deployRequest", () => {
  it("deploys a one-vs-rest batch by its id alone, with the default name and rule left to the API", () => {
    expect(deployRequest({ name: "  ", batchId: 4, runIds: null, models: ovr, rule: f1, overrides: {} })).toEqual({
      request: { name: null, batch_id: 4 },
      error: null,
    });
  });

  it("names a search batch's chosen rows with the batch, and a single run by itself", () => {
    expect(
      deployRequest({ name: "Pick", batchId: 7, runIds: [2, 1], models: ovr, rule: f1, overrides: {} }).request,
    ).toEqual({ name: "Pick", batch_id: 7, run_ids: [2, 1] });
    expect(
      deployRequest({
        name: "",
        batchId: null,
        runIds: [12],
        models: [{ runId: 12, label: "vehicle", binary: false }],
        rule: f1,
        overrides: {},
      }).request,
    ).toEqual({ name: null, run_ids: [12] });
  });

  it("sends the rule for every model and each model's own rule as numbers", () => {
    const out = deployRequest({
      name: "",
      batchId: 4,
      runIds: null,
      models: ovr,
      rule: { kind: "precision", value: "0.9" },
      overrides: { 2: { kind: "fixed", value: ".35" } },
    });
    expect(out.request).toEqual({
      name: null,
      batch_id: 4,
      threshold_rule: { kind: "precision", min: 0.9 },
      thresholds: { 2: { kind: "fixed", value: 0.35 } },
    });
  });

  it("sends no rule to a deployment without one-vs-rest models, and drops overrides of runs not deployed", () => {
    const single = [{ runId: 12, label: "vehicle", binary: false }];
    expect(
      deployRequest({
        name: "",
        batchId: null,
        runIds: [12],
        models: single,
        rule: { kind: "fixed", value: "0.4" },
        overrides: { 12: { kind: "fixed", value: "0.2" }, 99: { kind: "f1", value: "" } },
      }).request,
    ).toEqual({ name: null, run_ids: [12] });
    expect(
      deployRequest({
        name: "",
        batchId: 4,
        runIds: [1],
        models: [ovr[0]],
        rule: f1,
        overrides: { 2: { kind: "fixed", value: "0.2" } },
      }).request,
    ).toEqual({ name: null, batch_id: 4, run_ids: [1] });
  });

  it("refuses numbers the API would refuse, naming the model for its own rule", () => {
    const run = (rule: { kind: "precision" | "fixed"; value: string }, overrides = {}) =>
      deployRequest({ name: "", batchId: 4, runIds: null, models: ovr, rule, overrides });
    expect(run({ kind: "precision", value: "0" }).error).toBe("Precision must be above 0 and at most 1");
    expect(run({ kind: "precision", value: "1.2" }).error).toBe("Precision must be above 0 and at most 1");
    expect(run({ kind: "precision", value: "" }).error).toBe("Precision must be above 0 and at most 1");
    expect(run({ kind: "fixed", value: "-0.1" }).error).toBe("A fixed threshold must be from 0 to 1");
    expect(run({ kind: "fixed", value: "abc" }).error).toBe("A fixed threshold must be from 0 to 1");
    expect(run({ kind: "fixed", value: "1" }).error).toBeNull();
    expect(run({ kind: "fixed", value: "0.5" }, { 1: { kind: "fixed", value: "2" } }).error).toBe(
      "V01: A fixed threshold must be from 0 to 1",
    );
    expect(run({ kind: "fixed", value: "0.5" }, { 1: { kind: "fixed", value: "2" } }).request).toBeNull();
  });

  it("needs runs or a batch", () => {
    expect(deployRequest({ name: "", batchId: null, runIds: null, models: [], rule: f1, overrides: {} }).error).toBe(
      "Choose the runs to deploy",
    );
  });
});

const row = (run_id: number, cls: string | null, status: BatchRow["status"] = "done"): BatchRow => ({
  run_id,
  status,
  status_message: null,
  error: null,
  label: cls ? `${cls} vs rest` : "LightGBM",
  positive_class: cls,
  retried: false,
  fit_seconds: 1,
  finished_at: null,
});
const batch = (extra: Partial<BatchDetail>): BatchDetail =>
  ({
    id: 4,
    kind: "one_vs_rest",
    name: "Per class",
    status: "done",
    rows: [row(1, "V01"), row(2, "V02"), row(3, "V03", "failed")],
    ...extra,
  }) as BatchDetail;

describe("batchDeployment", () => {
  it("deploys a finished one-vs-rest batch's finished runs when nothing is selected", () => {
    expect(batchDeployment(batch({}), [])).toEqual({
      runIds: null,
      models: [
        { runId: 1, label: "V01", binary: true },
        { runId: 2, label: "V02", binary: true },
      ],
      blocker: null,
    });
  });

  it("deploys the selected rows, in the table's order", () => {
    expect(batchDeployment(batch({}), [2])).toMatchObject({
      runIds: [2],
      models: [{ runId: 2, label: "V02", binary: true }],
      blocker: null,
    });
  });

  it("asks for rows on a search, and waits for a running batch unless rows are chosen", () => {
    const search = batch({ kind: "search", rows: [row(5, null), row(6, null)] });
    expect(batchDeployment(search, []).blocker).toBe("Select the rows to deploy");
    expect(batchDeployment(search, [6])).toMatchObject({
      runIds: [6],
      models: [{ runId: 6, label: "LightGBM", binary: false }],
      blocker: null,
    });
    expect(batchDeployment(batch({ status: "running" }), []).blocker).toBe(
      "The batch is still running: deploy it once it ends, or select its finished runs",
    );
    expect(batchDeployment(batch({ status: "running" }), [1]).blocker).toBeNull();
    expect(batchDeployment(batch({ rows: [row(3, "V03", "failed")] }), []).blocker).toBe(
      "None of the batch's runs finished",
    );
  });
});

describe("runDeployable", () => {
  const run = (extra: Partial<ModelRun>) =>
    ({ status: "done", config: {}, report: { order: ["vehicle"] }, ...extra }) as ModelRun;
  it("offers a finished run with one target, one-vs-rest or not", () => {
    expect(runDeployable(run({}))).toBe(true);
    expect(runDeployable(run({ config: { positive_class: "V01" } }))).toBe(true);
  });
  it("not a chain, nor an unfinished run", () => {
    expect(runDeployable(run({ report: { order: ["vehicle", "speed"] } }))).toBe(false);
    expect(runDeployable(run({ status: "running", report: {} as ModelRun["report"] }))).toBe(false);
    expect(runDeployable(run({ status: "failed" }))).toBe(false);
  });
});

describe("runDeployLabel", () => {
  const run = (extra: Partial<ModelRun>) =>
    ({ id: 12, config: {}, report: { order: ["vehicle"], model_label: "LightGBM" }, ...extra }) as ModelRun;
  it("names a one-vs-rest run by its class, another by its model and what it predicts", () => {
    expect(runDeployLabel(run({ config: { positive_class: "V01" } }))).toBe("V01");
    expect(runDeployLabel(run({}))).toBe("LightGBM, predicting vehicle's class");
    expect(
      runDeployLabel(
        run({ report: { order: ["speed"], model_label: "LightGBM", target_kinds: { speed: "regression" } } }),
      ),
    ).toBe("LightGBM, predicting speed as a number");
    expect(runDeployLabel(run({ report: { order: ["speed"] } }))).toBe("Run 12, predicting speed's class");
  });
});

describe("curlExample", () => {
  const fields = [
    { name: "x1", dtype: "Float64" },
    { name: "region", dtype: "String" },
  ];
  it("posts two of the dataset's records, in the fields' order, to this origin with a key placeholder", () => {
    const text = curlExample("https://rf.example", 12, fields, [
      { region: "north", x1: 1.5, extra: 1 },
      { x1: null, region: "south" },
    ]);
    expect(text).toBe(
      [
        "curl -X POST https://rf.example/api/deployments/12/predict \\",
        '  -H "Authorization: Bearer rf_…" \\',
        '  -H "Content-Type: application/json" \\',
        `  -d '{"records": [`,
        '    {"x1":1.5,"region":"north"},',
        '    {"x1":null,"region":"south"}',
        "  ]}'",
      ].join("\n"),
    );
  });

  it("sends null for what the example lacks, and quotes an apostrophe for the shell", () => {
    const text = curlExample("http://localhost:5173", 3, fields, [{ x1: 2, region: "O'Neil" }]);
    expect(text).toContain(`{"x1":2,"region":"O'\\''Neil"},`);
    expect(text).toContain('{"x1":null,"region":null}');
    expect(curlExample("http://h", 3, fields, null)).toContain('{"x1":null,"region":null},');
  });
});

describe("scoreColumns", () => {
  const labels = (n: number) => Array.from({ length: n }, (_, i) => `C${i + 1}`);
  it("shows every score up to 10, with no toggle", () => {
    expect(scoreColumns(labels(10), false)).toEqual({ shown: labels(10), hidden: 0, toggle: false });
  });
  it("hides them past 10 until Show all scores", () => {
    expect(scoreColumns(labels(11), false)).toEqual({ shown: [], hidden: 11, toggle: true });
    expect(scoreColumns(labels(11), true)).toEqual({ shown: labels(11), hidden: 0, toggle: true });
  });
});

describe("ruleText and modelThreshold", () => {
  it("says each rule in words", () => {
    expect(ruleText({ kind: "f1" })).toBe("best F1 on validation");
    expect(ruleText({ kind: "precision", min: 0.9 })).toBe("the lowest with precision at least 0.9 on validation");
    expect(ruleText({ kind: "fixed", value: 0.35 })).toBe("fixed");
    expect(ruleText({ kind: "precision", min: 0.9 }, true)).toBe(
      "the F1 choice: no threshold reaches precision 0.9 on validation",
    );
    expect(ruleText({ kind: "f1" }, true)).toBe("0.5: no threshold gives a positive F1 on validation");
  });

  const model = (extra: Partial<DeploymentModel>): DeploymentModel => ({
    run_id: 1,
    label: "V01",
    model_label: null,
    kind: "binary",
    target: "vehicle",
    threshold: 0.31,
    threshold_rule: { kind: "f1" },
    ...extra,
  });
  it("gives a binary model's threshold and rule, a multiclass model's top class, a number run's number", () => {
    expect(modelThreshold(model({}))).toEqual({ value: "0.31", rule: "best F1 on validation", fallback: false });
    expect(modelThreshold(model({ threshold: 0.4, threshold_rule: { kind: "fixed", value: 0.4 } }))).toEqual({
      value: "0.40",
      rule: "fixed",
      fallback: false,
    });
    expect(
      modelThreshold(
        model({
          threshold: 0.23,
          threshold_rule: { kind: "precision", min: 0.99 },
          threshold_note: "No threshold reaches precision 0.99 on validation; the F1 choice (0.23) is used",
        }),
      ),
    ).toEqual({
      value: "0.23",
      rule: "the F1 choice: no threshold reaches precision 0.99 on validation",
      fallback: true,
    });
    expect(modelThreshold(model({ kind: "multiclass", threshold: null, threshold_rule: null }))).toEqual({
      value: "Top class",
      rule: null,
      fallback: false,
    });
    expect(modelThreshold(model({ kind: "regression", threshold: null, threshold_rule: null }))).toEqual({
      value: "A number",
      rule: null,
      fallback: false,
    });
  });
});

describe("fieldWarnings", () => {
  it("flags a field where more than half the values failed, exactly half not", () => {
    expect(moreThanHalf({ failed: 6, values: 10 })).toBe(true);
    expect(moreThanHalf({ failed: 5, values: 10 })).toBe(false);
    expect(moreThanHalf({ failed: 1, values: 0 })).toBe(true);
  });

  it("lists the flagged fields first, then by the share that failed", () => {
    expect(
      fieldWarnings({
        a: { failed: 1, values: 100, flagged: false },
        b: { failed: 30, values: 40, flagged: true },
        c: { failed: 20, values: 50, flagged: false },
      }),
    ).toEqual([
      { field: "b", failed: 30, values: 40, share: 0.75, flagged: true },
      { field: "c", failed: 20, values: 50, share: 0.4, flagged: false },
      { field: "a", failed: 1, values: 100, share: 0.01, flagged: false },
    ]);
    expect(fieldWarnings(null)).toEqual([]);
  });
});
