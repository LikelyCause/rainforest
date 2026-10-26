import { describe, expect, it } from "vitest";
import { eligibleTarget, setupNote, startBaselineDefaults } from "./baselines";
import { baselineLabel } from "./format";

const me = { id: 7 };
const f = (name: string, over: Record<string, unknown> = {}) =>
  ({ id: name.length, name, action: "keep", semantic_type: "categorical", n_unique: 3, ...over }) as any;

describe("baselineLabel", () => {
  it("names a baseline by its model type and version", () => {
    expect(baselineLabel("tree", 2)).toBe("Trees baseline v2");
    expect(baselineLabel("neural", 1, true)).toBe("Neural v1");
  });
  it("puts a named baseline's name first, with its model type in the long form", () => {
    expect(baselineLabel("tree", 2, true, "Dropout risk")).toBe("Dropout risk v2");
    expect(baselineLabel("tree", 2, false, "Dropout risk")).toBe("Dropout risk v2 (Trees)");
    expect(baselineLabel("tree", 2, false, null)).toBe("Trees baseline v2");
  });
});

describe("startBaselineDefaults", () => {
  it("offers the first type without a baseline and populates from the latest finalized one", () => {
    const ds = {
      baseline: { setup: false },
      baselines: [{ family: "tree", id: 12, version: 2, phase: "finalized", latest_finalized_id: 12 }],
    } as any;
    expect(startBaselineDefaults(ds, me, [f("y", { action: "target" }), f("c")])).toEqual({
      family: "linear",
      start: "baseline",
      source_recipe_id: 12,
      targets: ["y"],
      assessors: [7],
      name: "",
    });
  });
  it("sets up a new dataset from scratch, as Trees", () => {
    const ds = {
      baseline: { setup: true },
      baselines: [{ family: "tree", id: 3, version: 1, phase: "assessing", latest_finalized_id: null }],
    } as any;
    expect(startBaselineDefaults(ds, me, [])).toMatchObject({
      family: "tree",
      start: "scratch",
      source_recipe_id: null,
    });
  });
});

describe("eligibleTarget", () => {
  it("allows class-like fields with at most 100 values", () => {
    expect(eligibleTarget(f("c"))).toBeNull();
    expect(eligibleTarget(f("x", { semantic_type: "numeric_continuous", n_unique: 5000 }))).toBeNull();
    expect(eligibleTarget(f("g", { semantic_type: "numeric_discrete", n_unique: 5000 }))).toBeNull();
    expect(eligibleTarget(f("t", { semantic_type: "text" }))).toMatch(/text/i);
    expect(eligibleTarget(f("id", { semantic_type: "categorical", n_unique: 5000 }))).toMatch(/100/);
  });
});

describe("setupNote", () => {
  it("gives a lead the setup form", () => {
    expect(setupNote({ role: "lead" }, false)).toBeNull();
  });

  it("holds the form back from a lead while a refresh runs", () => {
    expect(setupNote({ role: "lead" }, true)).toBe(
      "A refresh is running. Set up the baseline once it finishes, so its targets come from the new pull's fields.",
    );
  });

  it("tells anyone else that a lead sets it up", () => {
    for (const role of ["member", "viewer"] as const)
      expect(setupNote({ role }, false)).toMatch(/^A lead sets up this dataset's first baseline/);
    expect(setupNote(undefined, false)).toMatch(/^A lead sets up/);
  });
});
