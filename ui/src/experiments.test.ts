import { describe, expect, it } from "vitest";
import { addBlocked, canEdit, filterExperiments, mlflowExperimentUrl, privacyNote, sourceLabel, sourceOptions } from "./experiments";
import type { Experiment } from "./types";

const me = { id: 2, role: "member" as const };
const lead = { id: 1, role: "lead" as const };
const x = (id: number, over: Partial<Experiment> = {}): Experiment => ({
  id, dataset_id: 1, name: `X${id}`, owner: null, owner_name: null, recipe_id: 10, recipe_version: 1, recipe_family: "tree" as const,
  is_default: false, recipe_phase: "finalized", assessor_id: null, source: "recipe:10", notes: null, created_at: "", archived_at: null,
  private: false, ...over,
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
    expect(privacyNote(x(2, { assessor_id: 2, private: true, recipe_version: 2 }))).toMatch(/until Trees baseline v2 reaches team review/);
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
});

describe("sourceOptions", () => {
  const versions = [
    { id: 10, family: "tree", version: 1, phase: "finalized" as const },
    { id: 12, family: "tree", version: 3, phase: "assessing" as const },
    { id: 11, family: "tree", version: 2, phase: "finalized" as const },
    { id: 13, family: "linear", version: 1, phase: "finalized" as const },
  ];
  const team = (phase: "assessing" | "review" | "finalized", assessor: boolean) =>
    ({ recipe: { id: 12, family: "tree", version: 3, phase }, me: assessor ? { submitted: false } : null });

  it("offers my assessment first while I assess, then finalized versions newest first", () => {
    expect(sourceOptions(versions, team("assessing", true), me)).toEqual([
      { value: "assessment:12:2", label: "My assessment (Trees v3)" },
      { value: "recipe:11", label: "Trees baseline v2 (finalized)" },
      { value: "recipe:10", label: "Trees baseline v1 (finalized)" },
      { value: "recipe:13", label: "Linear baseline v1 (finalized)" },
    ]);
  });

  it("offers only finalized versions to someone who doesn't assess", () => {
    expect(sourceOptions(versions, team("assessing", false), me).map((o) => o.value)).toEqual(["recipe:11", "recipe:10", "recipe:13"]);
    expect(sourceOptions([], undefined, me)).toEqual([]);
  });
});

describe("mlflowExperimentUrl", () => {
  it("filters the dataset's MLflow experiment to this experiment's runs", () => {
    expect(mlflowExperimentUrl(null, 4)).toBeNull();
    const url = mlflowExperimentUrl("7", 4)!;
    expect(url.startsWith("/mlflow/#/experiments/7?searchFilter=")).toBe(true);
    expect(decodeURIComponent(url.split("=").slice(1).join("="))).toBe("tags.`rainforest.experiment_id` = '4'");
  });
});
