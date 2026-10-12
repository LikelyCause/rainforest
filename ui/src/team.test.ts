import { describe, expect, it } from "vitest";
import { accessOf, differing, finalizeQuestion, nextOpen, oddOnes, shortNames, splitAgreement, spread, submittedText, valueGroups, voteLine } from "./team";
import type { Assessment } from "./types";

const lead = { id: 1, role: "lead" as const };
const member = { id: 2, role: "member" as const };
const ds = (phase: "assessing" | "review" | "finalized") => ({ phase, baseline: { label: "Trees baseline v2" } });
const mine = (submitted: boolean) => ({ me: { submitted, reviewed: 3, total: 5 } });

describe("accessOf", () => {
  it("lets an assessor edit their own assessment until they submit", () => {
    expect(accessOf(ds("assessing"), member, mine(false))).toMatchObject({ source: "mine", canEdit: true });
    const done = accessOf(ds("assessing"), member, mine(true));
    expect(done).toMatchObject({ source: "mine", canEdit: false });
    expect(done.note).toMatch(/Unsubmit it on the Baseline tab/);
  });

  it("shows everyone else the suggestions, read-only, while assessing (leads too)", () => {
    for (const who of [lead, member]) {
      const a = accessOf(ds("assessing"), who, { me: null });
      expect(a).toMatchObject({ source: "team", canEdit: false });
      expect(a.note).toMatch(/Trees baseline v2 is being assessed/);
    }
  });

  it("gives the team decision to leads in review and freezes it once finalized", () => {
    expect(accessOf(ds("review"), lead, undefined)).toMatchObject({ source: "team", canEdit: true });
    expect(accessOf(ds("review"), member, undefined)).toMatchObject({ source: "team", canEdit: false });
    expect(accessOf(ds("finalized"), lead, undefined)).toMatchObject({ canEdit: false });
  });

  it("names the places by their new names", () => {
    const notes = [
      accessOf(ds("assessing"), member, mine(true)).note, accessOf(ds("review"), member, undefined).note,
      accessOf(ds("finalized"), lead, undefined).note,
      accessOf({ phase: "assessing", baseline: { label: "Trees baseline v1", setup: true } }, member, undefined).note,
    ];
    for (const n of notes) expect(n).not.toMatch(/Team page|Baselines page/);
    expect(notes[3]).toMatch(/Baseline tab/);
  });
});

const a = (user_id: number, action: Assessment["action"], semantic_type = "categorical"): Assessment =>
  ({ user_id, user_name: `U${user_id}`, action, semantic_type, notes: null });

describe("spread", () => {
  it("counts actions, most common first", () => {
    expect(spread({ assessments: [a(1, "drop"), a(2, "keep"), a(3, "drop")] })).toBe("2 drop · 1 keep");
  });
  it("names the type when the assessors disagree on it", () => {
    expect(spread({ assessments: [a(1, "keep", "binary"), a(2, "keep", "categorical")] }, (t) => t.toUpperCase()))
      .toBe("1 keep as BINARY · 1 keep as CATEGORICAL");
  });
  it("names how a target is predicted when the assessors disagree on it", () => {
    const t = (id: number, target_task: "regression" | "bins") =>
      ({ ...a(id, "target", "numeric_continuous"), target_task, target_bins: target_task === "bins" ? { k: 3 } : null });
    expect(spread({ assessments: [t(1, "regression"), t(2, "bins")] })).toBe("1 target as Bands · 3 equal-sized · 1 target as Number");
  });
});

describe("differing", () => {
  it("marks keys whose values differ, treating missing as null", () => {
    const rows = [{ x: 1, y: "a", z: null }, { x: 1, y: "b", z: undefined }];
    expect([...differing(rows, ["x", "y", "z"])]).toEqual(["y"]);
    expect(differing([{ p: { a: 1 } }, { p: { a: 1 } }], ["p"]).size).toBe(0);
  });
});

describe("finalizeQuestion", () => {
  it("asks only when disagreements are open", () => {
    expect(finalizeQuestion(0, "Trees baseline v1")).toBeNull();
    expect(finalizeQuestion(1, "Trees baseline v3")).toMatch(/^1 disagreement is still open\. Finalize Trees baseline v3 anyway, keeping its current team decision\?$/);
    expect(finalizeQuestion(2, "Trees baseline v3")).toMatch(/2 disagreements are .* their current team decisions/);
  });
});

describe("nextOpen", () => {
  const q = [{ field_id: 1, resolved: false }, { field_id: 2, resolved: true }, { field_id: 3, resolved: false }];
  it("moves to the next open item, wrapping, and never back to the same one", () => {
    expect(nextOpen(q, 1)).toBe(3);
    expect(nextOpen(q, 3)).toBe(1);
    expect(nextOpen([{ field_id: 1, resolved: false }], 1)).toBeNull();
    expect(nextOpen(q, null)).toBe(1);
  });
});

describe("valueGroups and oddOnes", () => {
  it("groups assessors by value, most common first", () => {
    expect(valueGroups(["a", "b", "a"], ["Maya", "Priya", "Dana"])).toEqual([
      { value: "a", who: ["Maya", "Dana"], at: [0, 2] }, { value: "b", who: ["Priya"], at: [1] },
    ]);
  });
  it("marks only those outside a strict majority", () => {
    expect([...oddOnes(["a", "b", "a"])]).toEqual([1]);
    expect(oddOnes(["a", "b"]).size).toBe(0);
    expect(oddOnes(["a", "a"]).size).toBe(0);
  });
});

describe("voteLine", () => {
  const names = ["Maya Chen", "Dana Okafor", "Priya Raman"];
  it("says when everyone agrees", () => {
    expect(voteLine(["keep", "keep", "keep"], names, "keep")).toBe("All 3 chose keep");
    expect(voteLine(["keep", "keep"], names.slice(0, 2))).toBe("Both chose keep");
  });
  it("names who chose what, and the value where it isn't the current one", () => {
    expect(voteLine(["category", "category", "number"], names, "category")).toBe("2 Maya Chen, Dana Okafor · 1 Priya Raman: number");
    expect(voteLine(["category", "category", "number"], names, "number")).toBe("2 Maya Chen, Dana Okafor: category · 1 Priya Raman");
  });
  it("has nothing to say when nobody set it", () => {
    expect(voteLine([null, undefined], names.slice(0, 2))).toBeNull();
    expect(voteLine([null, "none"], names.slice(0, 2), "none")).toBe("Dana Okafor chose none");
  });
});

describe("shortNames", () => {
  it("keeps full names, shortening only long ones to first name and initial", () => {
    expect(shortNames(["Admin", "Test Member", "Alexandra Montgomery"])).toEqual(["Admin", "Test Member", "Alexandra M."]);
  });
  it("keeps full names where the short ones would look the same", () => {
    expect(shortNames(["Alexandra Montgomery", "Alexandra Mendelssohn"])).toEqual(["Alexandra Montgomery", "Alexandra Mendelssohn"]);
  });
});

describe("splitAgreement", () => {
  it("separates differing settings from shared ones", () => {
    const rows = [{ k: "type", values: ["a", "b"] }, { k: "scale", values: ["x", "x"] }, { k: "flag", values: [null, undefined] }];
    const { differ, agree } = splitAgreement(rows);
    expect(differ.map((r) => r.k)).toEqual(["type"]);
    expect(agree.map((r) => r.k)).toEqual(["scale", "flag"]);
  });
  it("compares a setting only across the assessors in `among`", () => {
    // Priya dropped the field: her "—" doesn't make the scaling the others chose differ.
    const rows = [{ k: "scale", values: ["x", "x", null], among: [0, 1] }, { k: "enc", values: ["a", "b", null], among: [0, 1] },
      { k: "none", values: [null, null, null], among: [] }];
    const { differ, agree } = splitAgreement(rows);
    expect(differ.map((r) => r.k)).toEqual(["enc"]);
    expect(agree.map((r) => r.k)).toEqual(["scale"]);
  });
});

describe("submittedText", () => {
  const p = (name: string, done: boolean, reviewed = 37) => ({ name, submitted_at: done ? "2026-10-10T00:00:00Z" : null, reviewed, total: 37 });
  it("says who review waits on", () => {
    expect(submittedText([p("Dana", true), p("Maya", true), p("Priya", false, 21)]))
      .toBe("2 of 3 are in. Team review opens when Priya (21/37) submits.");
    expect(submittedText([p("Dana", true), p("Maya", false, 3), p("Priya", false, 21), p("Ann", false, 0)]))
      .toBe("1 of 4 is in. Team review opens when 3 more people submit.");
  });
});
