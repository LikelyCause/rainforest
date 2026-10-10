import { describe, expect, it } from "vitest";
import { accessOf, differing, finalizeQuestion, nextOpen, spread } from "./team";
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
    expect(done.note).toMatch(/Unsubmit/);
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
