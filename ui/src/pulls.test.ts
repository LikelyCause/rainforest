import { describe, expect, it } from "vitest";
import { flattenLines, parseReopen, pullHeadline, reopenPath, reopenedNames } from "./pulls";
import type { Pull, PullDrift } from "./types";

const pull = (over: Partial<Pull> = {}): Pull => ({
  id: 12,
  status: "done",
  started_at: "2026-10-25T10:00:00Z",
  finished_at: "2026-10-25T10:02:00Z",
  rows: 750,
  has_drift: false,
  per_source: [
    { label: "curated", documents: 500, removed_duplicates: 0 },
    { label: "mining", documents: 300, removed_duplicates: 50 },
  ],
  flatten_notes: null,
  drift: null,
  error: null,
  created_by_name: "Ada",
  ...over,
});

const drift: PullDrift = {
  tree: {
    carried: ["a"],
    reopened: [
      { name: "speed", reason: "drift" },
      { name: "colour", reason: "new" },
      { name: "speed", reason: "type_changed" },
    ],
    missing: [],
  },
  linear: { carried: [], reopened: [], missing: ["gone"] },
};

describe("reopenedNames", () => {
  it("lists a family's reopened fields once each, in the order given", () => {
    expect(reopenedNames(drift, "tree")).toEqual(["speed", "colour"]);
  });
  it("is empty for a family with nothing reopened, an absent family or no drift", () => {
    expect(reopenedNames(drift, "linear")).toEqual([]);
    expect(reopenedNames(drift, "neural")).toEqual([]);
    expect(reopenedNames(null, "tree")).toEqual([]);
  });
});

describe("reopenPath and parseReopen", () => {
  it("round-trips the names through the URL", () => {
    const path = reopenPath(7, "tree", ["speed", "colour code"]);
    expect(path.startsWith("/d/7/baselines?new=tree&reopen=")).toBe(true);
    const q = new URLSearchParams(path.split("?")[1]);
    expect(parseReopen(q.get("reopen"))).toEqual(["speed", "colour code"]);
  });
  it("leaves reopen out when nothing is reopened and ignores blanks", () => {
    expect(reopenPath(7, "linear", [])).toBe("/d/7/baselines?new=linear");
    expect(parseReopen(null)).toEqual([]);
    expect(parseReopen("a,,b,a")).toEqual(["a", "b"]);
  });
});

describe("pullHeadline", () => {
  it("names the pull, its rows, collections and the duplicates removed", () => {
    expect(pullHeadline(pull())).toBe("Pull #12 · 750 rows · 2 collections · 50 duplicates removed");
  });
  it("leaves out what is zero and uses the singular", () => {
    const p = pull({ rows: 1, per_source: [{ label: "a", documents: 1, removed_duplicates: 1 }] });
    expect(pullHeadline(p)).toBe("Pull #12 · 1 row · 1 collection · 1 duplicate removed");
    expect(pullHeadline(pull({ per_source: [{ label: "a", documents: 5, removed_duplicates: 0 }] }))).toBe(
      "Pull #12 · 750 rows · 1 collection",
    );
  });
  it("says failed or running without counts", () => {
    expect(pullHeadline(pull({ id: 13, status: "failed", rows: null }))).toBe("Pull #13 failed");
    expect(pullHeadline(pull({ id: 14, status: "running", rows: null }))).toBe("Pull #14 running");
  });
});

describe("flattenLines", () => {
  it("turns the notes into readable lines", () => {
    expect(
      flattenLines({ arrays: ["tags"], mixed: ["speed"], renamed: { "a.b": "a_b" }, compared_on: ["vin"] }),
    ).toEqual([
      "Arrays kept as JSON text: tags",
      "Mixed types read as text: speed",
      "Renamed: a.b → a_b",
      "Duplicates compared on: vin",
    ]);
  });
  it("is empty without notes", () => {
    expect(flattenLines(null)).toEqual([]);
    expect(flattenLines({})).toEqual([]);
  });
});
