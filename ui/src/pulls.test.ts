import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  flattenLines,
  parseReopen,
  pullHeadline,
  pullStarter,
  pullState,
  reopenPath,
  reopenedNames,
  showsPull,
  uncomparedNote,
  versionInProgress,
} from "./pulls";
import type { Dataset, Pull, PullDrift } from "./types";

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

describe("versionInProgress", () => {
  it("is true while a version is being assessed or is in team review", () => {
    expect(versionInProgress({ phase: "assessing" })).toBe(true);
    expect(versionInProgress({ phase: "review" })).toBe(true);
  });
  it("is false once the current version is finalized", () => {
    expect(versionInProgress({ phase: "finalized" })).toBe(false);
    expect(versionInProgress({ phase: null })).toBe(false);
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

describe("pullState", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-25T10:05:00Z"));
  });
  afterEach(() => vi.useRealTimers());

  it("says when the newest pull finished, or that it failed, or that it is running", () => {
    expect(pullState(pull())).toBe("pulled 3 minutes ago");
    expect(pullState(pull({ status: "failed" }))).toBe("last pull failed 3 minutes ago");
    expect(pullState(pull({ status: "failed", finished_at: null }))).toBe("last pull failed 5 minutes ago");
    expect(pullState(pull({ status: "running", finished_at: null }))).toBe("pulling…");
    expect(pullState(null)).toBeNull();
  });
});

describe("pullStarter", () => {
  const running = pull({ id: 13, status: "running", finished_at: null, created_by_name: "Ada" });
  const done = pull({ id: 12, created_by_name: "Bo" });

  it("names who started the pull with that id", () => {
    expect(pullStarter([running, done], 13)).toBe("Ada");
  });

  it("is null when that pull isn't in the list yet, not the newest one listed", () => {
    expect(pullStarter([done], 13)).toBeNull();
    expect(pullStarter(undefined, 13)).toBeNull();
    expect(pullStarter([running], undefined)).toBeNull();
  });
});

describe("uncomparedNote", () => {
  const finalized = (family: "tree" | "linear", id: number | null) => ({ family, latest_finalized_id: id });

  it("says a pull without a comparison predates a finalized baseline", () => {
    expect(uncomparedNote(pull({ drift: null }), [finalized("tree", 4)])).toBe(
      "This pull predates the finalized Trees baseline; refresh to compare.",
    );
    expect(uncomparedNote(pull({ drift: {} }), [finalized("tree", 4), finalized("linear", 6)])).toBe(
      "This pull predates the finalized Trees and Linear baselines; refresh to compare.",
    );
  });

  it("names only the finalized baselines the pull didn't compare with", () => {
    expect(uncomparedNote(pull({ drift: { tree: drift.tree } }), [finalized("tree", 4), finalized("linear", 6)])).toBe(
      "This pull predates the finalized Linear baseline; refresh to compare.",
    );
    expect(uncomparedNote(pull({ drift: { tree: drift.tree } }), [finalized("tree", 4)])).toBeNull();
  });

  it("says there is nothing to compare with before any baseline is finalized", () => {
    expect(uncomparedNote(pull({ drift: null }), [finalized("tree", null)])).toBe(
      "No finalized baseline to compare with yet.",
    );
    expect(uncomparedNote(pull({ drift: {} }), [])).toBe("No finalized baseline to compare with yet.");
  });
});

describe("showsPull", () => {
  const ds = (over: Partial<Pick<Dataset, "source" | "last_pull">> = {}) => ({
    source: { connection: "vehicles" } as Dataset["source"],
    last_pull: pull({ id: 13, finished_at: "2026-10-25T10:02:00Z" }),
    ...over,
  });
  const none = { pull: false, field: false };

  it("opens on a pull that came after the version was finalized", () => {
    expect(showsPull(none, ds(), "2026-10-24T09:00:00Z")).toBe(true);
    expect(showsPull(none, ds({ last_pull: pull({ status: "failed" }) }), "2026-10-24T09:00:00Z")).toBe(true);
  });

  it("opens on the queue for a pull from before finalizing, a running pull or a version still in review", () => {
    expect(showsPull(none, ds(), "2026-10-26T09:00:00Z")).toBe(false);
    expect(showsPull(none, ds({ last_pull: pull({ status: "running", finished_at: null }) }), "2026-10-24")).toBe(
      false,
    );
    expect(showsPull(none, ds(), null)).toBe(false);
  });

  it("follows what the URL asks for", () => {
    expect(showsPull({ pull: true, field: false }, ds(), null)).toBe(true);
    expect(showsPull({ pull: false, field: true }, ds(), "2026-10-24T09:00:00Z")).toBe(false);
  });

  it("is off for a dataset that isn't connected or hasn't pulled", () => {
    expect(showsPull({ pull: true, field: false }, ds({ source: null }), null)).toBe(false);
    expect(showsPull({ pull: true, field: false }, ds({ last_pull: null }), null)).toBe(false);
  });
});
