import { describe, expect, it } from "vitest";
import { ACTIONS, can, lockedTitle, refreshBlocker } from "./can";

const connected = { source: { connection: "x" }, status: "ready", last_pull: { status: "done" } } as const;

describe("can", () => {
  it("lets a viewer change only their own picture", () => {
    for (const a of ACTIONS) expect(can({ role: "viewer" }, a)).toBe(a === "changeOwnPicture");
  });
  it("keeps lead-only actions for leads", () => {
    expect(can({ role: "member" }, "finalize")).toBe(false);
    expect(can({ role: "member" }, "train")).toBe(true);
    expect(can({ role: "member" }, "renameBaseline")).toBe(false);
    for (const a of ACTIONS) expect(can({ role: "lead" }, a)).toBe(true);
  });
  it("allows nothing before the user is known", () => {
    expect(can(null, "upload")).toBe(false);
  });
});

describe("can while a dataset is locked", () => {
  it("allows only what leaves the dataset alone", () => {
    expect(can({ role: "lead" }, "decide", true)).toBe(false);
    expect(can({ role: "lead" }, "finalize", true)).toBe(false);
    expect(can({ role: "member" }, "export", true)).toBe(false);
    expect(can({ role: "lead" }, "changeOwnPicture", true)).toBe(true);
    expect(can({ role: "lead" }, "querySql", true)).toBe(true);
    expect(can({ role: "lead" }, "decide", false)).toBe(true);
  });
});

describe("refreshBlocker", () => {
  const lead = { role: "lead" } as const;
  it("is null when a lead can refresh a ready connected dataset", () => {
    expect(refreshBlocker(lead, connected)).toBeNull();
    expect(refreshBlocker(lead, { ...connected, last_pull: null })).toBeNull();
  });
  it("says a refresh is running", () => {
    expect(refreshBlocker(lead, { ...connected, status: "refreshing" })).toBe("A refresh is already running");
    expect(refreshBlocker(lead, { ...connected, last_pull: { status: "running" } })).toBe(
      "A refresh is already running",
    );
  });
  it("says a dataset whose connect failed has to be connected again", () => {
    expect(refreshBlocker(lead, { ...connected, status: "failed", last_pull: { status: "failed" } })).toBe(
      "The connect failed, so there is nothing to refresh: delete this dataset and connect it again",
    );
  });
  it("asks to wait while the dataset is still being set up", () => {
    expect(refreshBlocker(lead, { ...connected, status: "profiling" })).toBe("Wait until the dataset is ready");
  });
  it("names who may refresh, and what", () => {
    for (const me of [{ role: "member" } as const, { role: "viewer" } as const, null])
      expect(refreshBlocker(me, connected)).toBe("Only a lead can refresh a dataset");
    expect(refreshBlocker(lead, { ...connected, source: null })).toBe("This dataset isn't connected to a source");
  });
});

describe("lockedTitle", () => {
  it("says why a dataset's row controls are off while it refreshes", () => {
    expect(lockedTitle({ status: "refreshing" })).toBe("A refresh is running; try again once it finishes");
  });
  it("is null otherwise", () => {
    for (const status of ["ready", "profiling", "failed"] as const) expect(lockedTitle({ status })).toBeNull();
  });
});
