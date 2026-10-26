import { describe, expect, it } from "vitest";
import { ACTIONS, can, canRefresh } from "./can";

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

describe("canRefresh", () => {
  it("lets a lead refresh a ready connected dataset", () => {
    expect(canRefresh({ role: "lead" }, connected)).toBe(true);
    expect(canRefresh({ role: "lead" }, { ...connected, last_pull: null })).toBe(true);
  });
  it("is for leads only", () => {
    expect(canRefresh({ role: "member" }, connected)).toBe(false);
    expect(canRefresh({ role: "viewer" }, connected)).toBe(false);
    expect(canRefresh(null, connected)).toBe(false);
  });
  it("is off while refreshing, while a pull runs or when the dataset isn't ready", () => {
    expect(canRefresh({ role: "lead" }, { ...connected, status: "refreshing" })).toBe(false);
    expect(canRefresh({ role: "lead" }, { ...connected, last_pull: { status: "running" } })).toBe(false);
    expect(canRefresh({ role: "lead" }, { ...connected, status: "profiling" })).toBe(false);
  });
  it("is off for a dataset that isn't connected", () => {
    expect(canRefresh({ role: "lead" }, { ...connected, source: null })).toBe(false);
  });
});
