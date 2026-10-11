import { describe, expect, it } from "vitest";
import { ACTIONS, can } from "./can";

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
