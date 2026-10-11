// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";
import { applyThemePref, getThemePref, setThemePref } from "./theme";
import { visibleSections } from "./pages/SettingsPage";

describe("theme preference", () => {
  beforeEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset.theme;
  });

  it("follows the system until someone picks a scheme", () => {
    expect(getThemePref()).toBe("system");
    applyThemePref();
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });

  it("pins the chosen scheme and remembers it", () => {
    setThemePref("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(getThemePref()).toBe("dark");
    delete document.documentElement.dataset.theme;
    applyThemePref();
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("going back to system unpins it and forgets the choice", () => {
    setThemePref("light");
    setThemePref("system");
    expect(document.documentElement.dataset.theme).toBeUndefined();
    expect(localStorage.getItem("rainforest.theme")).toBeNull();
  });

  it("ignores a stored value it doesn't know", () => {
    localStorage.setItem("rainforest.theme", "sepia");
    expect(getThemePref()).toBe("system");
  });
});

describe("settings sections", () => {
  it("shows Team to leads only", () => {
    expect(visibleSections({ role: "lead" }).map((s) => s.key)).toContain("team");
    for (const role of ["member", "viewer"] as const) {
      expect(visibleSections({ role }).map((s) => s.key)).toEqual(["profile", "appearance", "integrations", "application"]);
    }
  });
});
