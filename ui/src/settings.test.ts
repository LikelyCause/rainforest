import { describe, expect, it } from "vitest";
import { changed, isVisible, restore, runFields, settingProblem } from "./settings";
import type { ModelSetting } from "./types";

const S = (o: Partial<ModelSetting>): ModelSetting => ({ name: "x", label: "X", help: "", group: "trees", kind: "int", default: 1,
  min: 1, max: 10, step: null, log: false, options: null, applies: "both", when: null, none_value: null, none_label: null, ...o });
const settings = [
  S({ name: "seed", group: "run", default: 42, min: 0, max: 2 ** 31 - 1 }),
  S({ name: "cv", group: "run", kind: "choice", default: 5, options: [{ value: 0, label: "Off" }, { value: 5, label: "5" }] }),
  S({ name: "num_leaves", default: 63, min: 2, max: 4096 }),
  S({ name: "max_depth", kind: "int_or_none", default: -1, min: 1, max: 64, none_value: -1, none_label: "No limit" }),
  S({ name: "boosting_type", kind: "choice", default: "gbdt", options: [{ value: "gbdt", label: "GBDT" }, { value: "dart", label: "DART" }] }),
  S({ name: "drop_rate", kind: "float", default: 0.1, min: 0, max: 1, group: "advanced", when: { boosting_type: ["dart"] } }),
];

describe("settings", () => {
  it("checks values", () => {
    expect(settingProblem(settings[2], 1)).toMatch(/2/);
    expect(settingProblem(settings[2], 2.5)).toMatch(/whole/);
    expect(settingProblem(settings[3], -1)).toBeNull();
    expect(settingProblem(settings[3], 0)).toMatch(/No limit/);
    expect(settingProblem(settings[4], "nope")).toMatch(/one of/);
  });
  it("hides settings their condition doesn't meet", () => {
    expect(isVisible(settings[5], {}, settings)).toBe(false);
    expect(isVisible(settings[5], { boosting_type: "dart" }, settings)).toBe(true);
  });
  it("visibleValues drops hidden settings and defaults from what is sent", () => {
    expect(changed({ num_leaves: 63, drop_rate: 0.3, seed: 7 }, settings)).toEqual({});
    expect(changed({ num_leaves: 127, boosting_type: "dart", drop_rate: 0.3 }, settings))
      .toEqual({ num_leaves: 127, boosting_type: "dart", drop_rate: 0.3 });
    expect(changed({ max_depth: -1 }, settings)).toEqual({});
  });
  it("restore drops unknown and invalid values", () => {
    expect(restore({ num_leaves: 127, gone: 1, max_depth: 999, seed: 3 }, settings)).toEqual({ num_leaves: 127, seed: 3 });
    expect(restore("junk", settings)).toEqual({});
  });
  it("reads the run fields with their defaults", () => {
    expect(runFields({}, settings)).toEqual({ seed: 42, cv: 5 });
    expect(runFields({ seed: 7, cv: 0 }, settings)).toEqual({ seed: 7, cv: 0 });
  });
});
