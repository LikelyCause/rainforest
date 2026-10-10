import { describe, expect, it } from "vitest";
import { parseView, viewKey, withView } from "./view";

describe("view", () => {
  it("parses experiment and baseline views", () => {
    expect(parseView("experiment:12")).toEqual({ kind: "experiment", id: 12 });
    expect(parseView("baseline:3")).toEqual({ kind: "baseline", id: 3 });
    expect(parseView("x")).toBeNull();
    expect(parseView(null)).toBeNull();
  });
  it("adds the view to a path", () => {
    expect(withView("/d/1/review/5", "experiment:12")).toBe("/d/1/review/5?view=experiment%3A12");
    expect(withView("/d/1/review?x=1", "experiment:12")).toBe("/d/1/review?x=1&view=experiment%3A12");
    expect(withView("/d/1/review", null)).toBe("/d/1/review");
  });
  it("names the default view in query keys", () => {
    expect(viewKey(null)).toBe("current");
    expect(viewKey("experiment:4")).toBe("experiment:4");
  });
});
