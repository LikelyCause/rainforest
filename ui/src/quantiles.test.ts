import { describe, expect, it } from "vitest";
import { stripLabels } from "./quantiles";

const box = (p01: number, p50: number, p99: number) => ({ p01, p25: p01, p50, p75: p99, p99 });

describe("stripLabels", () => {
  it("puts the median on its own row, so it never sits on p1 or p99", () => {
    const labels = stripLabels(box(1, 1, 4));
    expect(labels.find((l) => l.key === "median")!.row).toBe("above");
    expect(labels.filter((l) => l.key !== "median").every((l) => l.row === "below")).toBe(true);
  });

  it("keeps a median at either end inside the strip", () => {
    expect(stripLabels(box(1, 1, 4)).find((l) => l.key === "median")!.anchor).toBe("start");
    expect(stripLabels(box(0, 99.5, 100)).find((l) => l.key === "median")!.anchor).toBe("end");
    expect(stripLabels(box(0, 50, 100)).find((l) => l.key === "median")!.anchor).toBe("middle");
  });

  it("places each label at its value", () => {
    const labels = stripLabels(box(0, 25, 100));
    expect(labels.map((l) => [l.key, l.at, l.text])).toEqual([
      ["p1", 0, "p1 0"],
      ["median", 0.25, "median 25"],
      ["p99", 1, "p99 100"],
    ]);
  });
});
