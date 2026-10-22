import { describe, expect, it } from "vitest";
import { parseExploreParams } from "./params";

const parse = (qs: string, targets = ["column_a", "column_b"]) => parseExploreParams(new URLSearchParams(qs), targets);

describe("parseExploreParams", () => {
  it("defaults to raw data, the first target and a 5k sample", () => {
    expect(parse("")).toEqual({ data: "raw", isRaw: true, kept: false, target: "column_a", sample: 5000 });
  });

  it("reads an export, a known target and a listed sample size", () => {
    expect(parse("data=export:12&target=column_b&sample=20000")).toEqual({
      data: "export:12",
      isRaw: false,
      kept: false,
      target: "column_b",
      sample: 20000,
    });
  });

  it("applies Kept only to raw data", () => {
    expect(parse("kept=1").kept).toBe(true);
    expect(parse("data=export:3&kept=1").kept).toBe(false);
  });

  it("replaces values the API would refuse", () => {
    expect(parse("data=export:abc").data).toBe("raw");
    expect(parse("data=raw;drop").data).toBe("raw");
    expect(parse("sample=7000").sample).toBe(5000);
    expect(parse("target=gone").target).toBe("column_a");
    expect(parse("target=column_a", []).target).toBeNull();
  });
});
