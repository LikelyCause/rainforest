import { describe, expect, it } from "vitest";
import { REFRESH_POLL_MS, pollInterval } from "./hooks";

describe("pollInterval", () => {
  it("polls fast while processing and slower while refreshing", () => {
    expect(pollInterval(["ready", "profiling"])).toBe(1000);
    expect(pollInterval(["ready", "refreshing"])).toBe(REFRESH_POLL_MS);
    expect(pollInterval(["refreshing", "pending"])).toBe(1000);
  });
  it("stops when nothing is in flight", () => {
    expect(pollInterval(["ready", "failed"])).toBe(false);
    expect(pollInterval([])).toBe(false);
  });
});
