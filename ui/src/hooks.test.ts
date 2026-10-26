import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import { REFRESH_POLL_MS, pollInterval, refetchDatasetsOnLock, refreshesDone } from "./hooks";

describe("refetchDatasetsOnLock", () => {
  it("marks every dataset and the list out of date, so a page learns of the lock, and nothing else", () => {
    const qc = new QueryClient();
    for (const key of [["dataset", 1], ["dataset", 2], ["datasets"], ["fields", 1]]) qc.setQueryData(key, {});
    refetchDatasetsOnLock(qc)();
    const stale = (key: unknown[]) => qc.getQueryState(key)?.isInvalidated;
    expect([stale(["dataset", 1]), stale(["dataset", 2]), stale(["datasets"])]).toEqual([true, true, true]);
    expect(stale(["fields", 1])).toBe(false);
  });
});

describe("refreshesDone", () => {
  it("names each dataset once, when a refresh seen running is over", () => {
    const seen = new Set<number>();
    expect(
      refreshesDone(seen, [
        { id: 5, status: "refreshing" },
        { id: 6, status: "ready" },
      ]),
    ).toEqual([]);
    expect(refreshesDone(seen, [{ id: 5, status: "refreshing" }])).toEqual([]);
    expect(
      refreshesDone(seen, [
        { id: 5, status: "ready" },
        { id: 6, status: "ready" },
      ]),
    ).toEqual([5]);
    expect(refreshesDone(seen, [{ id: 5, status: "ready" }])).toEqual([]);
  });

  it("counts a refreshed dataset that left the list as done", () => {
    const seen = new Set<number>();
    refreshesDone(seen, [{ id: 5, status: "refreshing" }]);
    expect(refreshesDone(seen, [])).toEqual([5]);
  });
});

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
