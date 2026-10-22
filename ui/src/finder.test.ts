import { describe, expect, it } from "vitest";
import { completeFilters, finderState, PRESETS, QUERY_LIMIT, selectionLabel } from "./finder";

const state = (qs: string) => finderState(new URLSearchParams(qs));

describe("finderState", () => {
  it("opens on the Unreviewed preset", () => {
    expect(state("")).toMatchObject({
      tab: "fields",
      preset: "unreviewed",
      modified: false,
      sort: "ordinal",
      filters: PRESETS[0].filters,
    });
  });

  it("uses a preset's filters and default sort", () => {
    const s = state("preset=missing");
    expect(s.filters).toEqual([{ column: "null_pct", op: ">", value: 0.2 }]);
    expect(s.sort).toBe("-null_pct");
    expect(state("preset=missing&sort=name").sort).toBe("name");
  });

  it("prefers explicit filters and marks them modified", () => {
    const f = [{ column: "n_unique", op: ">", value: 9 }];
    const s = state(`preset=text&f=${encodeURIComponent(JSON.stringify(f))}`);
    expect(s).toMatchObject({ preset: "text", modified: true, filters: f });
    expect(state(`f=${encodeURIComponent(JSON.stringify(f))}`).preset).toBeNull();
  });

  it("falls back to the preset when the filter param is unreadable", () => {
    expect(state("preset=text&f=%7Bnot-json").filters).toEqual(PRESETS.find((p) => p.key === "text")!.filters);
    expect(state("preset=text&f=%7B%7D").filters).toEqual(PRESETS.find((p) => p.key === "text")!.filters);
  });

  it("reads the SQL tab and ignores unknown tabs", () => {
    expect(state("tab=sql").tab).toBe("sql");
    expect(state("tab=nope").tab).toBe("fields");
  });
});

describe("completeFilters", () => {
  it("drops filters still being written", () => {
    expect(
      completeFilters([
        { column: "null_pct", op: ">" },
        { column: "null_pct", op: ">", value: "" },
        { column: "action", op: "in", value: [] },
        { column: "notes", op: "is_null" },
        { column: "action", op: "in", value: ["keep"] },
        { column: "n_unique", op: ">", value: 0 },
      ]),
    ).toEqual([
      { column: "notes", op: "is_null" },
      { column: "action", op: "in", value: ["keep"] },
      { column: "n_unique", op: ">", value: 0 },
    ]);
  });
});

describe("selectionLabel", () => {
  it("counts shown rows when everything is shown", () => {
    expect(selectionLabel(120, 0, false)).toBe("120 selected");
    expect(selectionLabel(120, 20, false)).toBe("100 selected");
  });

  it("says a truncated selection covers every match", () => {
    expect(selectionLabel(QUERY_LIMIT, 0, true)).toBe(`all matches (more than ${QUERY_LIMIT})`);
    expect(selectionLabel(QUERY_LIMIT, 3, true)).toBe(`all matches except 3 (more than ${QUERY_LIMIT})`);
  });
});
