import { describe, expect, it } from "vitest";
import {
  budgetMessage,
  comparedOnText,
  connectBlocker,
  editDraft,
  flattenKindLabel,
  matchColumns,
  newDraft,
  parseJsonDocument,
  previewKey,
  previewRequest,
  sourceSummary,
  unknownColumns,
  type ConnectEdit,
} from "./sources";
import type { PreviewResult } from "./types";

describe("flattenKindLabel", () => {
  it("says what the puller did with the less obvious kinds", () => {
    expect(flattenKindLabel("mixed")).toBe("mixed types, read as text");
    expect(flattenKindLabel("json")).toBe("array, stored as JSON text");
    expect(flattenKindLabel("boolean")).toBe("yes/no");
    expect(flattenKindLabel("identifier")).toBe("identifier");
  });

  it("leaves the plain kinds as they are", () => {
    for (const k of ["number", "text", "datetime"] as const) expect(flattenKindLabel(k)).toBe(k);
  });
});

describe("budgetMessage", () => {
  it("states an estimate that fits", () => {
    expect(budgetMessage(450_000_000, 600_000_000)).toBe("450,000,000 of 600,000,000 cells");
  });

  it("states an estimate over the limit", () => {
    expect(budgetMessage(700_000_000, 600_000_000)).toBe("700,000,000 cells exceeds the limit of 600,000,000");
  });

  it("treats exactly the limit as fitting", () => {
    expect(budgetMessage(600_000_000, 600_000_000)).toBe("600,000,000 of 600,000,000 cells");
  });
});

describe("parseJsonDocument", () => {
  it("accepts blank text and JSON objects", () => {
    expect(parseJsonDocument("")).toBeNull();
    expect(parseJsonDocument("  ")).toBeNull();
    expect(parseJsonDocument('{"year": {"$gt": 2020}}')).toBeNull();
    expect(parseJsonDocument('{"_id": {"$oid": "65f1a2b3c4d5e6f708192a3b"}}')).toBeNull();
  });

  it("names what is wrong with anything else", () => {
    expect(parseJsonDocument("[1, 2]")).toBe("Must be a JSON object");
    expect(parseJsonDocument("3")).toBe("Must be a JSON object");
    expect(parseJsonDocument("null")).toBe("Must be a JSON object");
    expect(parseJsonDocument("{year: 1}")).toBe("Not valid JSON");
  });
});

describe("sourceSummary", () => {
  it("joins the connection and the collection labels", () => {
    expect(
      sourceSummary({
        connection: "vehicles",
        collections: [
          { name: "curated_v2", label: "curated" },
          { name: "mining", label: "mining" },
        ],
      }),
    ).toBe("vehicles · curated, mining");
  });
});

const preview = (over: Partial<PreviewResult> = {}): PreviewResult => ({
  columns: [
    { name: "_id", kind: "identifier", present_share: 1 },
    { name: "vin", kind: "text", present_share: 1 },
    { name: "model_year", kind: "number", present_share: 0.9 },
  ],
  per_source: [],
  estimated_rows: 500,
  estimated_cells: 1500,
  budget: { max_cells: 600_000_000, fits: true },
  compared_on: ["vin", "model_year"],
  ...over,
});

describe("editDraft", () => {
  const edited = () => {
    let d = newDraft("vehicles");
    for (const e of [
      { type: "database", value: "fleet" },
      { type: "toggle", name: "curated" },
      { type: "toggle", name: "mining" },
      { type: "pick", name: "curated", patch: { label: "cur", filter: '{"year": 2020}' } },
      { type: "name", value: "Fleet" },
      { type: "addSource", value: true },
      { type: "dedupe", value: false },
      { type: "matchOn", value: "vin" },
    ] as ConnectEdit[])
      d = editDraft(d, e);
    return d;
  };

  it("keeps collections in the order they are ticked and moves one up", () => {
    const d = edited();
    expect(d.chosen).toEqual(["curated", "mining"]);
    expect(editDraft(d, { type: "moveUp", index: 1 }).chosen).toEqual(["mining", "curated"]);
    expect(editDraft(d, { type: "toggle", name: "curated" }).chosen).toEqual(["mining"]);
  });

  it("starts over on another connection", () => {
    expect(editDraft(edited(), { type: "connection", value: "archive" })).toEqual(newDraft("archive"));
  });

  it("drops the picks and option edits on another database", () => {
    expect(editDraft(edited(), { type: "database", value: "archive_db" })).toMatchObject({
      connection: "vehicles",
      database: "archive_db",
      chosen: [],
      picks: {},
      name: null,
      addSource: false,
      dedupe: null,
      matchOn: "",
    });
  });

  it("keeps everything when the same database is chosen again", () => {
    const d = edited();
    expect(editDraft(d, { type: "database", value: "fleet" })).toBe(d);
  });
});

describe("previewKey", () => {
  const one = editDraft(editDraft(newDraft("vehicles"), { type: "database", value: "fleet" }), {
    type: "toggle",
    name: "curated",
  });

  it("ignores labels, which the preview doesn't send", () => {
    const labelled = editDraft(one, { type: "pick", name: "curated", patch: { label: "trusted" } });
    expect(previewKey(labelled)).toBe(previewKey(one));
    expect(previewRequest(labelled).collections).toEqual([{ name: "curated" }]);
  });

  it("changes with a filter, a projection, the collections or their order", () => {
    const two = editDraft(one, { type: "toggle", name: "mining" });
    const keys = [
      previewKey(one),
      previewKey(two),
      previewKey(editDraft(two, { type: "moveUp", index: 1 })),
      previewKey(editDraft(one, { type: "pick", name: "curated", patch: { filter: '{"a": 1}' } })),
      previewKey(editDraft(one, { type: "pick", name: "curated", patch: { projection: '{"a": 0}' } })),
    ];
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe("matchColumns and unknownColumns", () => {
  it("splits the Match on list, trimmed and once each", () => {
    expect(matchColumns(" vin, model_year ,,vin ")).toEqual(["vin", "model_year"]);
    expect(matchColumns("  ")).toEqual([]);
  });

  it("names the columns the preview doesn't have", () => {
    expect(unknownColumns(["vin", "Vin", "colour"], preview())).toEqual(["Vin", "colour"]);
    expect(unknownColumns(["vin"], preview())).toEqual([]);
    expect(unknownColumns(["colour"], null)).toEqual([]);
  });
});

describe("comparedOnText", () => {
  it("lists the preview's comparison columns", () => {
    expect(comparedOnText(preview())).toBe("vin, model_year");
  });

  it("says when there is nothing to compare on, and is null without a preview", () => {
    expect(comparedOnText(preview({ compared_on: [] }))).toBe("no shared columns, so nothing is left out");
    expect(comparedOnText(null)).toBeNull();
  });
});

describe("connectBlocker", () => {
  const ok = { chosen: 2, invalid: false, preview: preview(), name: "Fleet", dedupe: true, unknownMatch: [] };

  it("is null when Connect can go ahead", () => {
    expect(connectBlocker(ok)).toBeNull();
  });

  it("says why Connect is off, in the dialog's order", () => {
    expect(connectBlocker({ ...ok, chosen: 0 })).toBe("Choose at least one collection.");
    expect(connectBlocker({ ...ok, invalid: true })).toBe("Fix the marked fields first.");
    expect(connectBlocker({ ...ok, preview: null })).toBe("Preview the collections first.");
    expect(connectBlocker({ ...ok, preview: preview({ budget: { max_cells: 1, fits: false } }) })).toBe(
      "Too large to pull: narrow a filter or add a projection.",
    );
    expect(connectBlocker({ ...ok, name: "  " })).toBe("Name the dataset.");
    expect(connectBlocker({ ...ok, unknownMatch: ["colour", "trim"] })).toBe(
      "Match on names columns the preview doesn't have: colour, trim.",
    );
  });

  it("ignores Match on when duplicates are kept", () => {
    expect(connectBlocker({ ...ok, dedupe: false, unknownMatch: ["colour"] })).toBeNull();
  });
});
