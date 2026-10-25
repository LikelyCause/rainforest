import { describe, expect, it } from "vitest";
import { budgetMessage, flattenKindLabel, parseJsonDocument, sourceSummary } from "./sources";

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
