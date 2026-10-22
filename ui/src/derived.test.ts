import { describe, expect, it } from "vitest";
import { describeDerived, groupsText, parseGroups } from "./derived";

describe("parseGroups", () => {
  it("reads ranges and exact values per line", () => {
    const { groups, error } = parseGroups("Circulatory: 390-459, 785\nDiabetes: 250-250.99\nInjury: V45, E880");
    expect(error).toBeNull();
    expect(groups).toEqual([
      { label: "Circulatory", ranges: [[390, 459]], values: ["785"] },
      { label: "Diabetes", ranges: [[250, 250.99]] },
      { label: "Injury", values: ["V45", "E880"] },
    ]);
  });
  it("keeps a colon in the label and round-trips", () => {
    const { groups } = parseGroups("Age: young: 0-30");
    expect(groups[0].label).toBe("Age: young");
    expect(parseGroups(groupsText(groups)).groups).toEqual(groups);
  });
  it("reports a line with no values or no colon", () => {
    expect(parseGroups("Circulatory:").error).toMatch(/no values/);
    expect(parseGroups("just text").error).toMatch(/Line 1/);
    expect(parseGroups("  \n").error).toMatch(/at least one/);
  });
});

it("describes each kind", () => {
  expect(describeDerived({ kind: "formula", spec: { expr: "a / b" } })).toBe("a / b");
  expect(describeDerived({ kind: "count", spec: { fields: ["x", "y"], values: ["Up", "Down"] } })).toBe(
    "count of Up / Down across 2 fields",
  );
});
