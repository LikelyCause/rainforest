import { describe, expect, it } from "vitest";
import { initials, tint } from "./avatar";

describe("avatar", () => {
  it("takes up to two initials", () => {
    expect(initials("Test Member")).toBe("TM");
    expect(initials("tanner")).toBe("T");
    expect(initials("  Ana  María  López ")).toBe("AL");
    expect(initials("")).toBe("?");
  });
  it("gives each account a stable colour from the palette", () => {
    expect(tint(7)).toBe(tint(7));
    for (const id of [1, 2, 3, 99, 1234]) expect(tint(id)).toBeGreaterThanOrEqual(1), expect(tint(id)).toBeLessThanOrEqual(8);
    expect(new Set([1, 2, 3, 4, 5, 6, 7, 8].map(tint)).size).toBeGreaterThan(4);
  });
});
