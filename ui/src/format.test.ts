import { afterEach, describe, expect, it, vi } from "vitest";
import { edgeFormatter, esc } from "./charts";
import { familyShort, fmtAgo, fmtInt, fmtNum, fmtP, fmtPct } from "./format";

describe("fmtNum", () => {
  it("handles missing, tiny, huge, integer and fractional values", () => {
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(Number.NaN)).toBe("—");
    expect(fmtNum(0)).toBe("0");
    expect(fmtNum(0.00001234)).toBe("1.23e-5");
    expect(fmtNum(5e9)).toBe("5.00e+9");
    expect(fmtNum(2_500_000)).toBe("2.5M");
    expect(fmtNum(12345)).toBe("12,345");
    expect(fmtNum(3.14159, 2)).toBe("3.14");
  });
});

describe("fmtPct and fmtInt", () => {
  it("formats shares and counts", () => {
    expect(fmtPct(undefined)).toBe("—");
    expect(fmtPct(0)).toBe("0.0%");
    expect(fmtPct(0.0004)).toBe("<0.1%");
    expect(fmtPct(0.256, 0)).toBe("26%");
    expect(fmtInt(1234567)).toBe("1,234,567");
    expect(fmtInt(null)).toBe("—");
  });
});

describe("fmtP", () => {
  it("shows two significant digits, exponents below 0.001 and a floor for underflow", () => {
    expect(fmtP(null)).toBe("—");
    expect(fmtP(Number.NaN)).toBe("—");
    expect(fmtP(0.04213)).toBe("0.042");
    expect(fmtP(0.5)).toBe("0.50");
    expect(fmtP(0.00003)).toBe("3e-5");
    expect(fmtP(0)).toBe("<1e-300");
    expect(fmtP(1e-320)).toBe("<1e-300");
  });
});

describe("fmtAgo", () => {
  afterEach(() => vi.useRealTimers());
  it("speaks relative time", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-09T12:00:00Z"));
    expect(fmtAgo(null)).toBe("—");
    expect(fmtAgo("2026-10-09T11:59:30Z")).toBe("30 seconds ago");
    expect(fmtAgo("2026-10-09T09:00:00Z")).toBe("3 hours ago");
    expect(fmtAgo("2026-10-07T12:00:00Z")).toBe("2 days ago");
  });
});

describe("familyShort", () => {
  it("names the families the same way everywhere", () => {
    expect(familyShort("tree")).toBe("Trees");
    expect(familyShort("other")).toBe("other");
  });
});

describe("edgeFormatter", () => {
  it("keeps neighbouring large edges apart", () => {
    const edges = [1_000_000, 1_001_000, 1_002_000];
    const f = edgeFormatter(edges);
    expect(new Set(edges.map(f)).size).toBe(3);
    expect(f(1_001_000)).toBe("1.001M");
  });

  it("gives decimals that follow the bin width", () => {
    expect(edgeFormatter([37, 48, 59])(48)).toBe("48");
    expect(edgeFormatter([0, 0.05, 0.1])(0.05)).toBe("0.050");
  });

  it("shows the time of day for bins narrower than a day", () => {
    const t0 = Date.UTC(2024, 0, 2, 8, 0, 0) / 1000;
    expect(edgeFormatter([t0, t0 + 86_400 * 2], true)(t0)).toBe("2024-01-02");
    expect(edgeFormatter([t0, t0 + 900], true)(t0 + 900)).toBe("2024-01-02 08:15");
    expect(edgeFormatter([t0, t0 + 5], true)(t0 + 5)).toBe("2024-01-02 08:00:05");
  });
});

describe("esc", () => {
  it("escapes markup in data-derived tooltip strings", () => {
    expect(esc(`<img src=x onerror="a('b')">&`)).toBe("&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;");
  });
});
