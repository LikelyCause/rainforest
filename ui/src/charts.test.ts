import { describe, expect, it } from "vitest";
import { LEGEND_DASH, LEGEND_LINE, thresholdCurveOption } from "./charts";
import type { ChartTheme } from "./theme";

const t = { categorical: ["#1", "#2", "#3"], textMuted: "#m", textSecondary: "#s", grid: "#g" } as ChartTheme;
const point = (threshold: number) => ({ threshold, precision: 0.5, recall: 0.5, f1: 0.5 });

describe("thresholdCurveOption", () => {
  it("draws the test F1 dashed and tells it from the validation F1 in the legend", () => {
    const option = thresholdCurveOption(t, { valid: [point(0), point(1)], test: [point(0), point(1)] }, 0.3) as {
      legend: { data: { name: string; icon: string }[] };
      series: { name: string; lineStyle: { type: string } }[];
    };
    expect(option.series.map((s) => [s.name, s.lineStyle.type])).toEqual([
      ["Precision", "solid"],
      ["Recall", "solid"],
      ["F1", "solid"],
      ["Test F1", "dashed"],
    ]);
    expect(option.legend.data).toEqual([
      { name: "Precision", icon: LEGEND_LINE },
      { name: "Recall", icon: LEGEND_LINE },
      { name: "F1", icon: LEGEND_LINE },
      { name: "Test F1", icon: LEGEND_DASH },
    ]);
    expect(LEGEND_DASH).not.toBe(LEGEND_LINE);
  });
});
