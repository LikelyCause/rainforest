import { fmtNum } from "./format";

export interface StripLabel {
  key: "p1" | "median" | "p99";
  text: string;
  /** Position along the strip, 0 (p1) to 1 (p99). */
  at: number;
  anchor: "start" | "middle" | "end";
  row: "above" | "below";
}

/** Labels for the p1–p99 strip. The median gets its own row, since it can sit right on p1 or p99, and is anchored
 * inward near an end so it stays inside the strip. */
export function stripLabels(box: { p01: number; p50: number; p99: number }): StripLabel[] {
  const at = (box.p50 - box.p01) / (box.p99 - box.p01);
  return [
    { key: "p1", text: `p1 ${fmtNum(box.p01)}`, at: 0, anchor: "start", row: "below" },
    {
      key: "median",
      text: `median ${fmtNum(box.p50)}`,
      at,
      anchor: at < 0.1 ? "start" : at > 0.9 ? "end" : "middle",
      row: "above",
    },
    { key: "p99", text: `p99 ${fmtNum(box.p99)}`, at: 1, anchor: "end", row: "below" },
  ];
}
