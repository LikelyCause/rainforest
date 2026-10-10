import type { EChartsCoreOption } from "echarts/core";
import type { BarsChart, Correlation, HistogramChart } from "./types";
import type { ChartTheme } from "./theme";
import { fmtCompact, fmtInt, fmtPct, METHOD_LABEL } from "./format";

// Tooltips render HTML; dataset values and column names are untrusted (#9).
export const esc = (v: unknown) =>
  String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

const BAR_RADIUS_UP: [number, number, number, number] = [4, 4, 0, 0];
export const BAR_RADIUS_RIGHT: [number, number, number, number] = [0, 4, 4, 0];

export function base(t: ChartTheme): EChartsCoreOption {
  return {
    // No morphing between fields: the previous field's bars must never ghost behind the new ones.
    animation: false,
    textStyle: { fontFamily: t.font, color: t.textSecondary },
    tooltip: {
      trigger: "item",
      backgroundColor: t.surface,
      borderColor: t.grid,
      textStyle: { color: t.text, fontSize: 12 },
      extraCssText: "box-shadow: 0 2px 10px rgba(0,0,0,0.12); border-radius: 6px;",
    },
  };
}

export const axisCommon = (t: ChartTheme) => ({
  axisLine: { lineStyle: { color: t.grid } },
  axisTick: { show: false },
  axisLabel: { color: t.textMuted, fontSize: 11 },
  splitLine: { lineStyle: { color: t.grid, type: "solid" as const, width: 1 } },
});

/**
 * Bin-edge labels precise enough that neighbouring edges differ: decimals follow the bin width (37–48, not
 * 37.00–48.53), large values keep enough significant digits (1.001M–1.002M, not 1M–1M), and time bins narrower
 * than a day show the time of day. Times are epoch seconds, shown in UTC like the profile.
 */
export function edgeFormatter(edges: number[], time = false): (v: number) => string {
  const width = edges.length > 1 ? Math.abs(edges[1] - edges[0]) : 1;
  if (time) {
    const chars = width >= 86_400 ? 10 : width >= 60 ? 16 : 19;
    return (v) => new Date(v * 1000).toISOString().slice(0, chars).replace("T", " ");
  }
  const maxAbs = Math.max(...edges.map(Math.abs), 0);
  const significant = Math.min(10, Math.max(2, Math.ceil(Math.log10(maxAbs / (width || 1))) + 1));
  const big = new Intl.NumberFormat(undefined, { notation: "compact", maximumSignificantDigits: significant });
  const decimals = width >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(width)) + 1);
  return (v) => (Math.abs(v) >= 1e6 ? big.format(v)
    : v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }));
}

/** Histogram: columns grow from one baseline; 2px surface gap between touching bins. */
export function histogramOption(
  t: ChartTheme, h: HistogramChart, opts: { time?: boolean; xLabel?: string } = {},
): EChartsCoreOption {
  const total = h.counts.reduce((a, b) => a + b, 0) + (h.below ?? 0) + (h.above ?? 0);
  const fmtEdge = edgeFormatter(h.edges, opts.time);
  const cats = h.counts.map((_, i) => fmtEdge(h.edges[i]));
  return {
    ...base(t),
    grid: { left: 8, right: 12, top: 12, bottom: opts.xLabel ? 40 : 24, containLabel: true },
    xAxis: {
      type: "category",
      data: cats,
      name: opts.xLabel,
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: t.textMuted, fontSize: 11 },
      ...axisCommon(t),
      axisLabel: { color: t.textMuted, fontSize: 11, hideOverlap: true },
    },
    yAxis: { type: "value", ...axisCommon(t), axisLabel: { color: t.textMuted, fontSize: 11, formatter: (v: number) => fmtCompact(v) } },
    tooltip: {
      ...(base(t) as any).tooltip,
      formatter: (p: any) => {
        const i = p.dataIndex;
        return `<b>${esc(fmtEdge(h.edges[i]))} – ${esc(fmtEdge(h.edges[i + 1]))}</b><br/>${fmtInt(h.counts[i])} rows · ${fmtPct(h.counts[i] / Math.max(total, 1))}`;
      },
    },
    series: [{
      type: "bar",
      data: h.counts,
      barCategoryGap: 0,
      barMaxWidth: 24,
      itemStyle: { color: t.series, borderRadius: BAR_RADIUS_UP, borderColor: t.surface, borderWidth: 1 },
      emphasis: { itemStyle: { color: t.series, opacity: 0.8 } },
    }],
  };
}

/** Horizontal bars for top levels; value at each bar tip. */
export function barsOption(t: ChartTheme, b: BarsChart, opts: { total?: number } = {}): EChartsCoreOption {
  const total = opts.total ?? b.counts.reduce((a, c) => a + c, 0) + (b.other ?? 0) + (b.missing ?? 0);
  const labels = b.labels.map((l) => (l.length > 28 ? `${l.slice(0, 27)}…` : l));
  return {
    ...base(t),
    grid: { left: 8, right: 56, top: 4, bottom: 4, containLabel: true },
    xAxis: { type: "value", show: false },
    yAxis: { type: "category", data: labels, inverse: true, ...axisCommon(t), axisLabel: { color: t.textSecondary, fontSize: 11.5 } },
    tooltip: {
      ...(base(t) as any).tooltip,
      formatter: (p: any) => `<b>${esc(b.labels[p.dataIndex])}</b><br/>${fmtInt(b.counts[p.dataIndex])} rows · ${fmtPct(b.counts[p.dataIndex] / Math.max(total, 1))}`,
    },
    series: [{
      type: "bar",
      data: b.counts,
      barMaxWidth: 18,
      barCategoryGap: "30%",
      itemStyle: { color: t.series, borderRadius: BAR_RADIUS_RIGHT },
      label: { show: true, position: "right", color: t.textSecondary, fontSize: 11, formatter: (p: any) => fmtPct(p.value / Math.max(total, 1)) },
    }],
  };
}

/** Top associations. Signed Spearman diverges blue (+) / red (−) around 0; unsigned measures are 0..1. */
export function correlationOption(t: ChartTheme, rows: Correlation[]): EChartsCoreOption {
  const anyNegative = rows.some((r) => r.value < 0);
  const labels = rows.map((r) => {
    const n = r.name.length > 26 ? `${r.name.slice(0, 25)}…` : r.name;
    return r.is_target ? `◎ ${n}` : n;
  });
  return {
    ...base(t),
    grid: { left: anyNegative ? 40 : 8, right: 48, top: 8, bottom: 20, containLabel: true },
    xAxis: {
      type: "value", min: anyNegative ? -1 : 0, max: 1, interval: 0.5, ...axisCommon(t),
      axisLabel: { color: t.textMuted, fontSize: 11, formatter: (v: number) => v.toFixed(1) },
    },
    yAxis: { type: "category", data: labels, inverse: true, ...axisCommon(t), axisLabel: { color: t.textSecondary, fontSize: 11.5 } },
    tooltip: {
      ...(base(t) as any).tooltip,
      formatter: (p: any) => {
        const r = rows[p.dataIndex];
        return `<b>${esc(r.name)}</b>${r.is_target ? " (target)" : ""}<br/>${METHOD_LABEL[r.method]}: ${r.value.toFixed(3)}<br/><span style="opacity:.7">${esc(r.type)} · n=${fmtInt(r.n_obs)} · click to open</span>`;
      },
    },
    series: [{
      type: "bar",
      data: rows.map((r) => ({
        value: r.value,
        itemStyle: {
          color: r.is_target ? t.target : r.value < 0 ? t.negative : t.series,
          borderRadius: r.value < 0 ? [4, 0, 0, 4] : BAR_RADIUS_RIGHT,
        },
        label: { position: r.value < 0 ? "left" : "right" },
      })),
      barMaxWidth: 16,
      barCategoryGap: "35%",
      cursor: "pointer",
      label: { show: true, position: "right", color: t.textSecondary, fontSize: 11, formatter: (p: any) => p.value.toFixed(2) },
      markLine: anyNegative ? { silent: true, symbol: "none", lineStyle: { color: t.textMuted, width: 1, type: "solid" }, label: { show: false }, data: [{ xAxis: 0 }] } : undefined,
    }],
  };
}
