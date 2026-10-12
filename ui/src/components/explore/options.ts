import type { EChartsCoreOption } from "echarts/core";
import { axisCommon, base, BAR_RADIUS_RIGHT, esc, FS } from "../../charts";
import { fmtInt, fmtNum, fmtPct, METHOD_LABEL } from "../../format";
import type { ChartTheme } from "../../theme";
import type { CorrMatrix, ImportanceResult, MapResult, TestDetail } from "../../types";

const short = (s: string, n = 24) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const tip = (t: ChartTheme) => (base(t) as any).tooltip;
// Fixed slot order; anything past the eighth folds into the muted colour rather than a generated hue.
const catColor = (t: ChartTheme, i: number) => t.categorical[i] ?? t.muted;
const legend = (t: ChartTheme) => ({
  type: "scroll", top: 0, left: 0, right: 0, icon: "circle", itemWidth: 10, itemHeight: 10,
  textStyle: { color: t.textSecondary, fontSize: FS.small }, pageTextStyle: { color: t.textMuted },
});

/** value = [col, row, strength, cell index]; cells below the threshold drop to the neutral colour. */
export function corrHeatmapOption(t: ChartTheme, m: CorrMatrix, threshold: number): EChartsCoreOption {
  const pos = new Map(m.order.map((id, i) => [id, i]));
  const byId = new Map(m.fields.map((f) => [f.id, f]));
  const labels = m.order.map((id) => {
    const f = byId.get(id);
    const n = short(f?.name ?? `#${id}`, 22);
    return f?.action === "target" ? `◎ ${n}` : n;
  });
  const data: [number, number, number, number][] = [];
  m.cells.forEach((c, k) => {
    const i = pos.get(c.a), j = pos.get(c.b);
    if (i === undefined || j === undefined) return;
    data.push([i, j, c.strength, k]);
    if (i !== j) data.push([j, i, c.strength, k]);
  });
  const axis = { type: "category", data: labels, ...axisCommon(t), splitLine: { show: false }, axisLine: { show: false } };
  return {
    ...base(t),
    grid: { left: 8, right: 8, top: 8, bottom: 8, containLabel: true },
    xAxis: { ...axis, position: "top", axisLabel: { color: t.textSecondary, fontSize: FS.small, rotate: 90, interval: 0, hideOverlap: true } },
    yAxis: { ...axis, inverse: true, axisLabel: { color: t.textSecondary, fontSize: FS.small, interval: 0, hideOverlap: true } },
    visualMap: {
      show: false, type: "continuous", dimension: 2, min: 0, max: 1, range: [threshold, 1],
      inRange: { color: [t.seq0, t.seqLo, t.seqHi] }, outOfRange: { color: [t.divMid] },
    },
    tooltip: {
      ...tip(t),
      formatter: (p: any) => {
        const c = m.cells[p.value[3]];
        const a = byId.get(c.a)?.name ?? c.a, b = byId.get(c.b)?.name ?? c.b;
        const v = c.method === "spearman" ? (c.value >= 0 ? "+" : "") + c.value.toFixed(3) : c.value.toFixed(3);
        return `<b>${esc(a)}</b> × <b>${esc(b)}</b><br/>${esc(METHOD_LABEL[c.method] ?? c.method)}: ${v}<br/>` +
          `<span style="opacity:.7">strength ${c.strength.toFixed(2)} · n=${fmtInt(c.n)} · click for details</span>`;
      },
    },
    series: [{
      type: "heatmap", data, cursor: "pointer", progressive: 0,
      itemStyle: { borderColor: t.surface, borderWidth: 1 },
      emphasis: { itemStyle: { borderColor: t.text, borderWidth: 1 } },
    }],
  };
}

/** Embedding scatter. The last value of every point is its row index, for the hover lookup. */
export function mapOption(t: ChartTheme, r: MapResult): EChartsCoreOption {
  const big = r.points.length > 5000;
  const common = {
    type: "scatter", symbolSize: big ? 3 : 5, large: true, largeThreshold: 2000, progressive: 5000,
    itemStyle: { opacity: big ? 0.65 : 0.85 },
  };
  const hover = (idx: number) => r.hover_fields.map((f, j) => {
    const v = r.hover_values[idx]?.[j];
    return `<span style="opacity:.7">${esc(short(f, 28))}</span> ${esc(v === null || v === undefined ? "—" : typeof v === "number" ? fmtNum(v) : v)}`;
  }).join("<br/>");
  const head = (c: number | null) => {
    if (!r.target) return "";
    const shown = c === null ? "missing" : r.target_kind === "categorical" ? r.classes[c] ?? c : fmtNum(c);
    return `<b>${esc(r.target)}: ${esc(shown)}</b><br/>`;
  };
  const axes = { xAxis: { type: "value", show: false, scale: true }, yAxis: { type: "value", show: false, scale: true } };
  const missing = r.points.map((p, i) => [p[0], p[1], i]).filter((_, i) => r.points[i][2] === null);
  const missingSeries = r.target && missing.length
    ? [{ ...common, name: "(missing)", data: missing, itemStyle: { ...common.itemStyle, color: t.muted } }] : [];

  if (r.target_kind === "numeric") {
    const vals = r.points.map((p) => p[2]).filter((v): v is number => v !== null);
    const lo = vals.reduce((a, b) => Math.min(a, b), Infinity), hi = vals.reduce((a, b) => Math.max(a, b), -Infinity);
    return {
      ...base(t), ...axes,
      grid: { left: 8, right: 84, top: 8, bottom: 8 },
      visualMap: {
        type: "continuous", dimension: 2, seriesIndex: 0, min: lo, max: hi, calculable: true, right: 0, top: "middle",
        inRange: { color: [t.seqLo, t.seqHi] }, textStyle: { color: t.textSecondary, fontSize: FS.small },
        itemHeight: 160, formatter: (v: number) => fmtNum(v),
      },
      tooltip: { ...tip(t), formatter: (p: any) => { const v = p.value; return head(v.length === 4 ? v[2] : null) + hover(v[v.length - 1]); } },
      series: [
        // Large mode ignores per-point visualMap colours, so this series renders progressively instead.
        { ...common, large: false, progressive: 4000, name: r.target, data: r.points.flatMap((p, i) => (p[2] === null ? [] : [[p[0], p[1], p[2], i]])) },
        ...missingSeries,
      ],
    };
  }
  if (r.target_kind === "categorical") {
    const series = r.classes.map((cls, k) => ({
      ...common, name: cls,
      data: r.points.flatMap((p, i) => (p[2] === k ? [[p[0], p[1], i]] : [])),
      itemStyle: { ...common.itemStyle, color: catColor(t, k) },
    }));
    return {
      ...base(t), ...axes,
      legend: legend(t),
      grid: { left: 8, right: 8, top: 36, bottom: 8 },
      tooltip: {
        ...tip(t),
        formatter: (p: any) => {
          const k = p.seriesIndex < r.classes.length ? p.seriesIndex : null;
          return head(k) + hover(p.value[2]);
        },
      },
      series: [...series, ...missingSeries],
    };
  }
  return {
    ...base(t), ...axes,
    grid: { left: 8, right: 8, top: 8, bottom: 8 },
    tooltip: { ...tip(t), formatter: (p: any) => hover(p.value[2]) },
    series: [{ ...common, data: r.points.map((p, i) => [p[0], p[1], i]), itemStyle: { ...common.itemStyle, color: t.series } }],
  };
}

export function testDetailOption(t: ChartTheme, d: TestDetail, labels: { field: string; target: string; classes: string[] }): EChartsCoreOption {
  const valueAxis = (name?: string) => ({
    type: "value", scale: true, name, nameLocation: "middle", nameGap: 36,
    nameTextStyle: { color: t.textMuted, fontSize: FS.small }, ...axisCommon(t),
    axisLabel: { color: t.textMuted, fontSize: FS.small, formatter: (v: number) => fmtNum(v) },
  });
  if (d.kind === "box") {
    return {
      ...base(t),
      grid: { left: 8, right: 12, top: 12, bottom: 8, containLabel: true },
      xAxis: { type: "category", data: d.groups.map((g) => short(g.class, 16)), ...axisCommon(t), splitLine: { show: false },
        axisLabel: { color: t.textSecondary, fontSize: FS.small, interval: 0, rotate: d.groups.length > 6 ? 30 : 0 } },
      yAxis: valueAxis(),
      tooltip: {
        ...tip(t),
        formatter: (p: any) => {
          const g = d.groups[p.dataIndex];
          return `<b>${esc(g.class)}</b> · n=${fmtInt(g.n)}<br/>max ${fmtNum(g.max)}<br/>Q3 ${fmtNum(g.q3)}<br/>median ${fmtNum(g.median)}<br/>Q1 ${fmtNum(g.q1)}<br/>min ${fmtNum(g.min)}`;
        },
      },
      series: [{
        type: "boxplot", data: d.groups.map((g) => [g.min, g.q1, g.median, g.q3, g.max]), boxWidth: [8, 40],
        itemStyle: { color: t.accentWeak, borderColor: t.series, borderWidth: 1.5 },
      }],
    };
  }
  if (d.kind === "levels") {
    const totals = d.counts.map((row) => row.reduce((a, b) => a + b, 0) || 1);
    const cats = d.counts.map((_, i) => short(labels.classes[i] ?? `#${i}`, 16));
    return {
      ...base(t),
      legend: legend(t),
      grid: { left: 8, right: 12, top: 36, bottom: 8, containLabel: true },
      xAxis: { type: "category", data: cats, ...axisCommon(t), splitLine: { show: false },
        axisLabel: { color: t.textSecondary, fontSize: FS.small, interval: 0, rotate: cats.length > 6 ? 30 : 0 } },
      yAxis: { type: "value", max: 1, ...axisCommon(t), axisLabel: { color: t.textMuted, fontSize: FS.small, formatter: (v: number) => fmtPct(v, 0) } },
      tooltip: {
        ...tip(t),
        formatter: (p: any) => {
          const n = d.counts[p.dataIndex][p.seriesIndex];
          return `<b>${esc(d.levels[p.seriesIndex])}</b> in ${esc(labels.classes[p.dataIndex] ?? p.dataIndex)}<br/>${fmtInt(n)} rows · ${fmtPct(n / totals[p.dataIndex])}`;
        },
      },
      series: d.levels.map((lvl, j) => ({
        type: "bar", name: lvl, stack: "levels", barMaxWidth: 48,
        data: d.counts.map((row, i) => row[j] / totals[i]),
        itemStyle: { color: catColor(t, j), borderColor: t.surface, borderWidth: 1 },
      })),
    };
  }
  return {
    ...base(t),
    grid: { left: 16, right: 12, top: 12, bottom: 28, containLabel: true },
    xAxis: valueAxis(labels.field),
    yAxis: { ...valueAxis(labels.target), nameGap: 48 },
    tooltip: { ...tip(t), formatter: (p: any) => `${esc(labels.field)} ${fmtNum(p.value[0])}<br/>${esc(labels.target)} ${fmtNum(p.value[1])}` },
    series: [{ type: "scatter", data: d.points, symbolSize: 4, large: true, largeThreshold: 2000, itemStyle: { color: t.series, opacity: 0.6 } }],
  };
}

export function miOption(t: ChartTheme, rows: ImportanceResult["mi"]): EChartsCoreOption {
  return {
    ...base(t),
    grid: { left: 8, right: 56, top: 4, bottom: 4, containLabel: true },
    xAxis: { type: "value", show: false },
    yAxis: { type: "category", data: rows.map((r) => short(r.name, 30)), inverse: true, ...axisCommon(t), axisLabel: { color: t.textSecondary, fontSize: FS.small } },
    tooltip: {
      ...tip(t),
      formatter: (p: any) => {
        const r = rows[p.dataIndex];
        return `<b>${esc(r.name)}</b><br/>Mutual information ${fmtNum(r.value)}${r.field_id ? '<br/><span style="opacity:.7">click to open the field</span>' : ""}`;
      },
    },
    series: [{
      type: "bar", data: rows.map((r) => r.value), barMaxWidth: 16, barCategoryGap: "35%", cursor: "pointer",
      itemStyle: { color: t.series, borderRadius: BAR_RADIUS_RIGHT },
      label: { show: true, position: "right", color: t.textSecondary, fontSize: FS.small, formatter: (p: any) => fmtNum(p.value, 2) },
    }],
  };
}

/** Explained variance and its running total share one 0–1 axis. */
export function pcaOption(t: ChartTheme, explained: number[], cumulative: number[]): EChartsCoreOption {
  const names = explained.map((_, i) => `PC${i + 1}`);
  return {
    ...base(t),
    legend: { ...legend(t), data: ["Explained", "Cumulative"] },
    grid: { left: 8, right: 12, top: 36, bottom: 8, containLabel: true },
    xAxis: { type: "category", data: names, ...axisCommon(t), splitLine: { show: false }, axisLabel: { color: t.textMuted, fontSize: FS.small, hideOverlap: true } },
    yAxis: { type: "value", min: 0, max: 1, ...axisCommon(t), axisLabel: { color: t.textMuted, fontSize: FS.small, formatter: (v: number) => fmtPct(v, 0) } },
    tooltip: {
      ...tip(t), trigger: "axis",
      formatter: (ps: any[]) => {
        const i = ps[0].dataIndex;
        return `<b>${names[i]}</b><br/>Explained ${fmtPct(explained[i])}<br/>Cumulative ${fmtPct(cumulative[i])}`;
      },
    },
    series: [
      { type: "bar", name: "Explained", data: explained, barMaxWidth: 24, itemStyle: { color: t.series, borderRadius: [4, 4, 0, 0] } },
      { type: "line", name: "Cumulative", data: cumulative, symbolSize: 6, lineStyle: { width: 2, color: catColor(t, 1) }, itemStyle: { color: catColor(t, 1) } },
    ],
  };
}

/** Components × features, diverging around zero. */
export function loadingsOption(t: ChartTheme, comps: number[], features: string[], value: (c: number, f: string) => number | undefined): EChartsCoreOption {
  const data: [number, number, number][] = [];
  let maxAbs = 0;
  comps.forEach((c, i) => features.forEach((f, j) => {
    const v = value(c, f);
    if (v === undefined) return;
    data.push([i, j, v]);
    maxAbs = Math.max(maxAbs, Math.abs(v));
  }));
  maxAbs = maxAbs || 1;
  return {
    ...base(t),
    grid: { left: 8, right: 8, top: 8, bottom: 40, containLabel: true },
    xAxis: { type: "category", data: comps.map((c) => `PC${c + 1}`), position: "top", ...axisCommon(t), splitLine: { show: false }, axisLine: { show: false } },
    yAxis: { type: "category", data: features.map((f) => short(f, 30)), inverse: true, ...axisCommon(t), splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: t.textSecondary, fontSize: FS.small } },
    visualMap: {
      type: "continuous", min: -maxAbs, max: maxAbs, orient: "horizontal", left: "center", bottom: 0, itemWidth: 10, itemHeight: 160,
      inRange: { color: [t.negative, t.divMid, t.series] }, textStyle: { color: t.textSecondary, fontSize: FS.small },
      text: [`+${fmtNum(maxAbs, 2)}`, `−${fmtNum(maxAbs, 2)}`], calculable: false,
    },
    tooltip: { ...tip(t), formatter: (p: any) => `<b>${esc(features[p.value[1]])}</b><br/>PC${comps[p.value[0]] + 1} loading ${p.value[2] >= 0 ? "+" : ""}${p.value[2].toFixed(3)}` },
    series: [{
      type: "heatmap", data,
      label: { show: true, color: t.text, fontSize: FS.small, formatter: (p: any) => p.value[2].toFixed(2) },
      itemStyle: { borderColor: t.surface, borderWidth: 1 },
    }],
  };
}
