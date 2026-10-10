const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
const grouped = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e9 || a < 1e-4)) return v.toExponential(2);
  if (a >= 1e6) return compact.format(v);
  if (Number.isInteger(v)) return grouped.format(v);
  return Number(v.toPrecision(digits + 1)).toLocaleString(undefined, { maximumFractionDigits: 4 });
}
/** Model metrics (AUC, F1, accuracy…): always three decimals, so columns line up. */
export const fmtMetric = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(3);
export const fmtInt = (v: number | null | undefined) => (v === null || v === undefined ? "—" : grouped.format(v));
export const fmtCompact = (v: number | null | undefined) => (v === null || v === undefined ? "—" : compact.format(v));
export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return "—";
  if (v > 0 && v < 0.001) return "<0.1%";
  return `${(v * 100).toFixed(digits)}%`;
}
export const fmtDate = (s: string | null) => (s ? new Date(s).toLocaleString() : "—");

export const TYPE_SHORT: Record<string, string> = {
  numeric_continuous: "cont",
  numeric_discrete: "disc",
  categorical: "cat",
  binary: "bin",
  text: "text",
  datetime: "date",
  identifier: "id",
  constant: "const",
};

export const METHOD_LABEL: Record<string, string> = {
  spearman: "Spearman ρ",
  cramers_v: "Cramér's V",
  correlation_ratio: "Correlation ratio η",
};

const FAMILY_SHORT: Record<string, string> = { tree: "Trees", linear: "Linear", neural: "Neural" };
/** The one vocabulary for model families in labels, badges and the top-bar switch. */
export const familyShort = (f: string) => FAMILY_SHORT[f] ?? f;

export const roleLabel = (r: string) => (r === "lead" ? "Lead" : r === "member" ? "Member" : r);

const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
export function fmtAgo(s: string | null): string {
  if (!s) return "—";
  const sec = (new Date(s).getTime() - Date.now()) / 1000;
  const steps: [Intl.RelativeTimeFormatUnit, number][] = [["second", 60], ["minute", 60], ["hour", 24], ["day", 30], ["month", 12], ["year", Infinity]];
  let v = sec;
  for (const [unit, size] of steps) {
    if (Math.abs(v) < size) return rtf.format(Math.round(v), unit);
    v /= size;
  }
  return fmtDate(s);
}

/** p-values: 0.042, 3e-5, <1e-300 (float underflow reads as 0). */
export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  if (p < 1e-300) return "<1e-300";
  if (p < 1e-3) return p.toExponential(0);
  return p.toPrecision(2);
}
