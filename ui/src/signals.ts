import { fmtInt, fmtNum, fmtPct } from "./format";
import type { FieldDetail, FieldListItem, TargetAssoc } from "./types";

export type Tone = "warn" | "bad" | undefined;

export interface Signal {
  key: string;
  label: string;
  value: string;
  tone: Tone;
  rule: string;
  /** A tighter value for the stat strip, with what it is measured against shown under it. */
  short?: string;
  detail?: string;
}

export const LEAK_AT = 0.95;
export const STRONG_TARGET_AT = 0.5;
export const MISSING_WARN = 0.2;
export const MISSING_BAD = 0.5;
export const LOW_CONFIDENCE = 0.7;

const NUMERIC = new Set(["numeric_continuous", "numeric_discrete"]);
const LEVELED = new Set(["categorical", "text", "numeric_discrete", "binary"]);

export const missingTone = (p: number | null | undefined): Tone =>
  p == null ? undefined : p >= MISSING_BAD ? "bad" : p >= MISSING_WARN ? "warn" : undefined;

/** The few numbers that decide a field, ordered by how much they matter, toned only past a stated threshold. */
export function fieldSignals(f: FieldDetail, targets: TargetAssoc[], decisionType: string): Signal[] {
  const out: Signal[] = [];
  const t = f.inferred_type;
  if (decisionType !== t) {
    out.push({
      key: "retyped",
      label: "Stats are for",
      value: t.replace("_", " "),
      tone: "warn",
      rule: `You set this field to ${decisionType.replace("_", " ")}; the profile below was computed for the inferred type.`,
    });
  }
  if (!f.is_target && targets.length) {
    const top = [...targets].sort((a, b) => b.strength - a.strength)[0];
    const leak = top.strength >= LEAK_AT;
    out.push({
      key: "target",
      label: leak ? "Leakage" : "Target",
      value: `${top.strength.toFixed(2)} → ${top.name}`,
      short: top.strength.toFixed(2),
      detail: `with ${top.name}`,
      tone: leak ? "bad" : top.strength >= STRONG_TARGET_AT ? "warn" : undefined,
      rule:
        `Strongest association with a target. Amber at ≥ ${STRONG_TARGET_AT}, red (possible leakage) at ≥ ${LEAK_AT}.\n` +
        targets.map((a) => `${a.name}: ${a.value.toFixed(3)}`).join(" · "),
    });
  }
  if (f.n_unique <= 1) {
    out.push({
      key: "constant",
      label: "Constant",
      value: "1 value",
      tone: "bad",
      rule: "Only one distinct value: carries no information.",
    });
  }
  out.push({
    key: "missing",
    label: "Missing",
    value: fmtPct(f.null_pct),
    tone: missingTone(f.null_pct),
    rule: `Amber at ≥ ${MISSING_WARN * 100}% missing, red at ≥ ${MISSING_BAD * 100}%.`,
  });
  if (NUMERIC.has(t)) {
    const skew = f.skewness ?? 0;
    out.push({
      key: "skew",
      label: "Skew",
      value: fmtNum(f.skewness, 2),
      tone: Math.abs(skew) >= 2 ? "warn" : undefined,
      rule: "Amber at |skew| ≥ 2.",
    });
    out.push({
      key: "outliers",
      label: "Outliers",
      value: fmtPct(f.outlier_pct),
      tone: (f.outlier_pct ?? 0) >= 0.05 ? "warn" : undefined,
      rule: "Share of values beyond 1.5 × IQR. Amber at ≥ 5%.",
    });
  }
  if (LEVELED.has(t) && f.n_unique > 1) {
    const idLike = f.unique_pct >= 0.999 && f.row_count > 100;
    const rare = f.stats?.rare_levels;
    out.push({
      key: "levels",
      label: idLike ? "ID-like" : "Levels",
      value: `${fmtInt(f.n_unique)}${rare ? ` (${fmtInt(rare)} rare)` : ""}`,
      tone: idLike ? "bad" : f.n_unique > 50 || f.unique_pct > 0.5 ? "warn" : undefined,
      rule: "Amber above 50 levels or 50% unique; red when every value is unique.",
    });
  }
  const unparseable = f.stats?.unparseable_count ?? 0;
  const infinite = f.stats?.inf_count ?? 0;
  if (unparseable || infinite) {
    const examples = f.stats?.unparseable_examples?.length
      ? `\nExamples: ${f.stats.unparseable_examples.join(", ")}`
      : "";
    out.push({
      key: "coerced",
      label: "Read as missing",
      value: fmtInt(unparseable + infinite),
      tone: "warn",
      rule:
        `Values the export treats as missing, already counted in Missing: ${fmtInt(unparseable)} that don't parse as ` +
        `${t === "datetime" ? "dates" : "numbers"}, ${fmtInt(infinite)} infinite.${examples}`,
    });
  }
  if (f.inference_confidence !== null && f.inference_confidence < LOW_CONFIDENCE) {
    out.push({
      key: "confidence",
      label: "Type confidence",
      value: fmtPct(f.inference_confidence, 0),
      tone: "warn",
      rule: `Inference below ${LOW_CONFIDENCE * 100}% confident: check the type.`,
    });
  }
  return out;
}

export type Attention = "leak" | "missing" | "confidence" | "edited" | null;

/** Why a list row deserves a look, highest priority first. */
export function attentionOf(f: FieldListItem): Attention {
  if (f.action === "target") return null;
  if (f.leakage) return "leak";
  if ((f.null_pct ?? 0) >= MISSING_BAD || (f.n_unique ?? 2) <= 1) return "missing";
  if (f.inference_confidence !== null && f.inference_confidence < LOW_CONFIDENCE) return "confidence";
  if (f.source === "user" && !f.reviewed) return "edited";
  return null;
}
