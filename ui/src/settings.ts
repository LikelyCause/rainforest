import type { ModelSetting, SettingValue } from "./types";

export const GROUP_ORDER = ["run", "trees", "learning", "sampling", "regularization", "classes", "numbers", "advanced"] as const;
export const GROUP_LABEL: Record<string, string> = {
  run: "Run", trees: "Trees", learning: "Learning", sampling: "Sampling", regularization: "Regularization",
  classes: "For class targets", numbers: "For number targets", advanced: "Advanced",
};

const fmt = (v: number | null) => (v === null ? "" : String(Number(v.toPrecision(6))));

/** What's wrong with a value for a setting (the API's rule), or null. */
export function settingProblem(s: ModelSetting, v: SettingValue | undefined): string | null {
  if (s.kind === "choice") {
    return (s.options ?? []).some((o) => o.value === v) ? null : `must be one of ${(s.options ?? []).map((o) => o.label).join(", ")}`;
  }
  if (s.kind === "bool") return typeof v === "boolean" ? null : "must be on or off";
  if (s.kind === "int_or_none" && v === s.none_value) return null;
  if (typeof v !== "number" || !Number.isFinite(v)) return "must be a number";
  if ((s.kind === "int" || s.kind === "int_or_none") && !Number.isInteger(v)) return "must be a whole number";
  if ((s.min !== null && v < s.min) || (s.max !== null && v > s.max)) {
    return `must be from ${fmt(s.min)} to ${fmt(s.max)}${s.none_label ? ` (or ${s.none_label})` : ""}`;
  }
  return null;
}

const current = (name: string, values: Record<string, SettingValue>, settings: ModelSetting[]) =>
  name in values ? values[name] : settings.find((s) => s.name === name)?.default ?? null;

/** Whether a setting shows: every setting it depends on has one of the values it needs. */
export function isVisible(s: ModelSetting, values: Record<string, SettingValue>, settings: ModelSetting[]): boolean {
  return Object.entries(s.when ?? {}).every(([other, allowed]) => allowed.includes(current(other, values, settings)));
}

/** The settings a run sends: visible, not run-level, and different from their default. */
export function changed(values: Record<string, SettingValue>, settings: ModelSetting[]): Record<string, SettingValue> {
  const out: Record<string, SettingValue> = {};
  for (const s of settings) {
    if (s.group === "run" || !(s.name in values) || values[s.name] === s.default) continue;
    if (isVisible(s, values, settings)) out[s.name] = values[s.name];
  }
  return out;
}

/** Remembered values that still fit the catalog: unknown names and values that break their rule are dropped. */
export function restore(saved: unknown, settings: ModelSetting[]): Record<string, SettingValue> {
  if (!saved || typeof saved !== "object" || Array.isArray(saved)) return {};
  const out: Record<string, SettingValue> = {};
  for (const [k, v] of Object.entries(saved as Record<string, SettingValue>)) {
    const s = settings.find((x) => x.name === k);
    if (s && settingProblem(s, v) === null) out[k] = v;
  }
  return out;
}

/** Seed and chain folds, from the values or their defaults. */
export function runFields(values: Record<string, SettingValue>, settings: ModelSetting[]): { seed: number; cv: number } {
  return { seed: Number(current("seed", values, settings) ?? 42), cv: Number(current("cv", values, settings) ?? 5) };
}

/** Visible settings with a problem, by name. */
export function problems(values: Record<string, SettingValue>, settings: ModelSetting[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const s of settings) {
    if (!isVisible(s, values, settings)) continue;
    const p = settingProblem(s, current(s.name, values, settings));
    if (p) out[s.name] = p;
  }
  return out;
}

/** "num_leaves 127 · learning_rate 0.02", labelled when the catalog is known. */
export function describeSettings(params: Record<string, SettingValue>, settings: ModelSetting[] | null | undefined): string {
  const entries = Object.entries(params);
  if (!entries.length) return "defaults";
  return entries.map(([k, v]) => {
    const s = settings?.find((x) => x.name === k);
    const shown = s?.kind === "choice" ? s.options?.find((o) => o.value === v)?.label ?? String(v)
      : s?.kind === "int_or_none" && v === s.none_value ? s.none_label ?? String(v)
      : typeof v === "boolean" ? (v ? "on" : "off") : String(v);
    return `${s?.label ?? k} ${shown}`;
  }).join(" · ");
}
