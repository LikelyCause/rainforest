import { settingProblem } from "./settings";
import type { ModelSetting, SettingValue } from "./types";

/** A sweep starts one run per combination, so it is kept to what a person would read through. */
export const SWEEP_MAX = 50;

export interface SweepAxis { name: string; values: SettingValue[] }

/** The values typed for one setting (comma or space separated), each checked against the setting's own rule. */
export function parseSweep(s: ModelSetting, text: string): { values: SettingValue[]; error: string | null } {
  const byComma = s.kind === "choice" || Boolean(s.none_label);
  const tokens = text.split(byComma ? "," : /[\s,]+/).map((t) => t.trim()).filter(Boolean);
  const values: SettingValue[] = [];
  for (const t of tokens) {
    let v: SettingValue;
    if (s.kind === "choice") v = (s.options ?? []).find((o) => o.label.toLowerCase() === t.toLowerCase() || String(o.value) === t)?.value ?? t;
    else if (s.kind === "bool") v = /^(on|true|yes)$/i.test(t) ? true : /^(off|false|no)$/i.test(t) ? false : t;
    else if (s.none_label && t.toLowerCase() === s.none_label.toLowerCase()) v = s.none_value;
    else v = /^-?\d*\.?\d+(e-?\d+)?$/i.test(t) ? Number(t) : t;
    const problem = settingProblem(s, v);
    if (problem) return { values, error: `${t}: ${problem}` };
    if (!values.includes(v)) values.push(v);
  }
  return { values, error: values.length < 2 ? "Give two or more values to compare" : null };
}

/** Every combination of the axes' values, each laid over the base settings. */
export function combinations(base: Record<string, SettingValue>, axes: SweepAxis[]): Record<string, SettingValue>[] {
  return axes.reduce<Record<string, SettingValue>[]>(
    (acc, axis) => acc.flatMap((c) => axis.values.map((v) => ({ ...c, [axis.name]: v }))), [{ ...base }]);
}
