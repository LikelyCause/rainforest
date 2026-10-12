import { settingProblem } from "./settings";
import type { ModelSetting, SettingValue } from "./types";

/** A sweep starts one run per combination, so it is kept to what a person would read through. */
export const SWEEP_MAX = 50;

export interface SweepAxis { name: string; values: SettingValue[] }

/** Typed text cut into values: by comma for choices and settings with a "none" label (whose labels can hold spaces),
 * else by comma or space. */
export function splitSweep(s: ModelSetting, text: string): string[] {
  const byComma = s.kind === "choice" || Boolean(s.none_label);
  return text.split(byComma ? "," : /[\s,]+/).map((t) => t.trim()).filter(Boolean);
}

/** One typed value as the setting's own kind (a choice by its label or value, on/off, the none label, a number), and
 * what's wrong with it, if anything. */
export function readSweepValue(s: ModelSetting, t: string): { value: SettingValue; problem: string | null } {
  let v: SettingValue;
  if (s.kind === "choice") v = (s.options ?? []).find((o) => o.label.toLowerCase() === t.toLowerCase() || String(o.value) === t)?.value ?? t;
  else if (s.kind === "bool") v = /^(on|true|yes)$/i.test(t) ? true : /^(off|false|no)$/i.test(t) ? false : t;
  else if (s.none_label && t.toLowerCase() === s.none_label.toLowerCase()) v = s.none_value;
  else v = /^-?\d*\.?\d+(e-?\d+)?$/i.test(t) ? Number(t) : t;
  return { value: v, problem: settingProblem(s, v) };
}

/** The values typed for one setting (comma or space separated), each checked against the setting's own rule. */
export function parseSweep(s: ModelSetting, text: string): { values: SettingValue[]; error: string | null } {
  const values: SettingValue[] = [];
  for (const t of splitSweep(s, text)) {
    const { value: v, problem } = readSweepValue(s, t);
    if (problem) return { values, error: `${t}: ${problem}` };
    if (!values.includes(v)) values.push(v);
  }
  return { values, error: values.length < 2 ? "Give two or more values to compare" : null };
}

/** Adds what was typed to a setting's value list, one entry per value, skipping any it already holds. */
export function addSweepValues(s: ModelSetting, list: string[], text: string): string[] {
  const out = [...list];
  const held = new Set(out.map((t) => String(readSweepValue(s, t).value)));
  for (const t of splitSweep(s, text)) {
    const key = String(readSweepValue(s, t).value);
    if (!held.has(key)) { held.add(key); out.push(t); }
  }
  return out;
}

/** Every combination of the axes' values, each laid over the base settings. */
export function combinations(base: Record<string, SettingValue>, axes: SweepAxis[]): Record<string, SettingValue>[] {
  return axes.reduce<Record<string, SettingValue>[]>(
    (acc, axis) => acc.flatMap((c) => axis.values.map((v) => ({ ...c, [axis.name]: v }))), [{ ...base }]);
}
