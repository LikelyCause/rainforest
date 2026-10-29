import { useState } from "react";
import { fmtSetting } from "../settings";
import { addSweepValues, readSweepValue } from "../sweep";
import type { ModelSetting } from "../types";
import { Icon } from "./kit";

/** One setting's values to try: a chip per value (with its own remove), and a box that adds what's typed on Enter, on
 * a comma's worth of values, or when it loses focus. Backspace in the empty box takes the last value back. */
export function SweepValues({
  s,
  values,
  onChange,
}: {
  s: ModelSetting;
  values: string[];
  onChange: (values: string[]) => void;
}) {
  const [draft, setDraft] = useState("");
  const add = () => {
    if (!draft.trim()) return;
    onChange(addSweepValues(s, values, draft));
    setDraft("");
  };
  const remove = (i: number) => onChange(values.filter((_, j) => j !== i));
  const example = s.options
    ? s.options
        .map((o) => o.label)
        .slice(0, 3)
        .join(", ")
    : s.kind === "float"
      ? "0.01, 0.05, 0.1"
      : "15, 31, 63";
  return (
    <div className="sweep-values" role="group" aria-label={`Values for ${s.label}`}>
      {values.map((t, i) => {
        const { value, problem } = readSweepValue(s, t);
        const label = problem ? t : fmtSetting(s, value);
        return (
          <span
            key={`${t}:${i}`}
            className={`sweep-chip num${problem ? " bad" : ""}`}
            title={problem ? `${t}: ${problem}` : undefined}
          >
            {label}
            <button type="button" className="ghost" aria-label={`Remove ${label}`} onClick={() => remove(i)}>
              <Icon name="x" size={12} />
            </button>
          </span>
        );
      })}
      <input
        aria-label={`Add a value for ${s.label}`}
        value={draft}
        placeholder={values.length ? "Add a value" : `e.g. ${example}`}
        onChange={(ev) => setDraft(ev.target.value)}
        onBlur={add}
        onKeyDown={(ev) => {
          if (ev.key === "Enter") {
            ev.preventDefault();
            add();
          } else if (ev.key === "Backspace" && !draft && values.length) {
            ev.preventDefault();
            remove(values.length - 1);
          }
        }}
      />
      <button
        type="button"
        className="ghost small sweep-add"
        disabled={!draft.trim()}
        onMouseDown={(ev) => ev.preventDefault()}
        onClick={add}
      >
        <Icon name="plus" />
        Add a value
      </button>
    </div>
  );
}
