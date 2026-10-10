import { useState } from "react";
import { changed, GROUP_LABEL, GROUP_ORDER, isVisible, settingProblem } from "../settings";
import type { ModelSetting, SettingValue } from "../types";

type Values = Record<string, SettingValue>;

const shown = (s: ModelSetting, v: SettingValue): string => {
  if (s.kind === "choice") return s.options?.find((o) => o.value === v)?.label ?? String(v);
  if (s.kind === "bool") return v ? "on" : "off";
  if (s.kind === "int_or_none" && v === s.none_value) return s.none_label ?? String(v);
  return String(v);
};

/** A number typed as text, so partial input ("0.", "-") can pass through on the way; committed once it parses. */
function NumberInput({ s, value, onChange, id }: { s: ModelSetting; value: number; onChange: (v: number) => void; id: string }) {
  const [text, setText] = useState(String(value));
  const [last, setLast] = useState(value);
  if (value !== last) {
    setLast(value);
    setText(String(value));
  }
  return (
    <input id={id} type="number" inputMode="decimal" value={text} min={s.min ?? undefined} max={s.max ?? undefined}
      step={s.step ?? (s.kind === "float" ? "any" : 1)}
      onChange={(e) => {
        setText(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value.trim() !== "" && Number.isFinite(n)) {
          setLast(n);
          onChange(n);
        }
      }}
      onBlur={() => setText(String(value))} />
  );
}

function Control({ s, value, onChange }: { s: ModelSetting; value: SettingValue; onChange: (v: SettingValue) => void }) {
  const id = `ms-${s.name}`;
  if (s.kind === "choice") {
    return (
      <select id={id} value={JSON.stringify(value)} onChange={(e) => onChange(JSON.parse(e.target.value))}>
        {(s.options ?? []).map((o) => <option key={JSON.stringify(o.value)} value={JSON.stringify(o.value)} title={o.help}>{o.label}</option>)}
      </select>
    );
  }
  if (s.kind === "bool") {
    return (
      <label className="check">
        <input id={id} type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        {value ? "On" : "Off"}
      </label>
    );
  }
  if (s.kind === "int_or_none") {
    const none = value === s.none_value;
    return (
      <div className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
        <select aria-label={`${s.label}: limit`} value={none ? "none" : "set"}
          onChange={(e) => onChange(e.target.value === "none" ? s.none_value : Math.max(s.min ?? 1, Math.min(s.max ?? 8, 8)))}>
          <option value="none">{s.none_label}</option>
          <option value="set">Set</option>
        </select>
        {!none && <NumberInput s={s} id={id} value={Number(value)} onChange={onChange} />}
      </div>
    );
  }
  return <NumberInput s={s} id={id} value={Number(value)} onChange={onChange} />;
}

function Row({ s, values, onChange }: { s: ModelSetting; values: Values; onChange: (name: string, v: SettingValue) => void }) {
  const value = s.name in values ? values[s.name] : s.default;
  const isChanged = value !== s.default;
  const problem = settingProblem(s, value);
  return (
    <div className={`setting ms-row${isChanged ? " edited" : ""}`}>
      <div className="ms-line">
        <label htmlFor={`ms-${s.name}`} title={s.help}>{s.label}</label>
        <Control s={s} value={value} onChange={(v) => onChange(s.name, v)} />
        {isChanged
          ? <button type="button" className="ghost small" title={`Back to the default (${shown(s, s.default)})`} onClick={() => onChange(s.name, s.default)}>↺</button>
          : <span />}
      </div>
      <div className="ms-help muted small-text">
        {problem ? <span className="ms-problem">{s.label} {problem}</span> : <>{s.help} <span className="ms-default">default {shown(s, s.default)}</span></>}
      </div>
    </div>
  );
}

/** A model's settings, grouped as the catalog says. Groups for class or number targets show only when the export
 * has such a target; Advanced is folded, with a search. */
export function ModelSettings({ settings, values, onChange, kinds }: {
  settings: ModelSetting[]; values: Values; onChange: (v: Values) => void; kinds: string[];
}) {
  const [open, setOpen] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [q, setQ] = useState("");
  const diff = changed(values, settings);
  const runChanged = settings.filter((s) => s.group === "run" && s.name in values && values[s.name] !== s.default).length;
  const n = Object.keys(diff).length + runChanged;
  const hasClasses = kinds.some((k) => k !== "regression");
  const hasNumbers = kinds.includes("regression");
  const set = (name: string, v: SettingValue) => onChange({ ...values, [name]: v });
  const visible = (s: ModelSetting) => isVisible(s, values, settings)
    && (s.applies !== "classes" || hasClasses) && (s.applies !== "numbers" || hasNumbers);
  const inAdvanced = settings.filter((s) => s.group === "advanced" && visible(s));
  const advancedChanged = inAdvanced.filter((s) => s.name in diff).length;
  const query = q.trim().toLowerCase();
  return (
    <div className="stack model-settings" style={{ gap: 8 }}>
      <button type="button" className="ms-summary" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>{open ? "▾" : "▸"} Model settings</span>
        <span className={n ? "badge accent" : "muted small-text"}>{n ? `${n} changed` : "Defaults"}</span>
      </button>
      {open && (
        <div className="stack" style={{ gap: 12 }}>
          {n > 0 && (
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button type="button" className="link small-text" onClick={() => onChange({})}>Reset all</button>
            </div>
          )}
          {GROUP_ORDER.filter((g) => g !== "advanced").map((g) => {
            const rows = settings.filter((s) => s.group === g && visible(s));
            if (!rows.length) return null;
            return (
              <section key={g} className="stack ms-group" style={{ gap: 2 }} aria-label={GROUP_LABEL[g]}>
                <h4>{GROUP_LABEL[g]}</h4>
                {rows.map((s) => <Row key={s.name} s={s} values={values} onChange={set} />)}
              </section>
            );
          })}
          {inAdvanced.length > 0 && (
            <section className="stack ms-group" style={{ gap: 2 }} aria-label="Advanced">
              <button type="button" className="ms-summary" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>
                <span>{advanced ? "▾" : "▸"} Advanced · {inAdvanced.length} settings</span>
                {advancedChanged > 0 && <span className="badge accent">{advancedChanged} changed</span>}
              </button>
              {advanced && (
                <>
                  <input type="search" placeholder="Search settings" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search settings" />
                  {inAdvanced.filter((s) => !query || `${s.label} ${s.name} ${s.help}`.toLowerCase().includes(query))
                    .map((s) => <Row key={s.name} s={s} values={values} onChange={set} />)}
                </>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
