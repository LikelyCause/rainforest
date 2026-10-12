import { useState } from "react";
import { changed, fmtSetting, GROUP_LABEL, GROUP_ORDER, groupSummary, isVisible, problems, suitsTargets } from "../settings";
import type { ModelSetting, SettingValue } from "../types";
import { Icon } from "./kit";

type Values = Record<string, SettingValue>;

/** A number typed as text, so partial input ("0.", "-") can pass through on the way; committed once it parses. */
function NumberInput({ s, value, onChange, id }: { s: ModelSetting; value: number; onChange: (v: number) => void; id: string }) {
  const [text, setText] = useState(String(value));
  const [last, setLast] = useState(value);
  if (value !== last) {
    setLast(value);
    setText(String(value));
  }
  return (
    // No min/max on the element: a browser counts steps from min, so arrows from a default like 600 would land on 601.
    // The range is checked in settingProblem instead.
    <input id={id} type="number" inputMode="decimal" value={text} step={s.step ?? (s.kind === "float" ? "any" : 1)}
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
      <div className="row exp-nowrap">
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

function Row({ s, values, onChange, problem }: {
  s: ModelSetting; values: Values; onChange: (name: string, v: SettingValue) => void; problem: string | undefined;
}) {
  const value = s.name in values ? values[s.name] : s.default;
  const isChanged = value !== s.default;
  return (
    <div className={`setting ms-row${isChanged ? " edited" : ""}${problem ? " invalid" : ""}`}>
      <label htmlFor={`ms-${s.name}`} title={s.help}>{s.label}</label>
      <Control s={s} value={value} onChange={(v) => onChange(s.name, v)} />
      <div className="ms-help muted small-text">
        {problem ? <span className="ms-problem" role="alert">{s.label} {problem}</span>
          : isChanged ? (
            <span className="ms-default">
              Default {fmtSetting(s, s.default)}{" "}
              <button type="button" className="link small-text" aria-label={`Reset ${s.label} to its default, ${fmtSetting(s, s.default)}`}
                onClick={() => onChange(s.name, s.default)}>Reset</button>
            </span>
          ) : s.help}
      </div>
    </div>
  );
}

/** A model's settings, one folding group per catalog group, each summed up in a line when folded. Groups for class
 * or number targets show only when the run has such a target. A group with a change starts open; Find a setting
 * searches every group. */
export function ModelSettings({ settings, values, onChange, kinds }: {
  settings: ModelSetting[]; values: Values; onChange: (v: Values) => void; kinds: string[];
}) {
  const [q, setQ] = useState("");
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const diff = changed(values, settings, kinds);
  const issues = problems(values, settings, kinds);
  const runChanged = settings.filter((s) => s.group === "run" && s.name in values && values[s.name] !== s.default).length;
  const n = Object.keys(diff).length + runChanged;
  const set = (name: string, v: SettingValue) => onChange({ ...values, [name]: v });
  const visible = (s: ModelSetting) => isVisible(s, values, settings) && suitsTargets(s, kinds);
  const query = q.trim().toLowerCase();
  const matches = (s: ModelSetting) => !query || `${s.label} ${s.name} ${s.help}`.toLowerCase().includes(query);
  const groups = GROUP_ORDER.map((g) => ({ g, rows: settings.filter((s) => s.group === g && visible(s)) })).filter((x) => x.rows.length);
  const found = groups.reduce((a, x) => a + x.rows.filter(matches).length, 0);
  return (
    <section className="card model-settings" aria-labelledby="ms-title">
      <div className="ms-head">
        <h2 id="ms-title">Settings <span className="muted num ms-count">{n ? `${n} changed from defaults` : "all at defaults"}</span></h2>
        <div className="row">
          <label className="ms-search">
            <Icon name="search" size={14} />
            <input type="search" placeholder="Find a setting" aria-label="Find a setting" value={q} onChange={(e) => setQ(e.target.value)} />
          </label>
          <button type="button" className="small" disabled={n === 0} onClick={() => onChange({})}>Reset all</button>
        </div>
      </div>
      {query && found === 0 && <div className="ms-none muted small-text">No setting matches “{q.trim()}”.</div>}
      {groups.map(({ g, rows }) => {
        const shownRows = rows.filter(matches);
        if (query && !shownRows.length) return null;
        const edited = rows.some((s) => s.name in diff || (s.group === "run" && s.name in values && values[s.name] !== s.default));
        const bad = rows.some((s) => issues[s.name]);
        const open = query ? true : toggled[g] ?? (edited || bad);
        return (
          <section key={g} className="ms-group" aria-label={GROUP_LABEL[g]}>
            <button type="button" className="ms-summary" aria-expanded={open}
              onClick={() => !query && setToggled({ ...toggled, [g]: !open })}>
              <Icon name={open ? "chevron-down" : "chevron-right"} />
              <b>{GROUP_LABEL[g]}</b>
              <span className={`ms-line-summary${bad ? " ms-problem" : ""}`}>{bad ? "A setting here needs fixing" : groupSummary(rows, values)}</span>
            </button>
            {open && (
              <div className="ms-grid">
                {shownRows.map((s) => <Row key={s.name} s={s} values={values} onChange={set} problem={issues[s.name]} />)}
              </div>
            )}
          </section>
        );
      })}
    </section>
  );
}
