import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type ChangeEvent } from "react";
import { pendingDraft, queueSave } from "../saveQueue";
import { NUMERIC, paramProblem, retype, toInput, withEncoding, type ParamRule } from "../decision";
import type { DecisionInput, Dim, FieldResponse, OptionCatalog } from "../types";

const DIMS: Dim[] = ["imputation", "transform", "scaling", "encoding"];
const DIM_LABEL: Record<Dim, string> = { imputation: "Missing values", transform: "Transform", scaling: "Scaling", encoding: "Encoding" };
const SAVE_DEBOUNCE_MS = 400;

export interface DecisionPanelHandle {
  confirm: () => void;
  revertAndConfirm: () => void;
  toggleKeep: () => void;
  focusNote: () => void;
}

export interface Confirmed {
  fieldId: number;
  name: string;
  before: DecisionInput;
  after: DecisionInput;
}

interface Props {
  data: FieldResponse;
  catalog: OptionCatalog;
  familyLabel: string;
  typing: boolean;
  onConfirmed: (c: Confirmed) => void;
  onShowShortcuts: () => void;
  /** Why this decision can't be changed here; the panel then only shows it. */
  readOnly?: string | null;
  /** Whose decision this is, when it isn't simply the team's ("Your assessment"). */
  owner?: string;
}

function retypeWarning(type: string, data: FieldResponse): string | null {
  const f = data.field;
  if (type === f.inferred_type) return null;
  if (type === "binary" && f.n_unique > 2) return `${f.n_unique.toLocaleString()} distinct values — Binary expects 2.`;
  if (type === "constant" && f.n_unique > 1) return `${f.n_unique.toLocaleString()} distinct values — not constant.`;
  if (NUMERIC.has(type) && !NUMERIC.has(f.inferred_type) && !f.stats?.numeric_like && !f.stats?.stored_as_text)
    return "Values weren't recognised as numbers; non-numeric ones become missing.";
  if (type === "categorical" && f.n_unique > 1000) return `${f.n_unique.toLocaleString()} levels — most will be grouped as rare.`;
  return null;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

export const DecisionPanel = forwardRef<DecisionPanelHandle, Props>(function DecisionPanel(
  { data, catalog, familyLabel, typing, onConfirmed, onShowShortcuts, readOnly, owner }, ref,
) {
  const f = data.field;
  const sugg = f.suggestion;
  const family = data.family;
  const server = toInput(data.decision, family);
  const [draft, setDraft] = useState<DecisionInput>(() => pendingDraft(f.id) ?? server);
  const draftRef = useRef(draft);
  const before = useRef(server);
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const [noteOpen, setNoteOpen] = useState(Boolean(draft.notes));

  const commit = (next: DecisionInput, delay = SAVE_DEBOUNCE_MS) => {
    if (readOnly) return;
    draftRef.current = next;
    setDraft(next);
    void queueSave(f.id, f.dataset_id, f.name, next, delay);
  };
  const update = (patch: Partial<DecisionInput>, delay?: number) => commit({ ...draftRef.current, ...patch }, delay);

  const suggested = toInput(sugg, family);
  const differs = (dim: Dim | "semantic_type" | "action" | "add_missing_indicator") => !same(draft[dim], suggested[dim]);
  const isTarget = draft.action === "target";
  const nonFeature = catalog.non_feature_types.includes(draft.semantic_type);
  const allowed = catalog.allowed[family][draft.semantic_type];
  const notes = catalog.family_notes[family] ?? {};
  const labels = catalog.labels;

  const editableDims = draft.action === "keep" ? DIMS.filter((d) => allowed[d].length > 1 || draft[d] !== allowed[d][0]) : [];
  const lockedDims = draft.action === "keep" ? DIMS.filter((d) => !editableDims.includes(d) && notes[d]) : [];
  const edited = (["action", "semantic_type", ...editableDims, ...(NUMERIC.has(draft.semantic_type) && draft.action === "keep" ? ["add_missing_indicator" as const] : [])] as const)
    .filter((k) => differs(k));

  const confirmLabel = (() => {
    const unchanged = same({ ...draft, notes: null }, { ...server, notes: null }) && draft.reviewed;
    if (unchanged) return "Next";
    const verb = { keep: "keep", drop: "drop", target: "target" }[draft.action];
    const parts = edited.filter((k) => k !== "action").map((k) =>
      k === "semantic_type" ? catalog.types.find((t) => t.value === draft.semantic_type)?.label.toLowerCase() ?? draft.semantic_type
        : k === "add_missing_indicator" ? "missing flag"
        : `${(labels[k as Dim][draft[k as Dim]] ?? draft[k as Dim]).split(" (")[0].toLowerCase()}${k === "imputation" ? " fill" : ""}`);
    const shown = parts.slice(0, 2).join(", ");
    return `Confirm · ${verb}${shown ? `, ${shown}` : ""}${parts.length > 2 ? ` +${parts.length - 2}` : ""}`;
  })();

  const confirm = () => {
    if (readOnly) return;
    const after = { ...draftRef.current, reviewed: true };
    commit(after, 0);
    onConfirmed({ fieldId: f.id, name: f.name, before: before.current, after });
  };
  const revertAndConfirm = () => {
    if (readOnly) return;
    const after = { ...suggested, notes: draftRef.current.notes, reviewed: true };
    commit(after, 0);
    onConfirmed({ fieldId: f.id, name: f.name, before: before.current, after });
  };
  const toggleKeep = () => {
    const d = draftRef.current;
    if (d.action === "target" || (d.action === "drop" && catalog.non_feature_types.includes(d.semantic_type))) return;
    update({ action: d.action === "keep" ? "drop" : "keep" }, 0);
  };
  const focusNote = () => {
    setNoteOpen(true);
    window.setTimeout(() => noteRef.current?.focus(), 0);
  };
  useImperativeHandle(ref, () => ({ confirm, revertAndConfirm, toggleKeep, focusNote }));

  // A select keeps keyboard focus after a mouse change, which used to swallow J/K (P2).
  const blurAfter = (fn: (e: ChangeEvent<HTMLSelectElement>) => void) => (e: ChangeEvent<HTMLSelectElement>) => {
    fn(e);
    e.currentTarget.blur();
  };
  const revert = (patch: Partial<DecisionInput>) => (
    <button type="button" className="ghost" onClick={() => update(patch, 0)} title="Revert this setting to the suggestion">↺</button>
  );
  const suggestedLine = (k: Dim | "semantic_type" | "action", text: string) =>
    differs(k) ? <div className="suggested">suggested: {text} {revert({ [k]: suggested[k] } as Partial<DecisionInput>)}</div> : null;

  const param = (key: string, title: string, step = 1) => (
    <ParamInput key={`${draft.encoding}-${key}`} id={`p-${key}`} title={title} step={step}
      value={draft.params?.[key] ?? catalog.default_params[draft.encoding]?.[key]}
      rule={catalog.param_rules?.[draft.encoding]?.[key]}
      onChange={(n) => update({ params: { ...draftRef.current.params, [key]: n } })} />
  );

  const warning = retypeWarning(draft.semantic_type, data);
  const typeLabel = (v: string) => catalog.types.find((t) => t.value === v)?.label ?? v;

  return (
    <aside className="decision" aria-label="Decision">
      <div className="decision-body">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>{owner ?? "Decision"}</h3>
          <span className="badge" title="Keep / drop, type and notes are shared; the settings below are for this model">{familyLabel}</span>
        </div>
        <fieldset className="bare" disabled={Boolean(readOnly)}>

          <div className={`setting${differs("action") ? " edited" : ""}`}>
            <div className="segmented" role="radiogroup" aria-label="Keep, drop or target">
              {(["keep", "drop", "target"] as const).map((a) => (
                <button key={a} type="button" role="radio" aria-checked={draft.action === a}
                  className={draft.action === a ? `on ${a}` : ""}
                  disabled={a === "keep" && nonFeature && !isTarget}
                  title={a === "keep" && nonFeature ? `${draft.semantic_type} fields can't be model features` : a === "target" ? "Use as a target column" : "X toggles keep / drop"}
                  onClick={() => update({ action: a }, 0)}>
                  {a[0].toUpperCase() + a.slice(1)}
                </button>
              ))}
            </div>
            {suggestedLine("action", sugg.action)}
          </div>

          <div className={`setting${differs("semantic_type") ? " edited" : ""}`}>
            <div className="setting-row">
              <label htmlFor="d-type">Type</label>
              <select id="d-type" value={draft.semantic_type}
                onChange={blurAfter((e) => commit(retype(draftRef.current, e.target.value, catalog, sugg), 0))}>
                {catalog.types.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            {warning && (
              <div className="inline-warn">
                {warning}{" "}
                <button type="button" className="link small-text" onClick={() => commit(retype(draftRef.current, f.inferred_type, catalog, sugg), 0)}>
                  Undo retype
                </button>
              </div>
            )}
            {!warning && suggestedLine("semantic_type", typeLabel(sugg.semantic_type))}
          </div>

          {editableDims.map((dim) => (
            <div key={dim} className={`setting${differs(dim) ? " edited" : ""}`}>
              {allowed[dim].length > 1 ? (
                <div className="setting-row">
                  <label htmlFor={`d-${dim}`}>{DIM_LABEL[dim]}</label>
                  <select id={`d-${dim}`} value={draft[dim]}
                    onChange={blurAfter((e) => dim === "encoding"
                      ? commit(withEncoding(draftRef.current, e.target.value, catalog), 0)
                      : update({ [dim]: e.target.value } as Partial<DecisionInput>, 0))}>
                    {allowed[dim].map((o) => <option key={o} value={o}>{labels[dim][o] ?? o}</option>)}
                  </select>
                </div>
              ) : (
                <div className="setting-row">
                  <span>{DIM_LABEL[dim]}</span>
                  <span className="secondary" title={notes[dim]}>{labels[dim][draft[dim]] ?? draft[dim]} (fixed for {familyLabel.toLowerCase()}) {notes[dim] && "ⓘ"}</span>
                </div>
              )}
              {suggestedLine(dim, (labels[dim][sugg[dim]] ?? sugg[dim]).split(" (")[0])}
              {dim === "imputation" && draft.imputation === "constant" && (
                <div className="setting-row">
                  <label htmlFor="d-fill">{["categorical", "binary"].includes(draft.semantic_type) ? "Missing label" : "Fill value"}</label>
                  <input id="d-fill" value={draft.impute_value ?? ""} inputMode={NUMERIC.has(draft.semantic_type) ? "decimal" : undefined}
                    onChange={(e) => update({ impute_value: e.target.value })} />
                </div>
              )}
              {dim === "encoding" && draft.encoding === "onehot" && <>{param("max_categories", "Max categories")}{param("min_frequency", "Min frequency", 0.005)}</>}
              {dim === "encoding" && draft.encoding === "ordinal" && param("min_frequency", "Min frequency", 0.001)}
              {dim === "encoding" && draft.encoding === "tfidf_svd" && <>{param("svd_components", "SVD components")}{param("max_features", "Max vocabulary", 1000)}</>}
            </div>
          ))}

          {draft.action === "keep" && NUMERIC.has(draft.semantic_type) && (
            <div className={`setting${differs("add_missing_indicator") ? " edited" : ""}`}>
              <div className="setting-row">
                <span>Missing flag</span>
                <label className="check">
                  <input type="checkbox" checked={draft.add_missing_indicator}
                    onChange={(e) => update({ add_missing_indicator: e.target.checked }, 0)} />
                  add indicator column
                </label>
              </div>
            </div>
          )}

          {lockedDims.length > 0 && (
            <div className="locked-line" title={lockedDims.map((d) => notes[d]).join("\n")}>
              {lockedDims.map((d) => DIM_LABEL[d]).join(", ")}: not used for {familyLabel.toLowerCase()} ⓘ
            </div>
          )}

          {sugg.reasons.length > 0 && (
            <div className="why">
              <h3>Why these suggestions</h3>
              <ul>{sugg.reasons.map((r) => <li key={r}><Reason text={r} /></li>)}</ul>
            </div>
          )}

          {noteOpen ? (
            <label className="field">Note
              <textarea ref={noteRef} value={draft.notes ?? ""} placeholder="Why this choice?"
                onChange={(e) => update({ notes: e.target.value || null })} />
            </label>
          ) : (
            <button type="button" className="link" style={{ justifyContent: "space-between", width: "100%" }} onClick={focusNote}>
              + add a note <kbd>N</kbd>
            </button>
          )}
        </fieldset>
      </div>

      {readOnly ? (
        <div className="decision-foot"><span className="secondary small-text">{readOnly}</span></div>
      ) : (
        <div className="decision-foot">
          <div className="foot-line">
            <span>
              {typing ? "Esc to return to review"
                : draft.reviewed && server.reviewed ? "Reviewed ✓"
                : edited.length ? `${edited.length} setting${edited.length === 1 ? "" : "s"} differ${edited.length === 1 ? "s" : ""} from the suggestion`
                : "Matches the suggestion"}
            </span>
            <button type="button" className="link small-text" onClick={onShowShortcuts}>? shortcuts</button>
          </div>
          <button type="button" className="primary confirm" onClick={confirm}>
            <span>{confirmLabel}</span><kbd>↵</kbd>
          </button>
          {edited.length > 0 && (
            <button type="button" onClick={() => commit({ ...suggested, notes: draftRef.current.notes, reviewed: draftRef.current.reviewed }, 0)}>
              Revert all to suggestion ↺
            </button>
          )}
        </div>
      )}
    </aside>
  );
});

/** A number field that can be cleared and retyped: only valid values are saved, and leaving it while it's
 * empty or out of range puts the saved value back. */
function ParamInput({ id, title, step, value, rule, onChange }: {
  id: string; title: string; step: number; value: number | undefined; rule: ParamRule | undefined;
  onChange: (n: number) => void;
}) {
  const [text, setText] = useState(value === undefined ? "" : String(value));
  useEffect(() => {
    if (value !== undefined && Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only an outside change (revert, retype) resyncs
  }, [value]);
  const problem = paramProblem(text, rule);
  return (
    <div className="setting-row">
      <label htmlFor={id}>{title}</label>
      <input id={id} type="number" step={step} min={rule?.min} max={rule?.max} value={text}
        aria-invalid={problem !== null} title={problem ?? undefined}
        onChange={(e) => {
          setText(e.target.value);
          if (paramProblem(e.target.value, rule) === null) onChange(Number(e.target.value));
        }}
        onBlur={() => problem && setText(value === undefined ? "" : String(value))} />
    </div>
  );
}

/** "Setting: explanation" with the setting in bold; reasons from before the labels show as they are. */
function Reason({ text }: { text: string }) {
  const m = /^([A-Z][\w /-]{1,24}): (.*)$/.exec(text);
  return m ? <><b>{m[1]}</b>: {m[2]}</> : <>{text}</>;
}
