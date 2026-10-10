import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type ChangeEvent } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { pendingDraft, queueSave } from "../saveQueue";
import { defaultTask, NUMERIC, paramProblem, parseEdges, retype, taskLabel, toInput, withEncoding, type ParamRule } from "../decision";
import { baselineDiff, revertedInput } from "../experiments";
import type { DecisionInput, Dim, FieldResponse, OptionCatalog, TargetBins, TargetTask } from "../types";
import type { View } from "../view";

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
  /** The decision source (see view.ts); null for the baseline in progress. */
  view?: View;
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
  { data, catalog, familyLabel, typing, onConfirmed, onShowShortcuts, readOnly, owner, view = null }, ref,
) {
  const f = data.field;
  const sugg = f.suggestion;
  const family = data.family;
  const server = toInput(data.decision, family);
  const [draft, setDraft] = useState<DecisionInput>(() => pendingDraft(f.id, view) ?? server);
  const draftRef = useRef(draft);
  const before = useRef(server);
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const [noteOpen, setNoteOpen] = useState(Boolean(draft.notes));

  const commit = (next: DecisionInput, delay = SAVE_DEBOUNCE_MS) => {
    if (readOnly) return;
    draftRef.current = next;
    setDraft(next);
    void queueSave(f.id, f.dataset_id, f.name, next, delay, view);
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

          {isTarget && (() => {
            const fallback = defaultTask(draft.semantic_type, f.n_unique);
            const task = draft.target_task ?? fallback;
            const allowedTask = (t: TargetTask) => (t === "classes"
              ? draft.semantic_type !== "numeric_continuous" && (!NUMERIC.has(draft.semantic_type) || (f.n_unique ?? 0) <= catalog.target_rule.max_levels)
              : NUMERIC.has(draft.semantic_type));
            const hint = (t: TargetTask) => (allowedTask(t) ? undefined
              : t === "classes" ? (draft.semantic_type === "numeric_continuous" ? "A continuous field is predicted as a number or as bands"
                : `More than ${catalog.target_rule.max_levels} distinct values`)
              : "Only numeric fields can be predicted as a number or as bands");
            return (
              <div className={`setting predict-as${task !== fallback ? " edited" : ""}`}>
                <span className="setting-label">Predict as</span>
                <div className="segmented full" role="radiogroup" aria-label="Predict as">
                    {(["classes", "regression", "bins"] as const).map((t) => (
                      <button key={t} type="button" role="radio" aria-checked={task === t} className={task === t ? "on" : ""}
                        disabled={!allowedTask(t)} title={hint(t)}
                        onClick={() => update({ target_task: t, target_bins: t === "bins" ? draftRef.current.target_bins ?? { k: catalog.target_rule.default_k } : null }, 0)}>
                        {taskLabel(t)}
                      </button>
                    ))}
                </div>
                {task !== fallback && fallback && <div className="suggested">default for this type: {taskLabel(fallback).toLowerCase()}</div>}
                {task === "bins" && (
                  <BandSettings fieldId={f.id} bins={draft.target_bins ?? { k: catalog.target_rule.default_k }} readOnly={Boolean(readOnly)}
                    onChange={(b) => update({ target_task: "bins", target_bins: b }, 0)} />
                )}
              </div>
            );
          })()}

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

          {data.baseline !== undefined && <BaselineLine draft={draft} data={data} readOnly={Boolean(readOnly)}
            onRevert={() => data.baseline && commit(revertedInput(draftRef.current, data.baseline), 0)} />}

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

/** In an experiment: whether this field differs from its baseline, and a way back. */
function BaselineLine({ draft, data, readOnly, onRevert }: {
  draft: DecisionInput; data: FieldResponse; readOnly: boolean; onRevert: () => void;
}) {
  if (!data.baseline) return <div className="baseline-line small-text muted">Not in the baseline.</div>;
  const changed = baselineDiff(draft, data.baseline);
  return (
    <div className={`baseline-line small-text${changed.length ? " changed" : ""}`}>
      {changed.length === 0 ? <span className="muted">Same as the baseline</span> : (
        <>
          <span className="grow">Changed from the baseline: <b>{changed.join(", ")}</b></span>
          {!readOnly && <button type="button" className="link" onClick={onRevert}>Revert to baseline</button>}
        </>
      )}
    </div>
  );
}

/** Equal-sized bands or chosen cut points, with how the field's rows fall into them. Cut points are saved once they
 * parse; until then the last valid setting stays. */
function BandSettings({ fieldId, bins, readOnly, onChange }: {
  fieldId: number; bins: TargetBins; readOnly: boolean; onChange: (b: TargetBins) => void;
}) {
  const [mode, setMode] = useState<"k" | "edges">("k" in bins ? "k" : "edges");
  const [text, setText] = useState("edges" in bins ? bins.edges.join(", ") : "");
  const parsed = parseEdges(text);
  const preview = useQuery({
    queryKey: ["bands", fieldId, JSON.stringify(bins)],
    queryFn: () => api.bands(fieldId, bins),
    placeholderData: keepPreviousData,
  });
  const max = Math.max(1, ...(preview.data?.counts ?? [1]));
  const pick = (m: "k" | "edges") => {
    setMode(m);
    if (m === "k") onChange({ k: "k" in bins ? bins.k : 5 });
    else if (typeof parsed !== "string") onChange({ edges: parsed });
  };
  return (
    <div className="stack band-settings" style={{ gap: 6 }}>
      <div className="segmented full soft" role="radiogroup" aria-label="Bands">
        {(["k", "edges"] as const).map((m) => (
          <button key={m} type="button" role="radio" aria-checked={mode === m} className={mode === m ? "on" : ""} disabled={readOnly}
            onClick={() => pick(m)}>{m === "k" ? "Equal-sized" : "Cut points"}</button>
        ))}
      </div>
      {mode === "k" ? (
        <div className="setting-row">
          <label htmlFor="d-bands-k">Bands</label>
          <input id="d-bands-k" type="number" min={2} max={20} step={1} value={"k" in bins ? bins.k : 5} disabled={readOnly}
            onChange={(e) => { const k = Number(e.target.value); if (Number.isInteger(k) && k >= 2 && k <= 20) onChange({ k }); }} />
        </div>
      ) : (
        <div className="setting-row">
          <label htmlFor="d-bands-edges">Cut at</label>
          <input id="d-bands-edges" value={text} placeholder="10, 25, 80" disabled={readOnly}
            onChange={(e) => { setText(e.target.value); const p = parseEdges(e.target.value); if (typeof p !== "string") onChange({ edges: p }); }} />
        </div>
      )}
      {mode === "edges" && text.trim() !== "" && typeof parsed === "string" && <div className="inline-warn">{parsed}</div>}
      {mode === "edges" && !("edges" in bins) ? <span className="muted small-text">Type cut points to use them.</span>
        : preview.error ? <div className="inline-warn">{(preview.error as Error).message}</div> : preview.data && (
        <ul className="band-preview" aria-label="Rows per band">
          {preview.data.labels.map((l, i) => (
            <li key={l}>
              <span className="mono">{l}</span>
              <span className="band-bar" style={{ width: `${Math.round((preview.data!.counts[i] / max) * 100)}%` }} />
              <span className="num">{preview.data!.counts[i].toLocaleString()}</span>
            </li>
          ))}
        </ul>
      )}
      <span className="muted small-text">Equal-sized bands are cut again on each export's training rows.</span>
    </div>
  );
}
