import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../api";
import type { Decision, DecisionInput, Dim, FieldListItem, FieldResponse, OptionCatalog, Suggestion } from "../types";
const NUMERIC = new Set(["numeric_continuous", "numeric_discrete"]);
const SAVE_DEBOUNCE_MS = 400;

export interface DecisionPanelHandle {
  markReviewed: () => void;
  acceptSuggestion: () => void;
  toggleKeep: () => void;
}

interface Props {
  data: FieldResponse;
  catalog: OptionCatalog;
  onAdvance: () => void;
}

function toInput(d: Decision | Suggestion, family: string): DecisionInput {
  return {
    family,
    action: d.action,
    semantic_type: d.semantic_type,
    imputation: d.imputation,
    impute_value: d.impute_value ?? null,
    add_missing_indicator: d.add_missing_indicator ?? false,
    transform: d.transform,
    scaling: d.scaling,
    encoding: d.encoding,
    params: d.params ?? {},
    notes: "notes" in d ? d.notes : null,
    reviewed: "reviewed" in d ? d.reviewed : false,
  };
}

/** Re-type a draft: keep compatible settings, otherwise fall back to the suggestion or the first valid option. */
function retype(d: DecisionInput, type: string, catalog: OptionCatalog, sugg: Suggestion): DecisionInput {
  const allowed = catalog.allowed[d.family][type];
  const handlesNan = catalog.families.find((f) => f.value === d.family)?.handles_nan ?? true;
  const pick = (dim: Dim) => {
    if (sugg.semantic_type === type) return sugg[dim];
    if (allowed[dim].includes(d[dim])) return d[dim];
    if (dim === "imputation" && NUMERIC.has(type)) return handlesNan ? "none" : "median";
    return allowed[dim][0];
  };
  const encoding = pick("encoding");
  const nonFeature = catalog.non_feature_types.includes(type);
  return {
    ...d,
    semantic_type: type,
    action: d.action === "target" ? "target" : nonFeature ? "drop" : d.action,
    imputation: pick("imputation"),
    transform: pick("transform"),
    scaling: pick("scaling"),
    encoding,
    impute_value: type === "categorical" || type === "binary" ? "__missing__" : type === "text" ? "" : null,
    params: sugg.semantic_type === type ? sugg.params : { ...(catalog.default_params[encoding] ?? {}) },
    add_missing_indicator: NUMERIC.has(type) ? d.add_missing_indicator : false,
  };
}

function suggestionParts(s: Suggestion): string[] {
  return [
    s.imputation !== "none" && `impute ${s.imputation}`,
    s.transform !== "none" && s.transform,
    s.scaling !== "none" && `${s.scaling} scale`,
    s.encoding !== "none" && s.encoding,
    s.add_missing_indicator && "+ missing flag",
  ].filter((x): x is string => Boolean(x));
}

export const DecisionPanel = forwardRef<DecisionPanelHandle, Props>(function DecisionPanel({ data, catalog, onAdvance }, ref) {
  const qc = useQueryClient();
  const fieldId = data.field.id;
  const datasetId = data.field.dataset_id;
  const sugg = data.field.suggestion;
  const family = data.family;
  const familyInfo = catalog.families.find((f) => f.value === family);
  const notes = catalog.family_notes[family] ?? {};
  const [draft, setDraft] = useState<DecisionInput>(() => toInput(data.decision, family));
  const draftRef = useRef(draft);
  const [state, setState] = useState<{ kind: "idle" | "saving" | "saved" | "error"; msg?: string }>({ kind: "idle" });
  const pending = useRef<DecisionInput | null>(null);
  const timer = useRef<number | undefined>(undefined);

  const applyToCache = useCallback((saved: Decision) => {
    qc.setQueryData<FieldResponse>(["field", fieldId, family], (old) => (old ? { ...old, decision: saved } : old));
    // Shared parts (keep/drop, type, notes) changed for every family; refetch the others when shown.
    qc.invalidateQueries({ queryKey: ["field", fieldId], predicate: (q) => q.queryKey[2] !== family });
    qc.setQueryData<FieldListItem[]>(["fields", datasetId], (list) =>
      list?.map((f) => (f.id === fieldId ? { ...f, action: saved.action, semantic_type: saved.semantic_type, reviewed: saved.reviewed, source: saved.source } : f)),
    );
    qc.invalidateQueries({ queryKey: ["summary", datasetId] });
    qc.invalidateQueries({ queryKey: ["exportPreview", datasetId] });
  }, [qc, fieldId, datasetId, family]);

  const send = useCallback(async (d: DecisionInput) => {
    setState({ kind: "saving" });
    try {
      const saved = await api.saveDecision(fieldId, d);
      applyToCache(saved);
      setState({ kind: "saved" });
    } catch (e) {
      const err = e as ApiError;
      setState({ kind: "error", msg: [err.message, ...(err.details ?? [])].join(" — ") });
    }
  }, [fieldId, applyToCache]);

  const flush = useCallback(() => {
    window.clearTimeout(timer.current);
    if (pending.current) {
      const d = pending.current;
      pending.current = null;
      return send(d);
    }
    return Promise.resolve();
  }, [send]);

  // Persist any unsaved edit when leaving this field (the panel is keyed by field id).
  useEffect(() => () => { void flush(); }, [flush]);

  const commit = (next: DecisionInput, immediate: boolean) => {
    draftRef.current = next;
    setDraft(next);
    pending.current = next;
    window.clearTimeout(timer.current);
    if (immediate) void flush();
    else timer.current = window.setTimeout(() => void flush(), SAVE_DEBOUNCE_MS);
    setState({ kind: "idle" });
  };
  const update = (patch: Partial<DecisionInput>, opts: { immediate?: boolean } = {}) =>
    commit({ ...draftRef.current, ...patch }, opts.immediate ?? false);

  const markReviewed = () => {
    update({ reviewed: true }, { immediate: true });
    onAdvance();
  };
  const acceptSuggestion = () => {
    commit({ ...toInput(sugg, family), notes: draftRef.current.notes, reviewed: true }, true);
    onAdvance();
  };
  const nonFeature = catalog.non_feature_types.includes(draft.semantic_type);
  const toggleKeep = () => {
    if (draft.action === "target") return;
    if (draft.action === "drop" && nonFeature) return;
    update({ action: draft.action === "keep" ? "drop" : "keep" });
  };
  useImperativeHandle(ref, () => ({ markReviewed, acceptSuggestion, toggleKeep }));

  const allowed = catalog.allowed[family][draft.semantic_type];
  const labels = catalog.labels;
  const typeAllowsAny = (dim: Dim) =>
    catalog.families.some((f) => catalog.allowed[f.value][draft.semantic_type][dim].length > 1);
  const select = (dim: Dim, title: string) => {
    if (allowed[dim].length === 1 && draft[dim] === allowed[dim][0]) {
      // Locked by the model family (e.g. scaling for trees): say why instead of hiding it silently.
      return notes[dim] && typeAllowsAny(dim) ? (
        <div className="locked"><b>{title}</b>{notes[dim]}</div>
      ) : null;
    }
    return (
      <label className="field">{title}
        <select value={draft[dim]} onChange={(e) => update({ [dim]: e.target.value } as Partial<DecisionInput>)}>
          {allowed[dim].map((o) => <option key={o} value={o}>{labels[dim][o] ?? o}</option>)}
        </select>
        {notes[dim] && allowed[dim].length > 1 && <span className="muted" style={{ fontWeight: 400 }}>{notes[dim]}</span>}
      </label>
    );
  };
  const param = (key: string, title: string, step = 1) => (
    <label className="field" key={key}>{title}
      <input type="number" step={step} value={draft.params?.[key] ?? catalog.default_params[draft.encoding]?.[key] ?? ""}
        onChange={(e) => update({ params: { ...draft.params, [key]: e.target.value === "" ? undefined : Number(e.target.value) } })} />
    </label>
  );

  const isTarget = draft.action === "target";
  const differsFromSuggestion =
    (["action", "semantic_type", "imputation", "transform", "scaling", "encoding"] as const).some((k) => draft[k] !== sugg[k]) ||
    draft.add_missing_indicator !== (sugg.add_missing_indicator ?? false);

  return (
    <aside className="decision" aria-label="Decision">
      <div className="section">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>Decision</h3>
          <span className="family-tag" title="Keep / drop, type and notes are shared; the preprocessing below is per model family">
            for {familyInfo?.label.toLowerCase() ?? family}
          </span>
        </div>
        <div className="segmented" role="group" aria-label="Keep, drop or target">
          <button className={draft.action === "keep" ? "on" : ""} disabled={nonFeature && !isTarget} onClick={() => update({ action: "keep" })}
            title={nonFeature ? `${draft.semantic_type} fields can't be model features` : "Keep (X toggles)"}>Keep</button>
          <button className={draft.action === "drop" ? "on drop" : ""} onClick={() => update({ action: "drop" })}>Drop</button>
          <button className={isTarget ? "on" : ""} onClick={() => update({ action: "target" })} title="Use as a target column">Target</button>
        </div>
        <label className="field">Semantic type
          <select value={draft.semantic_type} onChange={(e) => {
            const next = retype(draft, e.target.value, catalog, sugg);
            update(next);
          }}>
            {catalog.types.map((t) => <option key={t.value} value={t.value}>{t.label}{t.value === data.field.inferred_type ? " (inferred)" : ""}</option>)}
          </select>
        </label>
      </div>

      {draft.action === "keep" && (
        <div className="section">
          {select("imputation", "Missing values")}
          {draft.imputation === "constant" && !["categorical", "binary", "text"].includes(draft.semantic_type) && (
            <label className="field">Fill value
              <input value={draft.impute_value ?? ""} inputMode="decimal" onChange={(e) => update({ impute_value: e.target.value })} />
            </label>
          )}
          {draft.imputation === "constant" && ["categorical", "binary"].includes(draft.semantic_type) && (
            <label className="field">Missing-level label
              <input value={draft.impute_value ?? ""} onChange={(e) => update({ impute_value: e.target.value })} />
            </label>
          )}
          {NUMERIC.has(draft.semantic_type) && (
            <label className="check">
              <input type="checkbox" checked={draft.add_missing_indicator} onChange={(e) => update({ add_missing_indicator: e.target.checked })} />
              Add “was missing” indicator column
            </label>
          )}
          {select("transform", "Transform")}
          {select("scaling", "Scaling")}
          {select("encoding", "Encoding")}
          {draft.encoding === "onehot" && <>{param("max_categories", "Max categories")}{param("min_frequency", "Min level frequency", 0.005)}</>}
          {draft.encoding === "ordinal" && param("min_frequency", "Min level frequency", 0.001)}
          {draft.encoding === "tfidf_svd" && <>{param("svd_components", "SVD components")}{param("max_features", "Max vocabulary", 1000)}</>}
        </div>
      )}

      <label className="field">Notes
        <textarea value={draft.notes ?? ""} placeholder="Why this choice?" onChange={(e) => update({ notes: e.target.value || null })} />
      </label>

      <div className="row" style={{ justifyContent: "space-between" }}>
        <label className="check">
          <input type="checkbox" checked={draft.reviewed} onChange={(e) => update({ reviewed: e.target.checked }, { immediate: true })} /> Reviewed
        </label>
        <span className={`save-state${state.kind === "error" ? " error" : ""}`} aria-live="polite">
          {state.kind === "saving" ? "Saving…" : state.kind === "saved" ? "✓ Saved" : state.kind === "error" ? "" : ""}
        </span>
      </div>
      {state.kind === "error" && <div className="alert bad">✕ {state.msg}</div>}
      <div className="row">
        <button className="primary" onClick={markReviewed} style={{ flex: 1 }}>Mark reviewed &amp; next <kbd>↵</kbd></button>
      </div>

      <div className="section card" style={{ padding: 12, boxShadow: "none" }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>Suggestion</h3>
          {differsFromSuggestion ? <span className="badge warn">differs</span> : <span className="badge good">✓ matches</span>}
        </div>
        <div className="secondary" style={{ fontSize: 13 }}>
          <b>{sugg.action}</b> · {sugg.semantic_type.replace("_", " ")}
          {sugg.action === "keep" && suggestionParts(sugg).map((p) => <span key={p}> · {p}</span>)}
        </div>
        {sugg.reasons.length > 0 && <ul className="reasons">{sugg.reasons.map((r) => <li key={r}>{r}</li>)}</ul>}
        <button onClick={acceptSuggestion}>Accept suggestion &amp; next <kbd>A</kbd></button>
      </div>

      <div className="shortcuts" aria-label="Keyboard shortcuts">
        <kbd>J</kbd><span>next field (or <kbd>→</kbd>)</span>
        <kbd>K</kbd><span>previous field (or <kbd>←</kbd>)</span>
        <kbd>U</kbd><span>next unreviewed</span>
        <kbd>X</kbd><span>toggle keep / drop</span>
        <kbd>A</kbd><span>accept suggestion &amp; next</span>
        <kbd>↵</kbd><span>mark reviewed &amp; next</span>
        <kbd>/</kbd><span>search fields</span>
      </div>
    </aside>
  );
});
