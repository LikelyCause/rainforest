import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useAccess, useDebounced, useOptions } from "../hooks";
import { JoinAssessment } from "../components/JoinAssessment";
import { ExperimentModeBar } from "../components/ExperimentModeBar";
import { useView, type View, viewKey, withView } from "../view";
import { useDatasetContext } from "../components/DatasetLayout";
import { Empty, ErrorAlert, StatusGlyph, glyphFor } from "../components/ui";
import { fmtNum, fmtPct, TYPE_SHORT } from "../format";
import { completeFilters, finderState, PRESETS, QUERY_LIMIT, selectionLabel, type Preset } from "../finder";
import { LEAK_AT } from "../signals";
import { showError, toast } from "../toast";
import type { BulkRequest, BulkResult, FieldFilter, OptionCatalog, SqlResult } from "../types";

type Kind = "num" | "pct" | "text" | "bool" | "enum";
interface Col { value: string; label: string; kind: Kind; group: "Decision" | "Stats" | "Type" }

const COLUMNS: Col[] = [
  { value: "reviewed", label: "Reviewed", kind: "bool", group: "Decision" },
  { value: "action", label: "Decision", kind: "enum", group: "Decision" },
  { value: "suggested_action", label: "Suggested decision", kind: "enum", group: "Decision" },
  { value: "decision_source", label: "Decided by", kind: "enum", group: "Decision" },
  { value: "imputation", label: "Missing values", kind: "enum", group: "Decision" },
  { value: "transform", label: "Transform", kind: "enum", group: "Decision" },
  { value: "scaling", label: "Scaling", kind: "enum", group: "Decision" },
  { value: "encoding", label: "Encoding", kind: "enum", group: "Decision" },
  { value: "notes", label: "Notes", kind: "text", group: "Decision" },
  { value: "null_pct", label: "Missing", kind: "pct", group: "Stats" },
  { value: "n_unique", label: "Distinct values", kind: "num", group: "Stats" },
  { value: "unique_pct", label: "Distinct share", kind: "pct", group: "Stats" },
  { value: "max_target_assoc", label: "Target association", kind: "num", group: "Stats" },
  { value: "skewness", label: "Skewness", kind: "num", group: "Stats" },
  { value: "outlier_pct", label: "Outliers", kind: "pct", group: "Stats" },
  { value: "zeros_pct", label: "Zeros", kind: "pct", group: "Stats" },
  { value: "top_value_pct", label: "Top value share", kind: "pct", group: "Stats" },
  { value: "avg_length", label: "Avg length", kind: "num", group: "Stats" },
  { value: "mean", label: "Mean", kind: "num", group: "Stats" },
  { value: "kurtosis", label: "Kurtosis", kind: "num", group: "Stats" },
  { value: "inference_confidence", label: "Type confidence", kind: "pct", group: "Type" },
  { value: "inferred_type", label: "Inferred type", kind: "enum", group: "Type" },
  { value: "semantic_type", label: "Chosen type", kind: "enum", group: "Type" },
  { value: "name", label: "Name", kind: "text", group: "Type" },
  { value: "physical_dtype", label: "Physical dtype", kind: "text", group: "Type" },
];
const colOf = (v: string) => COLUMNS.find((c) => c.value === v) ?? COLUMNS[0];
const OPS: Record<Kind, string[]> = {
  num: [">", ">=", "<", "<=", "=", "!=", "is_null", "not_null"],
  pct: [">", ">=", "<", "<="],
  text: ["contains", "=", "!=", "is_null", "not_null"],
  bool: ["="],
  enum: ["in"],
};

function enumValues(col: string, catalog: OptionCatalog | undefined): { value: string; label: string }[] {
  if (col === "action" || col === "suggested_action") return ["keep", "drop", "target"].map((v) => ({ value: v, label: v }));
  if (col === "decision_source") return [{ value: "user", label: "you" }, { value: "suggested", label: "suggestion" }];
  if (!catalog) return [];
  if (col === "inferred_type" || col === "semantic_type") return catalog.types.map((t) => ({ value: t.value, label: t.label }));
  const dim = col as keyof OptionCatalog["labels"];
  return Object.entries(catalog.labels[dim] ?? {}).map(([value, label]) => ({ value, label: label.split(" (")[0] }));
}

// Finder state lives in the URL so leaving Bulk edit and coming back loses nothing (P10).
function useFinderState() {
  const [params, setParams] = useSearchParams();
  const key = params.toString();
  const state = useMemo(() => finderState(params), [key]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) v === null ? next.delete(k) : next.set(k, v);
    setParams(next, { replace: true });
  };
  return {
    ...state,
    setTab: (t: string) => set({ tab: t === "sql" ? "sql" : null }),
    setPreset: (p: Preset) => set({ preset: p.key, f: null, sort: p.sort ?? null }),
    setFilters: (f: FieldFilter[]) => set({ f: JSON.stringify(f) }),
    setSort: (s: string) => set({ sort: s }),
  };
}

/** Freeze a list of fields as a Review working set and open the first one still to do. */
export function useOpenWorkingSet(datasetId: number) {
  const nav = useNavigate();
  const view = useView();
  return (label: string, rows: { field_id: number; reviewed?: boolean }[], startAt?: number) => {
    if (!rows.length) return;
    const key = Math.random().toString(36).slice(2, 10);
    try { sessionStorage.setItem(`set:${key}`, JSON.stringify({ label, ids: rows.map((r) => r.field_id) })); } catch { /* storage unavailable */ }
    const first = startAt ?? (rows.find((r) => !r.reviewed) ?? rows[0]).field_id;
    nav(withView(`/d/${datasetId}/review/${first}?set=${key}`, view));
  };
}

type BulkAction = { label: string; verb: string; body: Omit<BulkRequest, "filters" | "excluded_ids" | "dry_run">; danger?: boolean };
const ACTIONS: BulkAction[] = [
  { label: "Accept suggestions", verb: "Accepted suggestions for", body: { accept_suggestion: true } },
  { label: "Keep", verb: "Kept", body: { action: "keep" } },
  { label: "Drop", verb: "Dropped", body: { action: "drop" }, danger: true },
  { label: "Mark reviewed", verb: "Marked reviewed", body: { reviewed: true } },
];

function FieldFinder({ datasetId, presetLabel, canEdit, view }: {
  datasetId: number; presetLabel: (p: string | null) => string; canEdit: boolean; view: View;
}) {
  const qc = useQueryClient();
  const { data: catalog } = useOptions();
  const st = useFinderState();
  const openSet = useOpenWorkingSet(datasetId);
  const filters = useDebounced(st.filters, 250);
  const complete = completeFilters(filters);
  const sortCol = st.sort.replace(/^-/, "");
  const desc = st.sort.startsWith("-");

  const q = useQuery({
    queryKey: ["fieldQuery", datasetId, complete, st.sort, viewKey(view)],
    // One row past the limit tells a full page from a truncated one.
    queryFn: () => api.queryFields(datasetId, { filters: complete, order_by: sortCol, descending: desc, limit: QUERY_LIMIT + 1 }, view),
    placeholderData: (prev) => prev,
  });
  const truncated = (q.data?.length ?? 0) > QUERY_LIMIT;
  const rows = (q.data ?? []).slice(0, QUERY_LIMIT) as any[];
  const [excluded, setExcluded] = useState<Set<number>>(new Set());
  useEffect(() => setExcluded(new Set()), [JSON.stringify(complete)]);
  const selected = rows.filter((r) => !excluded.has(r.field_id));

  const [pending, setPending] = useState<{ action: BulkAction; preview: BulkResult } | null>(null);
  const invalidate = () => {
    for (const k of ["fields", "fieldQuery", "exportPreview", "corrMatrix", "analysis", "team", "reviewQueue"]) {
      qc.invalidateQueries({ queryKey: [k, datasetId] });
    }
    qc.invalidateQueries({ queryKey: ["field"] });
  };
  const request = (a: BulkAction, dry_run: boolean) =>
    api.bulk(datasetId, { ...a.body, filters: complete, excluded_ids: [...excluded], dry_run }, view);
  const preview = useMutation({
    mutationFn: (a: BulkAction) => request(a, true).then((preview) => ({ action: a, preview })),
    onSuccess: setPending,
  });
  const undo = useMutation({
    mutationFn: (batch: string) => api.bulkUndo(datasetId, batch, view),
    onSuccess: (r) => { invalidate(); toast({ message: `Restored ${r.restored} field${r.restored === 1 ? "" : "s"}` }); },
    onError: (e) => showError("Couldn't undo", e),
  });
  const apply = useMutation({
    mutationFn: (a: BulkAction) => request(a, false),
    onSuccess: (r, a) => {
      setPending(null);
      invalidate();
      toast({
        message: r.changed ? `${a.verb} ${r.changed} field${r.changed === 1 ? "" : "s"}` : "Nothing changed",
        duration: 10_000,
        action: r.batch_id ? { label: "Undo", run: () => undo.mutate(r.batch_id!) } : undefined,
      });
    },
  });
  const inline = useMutation({
    mutationFn: ({ id, action }: { id: number; action: "keep" | "drop" }) => api.bulk(datasetId, { field_ids: [id], action }, view),
    onSuccess: invalidate,
    onError: (e) => showError("Couldn't save", e),
  });

  useEffect(() => {
    if (!pending) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPending(null);
      else if (e.key === "Enter" && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); apply.mutate(pending.action); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pending, apply]);

  const label = presetLabel(st.preset) + (st.modified ? " (modified)" : "");
  const reviewSet = () => openSet(label || "Bulk edit selection", selected);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.key === "r" && !e.metaKey && !e.ctrlKey && !["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) && selected.length) reviewSet();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const setFilter = (i: number, patch: Partial<FieldFilter>) => st.setFilters(st.filters.map((f, j) => (j === i ? { ...f, ...patch } : f)));
  const sortBy = (col: string) => st.setSort(sortCol === col && desc ? col : `-${col}`);
  const th = (col: string, text: string, num = false) => (
    <th className={`sortable${num ? " num" : ""}`} onClick={() => sortBy(col)} aria-sort={sortCol === col ? (desc ? "descending" : "ascending") : "none"}>
      {text}{sortCol === col ? (desc ? " ▼" : " ▲") : ""}
    </th>
  );
  const allChecked = selected.length === rows.length;

  return (
    <section className="stack" style={{ gap: 12 }}>
      <div className="chips" role="group" aria-label="Presets">
        {PRESETS.map((p) => (
          <button key={p.key} type="button" className={`chip${st.preset === p.key ? " on" : ""}`}
            aria-pressed={st.preset === p.key} onClick={() => st.setPreset(p)}>{p.label}</button>
        ))}
      </div>
      <div className="card stack">
        {st.filters.map((f, i) => {
          const col = colOf(f.column);
          return (
            <div key={i} className="filter-row">
              <select value={f.column} aria-label="Column" onChange={(e) => {
                const c = colOf(e.target.value);
                // The value always resets with the column, so a stale value can't be sent under a new meaning (#25).
                setFilter(i, { column: c.value, op: OPS[c.kind][0], value: c.kind === "enum" ? [] : undefined });
              }}>
                {(["Decision", "Stats", "Type"] as const).map((g) => (
                  <optgroup key={g} label={g}>
                    {COLUMNS.filter((c) => c.group === g).map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
                  </optgroup>
                ))}
              </select>
              {col.kind === "enum" ? <span className="muted small-text">is any of</span> : col.kind === "bool" ? <span className="muted small-text">is</span> : (
                <select value={f.op} aria-label="Operator" onChange={(e) => setFilter(i, { op: e.target.value })}>
                  {OPS[col.kind].map((o) => <option key={o} value={o}>{o.replace("_", " ")}</option>)}
                </select>
              )}
              <ValueEditor col={col} filter={f} catalog={catalog} onChange={(value) => setFilter(i, { value })} />
              <button type="button" className="ghost" aria-label="Remove filter" onClick={() => st.setFilters(st.filters.filter((_, j) => j !== i))}>✕</button>
            </div>
          );
        })}
        <div><button type="button" className="ghost" onClick={() => st.setFilters([...st.filters, { column: "null_pct", op: ">", value: undefined }])}>+ Filter</button></div>
      </div>

      <ErrorAlert error={q.error} />
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="secondary">
          {q.isFetching && !q.data ? "Searching…"
            : truncated ? `More than ${QUERY_LIMIT} fields match; the first ${QUERY_LIMIT} are shown`
            : `${rows.length} field${rows.length === 1 ? "" : "s"} match`}
          {(excluded.size > 0 || truncated) && ` · bulk actions: ${selectionLabel(rows.length, excluded.size, truncated)}`}
        </span>
        <button type="button" className="primary" disabled={!selected.length} onClick={reviewSet}>
          Review these {selected.length} → <kbd>R</kbd>
        </button>
      </div>

      {rows.length > 0 ? (
        <div>
          <div className="table-wrap" style={{ maxHeight: "calc(100vh - 380px)", minHeight: 240, borderRadius: "6px 6px 0 0" }}>
            <table className="data compact">
              <thead><tr>
                <th style={{ width: 32 }}>
                  <input type="checkbox" aria-label="Select all" checked={allChecked}
                    ref={(el) => { if (el) el.indeterminate = !allChecked && selected.length > 0; }}
                    onChange={() => setExcluded(allChecked ? new Set(rows.map((r) => r.field_id)) : new Set())} />
                </th>
                {th("name", "Name")}<th>Type</th><th>Decision</th><th>Suggestion</th>
                {th("null_pct", "Missing", true)}{th("n_unique", "Distinct", true)}{th("skewness", "Skew", true)}
                {th("max_target_assoc", "Target assoc.", true)}<th title="Reviewed">✓</th>
              </tr></thead>
              <tbody>
                {rows.map((r) => {
                  const off = excluded.has(r.field_id);
                  const differs = r.suggested_action && r.suggested_action !== r.action;
                  return (
                    <tr key={r.field_id} className={off ? "excluded" : undefined}>
                      <td>
                        <input type="checkbox" checked={!off} aria-label={`Include ${r.name}`} onChange={() => {
                          const n = new Set(excluded);
                          off ? n.delete(r.field_id) : n.add(r.field_id);
                          setExcluded(n);
                        }} />
                      </td>
                      <td><button type="button" className="link" onClick={() => openSet(label || "Bulk edit selection", selected, r.field_id)}>{r.name}</button></td>
                      <td><span className="type-tag">{TYPE_SHORT[r.semantic_type] ?? r.semantic_type}</span></td>
                      <td>
                        {r.action === "target" ? <span className="badge accent">◎ target</span> : (
                          <div className="segmented mini-seg" role="radiogroup" aria-label={`Decision for ${r.name}`}>
                            {(["keep", "drop"] as const).map((a) => (
                              <button key={a} type="button" role="radio" aria-checked={r.action === a} className={r.action === a ? `on ${a}` : ""}
                                disabled={!canEdit} onClick={() => r.action !== a && inline.mutate({ id: r.field_id, action: a })}>{a}</button>
                            ))}
                          </div>
                        )}
                      </td>
                      <td className={differs ? "" : "muted"} style={differs ? { color: "var(--warn)" } : undefined}>{r.suggested_action ?? "—"}</td>
                      <td className="num">{fmtPct(r.null_pct)}</td>
                      <td className="num">{fmtNum(r.n_unique)}</td>
                      <td className="num">{fmtNum(r.skewness, 2)}</td>
                      <td className="num">
                        {r.is_target ? <span className="muted">—</span> : <>
                          {fmtNum(r.max_target_assoc, 2)}
                          {(r.max_target_assoc ?? 0) >= LEAK_AT && <span className="badge bad" style={{ marginLeft: 6 }}>leak</span>}
                        </>}
                      </td>
                      <td><StatusGlyph kind={glyphFor(r)} reviewed={r.reviewed} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {pending ? (
            <div className="bulk-bar preview" role="alertdialog" aria-label="Confirm bulk change">
              <span className="grow">
                <b>{pending.action.label} · {pending.preview.matched} field{pending.preview.matched === 1 ? "" : "s"}</b>
                {" — "}{pending.preview.changed} will change
                {Object.entries(pending.preview.transitions).map(([t, n]) => ` · ${n} ${t}`).join("")}
                {pending.preview.unchanged > 0 && ` · ${pending.preview.unchanged} already so`}
                {pending.preview.reviewed_changed > 0 && <> · <b>{pending.preview.reviewed_changed} reviewed field{pending.preview.reviewed_changed === 1 ? "" : "s"} will change</b></>}
              </span>
              <button type="button" className={pending.action.danger ? "danger" : "primary"} disabled={apply.isPending || !pending.preview.changed}
                onClick={() => apply.mutate(pending.action)} autoFocus>
                Apply {pending.action.label.toLowerCase()} <kbd>↵</kbd>
              </button>
              <button type="button" onClick={() => setPending(null)}>Cancel <kbd>Esc</kbd></button>
              <ErrorAlert error={apply.error} />
            </div>
          ) : (
            <div className="bulk-bar">
              <span className="grow num">{selectionLabel(rows.length, excluded.size, truncated)}</span>
              {ACTIONS.map((a) => (
                <button key={a.label} type="button" className="small" disabled={!canEdit || !selected.length || preview.isPending}
                  onClick={() => preview.mutate(a)}>{a.label}</button>
              ))}
              <ErrorAlert error={preview.error} />
            </div>
          )}
        </div>
      ) : q.data && (
        <Empty title="No fields match" action={<button type="button" onClick={() => st.setPreset(PRESETS[0])}>Show unreviewed</button>} />
      )}
    </section>
  );
}

function ValueEditor({ col, filter, catalog, onChange }: {
  col: Col; filter: FieldFilter; catalog: OptionCatalog | undefined; onChange: (v: unknown) => void;
}) {
  if (filter.op === "is_null" || filter.op === "not_null") return <span />;
  if (col.kind === "bool") {
    const v = filter.value === undefined ? null : String(filter.value);
    return (
      <div className="segmented soft" role="radiogroup" aria-label={col.label}>
        {(["true", "false"] as const).map((x) => (
          <button key={x} type="button" role="radio" aria-checked={v === x} className={v === x ? "on" : ""} onClick={() => onChange(x)}>
            {x === "true" ? "Yes" : "No"}
          </button>
        ))}
      </div>
    );
  }
  if (col.kind === "enum") {
    const chosen = new Set(Array.isArray(filter.value) ? (filter.value as string[]) : []);
    return (
      <div className="enum-values" role="group" aria-label={col.label}>
        {enumValues(col.value, catalog).map((o) => (
          <button key={o.value} type="button" className={`chip${chosen.has(o.value) ? " on" : ""}`} aria-pressed={chosen.has(o.value)}
            onClick={() => onChange(chosen.has(o.value) ? [...chosen].filter((x) => x !== o.value) : [...chosen, o.value])}>
            {o.label}
          </button>
        ))}
      </div>
    );
  }
  if (col.kind === "pct") {
    const v = filter.value === undefined || filter.value === "" ? "" : String(+(Number(filter.value) * 100).toFixed(4));
    return (
      <label className="value-pct">
        <input type="number" step="any" value={v} aria-label={`${col.label} percent`}
          onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value) / 100)} />
        <span className="muted">%</span>
      </label>
    );
  }
  return (
    <input value={String(filter.value ?? "")} aria-label="Value" type={col.kind === "num" ? "number" : "text"} step="any"
      placeholder={col.kind === "num" ? "number" : "text"}
      onChange={(e) => onChange(e.target.value === "" ? undefined : col.kind === "num" ? Number(e.target.value) : e.target.value)} />
  );
}

const NAME_COLUMNS = new Set(["name", "field_name", "other_field_name"]);

function SqlConsole({ datasetId }: { datasetId: number }) {
  const examples = useQuery({ queryKey: ["sqlExamples"], queryFn: api.sqlExamples, staleTime: Infinity });
  const { data: fields } = useQuery({ queryKey: ["fields", datasetId], queryFn: () => api.fields(datasetId) });
  const openSet = useOpenWorkingSet(datasetId);
  const storeKey = `sql:${datasetId}`;
  const [sql, setSqlState] = useState(() => {
    try { return sessionStorage.getItem(storeKey) ?? ""; } catch { return ""; }
  }) || `SELECT name, inferred_type, null_pct, n_unique\nFROM meta.v_fields\nWHERE dataset_id = :dataset\nORDER BY null_pct DESC\nLIMIT 20`;
  const setSql = (v: string) => {
    setSqlState(v);
    try { sessionStorage.setItem(storeKey, v); } catch { /* storage unavailable */ }
  };
  const m = useMutation({ mutationFn: (q: string) => api.sql(q.split(":dataset").join(String(datasetId))) });
  const ranOnce = useRef(false);
  useEffect(() => {
    // Re-run on return only for metadata queries; raw-data scans can be slow.
    if (!ranOnce.current && sql && !/raw\./i.test(sql)) { ranOnce.current = true; m.mutate(sql); }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const byName = useMemo(() => new Map((fields ?? []).map((f) => [f.name, f])), [fields]);
  const nameCols = m.data?.columns.map((c, i) => (NAME_COLUMNS.has(c) ? i : -1)).filter((i) => i >= 0) ?? [];
  const resultFields = nameCols.length
    ? [...new Set(m.data!.rows.map((r) => byName.get(String(r[nameCols[0]]))).filter(Boolean))]
    : [];

  return (
    <section className="card stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="muted small-text">read-only role · <code>:dataset</code> = {datasetId}</span>
        <select aria-label="Examples" value="" onChange={(e) => { const ex = examples.data?.find((x) => x.title === e.target.value); if (ex) setSql(ex.sql); }}>
          <option value="">Examples ▾</option>
          {examples.data?.map((e) => <option key={e.title} value={e.title}>{e.title}</option>)}
        </select>
      </div>
      <textarea className="sql-editor" value={sql} spellCheck={false} aria-label="SQL"
        onChange={(e) => setSql(e.target.value)}
        onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); m.mutate(sql); } }} />
      <div className="row">
        <button type="button" className="primary" disabled={m.isPending} onClick={() => m.mutate(sql)}>Run <kbd>⌘↵</kbd></button>
        <span className="muted small-text">
          Views: <code>meta.v_fields</code>, <code>meta.v_correlations</code> · tables: <code>meta.fields</code>, <code>meta.field_decisions</code>,
          <code> meta.decision_history</code>, <code>raw.ds_{datasetId}</code>
        </span>
      </div>
      <ErrorAlert error={m.error} />
      {m.data && (
        <>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="muted small-text">
              {m.data.row_count} row{m.data.row_count === 1 ? "" : "s"}{m.data.truncated ? " (truncated)" : ""} · {m.data.elapsed_ms} ms
            </span>
            {resultFields.length > 0 && (
              <button type="button" className="primary"
                onClick={() => openSet("SQL result", resultFields.map((f) => ({ field_id: f!.id, reviewed: f!.reviewed })))}>
                Review these {resultFields.length} →
              </button>
            )}
          </div>
          <ResultTable result={m.data} nameCols={new Set(nameCols)} onOpen={(name) => {
            const f = byName.get(name);
            if (f) openSet("SQL result", resultFields.map((x) => ({ field_id: x!.id, reviewed: x!.reviewed })), f.id);
          }} />
        </>
      )}
    </section>
  );
}

function ResultTable({ result, nameCols, onOpen }: { result: SqlResult; nameCols: Set<number>; onOpen: (name: string) => void }) {
  if (!result.columns.length) return <div className="muted">Statement executed — no rows returned.</div>;
  return (
    <div className="table-wrap">
      <table className="data">
        <thead><tr>{result.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {result.rows.map((r, i) => (
            <tr key={i}>
              {r.map((v, j) => (
                <td key={j} className={typeof v === "number" ? "num" : undefined}>
                  {v === null ? <span className="muted">null</span>
                    : nameCols.has(j) ? <button type="button" className="link" onClick={() => onOpen(String(v))}>{String(v)}</button>
                    : typeof v === "number" ? fmtNum(v) : typeof v === "object" ? JSON.stringify(v) : String(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function BulkEditPage() {
  const ds = useDatasetContext();
  const view = useView();
  const access = useAccess(ds, view);
  const st = useFinderState();
  const presetLabel = (p: string | null) => PRESETS.find((x) => x.key === p)?.label ?? "Custom filter";
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Bulk edit</h1>
        <Link to={withView(`/d/${ds.id}/review`, view)} className="muted small-text">Changes here show up in Review at once</Link>
      </div>
      {view ? <ExperimentModeBar ds={ds} view={view} /> : <JoinAssessment ds={ds} />}
      {access.note && <div className={`alert ${access.canEdit ? "info" : "warn"}`}>{access.note}</div>}
      <div className="tabs-inline" role="tablist">
        <button type="button" role="tab" aria-selected={st.tab === "fields"} className={st.tab === "fields" ? "on" : ""} onClick={() => st.setTab("fields")}>Fields</button>
        <button type="button" role="tab" aria-selected={st.tab === "sql"} className={st.tab === "sql" ? "on" : ""} onClick={() => st.setTab("sql")}>SQL</button>
      </div>
      {st.tab === "fields" || view ? <FieldFinder datasetId={ds.id} presetLabel={presetLabel} canEdit={access.canEdit} view={view} /> : <SqlConsole datasetId={ds.id} />}
    </div>
  );
}
