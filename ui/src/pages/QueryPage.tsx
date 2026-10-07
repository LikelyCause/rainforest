import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { fmtNum, fmtPct } from "../format";
import type { FieldFilter, SqlResult } from "../types";

const COLUMNS: { value: string; label: string; kind: "num" | "text" | "bool" }[] = [
  { value: "inferred_type", label: "Inferred type", kind: "text" },
  { value: "semantic_type", label: "Chosen type", kind: "text" },
  { value: "action", label: "Action", kind: "text" },
  { value: "reviewed", label: "Reviewed", kind: "bool" },
  { value: "name", label: "Name", kind: "text" },
  { value: "null_pct", label: "Missing (0–1)", kind: "num" },
  { value: "n_unique", label: "Distinct values", kind: "num" },
  { value: "unique_pct", label: "Distinct (0–1)", kind: "num" },
  { value: "max_target_assoc", label: "Max target assoc.", kind: "num" },
  { value: "mean", label: "Mean", kind: "num" },
  { value: "std", label: "Std dev", kind: "num" },
  { value: "min_value", label: "Min", kind: "num" },
  { value: "max_value", label: "Max", kind: "num" },
  { value: "median", label: "Median", kind: "num" },
  { value: "skewness", label: "Skewness", kind: "num" },
  { value: "kurtosis", label: "Kurtosis", kind: "num" },
  { value: "zeros_pct", label: "Zeros (0–1)", kind: "num" },
  { value: "outlier_pct", label: "Outliers (0–1)", kind: "num" },
  { value: "top_value_pct", label: "Top value share", kind: "num" },
  { value: "avg_length", label: "Avg length", kind: "num" },
  { value: "inference_confidence", label: "Type confidence", kind: "num" },
  { value: "physical_dtype", label: "Physical dtype", kind: "text" },
  { value: "imputation", label: "Imputation", kind: "text" },
  { value: "scaling", label: "Scaling", kind: "text" },
  { value: "encoding", label: "Encoding", kind: "text" },
  { value: "notes", label: "Notes", kind: "text" },
];
const OPS: Record<string, string[]> = {
  num: [">", ">=", "<", "<=", "=", "!=", "is_null", "not_null"],
  text: ["=", "!=", "contains", "in", "is_null", "not_null"],
  bool: ["="],
};
const PRESETS: { label: string; filters: FieldFilter[]; order?: string; desc?: boolean }[] = [
  { label: "Text fields", filters: [{ column: "inferred_type", op: "=", value: "text" }] },
  { label: "Missing > 20%", filters: [{ column: "null_pct", op: ">", value: 0.2 }], order: "null_pct", desc: true },
  { label: "Unreviewed", filters: [{ column: "reviewed", op: "=", value: "false" }] },
  { label: "Skewed numerics", filters: [{ column: "skewness", op: ">", value: 2 }], order: "skewness", desc: true },
  { label: "Most target-associated", filters: [{ column: "max_target_assoc", op: "not_null" }], order: "max_target_assoc", desc: true },
  { label: "High-cardinality categoricals", filters: [{ column: "inferred_type", op: "=", value: "categorical" }, { column: "n_unique", op: ">", value: 50 }], order: "n_unique", desc: true },
];

function ResultTable({ result, onRow }: { result: SqlResult; onRow?: (row: unknown[]) => void }) {
  if (!result.columns.length) return <div className="muted">Statement executed — no rows returned.</div>;
  return (
    <div className="table-wrap">
      <table className="data">
        <thead><tr>{result.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {result.rows.map((r, i) => (
            <tr key={i} className={onRow ? "clickable" : undefined} onClick={() => onRow?.(r)}>
              {r.map((v, j) => (
                <td key={j} className={typeof v === "number" ? "num" : undefined}>
                  {v === null ? <span className="muted">null</span> : typeof v === "number" ? fmtNum(v) : typeof v === "object" ? JSON.stringify(v) : String(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FieldFinder({ datasetId }: { datasetId: number }) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [filters, setFilters] = useState<FieldFilter[]>([{ column: "inferred_type", op: "=", value: "text" }]);
  const [orderBy, setOrderBy] = useState("ordinal");
  const [desc, setDesc] = useState(false);
  const [ran, setRan] = useState<{ filters: FieldFilter[]; orderBy: string; desc: boolean } | null>(null);

  const q = useQuery({
    queryKey: ["fieldQuery", datasetId, ran],
    queryFn: () => api.queryFields(datasetId, { filters: ran!.filters, order_by: ran!.orderBy, descending: ran!.desc }),
    enabled: ran !== null,
  });
  const bulk = useMutation({
    mutationFn: (body: { accept_suggestion?: boolean; action?: string; reviewed?: boolean }) =>
      api.bulk(datasetId, { field_ids: (q.data ?? []).map((r) => r.field_id as number), ...body }),
    onSuccess: () => {
      for (const k of ["fields", "summary", "fieldQuery", "exportPreview"]) qc.invalidateQueries({ queryKey: [k, datasetId] });
      qc.invalidateQueries({ queryKey: ["field"] });
    },
  });

  const run = (f = filters, o = orderBy, d = desc) => setRan({ filters: f.filter((x) => x.column), orderBy: o, desc: d });
  const setFilter = (i: number, patch: Partial<FieldFilter>) => setFilters(filters.map((f, j) => (j === i ? { ...f, ...patch } : f)));
  const kindOf = (col: string) => COLUMNS.find((c) => c.value === col)?.kind ?? "text";

  return (
    <section className="card stack">
      <div className="card-head"><h2>Find fields</h2><span className="muted">queries <code>meta.v_fields</code></span></div>
      <div className="chips">
        {PRESETS.map((p) => (
          <button key={p.label} className="chip" onClick={() => {
            setFilters(p.filters); setOrderBy(p.order ?? "ordinal"); setDesc(!!p.desc); run(p.filters, p.order ?? "ordinal", !!p.desc);
          }}>{p.label}</button>
        ))}
      </div>
      {filters.map((f, i) => {
        const kind = kindOf(f.column);
        return (
          <div key={i} className="filter-row">
            <select value={f.column} onChange={(e) => setFilter(i, { column: e.target.value, op: OPS[kindOf(e.target.value)][0] })} aria-label="Column">
              {COLUMNS.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
            </select>
            <select value={f.op} onChange={(e) => setFilter(i, { op: e.target.value })} aria-label="Operator">
              {OPS[kind].map((o) => <option key={o} value={o}>{o.replace("_", " ")}</option>)}
            </select>
            {kind === "bool" ? (
              <select value={String(f.value ?? "true")} onChange={(e) => setFilter(i, { value: e.target.value })} aria-label="Value">
                <option value="true">true</option><option value="false">false</option>
              </select>
            ) : (
              <input value={String(f.value ?? "")} disabled={f.op === "is_null" || f.op === "not_null"} aria-label="Value"
                placeholder={f.op === "in" ? "a, b, c" : kind === "num" ? "number" : "value"}
                onChange={(e) => setFilter(i, { value: e.target.value })}
                onKeyDown={(e) => e.key === "Enter" && run()} />
            )}
            <button className="ghost" onClick={() => setFilters(filters.filter((_, j) => j !== i))} aria-label="Remove filter">✕</button>
          </div>
        );
      })}
      <div className="row">
        <button onClick={() => setFilters([...filters, { column: "null_pct", op: ">", value: "" }])}>+ Filter</button>
        <span className="muted" style={{ marginLeft: 8 }}>order by</span>
        <select value={orderBy} onChange={(e) => setOrderBy(e.target.value)} aria-label="Order by">
          <option value="ordinal">column order</option>
          {COLUMNS.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={desc} onChange={(e) => setDesc(e.target.checked)} /> descending</label>
        <div style={{ flex: 1 }} />
        <button className="primary" onClick={() => run()}>Run</button>
      </div>
      {q.error && <div className="alert bad">{(q.error as Error).message}</div>}
      {q.data && (
        <>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="secondary">{q.data.length} field{q.data.length === 1 ? "" : "s"}</span>
            {q.data.length > 0 && (
              <div className="row">
                <span className="muted" style={{ fontSize: 12.5 }}>Apply to all results:</span>
                <button className="small" disabled={bulk.isPending} onClick={() => bulk.mutate({ accept_suggestion: true })}>Accept suggestions</button>
                <button className="small" disabled={bulk.isPending} onClick={() => bulk.mutate({ action: "keep" })}>Keep</button>
                <button className="small danger" disabled={bulk.isPending} onClick={() => bulk.mutate({ action: "drop" })}>Drop</button>
                <button className="small" disabled={bulk.isPending} onClick={() => bulk.mutate({ reviewed: true })}>Mark reviewed</button>
              </div>
            )}
          </div>
          {bulk.data && <div className="alert info">Updated {bulk.data.updated} field(s).</div>}
          <div className="table-wrap">
            <table className="data">
              <thead><tr>
                <th>Name</th><th>Type</th><th>Action</th><th className="num">Missing</th><th className="num">Distinct</th>
                <th className="num">Skew</th><th className="num">Target assoc.</th><th>Reviewed</th>
              </tr></thead>
              <tbody>
                {q.data.map((r: any) => (
                  <tr key={r.field_id} className="clickable" onClick={() => nav(`/d/${datasetId}/review/${r.field_id}`)}>
                    <td>{r.name}</td>
                    <td><span className="type-tag">{r.semantic_type}</span></td>
                    <td>{r.action === "drop" ? <span className="badge bad">drop</span> : r.action === "target" ? <span className="badge accent">target</span> : <span className="badge">keep</span>}</td>
                    <td className="num">{fmtPct(r.null_pct)}</td>
                    <td className="num">{fmtNum(r.n_unique)}</td>
                    <td className="num">{fmtNum(r.skewness, 2)}</td>
                    <td className="num">{fmtNum(r.max_target_assoc, 2)}</td>
                    <td>{r.reviewed ? "✓" : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

function SqlConsole({ datasetId }: { datasetId: number }) {
  const examples = useQuery({ queryKey: ["sqlExamples"], queryFn: api.sqlExamples, staleTime: Infinity });
  const [sql, setSql] = useState(
    `SELECT name, inferred_type, null_pct, n_unique\nFROM meta.v_fields\nWHERE dataset_id = :dataset\nORDER BY null_pct DESC\nLIMIT 20`,
  );
  const m = useMutation({ mutationFn: (q: string) => api.sql(q.split(":dataset").join(String(datasetId))) });
  return (
    <section className="card stack">
      <div className="card-head">
        <h2>SQL console</h2>
        <span className="muted">read-only role · <code>:dataset</code> = {datasetId}</span>
      </div>
      <div className="chips">
        {examples.data?.map((e) => <button key={e.title} className="chip" onClick={() => setSql(e.sql)}>{e.title}</button>)}
      </div>
      <textarea className="sql-editor" value={sql} spellCheck={false} aria-label="SQL"
        onChange={(e) => setSql(e.target.value)}
        onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); m.mutate(sql); } }} />
      <div className="row">
        <button className="primary" disabled={m.isPending} onClick={() => m.mutate(sql)}>Run <kbd>⌘↵</kbd></button>
        <span className="muted" style={{ fontSize: 12.5 }}>
          Views: <code>meta.v_fields</code>, <code>meta.v_correlations</code> · tables: <code>meta.fields</code>, <code>meta.field_decisions</code>,
          <code> meta.decision_history</code>, <code>raw.ds_{datasetId}</code>
        </span>
      </div>
      {m.error && <div className="alert bad">✕ {(m.error as Error).message}</div>}
      {m.data && (
        <>
          <span className="muted" style={{ fontSize: 12.5 }}>
            {m.data.row_count} row{m.data.row_count === 1 ? "" : "s"}{m.data.truncated ? " (truncated)" : ""} · {m.data.elapsed_ms} ms
          </span>
          <ResultTable result={m.data} />
        </>
      )}
    </section>
  );
}

export function QueryPage() {
  const datasetId = Number(useParams().datasetId);
  return (
    <div className="page">
      <FieldFinder datasetId={datasetId} />
      <SqlConsole datasetId={datasetId} />
    </div>
  );
}
