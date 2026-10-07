import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, uploadDataset } from "../api";
import { PROCESSING } from "../hooks";
import { fmtDate, fmtInt, fmtPct } from "../format";
import type { Dataset } from "../types";

const STATUS_LABEL: Record<string, string> = {
  pending: "Queued",
  ingesting: "Ingesting",
  profiling: "Profiling",
  correlating: "Correlations",
  ready: "Ready",
  failed: "Failed",
};

function StatusBadge({ ds }: { ds: Dataset }) {
  const cls = ds.status === "ready" ? "good" : ds.status === "failed" ? "bad" : "accent";
  const icon = ds.status === "ready" ? "✓" : ds.status === "failed" ? "✕" : "…";
  return <span className={`badge ${cls}`}>{icon} {STATUS_LABEL[ds.status]}</span>;
}

function UploadCard() {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [targets, setTargets] = useState("column_a,column_b");
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pick = (f: File | undefined) => {
    if (!f) return;
    if (!/\.(csv|parquet|pq|txt)$/i.test(f.name)) {
      setError("Choose a .csv or .parquet file");
      return;
    }
    setError(null);
    setFile(f);
    if (!name) setName(f.name.replace(/\.[^.]+$/, ""));
  };

  const submit = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      await uploadDataset(file, name, targets, setProgress);
      setFile(null);
      setName("");
      qc.invalidateQueries({ queryKey: ["datasets"] });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setProgress(null);
    }
  };

  return (
    <section className="card stack">
      <div className="card-head"><h2>Upload a dataset</h2><span className="muted">CSV or Parquet</span></div>
      <div
        className={`dropzone${over ? " over" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}
      >
        {file ? (
          <><b>{file.name}</b><div className="muted">{fmtInt(Math.round(file.size / 1024))} KB · click to change</div></>
        ) : (
          <>Drop a file here or <span style={{ color: "var(--accent)" }}>browse</span></>
        )}
        <input ref={input} type="file" accept=".csv,.parquet,.pq,.txt" hidden onChange={(e) => pick(e.target.files?.[0])} />
      </div>
      <div className="grid-2" style={{ gap: 12 }}>
        <label className="field">Dataset name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-dataset" /></label>
        <label className="field">Target columns (comma-separated)<input value={targets} onChange={(e) => setTargets(e.target.value)} /></label>
      </div>
      {progress !== null && (
        <div className="stack" style={{ gap: 4 }}>
          <div className="meter"><span style={{ width: `${progress * 100}%` }} /></div>
          <span className="muted" style={{ fontSize: 12 }}>Uploading… {fmtPct(progress, 0)}</span>
        </div>
      )}
      {error && <div className="alert bad">{error}</div>}
      <div className="row"><button className="primary" disabled={!file || progress !== null} onClick={submit}>Upload &amp; profile</button></div>
    </section>
  );
}

function SampleCard() {
  const qc = useQueryClient();
  const [rows, setRows] = useState(50_000);
  const [fields, setFields] = useState(150);
  const [kind, setKind] = useState("mixed");
  const m = useMutation({
    mutationFn: () => api.createSample({ rows, fields, seed: 42, target_kind: kind }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
  return (
    <section className="card stack">
      <div className="card-head"><h2>Generate a sample dataset</h2></div>
      <p className="secondary" style={{ margin: 0 }}>
        Mixed numeric / categorical / text / date fields driven by shared latent factors, with missingness, skew,
        IDs, numbers-as-text, a constant column and one deliberately leaky feature.
      </p>
      <div className="row" style={{ gap: 12, alignItems: "flex-end" }}>
        <label className="field">Rows<input type="number" min={1000} max={2000000} step={1000} value={rows} onChange={(e) => setRows(Number(e.target.value))} style={{ width: 120 }} /></label>
        <label className="field">Fields<input type="number" min={30} max={400} value={fields} onChange={(e) => setFields(Number(e.target.value))} style={{ width: 90 }} /></label>
        <label className="field">Targets
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="mixed">column_a binary, column_b 3-class</option>
            <option value="binary">both binary</option>
            <option value="multiclass">both multiclass</option>
          </select>
        </label>
        <button className="primary" disabled={m.isPending} onClick={() => m.mutate()}>Generate</button>
      </div>
      {m.error && <div className="alert bad">{(m.error as Error).message}</div>}
    </section>
  );
}

export function DatasetsPage() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const { data, isLoading, error } = useQuery({
    queryKey: ["datasets"],
    queryFn: api.datasets,
    refetchInterval: (q) => ((q.state.data as Dataset[] | undefined)?.some((d) => PROCESSING.has(d.status)) ? 1000 : false),
  });
  const del = useMutation({
    mutationFn: (id: number) => api.deleteDataset(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });

  return (
    <div className="page">
      <div><h1>Datasets</h1><p className="secondary" style={{ margin: "4px 0 0" }}>Ingest a file, walk through every field, and export a model-ready dataset.</p></div>
      <div className="grid-2">
        <UploadCard />
        <SampleCard />
      </div>
      <section className="card">
        <div className="card-head"><h2>Your datasets</h2></div>
        {isLoading && <div className="empty">Loading…</div>}
        {error && <div className="alert bad">{(error as Error).message}</div>}
        {data && data.length === 0 && <div className="empty">No datasets yet. Upload a file or generate a sample above.</div>}
        {data?.map((ds) => (
          <div key={ds.id} className="dataset-row">
            <div className="stack" style={{ gap: 4, minWidth: 0 }}>
              <div className="row">
                {ds.status === "ready" ? <Link to={`/d/${ds.id}/review`} style={{ fontWeight: 600 }}>{ds.name}</Link> : <b>{ds.name}</b>}
                <StatusBadge ds={ds} />
                <span className="muted" style={{ fontSize: 12.5 }}>
                  {ds.row_count !== null && `${fmtInt(ds.row_count)} rows · `}
                  {ds.column_count !== null && `${ds.column_count} fields · `}
                  targets: {ds.target_columns.join(", ") || "none"} · {fmtDate(ds.created_at)}
                </span>
              </div>
              {PROCESSING.has(ds.status) && (
                <div className="stack" style={{ gap: 4, maxWidth: 520 }}>
                  <div className="meter"><span style={{ width: `${Math.max(ds.progress, 0.02) * 100}%` }} /></div>
                  <span className="muted" style={{ fontSize: 12 }}>{ds.status_message}</span>
                </div>
              )}
              {ds.warnings.map((w) => <div key={w} className="alert warn">⚠ {w}</div>)}
              {ds.status === "failed" && ds.error && (
                <details className="alert bad"><summary>✕ {ds.error.split("\n")[0]}</summary><pre>{ds.error}</pre></details>
              )}
            </div>
            <div className="row">
              {ds.status === "ready" && <button className="primary" onClick={() => nav(`/d/${ds.id}/review`)}>Review fields</button>}
              {!PROCESSING.has(ds.status) && (
                <button
                  className="ghost danger"
                  onClick={() => window.confirm(`Delete "${ds.name}", its raw table, decisions and exports?`) && del.mutate(ds.id)}
                >Delete</button>
              )}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
