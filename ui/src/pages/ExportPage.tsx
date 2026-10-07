import { Fragment, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useDataset, useOptions } from "../hooks";
import { fmtDate, fmtInt, fmtNum, fmtPct } from "../format";
import type { ExportJob, ModelRun } from "../types";

const FAMILY_EXPLAINER: Record<string, string> = {
  tree: "Missing values stay as NaN for the trees to route; no scaling or log transforms; mid/high-cardinality categoricals become compact ordinal codes.",
  linear: "Every numeric is imputed (with missing flags), log-transformed when skewed and standardized; categoricals are one-hot encoded.",
  neural: "Like linear, with quantile-normal scaling for skewed inputs so the network sees well-conditioned features.",
};

function TrainForm({ e, targets }: { e: ExportJob; targets: string[] }) {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["exportModels", e.id], queryFn: () => api.modelsForExport(e.id), staleTime: 60_000 });
  const [model, setModel] = useState<string>("");
  const [reverse, setReverse] = useState(false);
  useEffect(() => {
    // Default to a model built for this export's family (LightGBM for tree exports).
    if (!model && models.data) {
      const own = models.data.find((m) => m.compatible && m.family === e.model_family);
      setModel((own ?? models.data.find((m) => m.compatible))?.key ?? "");
    }
  }, [model, models.data, e.model_family]);
  const start = useMutation({
    mutationFn: () => api.startRun(e.id, { model, order: reverse ? [...targets].reverse() : targets, cv: 5 }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs", e.dataset_id] }),
  });
  const selected = models.data?.find((m) => m.key === model);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row">
        <select value={model} onChange={(ev) => setModel(ev.target.value)} aria-label="Model" style={{ maxWidth: 300 }}>
          {models.data?.map((m) => (
            <option key={m.key} value={m.key} disabled={!m.compatible}>
              {m.label}{m.compatible ? (m.family === "tree" ? "" : ` (${m.family})`) : " — not compatible"}
            </option>
          ))}
        </select>
        {targets.length === 2 && (
          <select value={reverse ? "rev" : "fwd"} onChange={(ev) => setReverse(ev.target.value === "rev")} aria-label="Chain order">
            <option value="fwd">{targets[0]} → {targets[1]}</option>
            <option value="rev">{targets[1]} → {targets[0]}</option>
          </select>
        )}
        <button className="primary" disabled={!model || !selected?.compatible || start.isPending} onClick={() => start.mutate()}>
          Train chain
        </button>
      </div>
      {selected && <span className="muted" style={{ fontSize: 12 }}>{selected.description}</span>}
      {start.error && <div className="alert bad">✕ {(start.error as Error).message}</div>}
    </div>
  );
}

function ExportRow({ e, familyLabel }: { e: ExportJob; familyLabel: string }) {
  const [copied, setCopied] = useState(false);
  const s = e.summary ?? {};
  const running = e.status === "pending" || e.status === "running";
  return (
    <div className="dataset-row">
      <div className="stack" style={{ gap: 6, minWidth: 0 }}>
        <div className="row">
          <b>Export #{e.id}</b>
          <span className="badge">{familyLabel}</span>
          <span className={`badge ${e.status === "done" ? "good" : e.status === "failed" ? "bad" : "accent"}`}>
            {e.status === "done" ? "✓ done" : e.status === "failed" ? "✕ failed" : "… " + e.status}
          </span>
          <span className="muted" style={{ fontSize: 12.5 }}>
            {fmtDate(e.created_at)}
            {e.n_rows !== null && ` · ${fmtInt(e.n_rows)} rows × ${fmtInt(e.n_features)} features`}
          </span>
        </div>
        {running && (
          <div className="stack" style={{ gap: 4, maxWidth: 520 }}>
            <div className="meter"><span style={{ width: `${Math.max(e.progress, 0.02) * 100}%` }} /></div>
            <span className="muted" style={{ fontSize: 12 }}>{e.status_message}</span>
          </div>
        )}
        {e.status === "done" && (
          <>
            <div className="secondary" style={{ fontSize: 13 }}>
              Split: train {fmtInt(s.split_counts?.train)} · valid {fmtInt(s.split_counts?.valid)} · test {fmtInt(s.split_counts?.test)}
              {s.split_info?.stratified ? " (stratified on both targets)" : ""}
              {s.split_info?.dropped_unlabeled ? ` · ${fmtInt(s.split_info.dropped_unlabeled)} rows without a target dropped` : ""}
              {s.contains_nan ? " · contains NaN (tree models only)" : ""}
            </div>
            {s.targets && (
              <div className="secondary" style={{ fontSize: 13 }}>
                {s.targets.map((t: string) => (
                  <span key={t} style={{ marginRight: 16 }}>◎ <b>{t}</b> ({s.target_kinds?.[t]}): {s.target_classes?.[t]?.map((c: string, i: number) => `${i}=${c}`).join(", ")}</span>
                ))}
              </div>
            )}
            <TrainForm e={e} targets={s.targets ?? []} />
            <div className="row">
              <span className="muted" style={{ fontSize: 12 }}>or from a shell:</span>
              <code className="badge" style={{ height: "auto", padding: "4px 8px" }}>{s.train_command}</code>
              <button className="small" onClick={() => { navigator.clipboard?.writeText(s.train_command); setCopied(true); }}>
                {copied ? "✓ Copied" : "Copy"}
              </button>
            </div>
          </>
        )}
        {e.status === "failed" && e.error && (
          <details className="alert bad"><summary>✕ {e.error.split("\n")[0]}</summary><pre>{e.error}</pre></details>
        )}
      </div>
      {e.status === "done" && (
        <div className="stack" style={{ alignItems: "stretch" }}>
          <a className="btn" href={`/api/exports/${e.id}/files/data`}>Dataset (.parquet)</a>
          <a className="btn" href={`/api/exports/${e.id}/files/preprocessor`}>Pipeline (.joblib)</a>
          <a className="btn" href={`/api/exports/${e.id}/files/manifest`}>Manifest (.json)</a>
        </div>
      )}
    </div>
  );
}

function RunsTable({ runs, targets }: { runs: ModelRun[]; targets: string[] }) {
  const metricFor = (r: ModelRun, t: string, m: "auc" | "accuracy") => {
    const x = r.metrics?.test?.[t];
    if (!x) return undefined;
    return m === "auc" ? x.roc_auc ?? x.roc_auc_ovr_macro : x.accuracy;
  };
  const best = useMemo(() => {
    const out: Record<string, number> = {};
    for (const t of targets) for (const m of ["auc", "accuracy"] as const) {
      const vals = runs.map((r) => metricFor(r, t, m)).filter((v): v is number => v !== undefined);
      if (vals.length) out[`${t}:${m}`] = Math.max(...vals);
    }
    return out;
  }, [runs, targets]);
  const cell = (r: ModelRun, t: string, m: "auc" | "accuracy") => {
    const v = metricFor(r, t, m);
    if (v === undefined) return <td className="num muted">—</td>;
    const isBest = runs.length > 1 && v === best[`${t}:${m}`];
    return <td className="num" style={isBest ? { fontWeight: 700 } : undefined}>{fmtNum(v, 3)}{isBest ? " ★" : ""}</td>;
  };
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Run</th><th>Model</th><th>Export</th><th>Chain order</th><th>Status</th>
            {targets.map((t) => <th key={t + "a"} className="num">{t} AUC</th>)}
            {targets.map((t) => <th key={t + "c"} className="num">{t} acc.</th>)}
            <th className="num">Fit</th><th />
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td>#{r.id}</td>
              <td>{r.report?.model_label ?? r.model}{r.family !== "tree" && <span className="type-tag"> {r.family}</span>}</td>
              <td>#{r.export_id}</td>
              <td className="secondary">{(r.report?.order ?? r.config?.order ?? targets).join(" → ")}</td>
              <td>
                {r.status === "done" ? <span className="badge good">✓ done</span>
                  : r.status === "failed" ? <span className="badge bad" title={r.error ?? ""}>✕ failed</span>
                  : <span className="badge accent">… {r.status_message ?? r.status}</span>}
              </td>
              {targets.map((t) => <Fragment key={t + "a"}>{cell(r, t, "auc")}</Fragment>)}
              {targets.map((t) => <Fragment key={t + "c"}>{cell(r, t, "accuracy")}</Fragment>)}
              <td className="num">{r.report?.fit_seconds !== undefined ? `${fmtNum(r.report.fit_seconds, 2)}s` : "—"}</td>
              <td>{r.status === "done" && <a href={`/api/runs/${r.id}/files/model`}>model</a>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ExportPage() {
  const datasetId = Number(useParams().datasetId);
  const qc = useQueryClient();
  const ds = useDataset(datasetId);
  const options = useOptions();
  const [family, setFamily] = useState<string | null>(null);
  const fam = family ?? ds.data?.model_family ?? "tree";
  const familyLabel = (f: string) => options.data?.families.find((x) => x.value === f)?.label ?? f;

  const preview = useQuery({ queryKey: ["exportPreview", datasetId, fam], queryFn: () => api.exportPreview(datasetId, fam) });
  const exports = useQuery({
    queryKey: ["exports", datasetId],
    queryFn: () => api.exports(datasetId),
    refetchInterval: (q) => ((q.state.data as ExportJob[] | undefined)?.some((e) => e.status === "pending" || e.status === "running") ? 1000 : false),
  });
  const runs = useQuery({
    queryKey: ["runs", datasetId],
    queryFn: () => api.runs(datasetId),
    refetchInterval: (q) => ((q.state.data as ModelRun[] | undefined)?.some((r) => r.status === "pending" || r.status === "running") ? 1500 : false),
  });
  const [cfg, setCfg] = useState({ valid_size: 0.15, test_size: 0.15, seed: 42, fit_sample_rows: 500_000 });
  const start = useMutation({
    mutationFn: () => api.startExport(datasetId, { ...cfg, model_family: fam }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["exports", datasetId] }),
  });
  const p = preview.data;
  const num = (key: keyof typeof cfg, label: string, step: number, min: number, max: number) => (
    <label className="field">{label}
      <input type="number" step={step} min={min} max={max} value={cfg[key]} style={{ width: 130 }}
        onChange={(e) => setCfg({ ...cfg, [key]: Number(e.target.value) })} />
    </label>
  );

  return (
    <div className="page">
      <div>
        <h1>Export &amp; train</h1>
        <p className="secondary" style={{ margin: "4px 0 0" }}>
          Your decisions become a scikit-learn ColumnTransformer. It is fitted on training rows only, then applied
          in chunks to every row. The output is all-numeric Parquet with a <code>__split</code> column and label-encoded targets,
          ready for a ClassifierChain.
        </p>
      </div>
      <section className="card stack">
        <div className="card-head">
          <h2>What will be exported</h2>
          <label className="row" style={{ gap: 6 }}>
            <span className="muted">Prepare features for</span>
            <select value={fam} onChange={(e) => setFamily(e.target.value)} aria-label="Model family">
              {options.data?.families.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
            </select>
          </label>
        </div>
        <p className="secondary" style={{ margin: 0, fontSize: 13 }}>{FAMILY_EXPLAINER[fam]}</p>
        {preview.error && <div className="alert bad">{(preview.error as Error).message}</div>}
        {p && (
          <>
            <div className="stats">
              <div className="stat"><div className="label">Fields kept</div><div className="value">{p.kept}</div></div>
              <div className="stat"><div className="label">Fields dropped</div><div className="value">{p.dropped}</div></div>
              <div className="stat"><div className="label">Targets</div><div className="value" style={{ fontSize: 14 }}>{p.targets.join(", ") || "—"}</div></div>
              <div className="stat"><div className="label">Customized settings</div><div className="value">{p.customized}</div></div>
              <div className={`stat${p.unreviewed ? " warn" : ""}`}><div className="label">Unreviewed</div><div className="value">{p.unreviewed}</div></div>
            </div>
            <div className="row" style={{ gap: 12 }}>
              {Object.entries(p.kept_by_type).map(([t, n]) => <span key={t} className="secondary" style={{ fontSize: 13 }}><span className="type-tag">{t}</span> {n}</span>)}
            </div>
            {p.warnings.map((w) => <div key={w} className="alert warn">⚠ {w}{w.includes("not reviewed") && <> <Link to={`/d/${datasetId}/review`}>Review them</Link></>}</div>)}
            {p.blocking.map((b) => <div key={b} className="alert bad">✕ {b}</div>)}
          </>
        )}
      </section>
      <section className="card stack">
        <div className="card-head"><h2>Split &amp; fitting</h2></div>
        <div className="row" style={{ gap: 12, alignItems: "flex-end" }}>
          {num("valid_size", "Validation share", 0.05, 0, 0.45)}
          {num("test_size", "Test share", 0.05, 0, 0.45)}
          {num("seed", "Random seed", 1, 0, 1e9)}
          {num("fit_sample_rows", "Max rows to fit on", 50_000, 1000, 5_000_000)}
          <button className="primary" disabled={!p || p.blocking.length > 0 || start.isPending} onClick={() => start.mutate()}>
            Export for {familyLabel(fam).toLowerCase()}
          </button>
        </div>
        <p className="muted" style={{ fontSize: 12.5, margin: 0 }}>
          Train share: {fmtPct(1 - cfg.valid_size - cfg.test_size, 0)}. Splits are stratified on the combination of both targets when every combination has at least 2 rows.
        </p>
        {start.error && <div className="alert bad">✕ {(start.error as Error).message}</div>}
      </section>
      <section className="card">
        <div className="card-head"><h2>Exports</h2></div>
        {exports.data?.length === 0 && <div className="empty">No exports yet.</div>}
        {exports.data?.map((e) => <ExportRow key={e.id} e={e} familyLabel={familyLabel(e.model_family)} />)}
      </section>
      <section className="card stack">
        <div className="card-head">
          <h2>Model runs</h2>
          <span className="muted">test split · ★ best per column · also in <code>meta.v_model_metrics</code></span>
        </div>
        {runs.data?.length ? (
          <RunsTable runs={runs.data} targets={ds.data?.target_columns ?? []} />
        ) : (
          <div className="empty">Train a chain on a finished export to compare models here.</div>
        )}
      </section>
    </div>
  );
}
