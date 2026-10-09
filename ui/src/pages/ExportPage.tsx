import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useFields, useOptions } from "../hooks";
import { useDatasetContext } from "../components/DatasetLayout";
import { ErrorAlert } from "../components/ui";
import { RunDetail } from "../components/RunDetail";
import { familyShort, fmtAgo, fmtDate, fmtInt, fmtNum, fmtPct } from "../format";
import { toast } from "../toast";
import type { ExportJob, ModelRun } from "../types";

const FAMILY_EXPLAINER: Record<string, string> = {
  tree: "Missing values stay as NaN for the trees to route; no scaling or log transforms; mid/high-cardinality categoricals become compact ordinal codes.",
  linear: "Every numeric is imputed (with missing flags), log-transformed when skewed and standardized; categoricals are one-hot encoded.",
  neural: "Like linear, with quantile-normal scaling for skewed inputs so the network sees well-conditioned features.",
};
const HOW_IT_WORKS =
  "Your decisions become a scikit-learn ColumnTransformer. It is fitted on training rows only, then applied in chunks to every row. " +
  "The output is all-numeric Parquet with a __split column and label-encoded targets, ready for a ClassifierChain.";
const MIN_TRAIN = 0.5;
const isActive = (s: string) => s === "pending" || s === "running";

/** A small anchored popover that closes on outside click or Esc. */
function Pop({ label, children, primary, align = "right" }: { label: ReactNode; children: (close: () => void) => ReactNode; primary?: boolean; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div ref={ref} style={{ position: "relative" }} onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <button type="button" className={`small${primary ? " primary" : ""}`} aria-expanded={open} onClick={() => setOpen(!open)}>{label}</button>
      {open && <div className="popover stack" style={{ minWidth: 280, padding: 12, gap: 8, ...(align === "left" ? { left: 0, right: "auto" } : {}) }}>{children(() => setOpen(false))}</div>}
    </div>
  );
}

function SplitBar({ valid, test, rows }: { valid: number; test: number; rows: number | null }) {
  const train = 1 - valid - test;
  const bad = train < MIN_TRAIN;
  const seg = (label: string, share: number, color: string) => share > 0 && (
    <span style={{ width: `${Math.max(share, 0) * 100}%`, background: color }} title={`${label} ${fmtPct(share, 0)}`}>
      {share >= 0.1 && <>{label} {fmtPct(share, 0)}{rows && share >= 0.3 ? ` · ${fmtInt(Math.round(rows * share))}` : ""}</>}
    </span>
  );
  return (
    <div className="split-bar" role="img" aria-label={`Train ${fmtPct(train, 0)}, validation ${fmtPct(valid, 0)}, test ${fmtPct(test, 0)}`}>
      {seg("train", Math.max(train, 0), bad ? "var(--bad)" : "var(--accent-fill)")}
      {seg("valid", valid, "var(--warn)")}
      {seg("test", test, "var(--text-secondary)")}
    </div>
  );
}

function BuildPanel({ datasetId, family, rows }: { datasetId: number; family: string; rows: number | null }) {
  const qc = useQueryClient();
  const fields = useFields(datasetId);
  const preview = useQuery({ queryKey: ["exportPreview", datasetId, family], queryFn: () => api.exportPreview(datasetId, family) });
  const exports = useQuery({ queryKey: ["exports", datasetId], queryFn: () => api.exports(datasetId) });
  const [valid, setValid] = useState("15");
  const [test, setTest] = useState("15");
  const [seed, setSeed] = useState(42);
  const [fitRows, setFitRows] = useState<number | null>(null);
  const v = Number(valid) / 100, t = Number(test) / 100;
  const train = 1 - v - t;
  const splitError = !Number.isFinite(v) || !Number.isFinite(t) || v < 0 || t < 0 ? "Shares must be numbers of 0% or more."
    : v >= 0.5 || t >= 0.5 ? "Validation and test must each stay under 50%."
    : train < MIN_TRAIN ? `Train would be ${fmtPct(train, 0)} — keep it at ${MIN_TRAIN * 100}% or more.` : null;
  const maxFit = rows ?? 500_000;
  const fit = Math.min(fitRows ?? maxFit, 5_000_000);

  const start = useMutation({
    mutationFn: (cfg: { valid_size: number; test_size: number; seed: number; fit_sample_rows: number }) =>
      api.startExport(datasetId, { ...cfg, model_family: family }),
    onSuccess: (e) => {
      qc.invalidateQueries({ queryKey: ["exports", datasetId] });
      toast({ message: `Export #${e.id} started` });
    },
  });
  const p = preview.data;
  const lastSame = exports.data?.find((e) => e.status === "done" && e.model_family === family);
  const firstTodo = fields.data?.find((f) => !f.reviewed);
  const reExport = () => lastSame && start.mutate({
    valid_size: lastSame.config.valid_size, test_size: lastSame.config.test_size,
    seed: lastSame.config.seed, fit_sample_rows: lastSame.config.fit_sample_rows,
  });

  return (
    <section className="card build-panel">
      <div className="stack" style={{ gap: 10, minWidth: 0 }}>
        <h2>What will be exported</h2>
        <ErrorAlert error={preview.error} />
        {p && (
          <>
            <div className="secondary num">
              <b>{p.kept}</b> kept · <b>{p.dropped}</b> dropped · targets {p.targets.length ? p.targets.map((x) => <code key={x} className="target-code">{x}</code>) : <span className="muted">none</span>} · <b>{p.customized}</b> customized
            </div>
            <div className="chips">
              {Object.entries(p.kept_by_type).map(([type, n]) => <span key={type} className="badge"><span className="mono">{type.replace("_", " ")}</span> {n}</span>)}
            </div>
            {p.unreviewed > 0 && (
              <div className="alert warn">
                <span className="grow">{p.unreviewed} unreviewed — their suggested settings will be used.</span>
                {firstTodo && <Link to={`/d/${datasetId}/review/${firstTodo.id}`}>Review next →</Link>}
              </div>
            )}
            {p.warnings.filter((w) => !w.includes("not reviewed")).map((w) => <div key={w} className="alert warn">⚠ {w}</div>)}
            {p.blocking.map((b) => <ErrorAlert key={b} error={{ message: b }} />)}
            {p.last_export && (p.changed_since_export ? (
              <div className="alert warn">
                <span className="grow">{p.changed_since_export} decision{p.changed_since_export === 1 ? "" : "s"} changed since Export #{p.last_export.id}</span>
                <button type="button" className="small" disabled={start.isPending || p.blocking.length > 0} onClick={reExport}>Re-export with same split</button>
              </div>
            ) : (
              <span className="badge good" style={{ alignSelf: "flex-start" }}>✓ Up to date with Export #{p.last_export.id}</span>
            ))}
          </>
        )}
      </div>
      <div className="stack" style={{ gap: 10 }}>
        <h2>Split</h2>
        <SplitBar valid={Math.max(v, 0) || 0} test={Math.max(t, 0) || 0} rows={rows} />
        <div className="row" style={{ gap: 12, alignItems: "flex-end" }}>
          <label className="field">Validation
            <span className="value-pct"><input type="number" min={0} max={45} step={5} value={valid} onChange={(e) => setValid(e.target.value)} style={{ width: 80 }} /> %</span>
          </label>
          <label className="field">Test
            <span className="value-pct"><input type="number" min={0} max={45} step={5} value={test} onChange={(e) => setTest(e.target.value)} style={{ width: 80 }} /> %</span>
          </label>
        </div>
        {splitError && <span className="small-text" style={{ color: "var(--bad)" }} role="alert">{splitError}</span>}
        <details className="advanced">
          <summary className="small-text secondary">Advanced</summary>
          <div className="row" style={{ gap: 12, marginTop: 8 }}>
            <label className="field">Random seed<input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: 110 }} /></label>
            <label className="field">Max rows to fit on
              <input type="number" min={1000} max={maxFit} step={10_000} value={fit} style={{ width: 130 }}
                onChange={(e) => setFitRows(Math.max(1000, Math.min(Number(e.target.value), maxFit)))} />
            </label>
          </div>
        </details>
        <div className="row" style={{ justifyContent: "flex-end", marginTop: "auto" }}>
          <span className="muted small-text">Model is set in the top bar</span>
          <button type="button" className="primary" disabled={!p || p.blocking.length > 0 || Boolean(splitError) || start.isPending}
            onClick={() => start.mutate({ valid_size: v, test_size: t, seed, fit_sample_rows: Math.max(fit, 1000) })}>
            Export for {family} models
          </button>
        </div>
        <ErrorAlert error={start.error} />
      </div>
    </section>
  );
}

function CopyCommand({ command }: { command: string }) {
  const [state, setState] = useState<"idle" | "copied" | "manual">("idle");
  const ref = useRef<HTMLElement>(null);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setState("copied");
    } catch {
      // Clipboard needs a secure context; over plain-http LAN access, select the text instead (#53).
      const sel = window.getSelection();
      if (ref.current && sel) { const r = document.createRange(); r.selectNodeContents(ref.current); sel.removeAllRanges(); sel.addRange(r); }
      setState("manual");
    }
  };
  return (
    <div className="stack" style={{ gap: 6 }}>
      <code ref={ref} className="cmd">{command}</code>
      <div className="row">
        <button type="button" className="small" onClick={copy}>{state === "copied" ? "✓ Copied" : "Copy"}</button>
        {state === "manual" && <span className="muted small-text">Selected — press ⌘C to copy</span>}
      </div>
    </div>
  );
}

function TrainForm({ e, targets, onStarted }: { e: ExportJob; targets: string[]; onStarted: (r: ModelRun) => void }) {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["exportModels", e.id], queryFn: () => api.modelsForExport(e.id), staleTime: 60_000 });
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  const [model, setModel] = useState("");
  const [gpuOn, setGpuOn] = useState(false);
  const [reverse, setReverse] = useState(false);
  const [showAll, setShowAll] = useState(false);
  useEffect(() => {
    if (!model && models.data) {
      const own = models.data.find((m) => m.compatible && m.family === e.model_family);
      setModel((own ?? models.data.find((m) => m.compatible))?.key ?? "");
    }
  }, [model, models.data, e.model_family]);
  const start = useMutation({
    mutationFn: () => api.startRun(e.id, { model, order: reverse ? [...targets].reverse() : targets, cv: 5, device: useGpu ? "cuda" : "cpu" }),
    onSuccess: (run) => {
      qc.setQueryData<ModelRun[]>(["runs", e.dataset_id], (old) => [run, ...(old ?? []).filter((r) => r.id !== run.id)]);
      qc.invalidateQueries({ queryKey: ["runs", e.dataset_id] });
      onStarted(run);
    },
  });
  const shown = models.data?.filter((m) => showAll || m.compatible) ?? [];
  const selected = models.data?.find((m) => m.key === model);
  const gpu = system.data?.gpu;
  const gpuChoice = Boolean(gpu?.available && selected?.devices?.includes("cuda"));
  const useGpu = gpuChoice && gpuOn;
  return (
    <>
      <label className="field">Model
        <select value={model} onChange={(ev) => setModel(ev.target.value)}>
          {shown.map((m) => <option key={m.key} value={m.key} disabled={!m.compatible}>{m.label}{m.compatible ? "" : " — not compatible"}</option>)}
        </select>
      </label>
      {selected && <span className="muted small-text">{selected.description}</span>}
      {gpuChoice && gpu && (
        <div className="stack" style={{ gap: 4 }}>
          <span className="secondary small-text" id={`device-label-${e.id}`}>Train on</span>
          <div className="segmented soft" role="radiogroup" aria-labelledby={`device-label-${e.id}`}>
            {([false, true] as const).map((on) => (
              <button key={String(on)} type="button" role="radio" aria-checked={gpuOn === on} className={gpuOn === on ? "on" : ""}
                onClick={() => setGpuOn(on)}>{on ? "GPU" : "CPU"}</button>
            ))}
          </div>
          <span className="muted small-text num">{gpu.name}{gpu.memory_mb ? ` · ${fmtInt(Math.round(gpu.memory_mb / 1024))} GB` : ""}</span>
        </div>
      )}
      {models.data?.some((m) => !m.compatible) && (
        <label className="check small-text"><input type="checkbox" checked={showAll} onChange={(ev) => setShowAll(ev.target.checked)} /> show incompatible models</label>
      )}
      {targets.length === 2 && (
        <label className="field">Chain order
          <select value={reverse ? "rev" : "fwd"} onChange={(ev) => setReverse(ev.target.value === "rev")}>
            <option value="fwd">{targets[0]} → {targets[1]}</option>
            <option value="rev">{targets[1]} → {targets[0]}</option>
          </select>
        </label>
      )}
      <ErrorAlert error={start.error} />
      <button type="button" className="primary" disabled={!model || !selected?.compatible || start.isPending} onClick={() => start.mutate()}>Train chain</button>
    </>
  );
}

function featuresByType(e: ExportJob): string {
  const by = e.summary?.features_by_type as Record<string, { fields: number; features: number }> | undefined;
  if (!by) return "";
  return Object.entries(by).map(([t, x]) => `${t.replace("_", " ")}: ${x.fields} → ${x.features}`).join("\n");
}

function ExportRow({ e, onStarted }: { e: ExportJob; onStarted: (r: ModelRun) => void }) {
  const s = e.summary ?? {};
  const by = s.features_by_type as Record<string, { fields: number }> | undefined;
  const inputs = by ? Object.values(by).reduce((a, x) => a + x.fields, 0) : null;
  return (
    <div className="export-row">
      <div className="stack" style={{ gap: 4, minWidth: 0 }}>
        <div className="row">
          <b>Export #{e.id}</b>
          <span className="badge">{familyShort(e.model_family)}</span>
          {e.status === "done" ? <span className="badge good">✓ done</span>
            : e.status === "failed" ? <span className="badge bad">✕ failed</span>
            : <span className="badge accent"><span className="spinner" aria-hidden="true" /> {e.status}</span>}
          <span className="muted small-text num" title={fmtDate(e.created_at)}>
            {fmtAgo(e.created_at)}
            {e.n_rows !== null && ` · ${fmtInt(e.n_rows)} rows · `}
            {e.n_features !== null && (
              <abbr title={featuresByType(e) || "Text, date and one-hot fields expand into several features each."}>
                {inputs !== null ? `${fmtInt(inputs)} fields → ` : ""}{fmtInt(e.n_features)} features ⓘ
              </abbr>
            )}
          </span>
        </div>
        {isActive(e.status) && (
          <div className="row" style={{ maxWidth: 520 }}>
            <div className="meter" style={{ flex: 1 }}><span style={{ width: `${Math.max(e.progress, 0.02) * 100}%` }} /></div>
            <span className="muted small-text">{e.status_message}</span>
          </div>
        )}
        {e.status === "done" && s.split_counts && (
          <span className="muted small-text num">
            train {fmtInt(s.split_counts.train)} · valid {fmtInt(s.split_counts.valid)} · test {fmtInt(s.split_counts.test)}
            {s.split_info?.stratified ? " · stratified on the targets" : ""}
            {s.contains_nan ? " · contains NaN (tree models only)" : ""}
          </span>
        )}
        {e.status === "failed" && e.error && (
          <details className="alert bad"><summary>✕ {e.error.split("\n")[0]}</summary><pre>{e.error}</pre></details>
        )}
      </div>
      {e.status === "done" && (
        <div className="row" style={{ flexWrap: "nowrap" }}>
          <Pop label="Train ▾" primary>{() => <TrainForm e={e} targets={s.targets ?? []} onStarted={onStarted} />}</Pop>
          <Pop label="Files ▾">{() => (
            <>
              <a className="btn" href={`/api/exports/${e.id}/files/data`}>Dataset (.parquet)</a>
              <a className="btn" href={`/api/exports/${e.id}/files/preprocessor`}>Pipeline (.joblib)</a>
              <a className="btn" href={`/api/exports/${e.id}/files/manifest`}>Manifest (.json)</a>
            </>
          )}</Pop>
          <Pop label="‹/› Use from code">{() => (
            <>
              <span className="secondary small-text">Train from a shell</span>
              <CopyCommand command={s.train_command ?? `docker compose exec api python -m app.modeling.train --export-id ${e.id}`} />
              {s.targets && (
                <span className="muted small-text">
                  {s.targets.map((t: string) => `${t}: ${s.target_classes?.[t]?.map((c: string, i: number) => `${i}=${c}`).join(", ")}`).join(" · ")}
                </span>
              )}
            </>
          )}</Pop>
        </div>
      )}
    </div>
  );
}

function RunsGroup({ e, runs, fresh }: { e: ExportJob; runs: ModelRun[]; fresh: number | null }) {
  const [open, setOpen] = useState<number | null>(null);
  const targets: string[] = e.summary?.targets ?? [];
  const metricFor = (r: ModelRun, t: string, m: "auc" | "f1") => {
    const x = r.metrics?.test?.[t];
    if (!x) return undefined;
    return m === "auc" ? x.roc_auc ?? x.roc_auc_ovr_macro : x.f1_macro;
  };
  // Best is judged within one export only: runs on different exports used different features (#52).
  const best = useMemo(() => {
    const out: Record<string, number> = {};
    for (const t of targets) for (const m of ["auc", "f1"] as const) {
      const vals = runs.map((r) => metricFor(r, t, m)).filter((v): v is number => v !== undefined);
      if (vals.length > 1) out[`${t}:${m}`] = Math.max(...vals);
    }
    return out;
  }, [runs, targets]);
  // Open a run's detail as soon as the run you just started finishes.
  const prevStatus = useRef(new Map<number, string>());
  useEffect(() => {
    for (const r of runs) {
      if (r.id === fresh && prevStatus.current.get(r.id) && isActive(prevStatus.current.get(r.id)!) && r.status === "done") setOpen(r.id);
      prevStatus.current.set(r.id, r.status);
    }
  }, [runs, fresh]);
  const cell = (r: ModelRun, t: string, m: "auc" | "f1") => {
    const v = metricFor(r, t, m);
    if (v === undefined) return <td className="num muted">—</td>;
    const isBest = v === best[`${t}:${m}`];
    return <td className="num" style={isBest ? { fontWeight: 700 } : undefined}>{fmtNum(v, 3)}{isBest ? " ★" : ""}</td>;
  };
  const sc = e.summary?.split_counts;
  return (
    <>
      <tr className="group-head">
        <td colSpan={6 + targets.length * 2}>
          Export #{e.id} · {familyShort(e.model_family)}{sc ? ` · ${fmtInt(sc.train)} / ${fmtInt(sc.valid)} / ${fmtInt(sc.test)}` : ""}
        </td>
      </tr>
      {runs.map((r) => {
        const order = r.report?.order ?? r.config?.order ?? targets;
        return (
          <Fragment key={r.id}>
            <tr className={`run-row${open === r.id ? " open" : ""}${fresh === r.id ? " fresh" : ""}`}
              onClick={() => r.status === "done" && setOpen(open === r.id ? null : r.id)} aria-expanded={open === r.id}>
              <td>{r.status === "done" ? (open === r.id ? "▾" : "▸") : ""} {r.report?.model_label ?? r.model}</td>
              <td className="secondary">{order.join(" → ")}</td>
              <td>
                {r.status === "done" ? <span className="badge good">✓ done</span>
                  : r.status === "failed" ? <span className="badge bad" title={r.error ?? ""}>✕ failed</span>
                  : <span className="row" style={{ flexWrap: "nowrap" }}><span className="badge accent"><span className="spinner" aria-hidden="true" /> {r.status_message ?? r.status}</span></span>}
              </td>
              {targets.map((t) => <Fragment key={t + "a"}>{cell(r, t, "auc")}</Fragment>)}
              {targets.map((t) => <Fragment key={t + "f"}>{cell(r, t, "f1")}</Fragment>)}
              <td className="num">{r.report?.fit_seconds !== undefined ? `${fmtNum(r.report.fit_seconds, 2)}s` : "—"}</td>
              <td>{(r.report?.device ?? r.config?.device) === "cuda" ? <span className="badge accent">GPU</span> : <span className="muted small-text">CPU</span>}</td>
              <td>{r.status === "done" && <a href={`/api/runs/${r.id}/files/model`} onClick={(ev) => ev.stopPropagation()}>model</a>}</td>
            </tr>
            {open === r.id && (
              <tr className="run-detail"><td colSpan={6 + targets.length * 2}><RunDetail run={r} targets={order} /></td></tr>
            )}
          </Fragment>
        );
      })}
    </>
  );
}

export function ExportPage() {
  const ds = useDatasetContext();
  const datasetId = ds.id;
  useOptions();
  const family = ds.model_family;
  const [fresh, setFresh] = useState<number | null>(null);
  const exports = useQuery({
    queryKey: ["exports", datasetId],
    queryFn: () => api.exports(datasetId),
    refetchInterval: (q) => ((q.state.data as ExportJob[] | undefined)?.some((e) => isActive(e.status)) ? 1000 : false),
  });
  const runs = useQuery({
    queryKey: ["runs", datasetId],
    queryFn: () => api.runs(datasetId),
    refetchInterval: (q) => ((q.state.data as ModelRun[] | undefined)?.some((r) => isActive(r.status)) ? 1500 : false),
  });
  const onStarted = (r: ModelRun) => {
    setFresh(r.id);
    toast({ message: `Training ${r.model} on Export #${r.export_id}` });
  };
  const groups = (exports.data ?? [])
    .map((e) => ({ e, runs: (runs.data ?? []).filter((r) => r.export_id === e.id) }))
    .filter((g) => g.runs.length);
  const headTargets: string[] = groups[0]?.e.summary?.targets ?? ds.target_columns;

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Export &amp; train <abbr className="muted" style={{ fontSize: 14, fontWeight: 400 }} title={`${HOW_IT_WORKS}\n\n${familyShort(family)}: ${FAMILY_EXPLAINER[family] ?? ""}`}>ⓘ How export works</abbr></h1>
      </div>
      <BuildPanel datasetId={datasetId} family={family} rows={ds.row_count} />

      <section className="card stack" style={{ gap: 0 }}>
        <div className="card-head"><h2>Exports</h2></div>
        <ErrorAlert error={exports.error} />
        {exports.data?.length === 0 && <div className="empty">No exports yet. Build one above, then train models on it.</div>}
        {exports.data?.map((e) => <ExportRow key={e.id} e={e} onStarted={onStarted} />)}
      </section>

      <section className="card stack">
        <div className="card-head">
          <h2>Model runs</h2>
          <span className="muted small-text">test split · ★ best within an export · click a run for precision, recall, F1 and confusion matrices</span>
        </div>
        <ErrorAlert error={runs.error} />
        {groups.length ? (
          <div className="table-wrap" style={{ maxHeight: "none" }}>
            <table className="data runs">
              <thead>
                <tr>
                  <th>Model</th><th>Chain order</th><th>Status</th>
                  {headTargets.map((t) => <th key={"a" + t} className="num">{t} AUC</th>)}
                  {headTargets.map((t) => <th key={"f" + t} className="num">{t} F1 (macro)</th>)}
                  <th className="num">Fit</th><th>Device</th><th />
                </tr>
              </thead>
              <tbody>
                {groups.map((g) => <RunsGroup key={g.e.id} e={g.e} runs={g.runs} fresh={fresh} />)}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">Train a chain on a finished export to compare models here.</div>
        )}
      </section>
    </div>
  );
}
