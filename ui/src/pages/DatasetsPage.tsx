import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, uploadDataset } from "../api";
import { PROCESSING, useMe, useOptions } from "../hooks";
import { baselineLabel, fmtAgo, fmtDate, fmtInt, fmtNum, fmtPct } from "../format";
import { ErrorAlert } from "../components/ui";
import { PhaseBadge } from "./DatasetTeamPage";
import { showError, toast } from "../toast";
import type { Dataset, FieldListItem, OptionCatalog } from "../types";

const ACCEPT = /\.(csv|parquet|pq|txt)$/i;

function StatusBadge({ ds }: { ds: Dataset }) {
  if (ds.status === "ready") return <span className="badge good">✓ ready</span>;
  if (ds.status === "failed") return <span className="badge bad">✕ failed</span>;
  return (
    <span className="badge accent" title={ds.status_message ?? ""}>
      <span className="spinner" aria-hidden="true" /> {ds.status_message ?? ds.status}
    </span>
  );
}

function lastVisited(id: number): string | null {
  try { return localStorage.getItem(`rf:last:${id}`); } catch { return null; }
}

/** Uploads start the moment a file is picked or dropped; name and targets are set afterwards (P11). */
function useUpload() {
  const qc = useQueryClient();
  const [progress, setProgress] = useState<{ name: string; frac: number } | null>(null);
  const start = async (file: File | undefined) => {
    if (!file) return;
    if (!ACCEPT.test(file.name)) {
      showError("Can't upload that file", { message: "Choose a .csv or .parquet file" });
      return;
    }
    setProgress({ name: file.name, frac: 0 });
    try {
      const ds = await uploadDataset(file, "", "", (frac) => setProgress({ name: file.name, frac }));
      qc.invalidateQueries({ queryKey: ["datasets"] });
      return ds;
    } catch (e) {
      showError(`Upload of ${file.name} failed`, e);
    } finally {
      setProgress(null);
    }
  };
  return { progress, start };
}

/** Why a field can't be a target (the API enforces the same rule), or null. */
function targetBlocker(f: FieldListItem, rule: OptionCatalog["target_rule"] | undefined): string | null {
  if (!rule) return null;
  if (!rule.types.includes(f.semantic_type))
    return `${f.semantic_type.replace("_", " ")}: a target must be binary, categorical or discrete numeric (change its type in Review first)`;
  if ((f.n_unique ?? 0) > rule.max_levels) return `${fmtInt(f.n_unique)} distinct values: a target can have at most ${rule.max_levels} classes`;
  return null;
}

function TargetPicker({ ds, onClose }: { ds: Dataset; onClose: () => void }) {
  const qc = useQueryClient();
  const fields = useQuery({ queryKey: ["fields", ds.id], queryFn: () => api.fields(ds.id) });
  const rule = useOptions().data?.target_rule;
  const [q, setQ] = useState("");
  const [chosen, setChosen] = useState<Set<string>>(new Set(ds.target_columns));
  const save = useMutation({
    mutationFn: (names: string[]) => api.patchDataset(ds.id, { target_columns: names }),
    onSuccess: () => {
      for (const k of ["datasets", "dataset", "fields", "field", "corr", "corrMatrix", "analysis", "fieldQuery", "exportPreview"]) {
        qc.invalidateQueries({ queryKey: [k] });
      }
      onClose();
    },
  });
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && onClose();
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [onClose]);
  const shown = (fields.data ?? []).filter((f) => f.name.toLowerCase().includes(q.toLowerCase())).slice(0, 200);
  const toggle = (n: string) => {
    const next = new Set(chosen);
    next.has(n) ? next.delete(n) : next.add(n);
    setChosen(next);
  };
  return (
    <div ref={ref} className="popover combobox" role="dialog" aria-label={`Targets for ${ds.name}`}
      onKeyDown={(e) => e.key === "Escape" && onClose()}>
      <input type="search" placeholder="Search columns" value={q} onChange={(e) => setQ(e.target.value)} autoFocus aria-label="Search columns" />
      <label className="check"><span className="row" style={{ gap: 8 }}>
        <input type="radio" checked={chosen.size === 0} onChange={() => setChosen(new Set())} />
        No target (unsupervised)
      </span></label>
      <div className="combo-list">
        {fields.isLoading && <span className="muted small-text" style={{ padding: 6 }}>Loading columns…</span>}
        {shown.map((f) => {
          const blocker = targetBlocker(f, rule);
          // A current target that no longer qualifies stays enabled so it can be unticked.
          const disabled = blocker !== null && !chosen.has(f.name);
          return (
            <label key={f.id} className="check" title={blocker ?? undefined} style={blocker ? { flexWrap: "wrap", ...(disabled && { opacity: 0.55, cursor: "not-allowed" }) } : undefined}>
              <span className="row" style={{ gap: 8, flexWrap: "nowrap", minWidth: 0 }}>
                <input type="checkbox" checked={chosen.has(f.name)} disabled={disabled} onChange={() => toggle(f.name)}
                  aria-describedby={blocker ? `tb-${f.id}` : undefined} />
                <span className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{f.name}</span>
              </span>
              <span className="type-tag">{f.semantic_type.replace("_", " ")}</span>
              {blocker && <span id={`tb-${f.id}`} className="muted small-text" style={{ flexBasis: "100%" }}>{blocker}</span>}
            </label>
          );
        })}
      </div>
      <ErrorAlert error={save.error} />
      <div className="row" style={{ justifyContent: "space-between", padding: "4px 2px 0" }}>
        <span className="muted small-text">{chosen.size} selected · associations are recomputed</span>
        <button type="button" className="primary small" disabled={save.isPending} onClick={() => save.mutate([...chosen])}>Save</button>
      </div>
    </div>
  );
}

function RowMenu({ ds, canDelete, onRename, onDelete }: { ds: Dataset; canDelete: boolean; onRename: () => void; onDelete: () => void }) {
  const nav = useNavigate();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div ref={ref} style={{ position: "relative" }} onClick={(e) => e.stopPropagation()}>
      <button type="button" className="ghost small" aria-label={`More actions for ${ds.name}`} aria-expanded={open} onClick={() => setOpen(!open)}>⋯</button>
      {open && (
        <div className="popover" role="menu" onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
          <button type="button" role="menuitem" className="ghost menu-item" onClick={() => { setOpen(false); onRename(); }}>Rename</button>
          {canDelete && ds.status === "ready" && ds.phase === "assessing" && (
            <button type="button" role="menuitem" className="ghost menu-item" onClick={() => nav(`/d/${ds.id}/team`)}>Assessors…</button>
          )}
          <button type="button" role="menuitem" className="ghost menu-item" style={{ color: "var(--bad)" }}
            disabled={!canDelete || PROCESSING.has(ds.status)} title={canDelete ? undefined : "Only a lead can delete a dataset"}
            onClick={() => { setOpen(false); onDelete(); }}>Delete…</button>
        </div>
      )}
    </div>
  );
}

function DeleteDialog({ ds, onClose }: { ds: Dataset; onClose: () => void }) {
  const qc = useQueryClient();
  const del = useMutation({
    mutationFn: () => api.deleteDataset(ds.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      toast({ message: `Deleted ${ds.name}` });
      onClose();
    },
  });
  const parts = [
    ds.field_count !== undefined && `${fmtInt(ds.field_count)} field decisions`,
    ds.export_count ? `${ds.export_count} export${ds.export_count === 1 ? "" : "s"}` : null,
    ds.run_count ? `${ds.run_count} model run${ds.run_count === 1 ? "" : "s"}` : null,
    "the raw table",
  ].filter(Boolean);
  return (
    <div className="overlay" onClick={onClose}>
      <div className="dialog" role="alertdialog" aria-modal="true" aria-label={`Delete ${ds.name}`} onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.key === "Escape" && onClose()}>
        <h2>Delete {ds.name}?</h2>
        <p className="secondary" style={{ margin: 0 }}>This removes {parts.join(", ")}. It can't be undone.</p>
        <ErrorAlert error={del.error} />
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button type="button" onClick={onClose} autoFocus>Cancel</button>
          <button type="button" className="danger" disabled={del.isPending} onClick={() => del.mutate()}>Delete</button>
        </div>
      </div>
    </div>
  );
}

function SampleMenu({ onCreated }: { onCreated: (ds: Dataset) => void }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState("50000");
  const [fields, setFields] = useState("150");
  const [kind, setKind] = useState("mixed");
  const ref = useRef<HTMLDivElement>(null);
  const limits = useQuery({ queryKey: ["sample-limits"], queryFn: api.sampleLimits, enabled: open, staleTime: Infinity });
  // rows × fields is capped server-side to bound generation memory.
  const maxRows = Math.min(2000000, Math.floor((limits.data?.max_cells ?? Infinity) / Math.max(Number(fields) || 30, 30)));
  const overBudget = Number(rows) > maxRows;
  const m = useMutation({
    mutationFn: () => api.createSample({ rows: Number(rows), fields: Number(fields), seed: 42, target_kind: kind }),
    onSuccess: (ds) => { qc.invalidateQueries({ queryKey: ["datasets"] }); setOpen(false); onCreated(ds); },
  });
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>Generate sample ▾</button>
      {open && (
        <form className="popover stack" style={{ width: 320, padding: 14, gap: 10 }}
          onSubmit={(e) => { e.preventDefault(); m.mutate(); }} onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
          <span className="secondary small-text">
            Mixed numeric, categorical, text and date fields with missingness, skew, IDs, numbers-as-text, a constant column and one leaky feature.
          </span>
          <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
            <label className="field" style={{ flex: 1 }}>Rows
              <input type="number" min={1000} max={maxRows} step={1000} value={rows} onChange={(e) => setRows(e.target.value)} />
              <span className="muted" style={{ fontWeight: 400 }}>1,000 – {fmtInt(maxRows)}{maxRows < 2000000 ? ` at ${fields} fields` : ""}</span>
            </label>
            <label className="field" style={{ width: 96 }}>Fields
              <input type="number" min={30} max={400} value={fields} onChange={(e) => setFields(e.target.value)} />
              <span className="muted" style={{ fontWeight: 400 }}>30 – 400</span>
            </label>
          </div>
          <label className="field">Targets
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="mixed">column_a binary, column_b 3-class</option>
              <option value="binary">both binary</option>
              <option value="multiclass">both multiclass</option>
            </select>
          </label>
          {overBudget && (
            <span className="small-text" role="alert">
              Too large: rows × fields is capped at {fmtInt(limits.data?.max_cells)}.{" "}
              <button type="button" className="link" onClick={() => setRows(String(Math.floor(maxRows / 1000) * 1000))}>
                Use {fmtInt(Math.floor(maxRows / 1000) * 1000)} rows
              </button>
            </span>
          )}
          <ErrorAlert error={m.error} />
          <button type="submit" className="primary" disabled={m.isPending || overBudget}>Generate</button>
        </form>
      )}
    </div>
  );
}

function FirstRun({ onBrowse, onDemo, busy }: { onBrowse: () => void; onDemo: () => void; busy: boolean }) {
  return (
    <section className="first-run card">
      <button type="button" className="dropzone first-drop" onClick={onBrowse}>
        <b>Drop a CSV or Parquet file</b>
        <span>or <span style={{ color: "var(--accent)" }}>browse</span></span>
        <span className="muted small-text">Targets are picked after upload</span>
      </button>
      <div className="stack" style={{ justifyContent: "center", gap: 12 }}>
        <h2>No data handy?</h2>
        <div><button type="button" className="primary" disabled={busy} onClick={onDemo}>Open the 50k × 150 demo</button></div>
        <span className="muted small-text">Or pick its size and targets with <b>Generate sample ▾</b> above.</span>
        <div className="chips">
          {["leaky feature", "ID columns", "numbers-as-text", "missingness"].map((c) => <span key={c} className="badge">{c}</span>)}
        </div>
      </div>
      <p className="muted" style={{ gridColumn: "1 / -1", margin: 0 }}>
        <b className="secondary">1</b> Review each field (<kbd>J</kbd>/<kbd>K</kbd>, <kbd>X</kbd>, <kbd>↵</kbd>) → <b className="secondary">2</b> Export and train in an experiment
      </p>
    </section>
  );
}

export function DatasetsPage() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const fileInput = useRef<HTMLInputElement>(null);
  const upload = useUpload();
  const me = useMe();
  const [dragging, setDragging] = useState(false);
  // ?target=<dataset id> opens that dataset's target picker (linked from Explore).
  const [targetFor, setTargetFor] = useState<number | null>(() => Number(new URLSearchParams(window.location.search).get("target")) || null);
  const [renaming, setRenaming] = useState<number | null>(null);
  const [deleting, setDeleting] = useState<Dataset | null>(null);
  const [fresh, setFresh] = useState<number | null>(null);
  const [details, setDetails] = useState<number | null>(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["datasets"],
    queryFn: api.datasets,
    refetchInterval: (q) => ((q.state.data as Dataset[] | undefined)?.some((d) => PROCESSING.has(d.status)) ? 1000 : false),
  });
  const demo = useMutation({
    mutationFn: () => api.createSample({ rows: 50_000, fields: 150, seed: 42, target_kind: "mixed" }),
    onSuccess: (ds) => { qc.invalidateQueries({ queryKey: ["datasets"] }); highlight(ds.id); },
  });
  const rename = useMutation({
    mutationFn: ({ id, name }: { id: number; name: string }) => api.patchDataset(id, { name }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["datasets"] }); qc.invalidateQueries({ queryKey: ["dataset"] }); setRenaming(null); },
    onError: (e) => showError("Couldn't rename", e),
  });

  const highlight = (id: number) => {
    setFresh(id);
    window.setTimeout(() => setFresh((f) => (f === id ? null : f)), 1500);
  };
  const uploaded = (ds: Dataset) => {
    highlight(ds.id);
    if (me.data?.role === "lead") {
      toast({ message: `${ds.name} uploaded`, action: { label: "Set up its baseline…", run: () => nav(`/d/${ds.id}/team`) } });
    }
  };
  const open = (ds: Dataset) => {
    const last = lastVisited(ds.id);
    nav(`/d/${ds.id}/review${last ? `/${last}` : ""}`);
  };

  // Tell the user when a dataset they're waiting on becomes ready.
  const wasProcessing = useRef<Set<number>>(new Set());
  useEffect(() => {
    for (const ds of data ?? []) {
      if (PROCESSING.has(ds.status)) wasProcessing.current.add(ds.id);
      else if (wasProcessing.current.delete(ds.id) && ds.status === "ready") {
        highlight(ds.id);
        toast({
          message: `✓ ${ds.name} is ready`, detail: `${fmtInt(ds.column_count)} fields`, tone: "good", duration: 8000,
          action: { label: "Start review", run: () => nav(`/d/${ds.id}/review`) },
        });
      }
    }
  }, [data, nav]);

  // A file dropped anywhere on the page uploads; without this the browser would navigate to it.
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const enter = (e: DragEvent) => { if (hasFiles(e)) { depth++; setDragging(true); } };
    const leave = (e: DragEvent) => { if (hasFiles(e) && --depth <= 0) { depth = 0; setDragging(false); } };
    const over = (e: DragEvent) => hasFiles(e) && e.preventDefault();
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      void upload.start(e.dataTransfer?.files[0]).then((ds) => ds && uploaded(ds));
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const browse = () => fileInput.current?.click();
  const rows = useMemo(() => data ?? [], [data]);

  return (
    <div className="page">
      <input ref={fileInput} type="file" accept=".csv,.parquet,.pq,.txt" hidden
        onChange={(e) => { void upload.start(e.target.files?.[0]).then((ds) => ds && uploaded(ds)); e.target.value = ""; }} />
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Datasets</h1>
        <div className="row">
          <SampleMenu onCreated={(ds) => highlight(ds.id)} />
          {rows.length > 0 && (
            <button type="button" className="primary" onClick={browse} disabled={upload.progress !== null}>↑ Upload CSV / Parquet</button>
          )}
        </div>
      </div>
      <ErrorAlert error={error} />
      {isLoading && <div className="stack"><span className="skeleton" /><span className="skeleton" /></div>}
      {data && rows.length === 0 && !upload.progress && <FirstRun onBrowse={browse} onDemo={() => demo.mutate()} busy={demo.isPending} />}
      <ErrorAlert error={demo.error} />

      {(rows.length > 0 || upload.progress) && (
        <div className="table-wrap" style={{ maxHeight: "none", overflow: "visible" }}>
          <table className="data datasets">
            <thead>
              <tr>
                <th>Name</th><th>Status</th><th className="num">Rows</th><th className="num">Fields</th><th>Targets</th>
                <th>Review</th><th>Last export</th><th>Updated</th><th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {upload.progress && (
                <tr className="fresh">
                  <td><b>{upload.progress.name}</b></td>
                  <td colSpan={8}>
                    <div className="row" style={{ flexWrap: "nowrap" }}>
                      <span className="badge accent"><span className="spinner" aria-hidden="true" /> uploading {fmtPct(upload.progress.frac, 0)}</span>
                      <div className="meter" style={{ width: 200 }}><span style={{ width: `${upload.progress.frac * 100}%` }} /></div>
                    </div>
                  </td>
                </tr>
              )}
              {rows.map((ds) => {
                const ready = ds.status === "ready";
                // An assessor of a later version only assesses the reopened fields.
                const total = (ds.my_assessment && ds.reopened != null ? ds.reopened : ds.field_count) ?? ds.column_count ?? 0;
                // An assessor sees their own progress; everyone else the team decision's.
                const reviewed = ds.my_assessment?.reviewed ?? ds.reviewed_count ?? 0;
                return (
                  <Fragment key={ds.id}>
                    <tr className={`${ready ? "clickable" : ""}${fresh === ds.id ? " fresh" : ""}`}
                      onClick={() => ready && renaming !== ds.id && open(ds)}
                      onKeyDown={(e) => e.key === "Enter" && e.target === e.currentTarget && ready && open(ds)}
                      tabIndex={ready ? 0 : -1} aria-label={ready ? `Open ${ds.name}` : undefined}>
                      <td className="name-cell">
                        {renaming === ds.id ? (
                          <input defaultValue={ds.name} autoFocus aria-label="Dataset name" onClick={(e) => e.stopPropagation()}
                            onKeyDown={(e) => {
                              e.stopPropagation();
                              if (e.key === "Enter") rename.mutate({ id: ds.id, name: e.currentTarget.value.trim() || ds.name });
                              if (e.key === "Escape") setRenaming(null);
                            }}
                            onBlur={(e) => e.currentTarget.value.trim() !== ds.name ? rename.mutate({ id: ds.id, name: e.currentTarget.value.trim() || ds.name }) : setRenaming(null)} />
                        ) : (
                          <b title={ds.name}>{ds.name}</b>
                        )}
                      </td>
                      <td>
                        <div className="row" style={{ flexWrap: "nowrap" }}>
                          <StatusBadge ds={ds} />
                          {ready && <PhaseBadge phase={ds.phase} label={ds.baseline ? baselineLabel(ds.baseline.family, ds.baseline.version, true) : null} />}
                          {ds.my_assessment?.submitted && <span className="badge good" title="You submitted your assessment">✓ submitted</span>}
                          {ds.warnings.length > 0 && (
                            <span className="badge warn" title={ds.warnings.join("\n")}>⚠ {ds.warnings.length}</span>
                          )}
                        </div>
                        {ds.status === "failed" && ds.error && (
                          <div className="small-text" style={{ color: "var(--bad)", marginTop: 4, maxWidth: 360 }}>
                            {ds.error.split("\n")[0]}{" "}
                            <button type="button" className="link small-text" onClick={(e) => { e.stopPropagation(); setDetails(details === ds.id ? null : ds.id); }}>
                              {details === ds.id ? "Hide details" : "Details"}
                            </button>
                          </div>
                        )}
                      </td>
                      <td className="num">{fmtInt(ds.row_count)}</td>
                      <td className="num">{fmtInt(ds.column_count)}</td>
                      <td style={{ position: "relative" }}>
                        {ready && (ds.target_columns.length ? (
                          <button type="button" className="ghost target-chips" title="Change targets"
                            onClick={(e) => { e.stopPropagation(); setTargetFor(ds.id); }}>
                            {ds.target_columns.map((t) => <code key={t}>{t}</code>)}
                          </button>
                        ) : (
                          <button type="button" className="chip attention on" onClick={(e) => { e.stopPropagation(); setTargetFor(ds.id); }}>Choose target ▾</button>
                        ))}
                        {targetFor === ds.id && (
                          <div onClick={(e) => e.stopPropagation()}><TargetPicker ds={ds} onClose={() => setTargetFor(null)} /></div>
                        )}
                      </td>
                      <td>
                        {ready && total > 0 && (
                          <div className="stack" style={{ gap: 3 }} title={`${ds.kept_count ?? 0} kept · ${ds.dropped_count ?? 0} dropped · ${ds.target_count ?? 0} targets`}>
                            <span className="num small-text">{reviewed === total ? "✓ " : ""}{fmtInt(reviewed)} / {fmtInt(total)}</span>
                            <span className="meter" style={{ width: 80, height: 4 }}><span style={{ width: `${(100 * reviewed) / total}%`, background: "var(--good)" }} /></span>
                          </div>
                        )}
                      </td>
                      <td className="small-text">
                        {ds.last_export ? (
                          <span title={`Export #${ds.last_export.id} · ${ds.last_export.status}`}>
                            v{ds.last_export.id}
                            {ds.last_run?.status === "done" && ds.last_run.value !== null && ` · ${ds.last_run.model_label ?? ds.last_run.model} ${fmtNum(ds.last_run.value, 2)} ${ds.last_run.metric?.startsWith("roc_auc") ? "AUC" : ds.last_run.metric}`}
                          </span>
                        ) : <span className="muted">—</span>}
                      </td>
                      <td className="small-text muted" title={fmtDate(ds.updated_at)}>{fmtAgo(ds.updated_at)}</td>
                      <td onClick={(e) => e.stopPropagation()}>
                        <div className="row" style={{ flexWrap: "nowrap", gap: 2 }}>
                          <RowMenu ds={ds} canDelete={me.data?.role === "lead"} onRename={() => setRenaming(ds.id)} onDelete={() => setDeleting(ds)} />
                          {ready && <span className="chevron" aria-hidden="true">→</span>}
                        </div>
                      </td>
                    </tr>
                    {details === ds.id && ds.error && (
                      <tr><td colSpan={9}><pre className="alert bad" style={{ margin: 0, whiteSpace: "pre-wrap" }}>{ds.error}</pre></td></tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {dragging && (
        <div className="drop-overlay" aria-hidden="true"><span>Drop to upload: CSV or Parquet</span></div>
      )}
      {deleting && <DeleteDialog ds={deleting} onClose={() => setDeleting(null)} />}
    </div>
  );
}
