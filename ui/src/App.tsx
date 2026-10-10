import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Navigate, Outlet, Route, Routes, useMatch, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useDataset, useFields, useOptions } from "./hooks";
import { ApiError, api, setUnauthorizedHandler } from "./api";
import { Logo } from "./components/Logo";
import { DatasetsPage } from "./pages/DatasetsPage";
import { LoginPage } from "./pages/LoginPage";
import { ReviewPage } from "./pages/ReviewPage";
import { BulkEditPage } from "./pages/BulkEditPage";
import { ExportPage } from "./pages/ExportPage";
import { ExplorePage } from "./pages/ExplorePage";
import { DatasetLayout } from "./components/DatasetLayout";
import { ErrorAlert, ReviewMeter } from "./components/ui";
import { familyShort } from "./format";
import { ToastRegion, showError, toast } from "./toast";
import type { Dataset, ExportJob, ModelRun } from "./types";

function ModelSwitch({ ds }: { ds: Dataset }) {
  const qc = useQueryClient();
  const { data: options } = useOptions();
  const m = useMutation({
    mutationFn: (f: string) => api.patchDataset(ds.id, { model_family: f }),
    onSuccess: (saved, f, ctx) => {
      qc.setQueryData(["dataset", ds.id], saved);
      for (const k of ["fieldQuery", "exportPreview"]) qc.invalidateQueries({ queryKey: [k, ds.id] });
      const from = (ctx as { from?: string } | undefined)?.from;
      toast({
        message: `Now preparing for ${familyShort(f).toLowerCase()} models`,
        detail: "Keep / drop, types and notes are kept; settings switch to this model's.",
        duration: 6000,
        action: from ? { label: "Undo", run: () => m.mutate(from) } : undefined,
      });
    },
    onMutate: () => ({ from: ds.model_family }),
    onError: (err, f) => showError(`Couldn't switch to ${familyShort(f)}`, err),
  });
  if (!options) return null;
  return (
    <div className="model-switch">
      <span id="model-label">Model</span>
      <div className="segmented soft" role="radiogroup" aria-labelledby="model-label">
        {options.families.map((f) => {
          const on = (m.isPending ? m.variables : ds.model_family) === f.value;
          return (
            <button key={f.value} type="button" role="radio" aria-checked={on} className={on ? "on" : ""}
              title={`${f.label}: ${f.examples}`} disabled={m.isPending}
              onClick={() => !on && m.mutate(f.value)}>
              {familyShort(f.value)}{m.isPending && on ? " …" : ""}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function DatasetSteps({ ds }: { ds: Dataset }) {
  const nav = useNavigate();
  const onReview = useMatch("/d/:datasetId/review/*");
  const onExport = useMatch("/d/:datasetId/export");
  const fields = useFields(ds.id);
  const exports = useQuery({ queryKey: ["exports", ds.id], queryFn: () => api.exports(ds.id) });
  const runs = useQuery({ queryKey: ["runs", ds.id], queryFn: () => api.runs(ds.id) });
  const all = fields.data ?? [];
  const reviewed = all.filter((f) => f.reviewed);
  const kept = reviewed.filter((f) => f.action === "keep").length;
  const dropped = reviewed.filter((f) => f.action === "drop").length;
  const targets = reviewed.filter((f) => f.action === "target").length;
  const running = [...(exports.data ?? []), ...(runs.data ?? [])]
    .filter((j: ExportJob | ModelRun) => j.status === "pending" || j.status === "running").length;
  const nextTodo = all.find((f) => !f.reviewed);
  const step = ({ isActive }: { isActive: boolean }) => `step${isActive ? " active" : ""}`;

  // U jumps back into review from Bulk edit, Explore and Export, matching its meaning on the Review page.
  useEffect(() => {
    if (onReview) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.key !== "u" || e.metaKey || e.ctrlKey || e.altKey || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return;
      if (nextTodo) nav(`/d/${ds.id}/review/${nextTodo.id}`);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onReview, nextTodo, nav, ds.id]);

  return (
    <>
      <nav className="steps" aria-label="Workflow">
        <NavLink to={`/d/${ds.id}/review`} className={step} title={`${kept} kept · ${dropped} dropped · ${targets} targets · ${all.length - reviewed.length} to do`}>
          <span><span className="n">1</span>Review<span className="count num">{reviewed.length}/{all.length}</span></span>
          <ReviewMeter kept={kept} dropped={dropped} targets={targets} total={all.length} />
        </NavLink>
        <NavLink to={`/d/${ds.id}/bulk`} className={step}><span>Bulk edit</span></NavLink>
        <NavLink to={`/d/${ds.id}/explore`} className={step}><span>Explore</span></NavLink>
        <NavLink to={`/d/${ds.id}/export`} className={step}>
          <span><span className="n">2</span>Export &amp; train{running > 0 && <span className="badge accent" style={{ marginLeft: 6 }} title={`${running} running`}>{running}</span>}</span>
        </NavLink>
      </nav>
      <div className="spacer" />
      <ModelSwitch ds={ds} />
      {!onReview && all.length > 0 && (nextTodo ? (
        <Link className="btn" to={`/d/${ds.id}/review/${nextTodo.id}`}>Next unreviewed › <kbd>U</kbd></Link>
      ) : !onExport && (
        <Link className="btn" to={`/d/${ds.id}/export`} style={{ background: "var(--good)", borderColor: "var(--good)", color: "#fff" }}>All reviewed → Export</Link>
      ))}
    </>
  );
}

function DatasetSwitcher({ ds }: { ds: Dataset }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const list = useQuery({ queryKey: ["datasets"], queryFn: api.datasets, enabled: open });
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div ref={ref} style={{ position: "relative" }} onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <button type="button" className="ghost" aria-expanded={open} onClick={() => setOpen(!open)} title={ds.name}
        style={{ padding: "0 6px", color: "var(--text-primary)" }}>
        <span className="ds-name">{ds.name}</span> ▾
      </button>
      {open && (
        <div className="popover" style={{ left: 0, right: "auto", minWidth: 260 }}>
          {list.data?.map((d) => (
            <Link key={d.id} to={`/d/${d.id}/review`} className="btn ghost" style={{ justifyContent: "flex-start", border: 0 }}
              onClick={() => setOpen(false)}>
              {d.id === ds.id ? "✓ " : ""}{d.name}
            </Link>
          ))}
          <Link to="/" className="btn ghost" style={{ justifyContent: "flex-start", border: 0 }} onClick={() => setOpen(false)}>All datasets…</Link>
        </div>
      )}
    </div>
  );
}

function TopBar() {
  const m = useMatch("/d/:datasetId/*");
  const datasetId = m ? Number(m.params.datasetId) : NaN;
  const ds = useDataset(datasetId, !Number.isNaN(datasetId));
  const d = ds.data;
  return (
    <header className="topbar">
      <NavLink to="/" className="brand"><Logo /> Rainforest</NavLink>
      {d && (
        <>
          <span className="muted">/</span>
          <DatasetSwitcher ds={d} />
          {d.status === "ready" && <DatasetSteps ds={d} />}
        </>
      )}
      <UserMenu />
    </header>
  );
}

function UserMenu() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, staleTime: Infinity });
  const out = useMutation({
    mutationFn: api.logout,
    onSuccess: () => {
      qc.clear();
      nav("/login", { replace: true });
    },
    onError: (err) => showError("Couldn't sign out", err),
  });
  return (
    <div className="user-menu">
      {me.data && <span className="muted small-text">{me.data.user}</span>}
      <button type="button" className="ghost small" disabled={out.isPending} onClick={() => out.mutate()}>Sign out</button>
    </div>
  );
}

/** Renders the app once the session checks out; a 401 has already sent the user to /login. */
function RequireAuth() {
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, staleTime: Infinity, retry: false });
  if (me.isPending) return null;
  if (me.error) {
    if (me.error instanceof ApiError && me.error.status === 401) return null;
    return (
      <div className="login">
        <div className="card login-card stack">
          <div className="login-brand"><Logo /> Rainforest</div>
          <ErrorAlert error={me.error} title={me.error instanceof ApiError && me.error.status === 503 ? "Not set up yet." : "Can't reach the API."} />
        </div>
      </div>
    );
  }
  return (
    <>
      <TopBar />
      <main className="main">
        <Outlet />
      </main>
    </>
  );
}

export default function App() {
  const nav = useNavigate();
  useEffect(() => {
    setUnauthorizedHandler((next) => nav(`/login?next=${encodeURIComponent(next)}`, { replace: true }));
  }, [nav]);
  return (
    <div className="app">
      <Routes>
        <Route path="/login" element={<main className="main"><LoginPage /></main>} />
        <Route element={<RequireAuth />}>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/d/:datasetId" element={<DatasetLayout />}>
            <Route index element={<Navigate to="review" replace />} />
            <Route path="review/:fieldId?" element={<ReviewPage />} />
            <Route path="bulk" element={<BulkEditPage />} />
            <Route path="query" element={<Navigate to="../bulk" replace />} />
            <Route path="explore/:view?" element={<ExplorePage />} />
            <Route path="export" element={<ExportPage />} />
          </Route>
          <Route path="*" element={<div className="empty">Page not found. <NavLink to="/">Go home</NavLink></div>} />
        </Route>
      </Routes>
      <ToastRegion />
    </div>
  );
}
