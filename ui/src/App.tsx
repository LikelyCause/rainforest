import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Navigate, Outlet, Route, Routes, useMatch, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAccess, useDataset, useExperiment, useFields, useMe } from "./hooks";
import { ApiError, api, setUnauthorizedHandler } from "./api";
import { can } from "./can";
import { Avatar } from "./components/Avatar";
import { Logo } from "./components/Logo";
import { PictureDialog } from "./components/PictureDialog";
import { DatasetsPage } from "./pages/DatasetsPage";
import { LoginPage } from "./pages/LoginPage";
import { ForbiddenPage } from "./pages/ForbiddenPage";
import { TeamPage } from "./pages/TeamPage";
import { BaselinesPage } from "./pages/BaselinesPage";
import { ReviewPage } from "./pages/ReviewPage";
import { BulkEditPage } from "./pages/BulkEditPage";
import { ExperimentPage, ExperimentsPage } from "./pages/ExperimentsPage";
import { ExplorePage } from "./pages/ExplorePage";
import { DatasetTeamPage, PhaseBadge } from "./pages/DatasetTeamPage";
import { DatasetLayout } from "./components/DatasetLayout";
import { ErrorAlert, ReviewMeter } from "./components/ui";
import { baselineLabel, roleLabel } from "./format";
import { ToastRegion, showError } from "./toast";
import { parseView, useView, withView } from "./view";
import type { Dataset, ExportJob, ModelRun } from "./types";

/** The baseline the dataset is working on (or last worked), linking to all of them. Replaces choosing a model type:
 * that is fixed per baseline. */
/** "Tries · Dropout risk v1", or just the name when it already says which baseline ("Team · Dropout risk v1"). */
const withBaseline = (name: string, label: string) => (name.includes(label) ? name : `${name} · ${label}`);

function BaselineChip({ ds }: { ds: Dataset }) {
  const v = parseView(useView());
  const x = useExperiment(v?.kind === "experiment" ? v.id : null);
  const b = ds.baseline;
  if (v?.kind === "experiment" && x.data) {
    return (
      <Link to={`/d/${ds.id}/experiments/${x.data.id}`} className="baseline-chip" title={`Adjusting ${x.data.name}, built from ${baselineLabel(x.data.recipe_family, x.data.recipe_version, false, x.data.recipe_name)}`}>
        <span className="muted">Experiment</span> <b>{withBaseline(x.data.name, baselineLabel(x.data.recipe_family, x.data.recipe_version, true, x.data.recipe_name))}</b>
      </Link>
    );
  }
  if (!b) return null;
  return (
    <Link to={`/d/${ds.id}/baselines`} className="baseline-chip baseline" title={b.setup ? "No baseline yet: a lead sets one up on the Team page. All baselines →" : `${b.label}. All baselines →`}>
      <span className="muted">Baseline</span> <b>{b.setup ? "none yet" : baselineLabel(b.family, b.version, true, b.name)}</b>
    </Link>
  );
}

function DatasetSteps({ ds }: { ds: Dataset }) {
  const nav = useNavigate();
  const onReview = useMatch("/d/:datasetId/review/*");
  const onExperiments = useMatch("/d/:datasetId/experiments/*");
  const onTeam = useMatch("/d/:datasetId/team");
  const view = useView();
  const access = useAccess(ds, view);
  const me = useMe();
  const fields = useFields(ds.id, true, view);
  const busy = (q: { state: { data?: { status: string }[] } }) =>
    (q.state.data ?? []).some((j) => j.status === "pending" || j.status === "running") ? 3000 : false;
  const exports = useQuery({ queryKey: ["exports", ds.id], queryFn: () => api.exports(ds.id), refetchInterval: busy });
  const runs = useQuery({ queryKey: ["runs", ds.id], queryFn: () => api.runs(ds.id), refetchInterval: busy });
  const all = fields.data ?? [];
  const reviewed = all.filter((f) => f.reviewed);
  const kept = reviewed.filter((f) => f.action === "keep").length;
  const dropped = reviewed.filter((f) => f.action === "drop").length;
  const targets = reviewed.filter((f) => f.action === "target").length;
  const running = [...(exports.data ?? []), ...(runs.data ?? [])]
    .filter((j: ExportJob | ModelRun) => j.status === "pending" || j.status === "running").length;
  const nextTodo = access.canEdit && !view ? all.find((f) => !f.reviewed) : undefined;
  const step = ({ isActive }: { isActive: boolean }) => `step${isActive ? " active" : ""}`;

  // U jumps back into review from Bulk edit, Explore and Experiments, matching its meaning on the Review page.
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
        <NavLink to={withView(`/d/${ds.id}/review`, view)} className={step} title={`${kept} kept · ${dropped} dropped · ${targets} targets · ${all.length - reviewed.length} to do`}>
          <span>Review<span className="count num">{reviewed.length}/{all.length}</span></span>
          <ReviewMeter kept={kept} dropped={dropped} targets={targets} total={all.length} />
        </NavLink>
        <NavLink to={withView(`/d/${ds.id}/bulk`, view)} className={step}><span>Bulk edit</span></NavLink>
        <NavLink to={`/d/${ds.id}/explore`} className={step}><span>Explore</span></NavLink>
        <NavLink to={`/d/${ds.id}/team`} className={step} title="Assessors, team review and finalizing">
          <span>Team <PhaseBadge phase={ds.phase} /></span>
        </NavLink>
        <NavLink to={`/d/${ds.id}/experiments`} className={step} title="Exports and model runs, grouped into experiments">
          <span>Experiments{running > 0 && <span className="running-dot" role="status" aria-label={`${running} in progress`} title={`${running} export${running === 1 ? " or run" : "s or runs"} in progress`} />}</span>
        </NavLink>
      </nav>
      <div className="spacer" />
      <BaselineChip ds={ds} />
      {!onReview && !view && all.length > 0 && (nextTodo ? (
        <Link className="btn" to={`/d/${ds.id}/review/${nextTodo.id}`}>Next unreviewed › <kbd>U</kbd></Link>
      ) : ds.phase === "finalized" ? !onExperiments && (
        <Link className="btn" to={`/d/${ds.id}/experiments`} style={{ background: "var(--good)", borderColor: "var(--good)", color: "#fff" }}>
          {can(me.data, "createExperiment") ? "Finalized → Start experimenting" : "Finalized → See experiments"}
        </Link>
      ) : !onTeam && access.source === "mine" && access.canEdit ? (
        <Link className="btn" to={`/d/${ds.id}/team`} style={{ background: "var(--good)", borderColor: "var(--good)", color: "#fff" }}>All reviewed → Submit</Link>
      ) : null)}
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
  const me = useMe();
  const [open, setOpen] = useState(false);
  const [picture, setPicture] = useState(false);
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  const out = useMutation({
    mutationFn: api.logout,
    onSuccess: ({ logout_url }) => {
      qc.clear();
      // OIDC: end the provider's session too, or the next sign-in would skip its login page.
      if (logout_url) window.location.assign(logout_url);
      else nav("/login", { replace: true });
    },
    onError: (err) => showError("Couldn't sign out", err),
  });
  const u = me.data;
  if (!u) return <div className="user-menu" />;
  const pick = u.mode === "pick";
  return (
    <div ref={ref} className="user-menu" style={{ position: "relative" }} onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <NavLink to="/team" className="team-link">Team</NavLink>
      <button type="button" className="ghost profile-button" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}
        title={u.email ?? undefined}>
        <Avatar user={u} size={28} />
        <span className="user-name">{u.name}</span>
        <span className={`badge role-${u.role}`}>{roleLabel(u.role)}</span>
      </button>
      {open && (
        <div className="popover profile-menu" role="menu">
          <div className="profile-menu-head">
            <Avatar user={u} size={44} />
            <div className="stack" style={{ gap: 2 }}>
              <b>{u.name}</b>
              <span className="muted small-text">{roleLabel(u.role)}</span>
            </div>
          </div>
          <button type="button" role="menuitem" className="ghost menu-item" onClick={() => { setOpen(false); setPicture(true); }}>Change picture…</button>
          <NavLink role="menuitem" className="menu-item" to="/team" onClick={() => setOpen(false)}>Team</NavLink>
          {system.data?.mlflow && (
            <a role="menuitem" className="menu-item" href="/mlflow/" target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>MLflow ↗</a>
          )}
          <button type="button" role="menuitem" className="ghost menu-item" disabled={out.isPending} onClick={() => out.mutate()}>
            {pick ? "Switch profile" : "Sign out"}
          </button>
        </div>
      )}
      {picture && <PictureDialog user={u} onClose={() => setPicture(false)} />}
    </div>
  );
}

/** Renders the app once the session checks out; a 401 has already sent the user to /login. */
function RequireAuth() {
  const me = useMe();
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
        <Route path="/forbidden" element={<main className="main"><ForbiddenPage /></main>} />
        <Route element={<RequireAuth />}>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/team" element={<TeamPage />} />
          <Route path="/d/:datasetId" element={<DatasetLayout />}>
            <Route index element={<Navigate to="review" replace />} />
            <Route path="review/:fieldId?" element={<ReviewPage />} />
            <Route path="bulk" element={<BulkEditPage />} />
            <Route path="query" element={<Navigate to="../bulk" replace />} />
            <Route path="explore/:view?" element={<ExplorePage />} />
            <Route path="export" element={<Navigate to="../experiments" replace />} />
            <Route path="experiments" element={<ExperimentsPage />} />
            <Route path="experiments/:experimentId" element={<ExperimentPage />} />
            <Route path="team" element={<DatasetTeamPage />} />
            <Route path="baselines" element={<BaselinesPage />} />
            <Route path="versions" element={<Navigate to="../baselines" replace />} />
          </Route>
          <Route path="*" element={<div className="empty">Page not found. <NavLink to="/">Go home</NavLink></div>} />
        </Route>
      </Routes>
      <ToastRegion />
    </div>
  );
}
