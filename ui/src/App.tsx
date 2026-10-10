import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Navigate, Outlet, Route, Routes, useMatch, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAccess, useDataset, useFields, useMe } from "./hooks";
import { ApiError, api, setUnauthorizedHandler } from "./api";
import { Logo } from "./components/Logo";
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
import { baselineLabel, familyShort, roleLabel } from "./format";
import { ToastRegion, showError } from "./toast";
import type { Dataset, ExportJob, ModelRun } from "./types";

/** The baseline the dataset is working on (or last worked), linking to all of them. Replaces choosing a model type:
 * that is fixed per baseline. */
function BaselineChip({ ds }: { ds: Dataset }) {
  const b = ds.baseline;
  if (!b) return null;
  return (
    <Link to={`/d/${ds.id}/baselines`} className="baseline-chip" title={`${b.label}${b.setup ? " · not set up yet" : ""}. All baselines →`}>
      <span className="muted">Baseline</span> <b>{b.setup ? `${familyShort(b.family)} · set up` : baselineLabel(b.family, b.version, true)}</b>
    </Link>
  );
}

function DatasetSteps({ ds }: { ds: Dataset }) {
  const nav = useNavigate();
  const onReview = useMatch("/d/:datasetId/review/*");
  const onExperiments = useMatch("/d/:datasetId/experiments/*");
  const onTeam = useMatch("/d/:datasetId/team");
  const access = useAccess(ds);
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
  const nextTodo = access.canEdit ? all.find((f) => !f.reviewed) : undefined;
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
        <NavLink to={`/d/${ds.id}/review`} className={step} title={`${kept} kept · ${dropped} dropped · ${targets} targets · ${all.length - reviewed.length} to do`}>
          <span><span className="n">1</span>Review<span className="count num">{reviewed.length}/{all.length}</span></span>
          <ReviewMeter kept={kept} dropped={dropped} targets={targets} total={all.length} />
        </NavLink>
        <NavLink to={`/d/${ds.id}/bulk`} className={step}><span>Bulk edit</span></NavLink>
        <NavLink to={`/d/${ds.id}/explore`} className={step}><span>Explore</span></NavLink>
        <NavLink to={`/d/${ds.id}/team`} className={step} title="Assessors, team review and finalizing">
          <span>Team <PhaseBadge phase={ds.phase} /></span>
        </NavLink>
        <NavLink to={`/d/${ds.id}/experiments`} className={step} title="Exports and model runs, grouped into experiments">
          <span><span className="n">2</span>Experiments{running > 0 && <span className="badge accent" style={{ marginLeft: 6 }} title={`${running} running`}>{running}</span>}</span>
        </NavLink>
      </nav>
      <div className="spacer" />
      <BaselineChip ds={ds} />
      {!onReview && all.length > 0 && (nextTodo ? (
        <Link className="btn" to={`/d/${ds.id}/review/${nextTodo.id}`}>Next unreviewed › <kbd>U</kbd></Link>
      ) : ds.phase === "finalized" ? !onExperiments && (
        <Link className="btn" to={`/d/${ds.id}/experiments`} style={{ background: "var(--good)", borderColor: "var(--good)", color: "#fff" }}>All reviewed → Experiments</Link>
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
  return (
    <div className="user-menu">
      {u && (
        <>
          <NavLink to="/team" className="team-link">Team</NavLink>
          <span className="user-name" title={u.email ?? undefined}>{u.name}</span>
          <span className={`badge${u.role === "lead" ? " accent" : ""}`}>{roleLabel(u.role)}</span>
        </>
      )}
      <button type="button" className="ghost small" disabled={out.isPending} onClick={() => out.mutate()}>Sign out</button>
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
