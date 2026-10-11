import { useCallback, useEffect, useRef, useState } from "react";
import { Link, NavLink, Navigate, Outlet, Route, Routes, useLocation, useMatch, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAccess, useDataset, useExperiment, useFields, useInvalidateTeam, useMe, useTeam } from "./hooks";
import { ApiError, api, setUnauthorizedHandler } from "./api";
import { can } from "./can";
import { Avatar } from "./components/Avatar";
import { Logo } from "./components/Logo";
import { PictureDialog } from "./components/PictureDialog";
import { ConfirmRegion, Icon, LensChip, PhaseRail, Tabs, confirmDialog, stageOf, useDismiss, type Lens, type Stage } from "./components/kit";
import { ExperimentBar } from "./components/ExperimentBar";
import { DatasetsPage } from "./pages/DatasetsPage";
import { LoginPage } from "./pages/LoginPage";
import { ForbiddenPage } from "./pages/ForbiddenPage";
import { TeamPage } from "./pages/TeamPage";
import { SettingsPage } from "./pages/SettingsPage";
import { BaselinesPage } from "./pages/BaselinesPage";
import { BaselinePage } from "./pages/BaselinePage";
import { ReviewPage } from "./pages/ReviewPage";
import { BulkEditPage } from "./pages/BulkEditPage";
import { ExperimentPage, ExperimentsPage } from "./pages/ExperimentsPage";
import { ExplorePage } from "./pages/ExplorePage";
import { DatasetTeamPage } from "./pages/DatasetTeamPage";
import { DatasetLayout, useDatasetContext } from "./components/DatasetLayout";
import { ErrorAlert } from "./components/ui";
import { baselineLabel, familyShort, fmtInt, roleLabel } from "./format";
import { ToastRegion, showError, toast } from "./toast";
import { parseView, useView, withView } from "./view";
import type { Dataset, ExportJob, ModelRun, TeamState } from "./types";
import type { Access } from "./team";

/* ------------------------------------------------------------------ first row: where you are */

/** Fields, Explore, Baseline, Experiments: the four places inside a dataset. */
function DatasetTabs({ ds }: { ds: Dataset }) {
  const nav = useNavigate();
  const view = useView();
  const access = useAccess(ds, view);
  const onReview = useMatch("/d/:datasetId/review/*");
  const onBulk = useMatch("/d/:datasetId/bulk");
  const onQueue = useMatch("/d/:datasetId/team");
  const onBaseline = useMatch("/d/:datasetId/baseline/*");
  const onVersions = useMatch("/d/:datasetId/baselines");
  const fields = useFields(ds.id, true, view);
  const team = useTeam(ds.id, ds.phase === "review");
  const busy = (q: { state: { data?: { status: string }[] } }) =>
    (q.state.data ?? []).some((j) => j.status === "pending" || j.status === "running") ? 3000 : false;
  const exports = useQuery({ queryKey: ["exports", ds.id], queryFn: () => api.exports(ds.id), refetchInterval: busy });
  const runs = useQuery({ queryKey: ["runs", ds.id], queryFn: () => api.runs(ds.id), refetchInterval: busy });
  const all = fields.data ?? [];
  const reviewed = all.filter((f) => f.reviewed);
  const running = [...(exports.data ?? []), ...(runs.data ?? [])]
    .filter((j: ExportJob | ModelRun) => j.status === "pending" || j.status === "running").length;
  const nextTodo = access.canEdit && !view ? all.find((f) => !f.reviewed) : undefined;
  const inReview = ds.phase === "review" && !view;
  const open = team.data?.review?.open;

  // U jumps back into Focus from the dataset's other pages, matching its meaning in Focus.
  useEffect(() => {
    if (onReview) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.key !== "u" || e.metaKey || e.ctrlKey || e.altKey || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable) return;
      if (nextTodo) nav(`/d/${ds.id}/review/${nextTodo.id}`);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onReview, nextTodo, nav, ds.id]);

  return (
    <Tabs label="Dataset" className="dataset-tabs" items={[
      {
        to: inReview ? `/d/${ds.id}/team` : withView(`/d/${ds.id}/review`, view), label: "Fields",
        active: Boolean(onReview || onBulk || onQueue),
        count: inReview ? (open != null ? `${open} open` : undefined) : all.length ? `${reviewed.length}/${all.length}` : undefined,
        title: inReview ? "The team review queue, then every field" : `${reviewed.length} of ${all.length} reviewed`,
      },
      { to: `/d/${ds.id}/explore`, label: "Explore" },
      { to: `/d/${ds.id}/baseline`, label: "Baseline", active: Boolean(onBaseline || onVersions), title: "Progress, assessors, versions" },
      {
        to: `/d/${ds.id}/experiments`,
        label: <>Experiments{running > 0 && <span className="running-dot" role="status" aria-label={`${running} in progress`}
          title={`${running} export${running === 1 ? " or run" : "s or runs"} in progress`} />}</>,
      },
    ]} />
  );
}

function DatasetSwitcher({ ds }: { ds: Dataset }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const list = useQuery({ queryKey: ["datasets"], queryFn: api.datasets, enabled: open });
  return (
    <div ref={ref} style={{ position: "relative", minWidth: 0 }}>
      <button type="button" className="ghost switcher" aria-expanded={open} aria-haspopup="true" onClick={() => setOpen(!open)} title={ds.name}>
        <span className="ds-name">{ds.name}</span><Icon name="chevron-down" />
      </button>
      {open && (
        <div className="popover" style={{ left: 0, right: "auto", minWidth: 260 }}>
          {list.data?.map((d) => (
            <Link key={d.id} to={`/d/${d.id}`} className="menu-link" aria-current={d.id === ds.id ? "true" : undefined} onClick={close}>
              <span className="grow">{d.name}</span>{d.id === ds.id && <Icon name="check" />}
            </Link>
          ))}
          <Link to="/" className="menu-link" onClick={close}>All datasets</Link>
        </div>
      )}
    </div>
  );
}

function TopBar() {
  const m = useMatch("/d/:datasetId/*");
  const datasetId = m ? Number(m.params.datasetId) : NaN;
  const ds = useDataset(datasetId, !Number.isNaN(datasetId));
  const d = m ? ds.data : undefined;
  const onHome = useMatch("/");
  return (
    <header className={`topbar${d ? " in-dataset" : ""}`}>
      <div className="topbar-row">
        {d ? (
          <Link to="/" className="brand" aria-label="Rainforest: all datasets"><Logo /><span className="brand-name">Rainforest</span></Link>
        ) : (
          <NavLink to="/" className="brand"><Logo /> Rainforest</NavLink>
        )}
        {d ? (
          <>
            <span className="crumb-sep" aria-hidden="true">/</span>
            <DatasetSwitcher ds={d} />
            {d.status === "ready" && <DatasetTabs ds={d} />}
          </>
        ) : (
          <Tabs label="Main" className="main-tabs" items={[{ to: "/", label: "Datasets", end: true, active: Boolean(onHome) }]} />
        )}
        <UserMenu compact={Boolean(d)} />
      </div>
      {d?.status === "ready" && <DatasetBar ds={d} />}
    </header>
  );
}

function UserMenu({ compact }: { compact?: boolean }) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const me = useMe();
  const [open, setOpen] = useState(false);
  const [picture, setPicture] = useState(false);
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
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
    <div ref={ref} className="user-menu" style={{ position: "relative" }}>
      <button type="button" className="ghost profile-button" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}
        aria-label={`Profile menu: ${u.name}, ${roleLabel(u.role)}`} title={u.email ?? undefined}>
        <Avatar user={u} size={28} />
        {!compact && <span className="user-name">{u.name}</span>}
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
          <button type="button" role="menuitem" className="ghost menu-item" onClick={() => { close(); setPicture(true); }}>Change picture…</button>
          <NavLink role="menuitem" className="menu-item" to="/settings" onClick={close}>Settings</NavLink>
          <NavLink role="menuitem" className="menu-item" to="/people" onClick={close}>People</NavLink>
          {system.data?.mlflow && (
            <a role="menuitem" className="menu-item" href="/mlflow/" target="_blank" rel="noopener noreferrer" onClick={close}>
              MLflow <Icon name="external" size={14} />
            </a>
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

/* ------------------------------------------------------------------ second row: where the version stands */

/** Every baseline and version of the dataset; replaces the Baseline chip. */
function VersionSwitcher({ ds, stage }: { ds: Dataset; stage: Stage }) {
  const me = useMe().data;
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const b = ds.baseline;
  if (!b) return null;
  const name = stage === "setup" ? "No baseline yet" : baselineLabel(b.family, b.version, true, b.name);
  return (
    <div ref={ref} style={{ position: "relative", minWidth: 0 }}>
      <button type="button" className="ghost switcher version-switcher" aria-expanded={open} aria-haspopup="true" onClick={() => setOpen(!open)}
        title={stage === "setup" ? "This dataset's first baseline isn't set up yet" : b.label}>
        <span className="ds-name">{name}</span>
        {stage !== "setup" && <span className="badge">{familyShort(b.family)}</span>}
        <Icon name="chevron-down" />
      </button>
      {open && (
        <div className="popover" style={{ left: 0, right: "auto", minWidth: 280 }}>
          {ds.baselines.map((l) => (
            <Link key={l.family} to={`/d/${ds.id}/baselines`} className="menu-link" onClick={close}>
              <span className="grow">{baselineLabel(l.family, l.version, true, l.name)} <span className="muted">· {familyShort(l.family)}</span></span>
              <span className="muted small-text">{l.phase === "finalized" ? "Finalized" : l.phase === "review" ? "Team review" : "Assessing"}</span>
            </Link>
          ))}
          <hr />
          <Link to={`/d/${ds.id}/baselines`} className="menu-link" onClick={close}>Versions and compare</Link>
          {can(me, "startBaseline") && <Link to={`/d/${ds.id}/baselines`} className="menu-link" onClick={close}>New version or baseline…</Link>}
        </div>
      )}
    </div>
  );
}

function lensOf(stage: Stage, access: Access, me: { role: string } | undefined, team: TeamState | undefined): { lens: Lens; text: string } {
  if (access.source === "mine") return { lens: "private", text: team?.me?.submitted ? "Private · submitted" : "Private · your assessment" };
  if (me?.role === "viewer") return stage === "finalized" ? { lens: "frozen", text: "Frozen snapshot" } : { lens: "suggestions", text: "Read-only" };
  if (stage === "setup") return { lens: "suggestions", text: "Suggestions only" };
  if (stage === "assessing") return { lens: "suggestions", text: "Suggestions · not assessing" };
  if (stage === "review") return { lens: "team", text: "Team decision · visible to all" };
  return { lens: "frozen", text: "Frozen snapshot" };
}

function gateText(stage: Stage, team: TeamState | undefined, lead: boolean): string | null {
  if (stage === "setup") return lead ? "Choose the model type, targets and assessors" : "A lead sets it up";
  if (!team) return null;
  if (stage === "assessing") {
    const n = team.assessors.length;
    const k = team.assessors.filter((a) => a.submitted_at).length;
    if (n === 0) return lead ? "Nobody is assigned yet" : "Nobody is assigned yet";
    return `Team review opens when ${n === 1 ? "the assessor submits" : `all ${n} submit`} · ${k} in`;
  }
  if (stage === "review" && team.review) return `${fmtInt(team.review.adopted)} adopted · ${fmtInt(team.review.open)} open`;
  if (stage === "finalized") {
    const who = team.recipe.finalized_by_name;
    return team.recipe.finalized_at ? `${who ? `by ${who}, ` : ""}${new Date(team.recipe.finalized_at).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}` : null;
  }
  return null;
}

/** The next step: always this slot, always the primary style; only the words change. */
function NextStep({ ds, stage, access, team }: { ds: Dataset; stage: Stage; access: Access; team: TeamState | undefined }) {
  const me = useMe().data;
  const nav = useNavigate();
  const loc = useLocation();
  const view = useView();
  const fields = useFields(ds.id, true, view);
  const invalidate = useInvalidateTeam(ds.id);
  const submit = useMutation({
    mutationFn: () => api.submitAssessment(ds.id),
    onSuccess: (t) => {
      invalidate();
      toast({ tone: "good", message: t.recipe.phase === "review" ? "Submitted. Everyone is in: team review is open." : "Assessment submitted" });
      nav(`/d/${ds.id}/baseline`);
    },
    onError: (e) => showError("Couldn't submit", e),
  });
  const finalize = useMutation({
    mutationFn: (acceptOpen: boolean) => api.finalize(ds.id, acceptOpen),
    onSuccess: (t) => {
      invalidate();
      toast({ tone: "good", message: `${t.recipe.label} finalized` });
      nav(`/d/${ds.id}/experiments?finalized=${t.recipe.id}`);
    },
    onError: (e) => showError("Couldn't finalize", e),
  });
  const here = (p: string) => loc.pathname === p || loc.pathname.startsWith(`${p}/`);
  const base = `/d/${ds.id}`;
  const lead = me?.role === "lead";
  const label = team?.recipe.label ?? ds.baseline?.label ?? "this baseline";
  const shortLabel = ds.baseline ? baselineLabel(ds.baseline.family, ds.baseline.version, true, ds.baseline.name) : label;

  if (stage === "setup") {
    return lead && !here(`${base}/baseline`) ? <Link className="btn primary" to={`${base}/baseline`}>Set up the baseline</Link> : null;
  }
  if (stage === "assessing" && access.source === "mine" && team?.me && !team.me.submitted) {
    const all = fields.data ?? [];
    const todo = all.filter((f) => !f.reviewed);
    if (todo.length > 0) {
      if (here(`${base}/review`)) return <button type="button" className="primary" disabled title={`Review every field first (${todo.length} left)`}>Submit · {todo.length} left</button>;
      return <Link className="btn primary" to={`${base}/review/${todo[0].id}`}>Next field · {fmtInt(todo.length)} left <kbd className="hide-phone">U</kbd></Link>;
    }
    if (all.length === 0) return null;
    const ask = async () => {
      const changed = team.me ? team.me.total : all.length;
      const ok = await confirmDialog({
        title: "Submit your assessment?",
        body: <p>{fmtInt(changed)} fields. Your choices stay private until everyone submits. You can unsubmit until then.</p>,
        confirm: "Submit", cancel: "Keep reviewing",
      });
      if (ok) submit.mutate();
    };
    return <button type="button" className="primary" disabled={submit.isPending} onClick={ask}>Submit assessment</button>;
  }
  if (stage === "review" && team?.review) {
    const open = team.review.open;
    const askFinalize = async () => {
      const r = team.review!;
      const ok = await confirmDialog({
        title: `Finalize ${shortLabel}?`,
        body: <>
          <p className="num">{fmtInt(r.adopted)} agreed and {fmtInt(r.disagreements - open)} settled, {fmtInt(open)} open. It freezes as a snapshot that experiments build on. A later change is a new version.</p>
          {open > 0 && <p><b>{open} disagreement{open === 1 ? " is" : "s are"} still open</b> and will keep {open === 1 ? "its" : "their"} current team decision{open === 1 ? "" : "s"}.</p>}
        </>,
        confirm: `Finalize ${shortLabel}`, cancel: "Not yet",
      });
      if (ok) finalize.mutate(open > 0);
    };
    if (!lead) return here(`${base}/team`) ? null : <Link className="btn primary" to={`${base}/team`}>See the review queue</Link>;
    if (open > 0 && !here(`${base}/team`)) return <Link className="btn primary" to={`${base}/team`}>Settle {fmtInt(open)} open field{open === 1 ? "" : "s"}</Link>;
    return <button type="button" className={open > 0 ? "" : "primary"} disabled={finalize.isPending} onClick={askFinalize}>Finalize {shortLabel}</button>;
  }
  if (stage === "finalized" && !here(`${base}/experiments`)) {
    return <Link className="btn primary" to={`${base}/experiments`}>{can(me, "createExperiment") ? "Start an experiment" : "See experiments"}</Link>;
  }
  return null;
}

/** Review and Bulk edit on an experiment's copy, or a finalized baseline: says so, and leads back. */
function ViewBar({ ds }: { ds: Dataset }) {
  const view = useView();
  const v = parseView(view);
  const x = useExperiment(v?.kind === "experiment" ? v.id : null);
  if (!v) return null;
  if (v.kind === "baseline") {
    const line = ds.baselines.find((l) => l.latest_finalized_id === v.id || l.id === v.id);
    return (
      <>
        <b className="version-name">{line ? baselineLabel(line.family, line.version, true, line.name) : "A finalized baseline"}</b>
        <span className="secondary small-text">Browsing a finalized baseline</span>
        <span className="spacer" />
        <LensChip lens="frozen">Frozen snapshot · read-only</LensChip>
        <Link className="btn primary" to={`/d/${ds.id}/baselines`}>Back to versions</Link>
      </>
    );
  }
  const d = x.data;
  return (
    <>
      <b className="version-name">{d?.name ?? "Experiment"}</b>
      {d && (
        <span className="secondary small-text">
          built from {baselineLabel(d.recipe_family, d.recipe_version, true, d.recipe_name)}
          {d.adjustments.count > 0 && <> · <b className="num">{d.adjustments.count}</b> field{d.adjustments.count === 1 ? "" : "s"} adjusted</>}
          {!d.can_adjust && <> · read-only{d.is_default ? ": start your own experiment to adjust fields" : `: only ${d.owner_name ?? "its owner"} adjusts it`}</>}
        </span>
      )}
      {d?.adjustments.targets_changed && <span className="badge warn">targets changed</span>}
      <span className="spacer" />
      <LensChip lens="experiment" title={d?.can_adjust ? "Changes apply to this experiment only; the baseline stays as the team agreed it." : undefined}>Experiment copy</LensChip>
      <Link className="btn primary" to={`/d/${ds.id}/experiments/${v.id}`}>Back to the experiment</Link>
    </>
  );
}

function DatasetBar({ ds }: { ds: Dataset }) {
  const me = useMe().data;
  const view = useView();
  const access = useAccess(ds, view);
  const onExperiment = useMatch("/d/:datasetId/experiments/:experimentId");
  const stage = stageOf(ds);
  const team = useTeam(ds.id, Boolean(stage && stage !== "setup"));
  let body: React.ReactNode;
  if (onExperiment) {
    body = <ExperimentBar ds={ds} experimentId={Number(onExperiment.params.experimentId)} />;
  } else if (view) {
    body = <ViewBar ds={ds} />;
  } else if (stage) {
    const lens = lensOf(stage, access, me, team.data);
    const gate = gateText(stage, team.data, me?.role === "lead");
    body = (
      <>
        <VersionSwitcher ds={ds} stage={stage} />
        <PhaseRail stage={stage} />
        {gate && <span className="gate-text num">{gate}</span>}
        <span className="spacer" />
        <LensChip lens={lens.lens} title={access.note || undefined}>{lens.text}</LensChip>
        <NextStep ds={ds} stage={stage} access={access} team={team.data} />
      </>
    );
  } else {
    return null;
  }
  return <div className="dataset-bar">{body}</div>;
}

/** Opening a dataset lands where its next step is: set-up, the review queue, or the fields. */
function DatasetIndex() {
  const ds = useDatasetContext();
  const stage = stageOf(ds);
  if (stage === "setup") return <Navigate to="baseline" replace />;
  if (stage === "review") return <Navigate to="team" replace />;
  return <Navigate to="review" replace />;
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
          <Route path="/people" element={<TeamPage />} />
          <Route path="/team" element={<Navigate to="/people" replace />} />
          <Route path="/settings/:section?" element={<SettingsPage />} />
          <Route path="/d/:datasetId" element={<DatasetLayout />}>
            <Route index element={<DatasetIndex />} />
            <Route path="review/:fieldId?" element={<ReviewPage />} />
            <Route path="fields" element={<Navigate to="../review" replace />} />
            <Route path="bulk" element={<BulkEditPage />} />
            <Route path="query" element={<Navigate to="../bulk?tab=sql" replace />} />
            <Route path="explore/:view?" element={<ExplorePage />} />
            <Route path="export" element={<Navigate to="../experiments" replace />} />
            <Route path="experiments" element={<ExperimentsPage />} />
            <Route path="experiments/:experimentId" element={<ExperimentPage />} />
            <Route path="team" element={<DatasetTeamPage />} />
            <Route path="baseline" element={<BaselinePage />} />
            <Route path="baselines" element={<BaselinesPage />} />
            <Route path="versions" element={<Navigate to="../baselines" replace />} />
          </Route>
          <Route path="*" element={<div className="empty">Page not found. <NavLink to="/">Go home</NavLink></div>} />
        </Route>
      </Routes>
      <ToastRegion />
      <ConfirmRegion />
    </div>
  );
}
