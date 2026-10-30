import { useEffect } from "react";
import { Link, NavLink, Navigate, Outlet, Route, Routes, useMatch, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useAccess, useDataset, useExperiment, useMe, useTeam } from "./hooks";
import { ApiError, api, setUnauthorizedHandler } from "./api";
import { Logo } from "./components/Logo";
import { SideRail } from "./components/SideRail";
import { ConfirmRegion, LensChip, stageOf, type Lens, type Stage } from "./components/kit";
import { NextStep } from "./components/NextStep";
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
import { BatchPage } from "./pages/BatchPage";
import { DeploymentsPage } from "./pages/DeploymentsPage";
import { DeploymentPage } from "./pages/DeploymentPage";
import { DatasetTeamPage } from "./pages/DatasetTeamPage";
import { DatasetLayout, useDatasetContext } from "./components/DatasetLayout";
import { ErrorAlert } from "./components/ui";
import { baselineLabel } from "./format";
import { ToastRegion } from "./toast";
import { parseView, useView } from "./view";
import type { Dataset, TeamState } from "./types";
import type { Access } from "./team";

/* ------------------------------------------------------------------ the page bar: where you are, whose decisions, the next step */

function lensOf(
  stage: Stage,
  access: Access,
  me: { role: string } | undefined,
  team: TeamState | undefined,
  onBaseline: boolean,
): { lens: Lens; text: string } {
  // The Baseline tab shows progress only while assessing, whoever is looking.
  if (onBaseline && stage === "assessing")
    return { lens: "suggestions", text: "Progress only · nobody's choices show" };
  if (access.source === "mine")
    return { lens: "private", text: team?.me?.submitted ? "Private · submitted" : "Private · your assessment" };
  if (me?.role === "viewer")
    return stage === "finalized"
      ? { lens: "frozen", text: "Frozen snapshot" }
      : { lens: "suggestions", text: "Read-only" };
  if (stage === "setup") return { lens: "suggestions", text: "Suggestions only" };
  if (stage === "assessing") return { lens: "suggestions", text: "Suggestions · not assessing" };
  if (stage === "review") return { lens: "team", text: "Team decision · visible to all" };
  return { lens: "frozen", text: "Frozen snapshot" };
}

/** Review and Bulk edit on an experiment's copy, or a finalized baseline: says so, and leads back. */
function ViewBar({ ds }: { ds: Dataset }) {
  const view = useView();
  const v = parseView(view);
  const x = useExperiment(v?.kind === "experiment" ? v.id : null);
  const lines = useQuery({
    queryKey: ["baselines", ds.id],
    queryFn: () => api.baselines(ds.id),
    enabled: v?.kind === "baseline",
  });
  if (!v) return null;
  if (v.kind === "baseline") {
    const ver = (lines.data ?? []).flatMap((l) => l.versions).find((r) => r.id === v.id);
    return (
      <>
        <b className="version-name">
          {ver ? baselineLabel(ver.family, ver.version, true, ver.name) : "A finalized baseline"}
        </b>
        <span className="secondary small-text">Browsing a finalized baseline</span>
        <span className="spacer" />
        <LensChip lens="frozen">Frozen snapshot · read-only</LensChip>
        <Link className="btn primary" to={`/d/${ds.id}/baselines?v=${v.id}`}>
          Back to versions
        </Link>
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
          {d.adjustments.count > 0 && (
            <>
              {" "}
              · <b className="num">{d.adjustments.count}</b> field{d.adjustments.count === 1 ? "" : "s"} adjusted
            </>
          )}
          {!d.can_adjust && (
            <>
              {" "}
              · read-only
              {d.is_default
                ? ": start your own experiment to adjust fields"
                : `: only ${d.owner_name ?? "its owner"} adjusts it`}
            </>
          )}
        </span>
      )}
      {d?.adjustments.targets_changed && <span className="badge warn">targets changed</span>}
      <span className="spacer" />
      <LensChip
        lens="experiment"
        title={
          d?.can_adjust ? "Changes apply to this experiment only; the baseline stays as the team agreed it." : undefined
        }
      >
        Experiment copy
      </LensChip>
      <Link className="btn primary" to={`/d/${ds.id}/experiments/${v.id}`}>
        Back to the experiment
      </Link>
    </>
  );
}

const SECTION: Record<string, string> = {
  review: "Fields",
  bulk: "Fields",
  team: "Team review",
  explore: "Explore",
  baseline: "Baseline",
  baselines: "Versions",
  experiments: "Experiments",
  deployments: "Deployments",
};

function DatasetBar({ ds }: { ds: Dataset }) {
  const me = useMe().data;
  const view = useView();
  const access = useAccess(ds, view);
  const onExperiment = useMatch("/d/:datasetId/experiments/:experimentId/*");
  const onBaseline = useMatch("/d/:datasetId/baseline");
  const section = useMatch("/d/:datasetId/:section/*")?.params.section;
  const stage = stageOf(ds);
  const team = useTeam(ds.id, Boolean(stage && stage !== "setup"));
  let body: React.ReactNode;
  if (onExperiment) {
    body = <ExperimentBar ds={ds} experimentId={Number(onExperiment.params.experimentId)} />;
  } else if (view) {
    body = <ViewBar ds={ds} />;
  } else if (stage) {
    const lens = lensOf(stage, access, me, team.data, Boolean(onBaseline));
    body = (
      <>
        {section && SECTION[section] && <span className="page-bar-title">{SECTION[section]}</span>}
        <span className="spacer" />
        <LensChip lens={lens.lens} title={access.note || undefined}>
          {lens.text}
        </LensChip>
        <NextStep ds={ds} stage={stage} access={access} team={team.data} />
      </>
    );
  } else {
    return null;
  }
  return <div className="dataset-bar">{body}</div>;
}

/** The bar above a ready dataset's pages; outside a dataset, pages carry their own heading. */
function PageBar() {
  const m = useMatch("/d/:datasetId/*");
  const datasetId = m ? Number(m.params.datasetId) : NaN;
  const ds = useDataset(datasetId, !Number.isNaN(datasetId));
  const d = m ? ds.data : undefined;
  return d && (d.status === "ready" || d.status === "refreshing") ? <DatasetBar ds={d} /> : null;
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
          <div className="login-brand">
            <Logo /> Rainforest
          </div>
          <ErrorAlert
            error={me.error}
            title={me.error instanceof ApiError && me.error.status === 503 ? "Not set up yet." : "Can't reach the API."}
          />
        </div>
      </div>
    );
  }
  return (
    <>
      <SideRail />
      <div className="app-body">
        <PageBar />
        <main className="main">
          <Outlet />
        </main>
      </div>
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
        <Route
          path="/login"
          element={
            <main className="main">
              <LoginPage />
            </main>
          }
        />
        <Route
          path="/forbidden"
          element={
            <main className="main">
              <ForbiddenPage />
            </main>
          }
        />
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
            <Route path="experiments/:experimentId/batches/:batchId" element={<BatchPage />} />
            <Route path="deployments" element={<DeploymentsPage />} />
            <Route path="deployments/:deploymentId" element={<DeploymentPage />} />
            <Route path="team" element={<DatasetTeamPage />} />
            <Route path="baseline" element={<BaselinePage />} />
            <Route path="baselines" element={<BaselinesPage />} />
            <Route path="versions" element={<Navigate to="../baselines" replace />} />
          </Route>
          <Route
            path="*"
            element={
              <div className="empty">
                Page not found. <NavLink to="/">Go home</NavLink>
              </div>
            }
          />
        </Route>
      </Routes>
      <ToastRegion />
      <ConfirmRegion />
    </div>
  );
}
