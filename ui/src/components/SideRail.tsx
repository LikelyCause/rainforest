import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useMatch, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAccess, useDataset, useFields, useMe, useTeam } from "../hooks";
import { ApiError, api } from "../api";
import { can } from "../can";
import { Avatar } from "./Avatar";
import { Logo } from "./Logo";
import { PictureDialog } from "./PictureDialog";
import { ErrorAlert } from "./ui";
import { Icon, PhaseRail, isModalOpen, stageOf, useDismiss, type IconName, type Stage } from "./kit";
import { baselineLabel, fmtInt, roleLabel } from "../format";
import { mlflowHome } from "../experiments";
import { showError } from "../toast";
import { useView, withView } from "../view";
import type { Dataset, ExportJob, ModelRun, RecipeVersion, TeamState } from "../types";

/* The left rail: which dataset and baseline you're on, that dataset's places, then the app's, and you.
 * Below 1360px it narrows to icons; labels stay in the accessibility tree. */

function RailLink({
  to,
  icon,
  label,
  count,
  badge,
  active,
  end,
  title,
}: {
  to: string;
  icon: IconName;
  label: ReactNode;
  count?: ReactNode;
  /** Stays visible when the rail narrows to icons. */
  badge?: ReactNode;
  active?: boolean;
  end?: boolean;
  title?: string;
}) {
  const body = (
    <>
      <Icon name={icon} />
      <span className="rail-label">{label}</span>
      {count != null && <span className="rail-count rail-label num">{count}</span>}
      {badge}
    </>
  );
  return active === undefined ? (
    <NavLink to={to} end={end} title={title} className={({ isActive }) => `rail-link${isActive ? " on" : ""}`}>
      {body}
    </NavLink>
  ) : (
    <Link
      to={to}
      title={title}
      className={`rail-link${active ? " on" : ""}`}
      aria-current={active ? "page" : undefined}
    >
      {body}
    </Link>
  );
}

const FOCUS_KEY = "rf:focus";

/** The dataset last opened, which the rail and the home page carry on with; NaN for none. */
export function storedFocus(): number {
  try {
    return Number(localStorage.getItem(FOCUS_KEY)) || NaN;
  } catch {
    return NaN;
  }
}

/** The dataset the rail works on: the one open, else the last one opened, so its places are a click away from
 *  anywhere. A remembered dataset that's gone or no longer shared is forgotten. */
function useFocusedDataset(): Dataset | undefined {
  const m = useMatch("/d/:datasetId/*");
  const routeId = m ? Number(m.params.datasetId) : NaN;
  const [stored, setStored] = useState(storedFocus);
  const id = Number.isNaN(routeId) ? stored : routeId;
  const ds = useDataset(id, !Number.isNaN(id));
  useEffect(() => {
    if (Number.isNaN(routeId) || !ds.data) return;
    try {
      localStorage.setItem(FOCUS_KEY, String(routeId));
    } catch {
      /* the focus just isn't remembered */
    }
    setStored(routeId);
  }, [routeId, ds.data]);
  const gone = ds.error instanceof ApiError && (ds.error.status === 404 || ds.error.status === 403);
  useEffect(() => {
    if (!gone || !Number.isNaN(routeId)) return;
    try {
      localStorage.removeItem(FOCUS_KEY);
    } catch {
      /* nothing to forget */
    }
    setStored(NaN);
  }, [gone, routeId]);
  return ds.data;
}

/** Fields, Explore, Experiments, Runs, Models: the focused dataset's places, greyed out until there is one. */
function DatasetNav({ ds }: { ds: Dataset | undefined }) {
  const ready = ds?.status === "ready" || ds?.status === "refreshing";
  if (!ds || !ready)
    return (
      <nav className="rail-group" aria-label="Dataset">
        {PLACES.map((p) => (
          <span
            key={p.label}
            className="rail-link off"
            aria-disabled="true"
            title={ds ? "Open once the dataset is ready" : "Choose a dataset first"}
          >
            <Icon name={p.icon} />
            <span className="rail-label">{p.label}</span>
          </span>
        ))}
      </nav>
    );
  return <ReadyNav ds={ds} />;
}

const PLACES: { label: string; icon: IconName }[] = [
  { label: "Fields", icon: "list" },
  { label: "Explore", icon: "chart" },
  { label: "Experiments", icon: "flask" },
  { label: "Runs", icon: "play" },
  { label: "Models", icon: "rocket" },
];

function ReadyNav({ ds }: { ds: Dataset }) {
  const nav = useNavigate();
  const view = useView();
  const access = useAccess(ds, view);
  const inDataset = useMatch("/d/:datasetId/*");
  const onReview = useMatch("/d/:datasetId/review/*");
  const onBulk = useMatch("/d/:datasetId/bulk");
  const onQueue = useMatch("/d/:datasetId/team");
  const fields = useFields(ds.id, true, view);
  const team = useTeam(ds.id, ds.phase === "review");
  const busy = (q: { state: { data?: { status: string }[] } }) =>
    (q.state.data ?? []).some((j) => j.status === "pending" || j.status === "running") ? 3000 : false;
  const exports = useQuery({ queryKey: ["exports", ds.id], queryFn: () => api.exports(ds.id), refetchInterval: busy });
  const runs = useQuery({ queryKey: ["runs", ds.id], queryFn: () => api.runs(ds.id), refetchInterval: busy });
  const all = fields.data ?? [];
  const reviewed = all.filter((f) => f.reviewed);
  const live = (j: ExportJob | ModelRun) => j.status === "pending" || j.status === "running";
  const runningRuns = (runs.data ?? []).filter(live).length;
  const running = (exports.data ?? []).filter(live).length + runningRuns;
  const nextTodo = access.canEdit && !view ? all.find((f) => !f.reviewed) : undefined;
  const inReview = ds.phase === "review" && !view;
  const open = team.data?.review?.open;

  // U jumps back into Focus from the dataset's other pages, matching its meaning in Focus.
  useEffect(() => {
    if (onReview || !inDataset) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (
        e.key !== "u" ||
        e.metaKey ||
        e.ctrlKey ||
        e.altKey ||
        ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) ||
        el.isContentEditable
      )
        return;
      if (isModalOpen()) return;
      if (nextTodo) nav(`/d/${ds.id}/review/${nextTodo.id}`);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onReview, inDataset, nextTodo, nav, ds.id]);

  const dot = (n: number, what: string) =>
    n > 0 ? (
      <span className="running-dot" role="status" aria-label={`${n} in progress`} title={`${n} ${what} in progress`} />
    ) : undefined;

  return (
    <nav className="rail-group" aria-label="Dataset">
      <RailLink
        to={inReview ? `/d/${ds.id}/team` : withView(`/d/${ds.id}/review`, view)}
        icon="list"
        label="Fields"
        active={Boolean(onReview || onBulk || onQueue)}
        count={
          inReview
            ? open != null
              ? `${open} open`
              : undefined
            : all.length
              ? `${reviewed.length}/${all.length}`
              : undefined
        }
        title={inReview ? "The team review queue, then every field" : `${reviewed.length} of ${all.length} reviewed`}
      />
      <RailLink to={`/d/${ds.id}/explore`} icon="chart" label="Explore" />
      <RailLink
        to={`/d/${ds.id}/experiments`}
        icon="flask"
        label="Experiments"
        badge={dot(running, running === 1 ? "export or run" : "exports or runs")}
      />
      <RailLink
        to={`/d/${ds.id}/runs`}
        icon="play"
        label="Runs"
        badge={dot(runningRuns, runningRuns === 1 ? "run" : "runs")}
      />
      <RailLink to={`/d/${ds.id}/models`} icon="rocket" label="Models" title="Deployed models: keys and tests" />
    </nav>
  );
}

const PHASE_WORD = { assessing: "Assessing", review: "Team review", finalized: "Finalized" } as const;
const STAGE_WORD = { setup: "Set up", ...PHASE_WORD } as const;

/** Where a dataset picked from the menu opens: the same place, when it's one every dataset has. */
const KEEP = new Set(["explore", "experiments", "runs", "models"]);

export function gateText(stage: Stage, team: TeamState | undefined, lead: boolean): string | null {
  if (stage === "setup") return lead ? "Choose the model type, targets and assessors" : "A lead sets it up";
  if (!team) return null;
  if (stage === "assessing") {
    const n = team.assessors.length;
    const k = team.assessors.filter((a) => a.submitted_at).length;
    if (n === 0) return "Nobody is assigned yet";
    return `Team review opens when ${n === 1 ? "the assessor submits" : n === 2 ? "both submit" : `all ${n} submit`} · ${k} in`;
  }
  if (stage === "review" && team.review)
    return `${fmtInt(team.review.adopted)} adopted · ${fmtInt(team.review.open)} open`;
  if (stage === "finalized") {
    const who = team.recipe.finalized_by_name;
    return team.recipe.finalized_at
      ? `${who ? `by ${who}, ` : ""}${new Date(team.recipe.finalized_at).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}`
      : null;
  }
  return null;
}

/** The baseline half of the focus menu: where the version being worked stands, and every version to switch to.
 *  The current version opens its progress, a finalized one its fields read-only, any other the versions page. */
function BaselineMenu({ ds, stage, close }: { ds: Dataset; stage: Stage; close: () => void }) {
  const me = useMe().data;
  const team = useTeam(ds.id, stage !== "setup");
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id) });
  const b = ds.baseline;
  if (!b) return null;
  const gate = gateText(stage, team.data, me?.role === "lead");
  const where = (v: RecipeVersion) =>
    v.id === b.id
      ? `/d/${ds.id}/baseline`
      : v.phase === "finalized"
        ? `/d/${ds.id}/review?view=baseline:${v.id}`
        : `/d/${ds.id}/baselines?v=${v.id}`;
  return (
    <section className="focus-baseline" aria-label="Baseline">
      <span className="menu-note muted small-text">Baseline</span>
      <div className="focus-phase">
        <PhaseRail stage={stage} />
        {gate && <span className="gate-text num">{gate}</span>}
      </div>
      {lines.isPending && <span className="menu-note muted small-text">Loading versions…</span>}
      <ErrorAlert error={lines.error} />
      {(lines.data ?? []).map((l) => (
        <div key={l.family} role="group" aria-label={`${l.name ?? l.label} baseline`}>
          {(lines.data ?? []).length > 1 && (
            <span className="menu-note muted small-text">
              {l.name ?? `${l.label} baseline`} · {l.label}
            </span>
          )}
          {l.versions.map((v) => {
            const current = v.id === b.id;
            return (
              <Link
                key={v.id}
                to={where(v)}
                className="menu-link"
                aria-current={current ? "true" : undefined}
                onClick={close}
                title={
                  current
                    ? "The version being worked: its progress"
                    : v.phase === "finalized"
                      ? "Browse its fields, read-only"
                      : "See it on the versions page"
                }
              >
                <span className="grow">
                  {baselineLabel(v.family, v.version, true, v.name)}
                  {current && <span className="muted"> · current</span>}
                </span>
                <span className="muted small-text">{PHASE_WORD[v.phase]}</span>
                {current ? (
                  <Icon name="check" />
                ) : v.phase === "finalized" ? (
                  <Icon name="lock" size={14} className="muted" />
                ) : null}
              </Link>
            );
          })}
        </div>
      ))}
      <Link to={`/d/${ds.id}/baselines`} className="menu-link" onClick={close}>
        Versions and compare
      </Link>
      {can(me, "startBaseline", ds.status === "refreshing") && (
        <Link to={`/d/${ds.id}/baselines`} className="menu-link" onClick={close}>
          New version or baseline…
        </Link>
      )}
    </section>
  );
}

/** The top-left menu: which dataset, and which of its baselines, the rail's places open on. */
function FocusSwitcher({ ds }: { ds: Dataset | undefined }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const section = useMatch("/d/:datasetId/:section/*")?.params.section;
  const list = useQuery({ queryKey: ["datasets"], queryFn: api.datasets, enabled: open });
  const ready = ds?.status === "ready" || ds?.status === "refreshing";
  const stage = ds && ready ? stageOf(ds) : null;
  const b = ds?.baseline;
  const focus =
    stage === "setup"
      ? "No baseline yet"
      : stage && b
        ? `${baselineLabel(b.family, b.version, true, b.name)} · ${STAGE_WORD[stage]}`
        : ds?.row_count != null
          ? `${fmtInt(ds.row_count)} rows · ${fmtInt(ds.column_count)} fields`
          : null;
  const to = (id: number) => (section && KEEP.has(section) ? `/d/${id}/${section}` : `/d/${id}`);
  return (
    <div ref={ref} className="rail-switch">
      <button
        type="button"
        className="rail-dataset"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen(!open)}
        title={ds ? [ds.name, focus].filter(Boolean).join(" · ") : "Choose a dataset"}
      >
        <Icon name="database" className="rail-icon" />
        <span className="rail-label rail-dataset-text">
          <span className="ds-name">{ds?.name ?? "Choose a dataset"}</span>
          {focus && <span className="muted small-text num">{focus}</span>}
        </span>
        <Icon name="chevron-down" className="rail-label muted" />
      </button>
      {open && (
        <div className="popover rail-popover focus-menu">
          <span className="menu-note muted small-text">Dataset</span>
          {list.isPending && <span className="menu-note muted small-text">Loading datasets…</span>}
          <ErrorAlert error={list.error} />
          {list.data?.map((d) => (
            <Link
              key={d.id}
              to={to(d.id)}
              className="menu-link"
              aria-current={d.id === ds?.id ? "true" : undefined}
              onClick={close}
            >
              <span className="grow">{d.name}</span>
              {d.id === ds?.id && <Icon name="check" />}
            </Link>
          ))}
          <Link to="/" className="menu-link" onClick={close}>
            All datasets
          </Link>
          {ds && stage && (
            <>
              <hr />
              <BaselineMenu ds={ds} stage={stage} close={close} />
            </>
          )}
        </div>
      )}
    </div>
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
  const button = useRef<HTMLButtonElement>(null);
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
    <div ref={ref} className="user-menu">
      <button
        ref={button}
        type="button"
        className="ghost profile-button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        aria-label={`Profile menu: ${u.name}, ${roleLabel(u.role)}`}
        title={u.email ?? undefined}
      >
        <Avatar user={u} size={28} />
        <span className="rail-label profile-text">
          <span className="user-name">{u.name}</span>
          <span className="muted small-text">{roleLabel(u.role)}</span>
        </span>
      </button>
      {open && (
        <div className="popover profile-menu rail-popover" role="menu">
          <div className="profile-menu-head">
            <Avatar user={u} size={44} />
            <div className="stack" style={{ gap: 2 }}>
              <b>{u.name}</b>
              <span className="muted small-text">{roleLabel(u.role)}</span>
            </div>
          </div>
          <button
            type="button"
            role="menuitem"
            className="ghost menu-item"
            onClick={() => {
              close();
              setPicture(true);
            }}
          >
            Change picture…
          </button>
          {system.data?.mlflow.enabled && (
            <a
              role="menuitem"
              className="menu-item"
              href={mlflowHome(system.data.mlflow.browseUrl) ?? undefined}
              target="_blank"
              rel="noopener noreferrer"
              onClick={close}
            >
              MLflow <Icon name="external" size={14} />
            </a>
          )}
          <button
            type="button"
            role="menuitem"
            className="ghost menu-item"
            disabled={out.isPending}
            onClick={() => out.mutate()}
          >
            {pick ? "Switch profile" : "Sign out"}
          </button>
        </div>
      )}
      {/* The menu item that opened it is gone by now, so focus returns to the profile button. */}
      {picture && (
        <PictureDialog
          user={u}
          onClose={() => {
            setPicture(false);
            button.current?.focus();
          }}
        />
      )}
    </div>
  );
}

export function SideRail() {
  const ds = useFocusedDataset();
  return (
    <aside className="rail">
      <Link to="/" className="brand" aria-label="Rainforest: all datasets" title="All datasets">
        <Logo />
        <span className="rail-label">Rainforest</span>
      </Link>
      <FocusSwitcher ds={ds} />
      <DatasetNav ds={ds} />
      <span className="spacer" />
      <nav className="rail-group" aria-label="Main">
        <RailLink to="/" icon="table" label="Datasets" end />
        <RailLink to="/settings" icon="settings" label="Settings" />
      </nav>
      <UserMenu />
    </aside>
  );
}
