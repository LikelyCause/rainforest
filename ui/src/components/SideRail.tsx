import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useMatch, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAccess, useDataset, useFields, useMe, useTeam } from "../hooks";
import { api } from "../api";
import { can } from "../can";
import { Avatar } from "./Avatar";
import { Logo } from "./Logo";
import { PictureDialog } from "./PictureDialog";
import { ErrorAlert } from "./ui";
import { Icon, PhaseRail, isModalOpen, stageOf, useDismiss, type IconName, type Stage } from "./kit";
import { baselineLabel, familyShort, fmtInt, roleLabel } from "../format";
import { mlflowHome } from "../experiments";
import { showError } from "../toast";
import { useView, withView } from "../view";
import type { Dataset, ExportJob, ModelRun, RecipeVersion, TeamState } from "../types";

/* The left rail: where you can go (the dataset's places, then the app's), the baseline's standing, and you.
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

/** Fields, Explore, Baseline, Experiments, Deployments: the places inside a dataset. */
function DatasetNav({ ds }: { ds: Dataset }) {
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
  const running = [...(exports.data ?? []), ...(runs.data ?? [])].filter(
    (j: ExportJob | ModelRun) => j.status === "pending" || j.status === "running",
  ).length;
  const nextTodo = access.canEdit && !view ? all.find((f) => !f.reviewed) : undefined;
  const inReview = ds.phase === "review" && !view;
  const open = team.data?.review?.open;

  // U jumps back into Focus from the dataset's other pages, matching its meaning in Focus.
  useEffect(() => {
    if (onReview) return;
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
  }, [onReview, nextTodo, nav, ds.id]);

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
        to={`/d/${ds.id}/baseline`}
        icon="layers"
        label="Baseline"
        active={Boolean(onBaseline || onVersions)}
        title="Progress, assessors, versions"
      />
      <RailLink
        to={`/d/${ds.id}/experiments`}
        icon="flask"
        label="Experiments"
        badge={
          running > 0 ? (
            <span
              className="running-dot"
              role="status"
              aria-label={`${running} in progress`}
              title={`${running} export${running === 1 ? " or run" : "s or runs"} in progress`}
            />
          ) : undefined
        }
      />
      <RailLink to={`/d/${ds.id}/deployments`} icon="rocket" label="Deployments" />
    </nav>
  );
}

function DatasetSwitcher({ ds }: { ds: Dataset }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const list = useQuery({ queryKey: ["datasets"], queryFn: api.datasets, enabled: open });
  const size = ds.row_count != null ? `${fmtInt(ds.row_count)} rows · ${fmtInt(ds.column_count)} fields` : null;
  return (
    <div ref={ref} className="rail-switch">
      <button
        type="button"
        className="rail-dataset"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen(!open)}
        title={ds.name}
      >
        <Icon name="database" className="rail-icon" />
        <span className="rail-label rail-dataset-text">
          <span className="ds-name">{ds.name}</span>
          {size && <span className="muted small-text num">{size}</span>}
        </span>
        <Icon name="chevron-down" className="rail-label muted" />
      </button>
      {open && (
        <div className="popover rail-popover">
          {list.data?.map((d) => (
            <Link
              key={d.id}
              to={`/d/${d.id}`}
              className="menu-link"
              aria-current={d.id === ds.id ? "true" : undefined}
              onClick={close}
            >
              <span className="grow">{d.name}</span>
              {d.id === ds.id && <Icon name="check" />}
            </Link>
          ))}
          <hr />
          <Link to="/" className="menu-link" onClick={close}>
            All datasets
          </Link>
        </div>
      )}
    </div>
  );
}

const PHASE_WORD = { assessing: "Assessing", review: "Team review", finalized: "Finalized" } as const;

/** Every version of every baseline of the dataset, with its phase. The current version opens the Baseline tab,
 *  a finalized one its fields read-only, any other the versions page with it in view. */
function VersionSwitcher({ ds, stage }: { ds: Dataset; stage: Stage }) {
  const me = useMe().data;
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id), enabled: open });
  const b = ds.baseline;
  if (!b) return null;
  const name = stage === "setup" ? "No baseline yet" : baselineLabel(b.family, b.version, true, b.name);
  const where = (v: RecipeVersion) =>
    v.id === b.id
      ? `/d/${ds.id}/baseline`
      : v.phase === "finalized"
        ? `/d/${ds.id}/review?view=baseline:${v.id}`
        : `/d/${ds.id}/baselines?v=${v.id}`;
  return (
    <div ref={ref} className="version-switch">
      <button
        type="button"
        className="ghost switcher version-switcher"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen(!open)}
        title={stage === "setup" ? "This dataset's first baseline isn't set up yet" : b.label}
      >
        <span className="ds-name">{name}</span>
        {stage !== "setup" && <span className="badge">{familyShort(b.family)}</span>}
        <Icon name="chevron-down" />
      </button>
      {open && (
        <div className="popover version-menu rail-popover">
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
          <hr />
          <Link to={`/d/${ds.id}/baselines`} className="menu-link" onClick={close}>
            Versions and compare
          </Link>
          {can(me, "startBaseline", ds.status === "refreshing") && (
            <Link to={`/d/${ds.id}/baselines`} className="menu-link" onClick={close}>
              New version or baseline…
            </Link>
          )}
        </div>
      )}
    </div>
  );
}

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

/** The baseline version being worked, where it stands in its four phases, and what it's waiting on. */
function BaselineStatus({ ds }: { ds: Dataset }) {
  const me = useMe().data;
  const stage = stageOf(ds);
  const team = useTeam(ds.id, Boolean(stage && stage !== "setup"));
  if (!stage) return null;
  const gate = gateText(stage, team.data, me?.role === "lead");
  return (
    <section className="rail-status" aria-label="Baseline">
      <VersionSwitcher ds={ds} stage={stage} />
      <PhaseRail stage={stage} />
      {gate && <span className="gate-text rail-label num">{gate}</span>}
    </section>
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
  const m = useMatch("/d/:datasetId/*");
  const datasetId = m ? Number(m.params.datasetId) : NaN;
  const ds = useDataset(datasetId, !Number.isNaN(datasetId));
  const d = m ? ds.data : undefined;
  const ready = d?.status === "ready" || d?.status === "refreshing";
  return (
    <aside className="rail">
      <Link to="/" className="brand" aria-label="Rainforest: all datasets" title="All datasets">
        <Logo />
        <span className="rail-label">Rainforest</span>
      </Link>
      {d && <DatasetSwitcher ds={d} />}
      {d && ready && <DatasetNav ds={d} />}
      {d && <hr className="rail-rule" />}
      <nav className="rail-group" aria-label="Main">
        <RailLink to="/" icon="database" label={d ? "All datasets" : "Datasets"} end />
        <RailLink to="/people" icon="users" label="People" />
        <RailLink to="/settings" icon="settings" label="Settings" />
      </nav>
      <span className="spacer" />
      {d && ready && <BaselineStatus ds={d} />}
      <UserMenu />
    </aside>
  );
}
