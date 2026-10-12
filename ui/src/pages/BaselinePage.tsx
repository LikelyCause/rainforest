import { useCallback, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { can } from "../can";
import { useInvalidateTeam, useMe, useTeam } from "../hooks";
import { Avatar } from "../components/Avatar";
import { useDatasetContext } from "../components/DatasetLayout";
import { Banner, Chip, Icon, ProgressBar, confirmDialog, stageOf, useDismiss } from "../components/kit";
import { ReuseCard } from "../components/ReuseCard";
import { StartBaseline } from "../components/StartBaseline";
import { ErrorAlert } from "../components/ui";
import { FAMILIES } from "../baselines";
import { baselineLabel, familyShort, fmtAgo, fmtDate, fmtInt } from "../format";
import { builtFrom } from "../recipes";
import { submittedText } from "../team";
import { showError, toast } from "../toast";
import type { AssessorProgress, Dataset, Experiment, RecipeVersion, TeamState } from "../types";
import { PhaseBadge, useUserMap } from "./DatasetTeamPage";

const dateOnly = (s: string | null) => (s ? new Date(s).toLocaleDateString(undefined, { day: "numeric", month: "short" }) : "—");

/** The Baseline tab: the version being worked (its progress, assessors, queue summary) beside every version. */
export function BaselinePage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const team = useTeam(ds.id);
  const stage = stageOf(ds);
  if (team.error) return <div className="page"><ErrorAlert error={team.error} /></div>;
  if (!team.data || !stage) return <div className="page" aria-busy="true"><span className="skeleton" style={{ width: 240, height: 22 }} /></div>;
  const t = team.data;

  if (stage === "setup") {
    return (
      <div className="page bl-page">
        {me?.role === "lead" ? <StartBaseline ds={ds} /> : (<>
          <div className="bl-title">
            <h1>Set up the baseline</h1>
            <p className="secondary">A baseline is the team's agreed preparation of this dataset for one model type: its targets, and each field kept, dropped or retyped.</p>
          </div>
          <Banner tone="info">A lead sets up this dataset's first baseline: its model type, targets and who assesses it. Until then you can browse the fields and Explore.</Banner>
        </>)}
      </div>
    );
  }

  return (
    <div className="page bl-page">
      <div className="bl-grid">
        <div className="bl-main">
          <Title ds={ds} team={t} />
          {t.recipe.phase === "assessing" ? <Assessing ds={ds} team={t} />
            : t.recipe.phase === "review" ? <InReview ds={ds} team={t} />
            : <Finalized ds={ds} team={t} />}
        </div>
        <Versions ds={ds} team={t} />
      </div>
    </div>
  );
}

/** The version's name and where it came from. */
function Title({ ds, team }: { ds: Dataset; team: TeamState }) {
  const r = team.recipe;
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id) });
  const v = (lines.data ?? []).flatMap((l) => l.versions).find((x) => x.id === r.id);
  const parts = [
    `Started ${dateOnly(r.created_at)}${r.created_by_name ? ` by ${r.created_by_name}` : ""}${v ? ` from ${builtFrom(v, true)}` : ""}`,
    // A later version nobody assesses ("Start v2 and edit directly") reopens nothing: the lead changes it in Fields.
    r.reopened == null ? (team.assessors.length === 0 && r.version > 1 && !r.reuse ? "nothing reopened, edited directly" : "every field open")
      : `${fmtInt(r.reopened)} field${r.reopened === 1 ? "" : "s"} reopened`,
    `settings for ${familyShort(r.family).toLowerCase()} models`,
  ];
  return (
    <div className="bl-title">
      <h1>{r.label ?? baselineLabel(r.family, r.version, false, r.name)}</h1>
      <p className="secondary" title={fmtDate(r.created_at)}>{parts.join(" · ")}</p>
    </div>
  );
}

function Assessing({ ds, team }: { ds: Dataset; team: TeamState }) {
  const r = team.recipe;
  const me = useMe().data;
  const lead = me?.role === "lead";
  return (
    <>
      {r.reopened != null && !r.reuse && (
        <Banner tone="info">
          {r.label} reopens {r.reopened} field{r.reopened === 1 ? "" : "s"}; only those are assessed. The rest are carried over from the version before and locked.
        </Banner>
      )}
      {team.me?.submitted && <Banner tone="good" title="Submitted.">{submittedText(team.assessors)}</Banner>}
      {team.me && <MyAssessment ds={ds} me={team.me} />}
      <Assessors ds={ds} team={team} />
      {r.reuse && <ReuseCard reuse={r.reuse} label={r.label ?? ""} />}
      <section className="bl-next" aria-labelledby="bl-then">
        <h2 id="bl-then" className="bl-label">What happens next</h2>
        <ol>
          <li className="card"><b>Everyone submits</b><span>Each assessment locks.</span></li>
          <li className="card"><b>Agreed fields adopted</b><span>Automatically, as chosen.</span></li>
          <li className="card"><b>{lead ? "You settle the rest" : "A lead settles the rest"}</b><span>In the team review, with comments.</span></li>
          <li className="card"><b>{lead ? "You finalize" : "A lead finalizes"} v{r.version}</b><span>It freezes for experiments.</span></li>
        </ol>
      </section>
    </>
  );
}

function MyAssessment({ ds, me }: { ds: Dataset; me: NonNullable<TeamState["me"]> }) {
  const invalidate = useInvalidateTeam(ds.id);
  const left = me.total - me.reviewed;
  const submit = useMutation({
    mutationFn: () => (me.submitted ? api.unsubmitAssessment(ds.id) : api.submitAssessment(ds.id)),
    onSuccess: (t) => {
      invalidate();
      toast({ tone: "good", message: t.recipe.phase === "review" ? "Submitted. Everyone is in: team review is open."
        : t.me?.submitted ? "Assessment submitted" : "Assessment reopened" });
    },
  });
  const ask = async () => {
    if (me.submitted) return submit.mutate();
    const ok = await confirmDialog({
      title: "Submit your assessment?",
      body: <p>{fmtInt(me.total)} fields. Your choices stay private until everyone submits. You can unsubmit until then.</p>,
      confirm: "Submit", cancel: "Keep reviewing",
    });
    if (ok) submit.mutate();
  };
  return (
    <section className="card bl-mine" aria-labelledby="bl-mine">
      <div className="stack grow" style={{ gap: 6, minWidth: 0 }}>
        <div className="row" style={{ gap: 8 }}>
          <h2 id="bl-mine">Your assessment</h2>
          <Chip tone="private" icon="user">Private</Chip>
          {me.submitted && <Chip tone="good" icon="check">Submitted</Chip>}
        </div>
        <div className="row bl-progress">
          <ProgressBar value={me.reviewed} max={me.total} label="Your progress" color="var(--lens-private)" width={220} />
          <span className="num small-text secondary">{fmtInt(me.reviewed)} of {fmtInt(me.total)} · {fmtInt(left)} left</span>
        </div>
        <p className="muted small-text">Nobody sees your choices until team review, the lead included.</p>
      </div>
      <div className="row bl-mine-actions">
        {!me.submitted && <Link className={`btn${left > 0 ? " primary" : ""}`} to={`/d/${ds.id}/review`}>{left ? "Continue" : "Look over your choices"}</Link>}
        <button type="button" className={!me.submitted && left === 0 ? "primary" : ""} disabled={submit.isPending || (!me.submitted && left > 0)}
          title={!me.submitted && left > 0 ? `Review every field first (${left} left)` : undefined} onClick={ask}>
          {me.submitted ? "Unsubmit to make changes" : "Submit"}
        </button>
      </div>
      <ErrorAlert error={submit.error} />
    </section>
  );
}

function Assessors({ ds, team }: { ds: Dataset; team: TeamState }) {
  const me = useMe().data;
  const lead = me?.role === "lead";
  const invalidate = useInvalidateTeam(ds.id);
  const users = useUserMap();
  const [adding, setAdding] = useState(false);
  const [pick, setPick] = useState("");
  const assigned = new Set(team.assessors.map((a) => a.user_id));
  const candidates = [...users.values()].filter((u) => u.active && u.role !== "viewer" && !assigned.has(u.id));
  const submitted = team.assessors.filter((a) => a.submitted_at).length;
  const assign = useMutation({
    mutationFn: (userId: number) => api.assign(ds.id, userId),
    onSuccess: () => { setPick(""); setAdding(false); invalidate(); },
  });
  const unassign = useMutation({
    mutationFn: (userId: number) => api.unassign(ds.id, userId),
    onSuccess: (t) => {
      invalidate();
      if (t.recipe.phase === "review") toast({ message: "Everyone left has submitted: team review is open." });
    },
    onError: (e) => showError("Couldn't unassign", e),
  });
  const askUnassign = async (a: AssessorProgress) => {
    const ok = await confirmDialog({
      title: `Unassign ${a.name}?`,
      body: <p>Their choices are kept but won't count. If everyone left has submitted, team review opens.</p>,
      confirm: "Unassign", cancel: "Keep them", danger: true,
    });
    if (ok) unassign.mutate(a.user_id);
  };
  return (
    <section className="card bl-who" aria-labelledby="bl-who">
      <div className="bl-card-head">
        <h2 id="bl-who">Assessors {team.assessors.length > 0
          ? <span className="num muted">{submitted} of {team.assessors.length} submitted</span>
          : <span className="muted">· nobody assigned</span>}</h2>
        {lead && !adding && <button type="button" className="small" onClick={() => setAdding(true)}><Icon name="plus" size={14} />Add an assessor</button>}
      </div>
      {lead && adding && (
        <div className="row bl-add">
          <select value={pick} onChange={(e) => setPick(e.target.value)} aria-label="User to assign" autoFocus>
            <option value="">Choose a person…</option>
            {candidates.map((u) => <option key={u.id} value={u.id}>{u.name} · {u.role}</option>)}
          </select>
          <button type="button" className="primary" disabled={!pick || assign.isPending} onClick={() => assign.mutate(Number(pick))}>Assign</button>
          <button type="button" className="ghost" onClick={() => { setAdding(false); setPick(""); }}>Cancel</button>
          <ErrorAlert error={assign.error} />
        </div>
      )}
      {team.assessors.length === 0 ? (
        <p className="bl-pad secondary">{lead ? `Nobody is assigned yet. Assign the people who should assess ${team.recipe.reopened != null ? "the reopened fields" : "every field"}.`
          : "A lead hasn't assigned anyone yet."}</p>
      ) : (
        <div className="table-wrap bl-table">
          <table className="data">
            <thead><tr><th scope="col">Person</th><th scope="col">Progress</th><th scope="col">Status</th>{lead && <th scope="col" aria-label="Actions" />}</tr></thead>
            <tbody>
              {team.assessors.map((a) => {
                const u = users.get(a.user_id);
                return (
                  <tr key={a.user_id}>
                    <td>
                      <span className="bl-person">
                        <Avatar user={{ id: a.user_id, name: a.name, avatar_url: u?.avatar_url }} size={28} />
                        <span>{a.name}{a.user_id === me?.id && <span className="muted"> (you)</span>}</span>
                      </span>
                    </td>
                    <td>
                      <span className="bl-person">
                        <ProgressBar value={a.reviewed} max={a.total} label={`${a.name}'s progress`} width={140}
                          color={a.submitted_at ? "var(--good)" : "var(--border-strong)"} />
                        <span className="num small-text secondary">{fmtInt(a.reviewed)} / {fmtInt(a.total)}</span>
                      </span>
                    </td>
                    <td>
                      {a.submitted_at ? <Chip tone="good" icon="check" title={fmtDate(a.submitted_at)}>Submitted {fmtAgo(a.submitted_at)}</Chip> : <Chip>Assessing</Chip>}
                    </td>
                    {lead && <td className="bl-actions"><RowMenu name={a.name} busy={unassign.isPending} onUnassign={() => askUnassign(a)} /></td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="bl-foot">You see progress, never choices, until everyone submits.</p>
    </section>
  );
}

function RowMenu({ name, busy, onUnassign }: { name: string; busy: boolean; onUnassign: () => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  return (
    <span ref={ref} className="bl-menu">
      <button type="button" className="ghost small" aria-label={`Actions for ${name}`} aria-haspopup="true" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Icon name="more" />
      </button>
      {open && (
        <div className="popover" role="menu">
          <button type="button" role="menuitem" className="menu-link danger-text" disabled={busy} onClick={() => { close(); onUnassign(); }}>Unassign {name}</button>
        </div>
      )}
    </span>
  );
}

function finalizeBody(team: TeamState, label: string) {
  const r = team.review;
  const open = r?.open ?? 0;
  return (
    <>
      {r && <p className="num">{fmtInt(r.adopted)} agreed and {fmtInt(r.disagreements - open)} settled, {fmtInt(open)} open. It freezes as a snapshot that experiments build on. A later change is a new version, v{team.recipe.version + 1}.</p>}
      {open > 0 && <p><b>{open} disagreement{open === 1 ? " is" : "s are"} still open</b> and will keep {open === 1 ? "its" : "their"} current team decision{open === 1 ? "" : "s"}.</p>}
      {!r && <p>{label}'s decisions will be frozen.</p>}
    </>
  );
}

function InReview({ ds, team }: { ds: Dataset; team: TeamState }) {
  const me = useMe().data;
  const nav = useNavigate();
  const lead = me?.role === "lead";
  const invalidate = useInvalidateTeam(ds.id);
  const users = useUserMap();
  const r = team.review;
  const short = baselineLabel(team.recipe.family, team.recipe.version, true, team.recipe.name);
  const finalize = useMutation({
    mutationFn: (acceptOpen: boolean) => api.finalize(ds.id, acceptOpen),
    onSuccess: (t) => {
      invalidate();
      toast({ tone: "good", message: `${t.recipe.label} finalized` });
      nav(`/d/${ds.id}/experiments?finalized=${t.recipe.id}`);
    },
    onError: (e) => showError("Couldn't finalize", e),
  });
  const ask = async () => {
    const ok = await confirmDialog({ title: `Finalize ${short}?`, body: finalizeBody(team, short), confirm: `Finalize ${short}`, cancel: "Not yet" });
    if (ok) finalize.mutate((r?.open ?? 0) > 0);
  };
  const direct = team.assessors.length === 0;
  const done = team.assessors.filter((a) => a.submitted_at);
  return (
    <>
      <section className="card bl-review" aria-labelledby="bl-review">
        <div className="bl-card-head">
          <h2 id="bl-review">Team review</h2>
          <Chip tone="team" icon="users">Visible to all</Chip>
        </div>
        {/* Nobody assessed a direct version, so there is nothing adopted or disputed to count. */}
        {r && !direct && (
          <dl className="bl-stats">
            <div><dt>Adopted</dt><dd className="num">{fmtInt(r.adopted)}</dd></div>
            <div><dt>Disagreements</dt><dd className="num">{fmtInt(r.disagreements)}</dd></div>
            <div><dt>Open</dt><dd className="num">{fmtInt(r.open)}</dd></div>
          </dl>
        )}
        <p className="bl-pad secondary">
          {direct ? (team.recipe.reuse ? "Every field was carried over from the reused baseline; finalize to confirm it."
            : "Nothing was reopened: change the team decision in Fields, then finalize.")
            : r && r.open > 0 ? `${lead ? "Settle" : "A lead settles"} the ${r.open} open disagreement${r.open === 1 ? "" : "s"} in the queue, where everyone can comment.`
            : lead ? "Every disagreement is settled. Finalize when you're ready." : "Every disagreement is settled; a lead finalizes."}
        </p>
        <div className="row bl-pad">
          <Link className="btn" to={`/d/${ds.id}/team`}>Open the queue <Icon name="arrow-right" size={14} /></Link>
          {lead && <button type="button" disabled={finalize.isPending} onClick={ask}>Finalize…</button>}
        </div>
      </section>
      {done.length > 0 && (
        <section className="card bl-who" aria-labelledby="bl-in">
          <div className="bl-card-head"><h2 id="bl-in">Assessors <span className="num muted">{done.length} submitted</span></h2></div>
          <ul className="bl-people">
            {done.map((a) => (
              <li key={a.user_id}>
                <Avatar user={{ id: a.user_id, name: a.name, avatar_url: users.get(a.user_id)?.avatar_url }} size={28} />
                <span className="grow">{a.name}{a.user_id === me?.id && <span className="muted"> (you)</span>}</span>
                <Chip tone="good" icon="check" title={fmtDate(a.submitted_at)}>Submitted {fmtAgo(a.submitted_at)}</Chip>
              </li>
            ))}
          </ul>
        </section>
      )}
      {team.recipe.reuse && <ReuseCard reuse={team.recipe.reuse} label={team.recipe.label ?? ""} />}
    </>
  );
}

/** "6 agreed, 2 settled in review, 1 kept as it was": disagreements left open at finalizing kept their team decision. */
function finalSummary(r: NonNullable<TeamState["review"]>): string {
  const parts = [`${fmtInt(r.adopted)} agreed`];
  if (r.disagreements > 0) parts.push(`${fmtInt(r.disagreements - r.open)} settled in review`);
  if (r.open > 0) parts.push(`${fmtInt(r.open)} kept as ${r.open === 1 ? "it was" : "they were"}`);
  return parts.join(", ");
}

function Finalized({ ds, team }: { ds: Dataset; team: TeamState }) {
  const me = useMe().data;
  const r = team.recipe;
  const lead = me?.role === "lead";
  return (
    <>
      <section className="card bl-review" aria-labelledby="bl-final">
        <div className="bl-card-head">
          <h2 id="bl-final">Finalized</h2>
          <Chip tone="frozen" icon="lock">Frozen snapshot</Chip>
        </div>
        <p className="bl-pad">
          {r.finalized_by_name ?? "A lead"} finalized {r.label} <span title={fmtDate(r.finalized_at)}>{fmtAgo(r.finalized_at)}</span>
          {team.review ? <span className="num">: {finalSummary(team.review)}.</span> : "."}
        </p>
        <p className="bl-pad secondary">Its decisions are frozen. Experiments build on it; a change is a new version.</p>
        <div className="row bl-pad">
          <Link className="btn" to={`/d/${ds.id}/experiments`}>{can(me, "createExperiment") ? "Go to Experiments" : "See its experiments"} <Icon name="arrow-right" size={14} /></Link>
          <Link className="btn ghost" to={`/d/${ds.id}/review?view=baseline:${r.id}`}>Browse its fields</Link>
          {lead && <Link className="btn ghost" to={`/d/${ds.id}/baselines?new=${r.family}`}>Start a new version</Link>}
        </div>
      </section>
      {r.reuse && <ReuseCard reuse={r.reuse} label={r.label ?? ""} />}
    </>
  );
}

/** Every baseline and version, newest first, with what each was built from and what it fed. */
function Versions({ ds, team }: { ds: Dataset; team: TeamState }) {
  const me = useMe().data;
  const lead = me?.role === "lead";
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id) });
  const experiments = useQuery({ queryKey: ["experiments", ds.id], queryFn: () => api.experiments(ds.id) });
  const cur = team.recipe;
  const inProgress = cur.phase !== "finalized";
  const busyReason = inProgress ? `One version is in progress at a time. Finalize v${cur.version} first.` : null;
  const taken = new Set((lines.data ?? []).map((l) => l.family));
  const free = FAMILIES.filter((f) => !taken.has(f));
  const byRecipe = (id: number) => (experiments.data ?? []).filter((x) => x.recipe_id === id && !x.archived_at);
  return (
    <aside className="bl-aside" aria-labelledby="bl-vers">
      <div className="bl-aside-head">
        <h2 id="bl-vers">Versions</h2>
        <Link className="btn small" to={`/d/${ds.id}/baselines`}>Compare</Link>
      </div>
      <ErrorAlert error={lines.error} />
      {(lines.data ?? []).map((l) => {
        const latest = l.versions.find((v) => v.id === l.latest_finalized_id);
        return (
          <div key={l.family} className="card bl-versions">
            {(lines.data ?? []).length > 1 && <div className="bl-line-name">{l.name ?? `${l.label} baseline`} <Chip>{familyShort(l.family)}</Chip></div>}
            {l.versions.map((v) => <VersionRow key={v.id} ds={ds} v={v} current={v.id === cur.id} experiments={byRecipe(v.id)} />)}
            {lead && (
              <div className="bl-version-foot">
                {latest && !busyReason ? (
                  <Link className="btn small" to={`/d/${ds.id}/baselines?new=${l.family}`}>New version</Link>
                ) : (
                  <button type="button" className="small" disabled title={busyReason ?? "Finalize a version first"}>New version</button>
                )}
                {busyReason && l.family === cur.family && <span className="muted small-text">{busyReason}</span>}
              </div>
            )}
          </div>
        );
      })}
      {lead && free.length > 0 && can(me, "startBaseline") && (
        <div className="card bl-other">
          <span className="bl-label">Other model types</span>
          <p className="secondary small-text">A {free.map((f) => familyShort(f).toLowerCase()).join(" or ")} baseline keeps its own versions, built from the same fields.</p>
          <div className="row" style={{ gap: 8 }}>
            {free.map((f) => busyReason ? (
              <button key={f} type="button" className="small" disabled title={busyReason}>Start a {familyShort(f).toLowerCase()} baseline</button>
            ) : (
              <Link key={f} className="btn small" to={`/d/${ds.id}/baselines?start=${f}`}>Start a {familyShort(f).toLowerCase()} baseline</Link>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}

function VersionRow({ ds, v, current, experiments }: { ds: Dataset; v: RecipeVersion; current: boolean; experiments: Experiment[] }) {
  const label = v.label ?? baselineLabel(v.family, v.version, false, v.name);
  const chosen = experiments.find((x) => x.chosen)?.chosen;
  return (
    <div className={`bl-version${current ? ` current ${v.phase}` : ""}`} aria-current={current ? "true" : undefined}>
      <span className="row" style={{ gap: 8 }}>
        {v.phase === "finalized" && !current
          ? <Link to={`/d/${ds.id}/review?view=baseline:${v.id}`} className="bl-version-name" title="Browse its fields, read-only">{label}</Link>
          : <b className="bl-version-name">{label}</b>}
        <PhaseBadge phase={v.phase} />
      </span>
      <span className="small-text secondary">
        {v.finalized_at
          ? <span title={fmtDate(v.finalized_at)}>Finalized {dateOnly(v.finalized_at)}{v.finalized_by_name ? ` by ${v.finalized_by_name}` : ""}</span>
          : <span title={fmtDate(v.created_at)}>Started {dateOnly(v.created_at)}{v.created_by_name ? ` by ${v.created_by_name}` : ""}</span>}
        {" · "}{v.reopened == null ? "all fields" : `${fmtInt(v.reopened)} reopened`} · from {builtFrom(v, true)}
      </span>
      {experiments.length > 0 && (
        <span className="small-text">
          <Link to={`/d/${ds.id}/experiments`}>{experiments.length} experiment{experiments.length === 1 ? "" : "s"}</Link>
          {chosen && <span className="muted"> · chosen model {chosen.model_label ?? chosen.model}</span>}
        </span>
      )}
    </div>
  );
}
