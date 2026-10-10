import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useAccess, useMe, useOptions, useTeam } from "../hooks";
import { useDatasetContext } from "../components/DatasetLayout";
import { DecisionPanel } from "../components/DecisionPanel";
import { ReuseCard } from "../components/ReuseCard";
import { StartBaseline } from "../components/StartBaseline";
import { Empty, ErrorAlert, StatusGlyph } from "../components/ui";
import { familyShort, fmtAgo, fmtDate, fmtInt } from "../format";
import { differing, finalizeQuestion, nextOpen, PHASE_LABEL, spread } from "../team";
import { showError, toast } from "../toast";
import type { Dataset, FieldReview, OptionCatalog, QueueItem, SettingsRow, TeamState } from "../types";

/** Everything that reads the decisions or the team state goes stale when the phase or an assessment changes. */
export function useInvalidateTeam(datasetId: number) {
  const qc = useQueryClient();
  return () => {
    for (const k of ["team", "reviewQueue", "dataset", "fields", "fieldQuery", "exportPreview", "corrMatrix", "analysis",
      "versions", "baselines", "recipeDiff", "experiments", "exports", "runs"]) {
      qc.invalidateQueries({ queryKey: [k, datasetId] });
    }
    qc.invalidateQueries({ queryKey: ["datasets"] });
    qc.invalidateQueries({ queryKey: ["finalizedRecipes"] });
    qc.invalidateQueries({ queryKey: ["field"] });
    qc.invalidateQueries({ queryKey: ["fieldReview"] });
    qc.invalidateQueries({ queryKey: ["experiment"] });
  };
}

/** The phase of a baseline version, with its short label ("Trees v2") when given. */
export function PhaseBadge({ phase, label }: { phase: Dataset["phase"]; label?: string | null }) {
  if (!phase) return null;
  const cls = phase === "finalized" ? "good" : phase === "review" ? "warn" : "accent";
  const text = label ? `${PHASE_LABEL[phase]} · ${label}` : phase === "review" ? "Review" : PHASE_LABEL[phase];
  return <span className={`badge ${cls}`} title={`Baseline ${PHASE_LABEL[phase].toLowerCase()}`}>{text}</span>;
}

export function DatasetTeamPage() {
  const ds = useDatasetContext();
  const team = useTeam(ds.id);
  if (team.error) return <div className="page"><ErrorAlert error={team.error} /></div>;
  if (!team.data) return <div className="page" aria-busy="true"><span className="skeleton" style={{ width: 240, height: 22 }} /></div>;
  return team.data.recipe.phase === "assessing" ? <Assessing ds={ds} team={team.data} /> : <Review ds={ds} team={team.data} />;
}

function Assessing({ ds, team }: { ds: Dataset; team: TeamState }) {
  const me = useMe().data;
  const r = team.recipe;
  if (r.setup) {
    return (
      <div className="page">
        <h1>Team</h1>
        {me?.role === "lead" ? <StartBaseline ds={ds} /> : (
          <div className="alert info">A lead sets up this dataset's first baseline: its model type, targets and who assesses it. Until then you can browse the fields and Explore.</div>
        )}
      </div>
    );
  }
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Team <PhaseBadge phase="assessing" label={r.label} /></h1>
        <span className="muted small-text">Review opens by itself once every assessor has submitted. <Link to={`/d/${ds.id}/baselines`}>Baselines →</Link></span>
      </div>
      {r.reopened != null && !r.reuse && (
        <div className="alert info"><span className="grow">
          {r.label} reopens {r.reopened} field{r.reopened === 1 ? "" : "s"}; only those are assessed. The rest are carried over from the version before and locked.
        </span></div>
      )}
      {r.reuse && <ReuseCard reuse={r.reuse} label={r.label ?? ""} />}
      {team.me && <MyAssessment ds={ds} me={team.me} />}
      <Assessors ds={ds} team={team} />
    </div>
  );
}

function MyAssessment({ ds, me }: { ds: Dataset; me: NonNullable<TeamState["me"]> }) {
  const invalidate = useInvalidateTeam(ds.id);
  const left = me.total - me.reviewed;
  const submit = useMutation({
    mutationFn: () => (me.submitted ? api.unsubmitAssessment(ds.id) : api.submitAssessment(ds.id)),
    onSuccess: (t) => {
      invalidate();
      toast({ message: t.recipe.phase === "review" ? "Submitted. Everyone is in: team review is open."
        : t.me?.submitted ? "Assessment submitted" : "Assessment reopened" });
    },
  });
  return (
    <section className="card stack" aria-label="Your assessment">
      <div className="card-head">
        <h2>Your assessment</h2>
        {me.submitted ? <span className="badge good">✓ submitted</span> : <span className="badge">in progress</span>}
      </div>
      <span className="secondary">Your choices are yours alone until team review opens; nobody else sees them, the lead included.</span>
      <div className="row">
        <span className="meter" style={{ width: 200 }}><span style={{ width: `${me.total ? (100 * me.reviewed) / me.total : 0}%`, background: "var(--good)" }} /></span>
        <span className="num small-text">{fmtInt(me.reviewed)} / {fmtInt(me.total)} reviewed</span>
      </div>
      <div className="row">
        {!me.submitted && <Link className="btn" to={`/d/${ds.id}/review`}>{left ? "Continue reviewing" : "Look over your choices"}</Link>}
        <button type="button" className={me.submitted ? "" : "primary"} disabled={submit.isPending || (!me.submitted && left > 0)}
          title={!me.submitted && left > 0 ? `Review every field first (${left} left)` : undefined}
          onClick={() => submit.mutate()}>
          {me.submitted ? "Unsubmit to make changes" : left > 0 ? `Submit (${left} left)` : "Submit assessment"}
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
  const users = useQuery({ queryKey: ["users"], queryFn: api.users, enabled: lead });
  const [pick, setPick] = useState("");
  const assigned = new Set(team.assessors.map((a) => a.user_id));
  const candidates = (users.data ?? []).filter((u) => u.active && !assigned.has(u.id));
  const assign = useMutation({
    mutationFn: (userId: number) => api.assign(ds.id, userId),
    onSuccess: () => { setPick(""); invalidate(); },
  });
  const unassign = useMutation({
    mutationFn: (userId: number) => api.unassign(ds.id, userId),
    onSuccess: (t) => {
      invalidate();
      if (t.recipe.phase === "review") toast({ message: "Everyone left has submitted: team review is open." });
    },
    onError: (e) => showError("Couldn't unassign", e),
  });
  return (
    <section className="card stack" aria-label="Assessors">
      <div className="card-head">
        <h2>Assessors</h2>
        <span className="muted small-text">Progress only: nobody's choices show before review.</span>
      </div>
      {team.assessors.length === 0 ? (
        <span className="secondary">{lead ? `Nobody is assigned yet. Assign the people who should assess ${team.recipe.reopened != null ? "the reopened fields" : "every field"}.`
          : "A lead hasn't assigned anyone yet."}</span>
      ) : (
        <table className="data team">
          <thead><tr><th>Assessor</th><th>Progress</th><th>Status</th>{lead && <th aria-label="Actions" />}</tr></thead>
          <tbody>
            {team.assessors.map((a) => (
              <tr key={a.user_id}>
                <td>{a.name}{a.user_id === me?.id && <span className="muted"> (you)</span>}</td>
                <td>
                  <div className="row" style={{ flexWrap: "nowrap" }}>
                    <span className="meter" style={{ width: 120 }}><span style={{ width: `${a.total ? (100 * a.reviewed) / a.total : 0}%`, background: "var(--good)" }} /></span>
                    <span className="num small-text">{fmtInt(a.reviewed)} / {fmtInt(a.total)}</span>
                  </div>
                </td>
                <td>{a.submitted_at ? <span className="badge good" title={fmtDate(a.submitted_at)}>✓ submitted {fmtAgo(a.submitted_at)}</span> : <span className="badge">assessing</span>}</td>
                {lead && (
                  <td>
                    <button type="button" className="ghost small" disabled={unassign.isPending}
                      onClick={() => window.confirm(`Unassign ${a.name}? Their choices are kept but won't count.`) && unassign.mutate(a.user_id)}>
                      Unassign
                    </button>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {lead && (
        <div className="row">
          <label className="field">Assign
            <select value={pick} onChange={(e) => setPick(e.target.value)} aria-label="User to assign">
              <option value="">Choose a person…</option>
              {candidates.map((u) => <option key={u.id} value={u.id}>{u.name} · {u.role}</option>)}
            </select>
          </label>
          <button type="button" className="primary" disabled={!pick || assign.isPending} onClick={() => assign.mutate(Number(pick))}>Assign</button>
        </div>
      )}
      <ErrorAlert error={assign.error} />
    </section>
  );
}

function Review({ ds, team }: { ds: Dataset; team: TeamState }) {
  const me = useMe().data;
  const { data: catalog } = useOptions();
  const access = useAccess(ds);
  const invalidate = useInvalidateTeam(ds.id);
  const [params, setParams] = useSearchParams();
  const queue = useQuery({ queryKey: ["reviewQueue", ds.id], queryFn: () => api.reviewQueue(ds.id) });
  const items = queue.data?.queue ?? [];
  const selected = Number(params.get("field")) || (items.find((q) => !q.resolved) ?? items[0])?.field_id || null;
  const item = items.find((q) => q.field_id === selected) ?? null;
  const select = (id: number | null) => id !== null && setParams({ field: String(id) }, { replace: true });
  const finalize = useMutation({
    mutationFn: (acceptOpen: boolean) => api.finalize(ds.id, acceptOpen),
    onSuccess: (t) => { invalidate(); toast({ message: `${t.recipe.label} finalized` }); },
    onError: (e) => showError("Couldn't finalize", e),
  });
  const summary = queue.data?.summary ?? team.review;
  const finalized = team.recipe.phase === "finalized";
  // A version nobody assesses: nothing reopened (the lead edits directly), or a reused recipe that carried everything.
  const direct = team.assessors.length === 0;
  const askFinalize = () => {
    const question = finalizeQuestion(summary?.open ?? 0, team.recipe.label ?? "this baseline");
    if (question ? window.confirm(question) : window.confirm(`Finalize ${team.recipe.label}? Its decisions will be frozen.`)) {
      finalize.mutate(Boolean(question));
    }
  };

  return (
    <div className="team-review">
      <aside className="sidebar" aria-label="Review queue">
        <div className="sidebar-head">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <b>Team review</b>
            <PhaseBadge phase={team.recipe.phase} label={team.recipe.label} />
          </div>
          {summary && (
            <span className="small-text secondary num">
              {summary.adopted} adopted · {summary.disagreements} disagreement{summary.disagreements === 1 ? "" : "s"} · <b>{summary.open} open</b>
            </span>
          )}
          {!finalized && direct && (
            <span className="small-text secondary">
              {team.recipe.reuse ? "Every field was carried over from the reused baseline; finalize to confirm it."
                : "Nothing was reopened: change the team decision in Review or Bulk edit, then finalize."}
            </span>
          )}
          {finalized ? (
            <span className="small-text secondary">
              Finalized {fmtAgo(team.recipe.finalized_at)}{team.recipe.finalized_by_name ? ` by ${team.recipe.finalized_by_name}` : ""}.{" "}
              <Link to={`/d/${ds.id}/experiments`}>Experiments →</Link>{" "}
              <Link to={`/d/${ds.id}/baselines`}>{me?.role === "lead" ? "Start a baseline or version →" : "Baselines →"}</Link>
            </span>
          ) : me?.role === "lead" ? (
            <button type="button" className={summary?.open ? "" : "primary"} disabled={finalize.isPending} onClick={askFinalize}>
              Finalize {team.recipe.label}
            </button>
          ) : (
            <span className="small-text secondary">A lead settles each disagreement and finalizes. You can comment.</span>
          )}
        </div>
        <div className="field-list" role="listbox" aria-label="Disagreements">
          {items.map((q) => (
            <div key={q.field_id} role="option" aria-selected={q.field_id === selected} tabIndex={0}
              className={`queue-item${q.field_id === selected ? " active" : ""}`}
              onClick={() => select(q.field_id)} onKeyDown={(e) => e.key === "Enter" && select(q.field_id)}>
              <StatusGlyph kind={q.resolved ? (q.team.action === "drop" ? "dropped" : q.team.action === "target" ? "target" : "kept") : "todo"} reviewed={q.resolved} />
              <span className="grow">
                <span className="name">{q.name}</span>
                <span className="muted small-text">{spread(q, (t) => catalog?.types.find((x) => x.value === t)?.label ?? t)}</span>
              </span>
              {q.comments > 0 && <span className="badge" title={`${q.comments} comment${q.comments === 1 ? "" : "s"}`}>{q.comments}</span>}
            </div>
          ))}
          {queue.data && items.length === 0 && <Empty title="No disagreements">Every field was adopted as the assessors agreed.</Empty>}
        </div>
      </aside>
      <section className="detail">
        <ErrorAlert error={queue.error} />
        {item && catalog ? (
          <Comparison item={item} family={ds.model_family} catalog={catalog} canComment={!finalized} />
        ) : queue.data && items.length > 0 ? null : team.recipe.reuse ? (
          <ReuseCard reuse={team.recipe.reuse} label={team.recipe.label ?? ""} />
        ) : (
          queue.data && <Empty title={finalized ? "Finalized" : "Nothing to settle"}>
            {finalized ? "This baseline version is frozen. A change is a new version, started on the Baselines page."
              : direct ? "Nobody assessed this version. Change the team decision in Review or Bulk edit, then finalize."
              : "Finalize when you're ready; every field took the assessors' shared choice."}
          </Empty>
        )}
      </section>
      {item && catalog ? (
        <TeamDecision key={item.field_id} item={item} ds={ds} catalog={catalog} readOnly={access.canEdit ? null : access.note}
          onSettled={() => select(nextOpen(items, item.field_id))} />
      ) : <aside className="decision" />}
    </div>
  );
}

const DIMS = ["imputation", "transform", "scaling", "encoding"] as const;

function Comparison({ item, family, catalog, canComment }: { item: QueueItem; family: string; catalog: OptionCatalog; canComment: boolean }) {
  const qc = useQueryClient();
  const review = useQuery({ queryKey: ["fieldReview", item.field_id], queryFn: () => api.fieldReview(item.field_id), placeholderData: keepPreviousData });
  const [body, setBody] = useState("");
  const comment = useMutation({
    mutationFn: () => api.addComment(item.field_id, body.trim()),
    onSuccess: (comments) => {
      setBody("");
      qc.setQueryData<FieldReview>(["fieldReview", item.field_id], (old) => (old ? { ...old, comments } : old));
      qc.invalidateQueries({ queryKey: ["reviewQueue"] });
    },
  });
  const typeLabel = (t: string) => catalog.types.find((x) => x.value === t)?.label ?? t;
  const rows = (review.data?.assessments ?? []).map((a) => ({ ...a, s: a.settings[family] as SettingsRow | undefined }));
  const flat: Record<string, unknown>[] = rows.map((r) => ({
    action: r.action, semantic_type: r.semantic_type, impute_value: r.s?.impute_value,
    add_missing_indicator: r.s?.add_missing_indicator, ...Object.fromEntries(DIMS.map((d) => [d, r.s?.[d]])),
  }));
  const diff = differing(flat, ["action", "semantic_type", ...DIMS, "impute_value", "add_missing_indicator"]);
  const line = (key: string, label: string, render: (r: (typeof rows)[number]) => string | null | undefined) => (
    <tr className={diff.has(key) ? "differs" : ""}>
      <th scope="row">{label}</th>
      {rows.map((r) => <td key={r.user_id}>{render(r) ?? <span className="muted">—</span>}</td>)}
    </tr>
  );
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="field-head">
        <div className="field-title"><h1>{item.name}</h1><span className="pos">inferred {typeLabel(item.inferred_type)}</span></div>
        <span className="field-meta">Each assessor's choice, side by side. Rows where they differ are marked. Settings are for {familyShort(family).toLowerCase()} models.</span>
      </div>
      <ErrorAlert error={review.error} />
      {rows.length > 0 && (
        <div className="table-wrap" style={{ maxHeight: "none" }}>
          <table className="data comparison">
            <thead><tr><th aria-label="Setting" />{rows.map((r) => <th key={r.user_id}>{r.user_name}</th>)}</tr></thead>
            <tbody>
              {line("action", "Decision", (r) => r.action)}
              {line("semantic_type", "Type", (r) => typeLabel(r.semantic_type))}
              {DIMS.map((d) => line(d, { imputation: "Missing values", transform: "Transform", scaling: "Scaling", encoding: "Encoding" }[d],
                (r) => (r.action === "keep" && r.s ? (catalog.labels[d][r.s[d]] ?? r.s[d]).split(" (")[0] : null)))}
              {line("impute_value", "Fill value", (r) => (r.s?.imputation === "constant" ? r.s.impute_value : null))}
              {line("add_missing_indicator", "Missing flag", (r) => (r.s?.add_missing_indicator ? "yes" : null))}
              {line("notes", "Notes", (r) => r.notes)}
            </tbody>
          </table>
        </div>
      )}
      <section className="stack" aria-label="Comments">
        <h3>Comments</h3>
        {(review.data?.comments ?? []).length === 0 && <span className="muted small-text">No comments yet.</span>}
        {(review.data?.comments ?? []).map((c) => (
          <div key={c.id} className="comment">
            <span className="small-text"><b>{c.user_name}</b> <span className="muted" title={fmtDate(c.created_at)}>{fmtAgo(c.created_at)}</span></span>
            <span style={{ whiteSpace: "pre-wrap" }}>{c.body}</span>
          </div>
        ))}
        {canComment && (
          <form className="stack" onSubmit={(e) => { e.preventDefault(); if (body.trim()) comment.mutate(); }}>
            <textarea value={body} onChange={(e) => setBody(e.target.value)} placeholder="Add to the discussion…" aria-label="Comment" maxLength={4000} />
            <div><button type="submit" disabled={!body.trim() || comment.isPending}>Comment</button></div>
            <ErrorAlert error={comment.error} />
          </form>
        )}
      </section>
    </div>
  );
}

function TeamDecision({ item, ds, catalog, readOnly, onSettled }: {
  item: QueueItem; ds: Dataset; catalog: OptionCatalog; readOnly: string | null; onSettled: () => void;
}) {
  const field = useQuery({ queryKey: ["field", item.field_id, "current"], queryFn: () => api.field(item.field_id) });
  if (!field.data) return <aside className="decision"><ErrorAlert error={field.error} /></aside>;
  return (
    <DecisionPanel data={field.data} catalog={catalog} familyLabel={familyShort(ds.model_family)} typing={false}
      owner="Team decision" readOnly={readOnly} onShowShortcuts={() => undefined}
      onConfirmed={() => { toast({ message: `${item.name}: team decision set` }); onSettled(); }} />
  );
}
