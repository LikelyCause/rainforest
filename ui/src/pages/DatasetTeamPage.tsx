import { forwardRef, useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useSearchParams } from "react-router-dom";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { can } from "../can";
import { useAccess, useFields, useMe, useOptions, useTeam } from "../hooks";
import { Avatar } from "../components/Avatar";
import { useDatasetContext } from "../components/DatasetLayout";
import { DecisionPanel, type DecisionPanelHandle, type VoteKey } from "../components/DecisionPanel";
import { FieldDetailView } from "../components/FieldDetailView";
import { Chip, Icon, Segmented } from "../components/kit";
import { ReuseCard } from "../components/ReuseCard";
import { Empty, ErrorAlert, StatusGlyph } from "../components/ui";
import { bandsText } from "../decision";
import { familyShort, fmtAgo, fmtDate, fmtInt } from "../format";
import { nextOpen, oddOnes, PHASE_LABEL, splitAgreement, spread, voteLine } from "../team";
import { toast } from "../toast";
import type { Dataset, FieldListItem, FieldResponse, FieldReview, OptionCatalog, QueueItem, SettingsRow, TeamState, TeamUser } from "../types";

export { useInvalidateTeam } from "../hooks";

/** The phase of a baseline version, with its short label ("Trees v2") when given; a long one is cut, in full on hover. */
export function PhaseBadge({ phase, label }: { phase: Dataset["phase"]; label?: string | null }) {
  if (!phase) return null;
  const tone = phase === "finalized" ? "good" : phase === "review" ? "team" : "private";
  const text = label ? `${PHASE_LABEL[phase]} · ${label}` : PHASE_LABEL[phase];
  return (
    <span className={`badge ${tone}`} title={label ? text : `Baseline ${PHASE_LABEL[phase].toLowerCase()}`}><span className="clip">{text}</span></span>
  );
}

/** The team review queue (/d/:id/team). While a version is set up or assessed, its place is the Baseline tab. */
export function DatasetTeamPage() {
  const ds = useDatasetContext();
  const team = useTeam(ds.id);
  if (team.error) return <div className="page"><ErrorAlert error={team.error} /></div>;
  if (!team.data) return <div className="page" aria-busy="true"><span className="skeleton" style={{ width: 240, height: 22 }} /></div>;
  if (team.data.recipe.phase === "assessing") return <Navigate to={`/d/${ds.id}/baseline`} replace />;
  return <Review ds={ds} team={team.data} />;
}

type Show = "open" | "settled" | "agreed";

/** Users by id, for avatars; everyone may list them. */
export function useUserMap(): Map<number, TeamUser> {
  const users = useQuery({ queryKey: ["users"], queryFn: api.users, staleTime: 60_000 });
  return useMemo(() => new Map((users.data ?? []).map((u) => [u.id, u])), [users.data]);
}

function Review({ ds, team }: { ds: Dataset; team: TeamState }) {
  const me = useMe().data;
  const { data: catalog } = useOptions();
  const access = useAccess(ds);
  const users = useUserMap();
  const [params, setParams] = useSearchParams();
  const queue = useQuery({ queryKey: ["reviewQueue", ds.id], queryFn: () => api.reviewQueue(ds.id) });
  const fields = useFields(ds.id);
  const panel = useRef<DecisionPanelHandle>(null);
  const items = queue.data?.queue ?? [];
  const open = items.filter((q) => !q.resolved);
  const settled = items.filter((q) => q.resolved);
  const inQueue = new Set(items.map((q) => q.field_id));
  // The queue lists only disagreements; agreed fields are the rest this version worked on, opened in Fields · Focus.
  const agreed = (fields.data ?? []).filter((f) => !inQueue.has(f.id) && !f.locked);
  const summary = queue.data?.summary ?? team.review;
  const finalized = team.recipe.phase === "finalized";
  const lead = me?.role === "lead";
  // A version nobody assesses: nothing reopened (the lead edits directly), or a reused recipe that carried everything.
  const direct = team.assessors.length === 0;

  const show: Show = (params.get("show") as Show | null) ?? (open.length ? "open" : settled.length ? "settled" : "agreed");
  const list = show === "open" ? open : show === "settled" ? settled : [];
  const selected = Number(params.get("field")) || (list[0] ?? open[0] ?? items[0])?.field_id || null;
  const item = items.find((q) => q.field_id === selected) ?? null;
  const setParam = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) if (v === null) next.delete(k); else next.set(k, v);
    setParams(next, { replace: true });
  };
  const select = (id: number | null) => id !== null && setParam({ field: String(id) });
  const at = list.findIndex((q) => q.field_id === selected);
  const step = (d: 1 | -1) => {
    if (!list.length) return;
    const i = at < 0 ? 0 : (at + d + list.length) % list.length;
    select(list[i].field_id);
  };

  const field = useQuery({ queryKey: ["field", item?.field_id, "current"], queryFn: () => api.field(item!.field_id), enabled: item !== null });
  const review = useQuery({
    queryKey: ["fieldReview", item?.field_id], queryFn: () => api.fieldReview(item!.field_id), enabled: item !== null, placeholderData: keepPreviousData,
  });

  // J and K walk the list shown; Enter settles (leads), as in Focus.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable || document.querySelector(".overlay")) return;
      if (e.key === "j") { e.preventDefault(); step(1); }
      else if (e.key === "k") { e.preventDefault(); step(-1); }
      else if (e.key === "Enter" && access.canEdit && item && !el.closest("button, a, summary, [role=option]")) { e.preventDefault(); panel.current?.confirm(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const typeLabel = (t: string) => catalog?.types.find((x) => x.value === t)?.label ?? t;
  const position = show === "agreed" || at < 0 ? null : `${show} ${at + 1} of ${list.length}`;

  return (
    <div className="team-review tr">
      <aside className="sidebar tr-queue" aria-label="Review queue">
        <div className="sidebar-head">
          <Segmented<Show> label="Show" pressed full size="small" value={show}
            onChange={(v) => setParam({ show: v, field: v === "open" ? String(open[0]?.field_id ?? "") || null : v === "settled" ? String(settled[0]?.field_id ?? "") || null : params.get("field") })}
            options={[
              { value: "open", label: <>Open <span className="num">{fmtInt(open.length)}</span></> },
              { value: "settled", label: <>Settled <span className="num">{fmtInt(settled.length)}</span></> },
              { value: "agreed", label: <>Agreed <span className="num">{fmtInt(summary?.adopted ?? agreed.length)}</span></> },
            ]} />
          {!finalized && direct && (
            <span className="small-text secondary">
              {team.recipe.reuse ? "Every field was carried over from the reused baseline; finalize to confirm it."
                : "Nothing was reopened: change the team decision in Fields, then finalize."}
            </span>
          )}
          {finalized ? (
            <span className="small-text secondary">
              Finalized {fmtAgo(team.recipe.finalized_at)}{team.recipe.finalized_by_name ? ` by ${team.recipe.finalized_by_name}` : ""}.{" "}
              <Link to={`/d/${ds.id}/experiments`}>Experiments</Link> · <Link to={`/d/${ds.id}/baseline`}>{lead ? "Start a new version" : "Baseline"}</Link>
            </span>
          ) : !lead && (
            <span className="small-text secondary">A lead settles each disagreement and finalizes. You can comment.</span>
          )}
        </div>
        {show === "agreed" ? (
          <nav className="field-list tr-list" aria-label="Agreed fields">
            {agreed.map((f) => <AgreedRow key={f.id} ds={ds} f={f} typeLabel={typeLabel} />)}
            {fields.data && agreed.length === 0 && <span className="muted small-text tr-pad">No field was agreed on by everyone.</span>}
          </nav>
        ) : (
          <div className="field-list tr-list" role="listbox" aria-label={show === "open" ? "Open disagreements" : "Settled disagreements"}>
            {list.map((q) => (
              <div key={q.field_id} role="option" aria-selected={q.field_id === selected} tabIndex={0}
                className={`queue-item${q.field_id === selected ? " active" : ""}`}
                onClick={() => select(q.field_id)} onKeyDown={(e) => e.key === "Enter" && select(q.field_id)}>
                <StatusGlyph kind={q.resolved ? (q.team.action === "drop" ? "dropped" : q.team.action === "target" ? "target" : "kept") : "todo"} reviewed={q.resolved} />
                <span className="grow">
                  <span className="name">{q.name}</span>
                  <span className="muted small-text">{spread(q, typeLabel)}</span>
                </span>
                {q.comments > 0 && (
                  <Chip icon="comment" title={`${q.comments} comment${q.comments === 1 ? "" : "s"}`}><span className="num">{q.comments}</span></Chip>
                )}
              </div>
            ))}
            {queue.data && list.length === 0 && (
              <span className="muted small-text tr-pad">
                {items.length === 0 ? "No disagreements: every field was adopted as the assessors agreed."
                  : show === "open" ? "Nothing left open. Every disagreement is settled." : "Nothing settled yet."}
              </span>
            )}
          </div>
        )}
        <p className="tr-foot muted small-text">
          {finalized ? "Agreed fields were adopted as chosen." : "Agreed fields are adopted as chosen. Open any of them from Agreed to change it before you finalize."}
        </p>
      </aside>

      <section className="detail tr-main">
        <ErrorAlert error={queue.error ?? fields.error} />
        {item && catalog ? (
          <>
            <header className="tr-head">
              <div className="stack" style={{ gap: 4, minWidth: 0 }}>
                <h1 className="tr-name">{item.name}</h1>
                <p className="secondary num small-text">
                  Inferred as {typeLabel(item.inferred_type).toLowerCase()}
                  {field.data && field.data.field.id === item.field_id && <> · {fmtInt(field.data.field.n_unique)} distinct · {fmtInt(field.data.field.row_count)} rows</>}
                  {position && <> · {position}</>}
                  {" · "}settings for {familyShort(ds.model_family).toLowerCase()} models
                </p>
              </div>
              {list.length > 1 && show !== "agreed" && (
                <div className="row tr-nav">
                  <button type="button" className="small" onClick={() => step(-1)}>Previous <kbd>K</kbd></button>
                  <button type="button" className="small" onClick={() => step(1)}>Next <kbd>J</kbd></button>
                </div>
              )}
            </header>
            <Comparison item={item} review={review.data} error={review.error} family={ds.model_family} catalog={catalog} users={users} meId={me?.id} />
            <Discussion item={item} review={review.data} users={users} canComment={!finalized && can(me, "comment")} />
            <FieldProfile ds={ds} id={item.field_id} data={field.data} />
          </>
        ) : queue.data && items.length > 0 ? null : team.recipe.reuse ? (
          <ReuseCard reuse={team.recipe.reuse} label={team.recipe.label ?? ""} />
        ) : (
          queue.data && <Empty title={finalized ? "Finalized" : "Nothing to settle"} action={finalized ? (
            <Link className="btn primary" to={`/d/${ds.id}/experiments`}>
              {can(me, "createExperiment") ? "Start experimenting" : "See its experiments"}
            </Link>
          ) : undefined}>
            {finalized ? "This baseline version is frozen: train models on it under Experiments. A change is a new version, started on the Baseline tab."
              : direct ? "Nobody assessed this version. Change the team decision in Fields, then finalize."
              : "Finalize when you're ready; every field took the assessors' shared choice."}
          </Empty>
        )}
      </section>

      {item && catalog ? (
        <TeamDecision key={item.field_id} ref={panel} item={item} ds={ds} catalog={catalog} data={field.data} error={field.error}
          review={review.data} readOnly={access.canEdit ? null : access.note}
          onSettled={() => select(nextOpen(items, item.field_id))} />
      ) : <aside className="decision" aria-label="Team decision" />}
    </div>
  );
}

function AgreedRow({ ds, f, typeLabel }: { ds: Dataset; f: FieldListItem; typeLabel: (t: string) => string }) {
  return (
    <Link className="queue-item" to={`/d/${ds.id}/review/${f.id}`} title={`Open ${f.name} in Fields · Focus to change it`}>
      <StatusGlyph kind={f.action === "drop" ? "dropped" : f.action === "target" ? "target" : "kept"} reviewed />
      <span className="grow">
        <span className="name">{f.name}</span>
        <span className="muted small-text">All chose {f.action}, {typeLabel(f.semantic_type).toLowerCase()}</span>
      </span>
      <Icon name="chevron-right" />
    </Link>
  );
}

const DIMS = ["imputation", "transform", "scaling", "encoding"] as const;
const DIM_LABEL = { imputation: "Missing values", transform: "Transform", scaling: "Scaling", encoding: "Encoding" } as const;

interface Row { key: string; label: string; values: (string | null)[]; agree?: (v: string) => string }

/** Each part of the decision, one value per assessor, in the order the comparison shows them. */
function compareRows(review: FieldReview, family: string, catalog: OptionCatalog): Row[] {
  const typeLabel = (t: string) => catalog.types.find((x) => x.value === t)?.label ?? t;
  const rows = review.assessments.map((a) => ({ ...a, s: a.settings[family] as SettingsRow | undefined }));
  const dim = (r: (typeof rows)[number], d: (typeof DIMS)[number]) =>
    (r.action === "keep" && r.s ? (catalog.labels[d][r.s[d]] ?? r.s[d]).split(" (")[0] : null);
  const out: Row[] = [
    { key: "action", label: "Decision", values: rows.map((r) => r.action), agree: (v) => v },
    { key: "semantic_type", label: "Type", values: rows.map((r) => typeLabel(r.semantic_type)), agree: (v) => `type ${v.toLowerCase()}` },
  ];
  if (rows.some((r) => r.action === "target")) {
    out.push({ key: "target", label: "Predict as", values: rows.map((r) => bandsText(r.target_task, r.target_bins) ?? null), agree: (v) => `predicted as ${v.toLowerCase()}` });
  }
  for (const d of DIMS) out.push({ key: d, label: DIM_LABEL[d], values: rows.map((r) => dim(r, d)), agree: (v) => `${DIM_LABEL[d].toLowerCase()} ${v.toLowerCase()}` });
  out.push({ key: "impute_value", label: "Fill value", values: rows.map((r) => (r.s?.imputation === "constant" ? r.s.impute_value : null)), agree: (v) => `fill ${v}` });
  out.push({ key: "add_missing_indicator", label: "Missing flag", values: rows.map((r) => (r.action === "keep" && r.s ? (r.s.add_missing_indicator ? "Yes" : "No") : null)),
    agree: (v) => (v === "Yes" ? "missing flag" : "no missing flag") });
  out.push({ key: "notes", label: "Note", values: rows.map((r) => r.notes || null) });
  return out;
}

function Comparison({ item, review, error, family, catalog, users, meId }: {
  item: QueueItem; review: FieldReview | undefined; error: unknown; family: string; catalog: OptionCatalog; users: Map<number, TeamUser>; meId?: number;
}) {
  if (!review) return <ErrorAlert error={error} />;
  const people = review.assessments.map((a) => ({ id: a.user_id, name: a.user_name, avatar_url: users.get(a.user_id)?.avatar_url }));
  const { differ, agree } = splitAgreement(compareRows(review, family, catalog));
  const shared = agree.filter((r) => r.agree && r.values[0]).map((r) => r.agree!(r.values[0]!));
  const who = (i: number) => (people[i].id === meId ? "you" : people[i].name.split(" ")[0]);
  return (
    <section className="card tr-compare" aria-labelledby="tr-diff">
      <div className="tr-card-head">
        <h2 id="tr-diff">Where the assessors differ</h2>
        <span className="muted small-text">Only differing settings are shown</span>
      </div>
      <ErrorAlert error={error} />
      {differ.length === 0 ? (
        <p className="tr-pad secondary">The assessors' choices on {item.name} match now.</p>
      ) : (
        <>
          <div className="table-wrap tr-table">
            <table className="data comparison">
              <thead>
                <tr>
                  <th scope="col">Setting</th>
                  {people.map((p) => (
                    <th key={p.id} scope="col"><span className="tr-person"><Avatar user={p} size={20} />{p.name}{p.id === meId && <span className="muted"> (you)</span>}</span></th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {differ.map((r) => {
                  const odd = oddOnes(r.values);
                  return (
                    <tr key={r.key}>
                      <th scope="row">{r.label}</th>
                      {r.values.map((v, i) => (
                        <td key={people[i].id} className={odd.has(i) ? "odd" : undefined}>{v ?? <span className="muted">—</span>}</td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {/* On a phone: each differing setting as stacked rows, grouped by value. */}
          <div className="tr-stacked">
            {differ.map((r) => {
              const odd = oddOnes(r.values);
              const groups = new Map<string, number[]>();
              r.values.forEach((v, i) => groups.set(v ?? "—", [...(groups.get(v ?? "—") ?? []), i]));
              return (
                <div key={r.key} className="tr-stack-row">
                  <span className="secondary small-text">{r.label}</span>
                  {[...groups].map(([v, at]) => (
                    <span key={v} className={at.some((i) => odd.has(i)) ? "odd-text" : undefined}><b>{v}</b> · {at.map(who).join(", ")}</span>
                  ))}
                </div>
              );
            })}
          </div>
        </>
      )}
      {shared.length > 0 && <p className="tr-agree"><b>All agree:</b> {shared.join(" · ")}</p>}
    </section>
  );
}

function Discussion({ item, review, users, canComment }: { item: QueueItem; review: FieldReview | undefined; users: Map<number, TeamUser>; canComment: boolean }) {
  const qc = useQueryClient();
  const me = useMe().data;
  const [body, setBody] = useState("");
  const comment = useMutation({
    mutationFn: () => api.addComment(item.field_id, body.trim()),
    onSuccess: (comments) => {
      setBody("");
      qc.setQueryData<FieldReview>(["fieldReview", item.field_id], (old) => (old ? { ...old, comments } : old));
      qc.invalidateQueries({ queryKey: ["reviewQueue"] });
    },
  });
  const send = () => { if (body.trim() && !comment.isPending) comment.mutate(); };
  const comments = review?.comments ?? [];
  const mac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
  return (
    <section className="card tr-discussion" aria-labelledby="tr-talk">
      <h2 id="tr-talk">Discussion <span className="num muted">{comments.length || ""}</span></h2>
      {comments.length === 0 && <span className="muted small-text">No comments yet.</span>}
      {comments.map((c) => (
        <div key={c.id} className="tr-comment">
          <Avatar user={{ id: c.user_id, name: c.user_name, avatar_url: users.get(c.user_id)?.avatar_url }} size={28} />
          <div className="stack" style={{ gap: 2, minWidth: 0 }}>
            <span className="small-text"><b>{c.user_name}</b> <span className="muted" title={fmtDate(c.created_at)}>· {fmtAgo(c.created_at)}</span></span>
            <p className="tr-body">{c.body}</p>
          </div>
        </div>
      ))}
      {canComment && me && (
        <form className="tr-composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <span className="tr-composer-avatar"><Avatar user={me} size={28} /></span>
          <div className="stack grow" style={{ gap: 8, minWidth: 0 }}>
            <textarea rows={2} value={body} onChange={(e) => setBody(e.target.value)} placeholder="Add to the discussion" aria-label="Comment"
              maxLength={4000} onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); } }} />
            <div className="row tr-composer-foot">
              <span className="muted small-text tr-hint"><kbd>{mac ? "⌘" : "Ctrl"}</kbd> <kbd>↵</kbd> to send</span>
              <button type="submit" className="small" disabled={!body.trim() || comment.isPending}>Comment</button>
            </div>
            <ErrorAlert error={comment.error} />
          </div>
        </form>
      )}
    </section>
  );
}

/** The field's distribution and associations, as Focus shows them, folded away until asked for. */
function FieldProfile({ ds, id, data }: { ds: Dataset; id: number; data: FieldResponse | undefined }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="card tr-profile" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary><b>Field profile</b> <span className="muted">· distribution and associations, as in Focus</span></summary>
      {open && (
        <div className="stack tr-profile-body">
          <Link to={`/d/${ds.id}/review/${id}`} className="small-text">Open in Fields · Focus</Link>
          {data && data.field.id === id ? <FieldDetailView data={data} position={null} stale={false} onOpenField={() => undefined} />
            : <span className="skeleton" style={{ width: 240, height: 22 }} />}
        </div>
      )}
    </details>
  );
}

/** How the assessors voted on each part of the decision the panel shows, against the team decision now. */
function votesFor(review: FieldReview | undefined, data: FieldResponse, catalog: OptionCatalog): Partial<Record<VoteKey, string>> {
  if (!review) return {};
  const as = review.assessments;
  const names = as.map((a) => a.user_name);
  const typeLabel = (t: string) => (catalog.types.find((x) => x.value === t)?.label ?? t).toLowerCase();
  const settings = (a: (typeof as)[number]) => (a.action === "keep" ? (a.settings[data.family] as SettingsRow | undefined) : undefined);
  const d = data.decision;
  const out: Partial<Record<VoteKey, string>> = {};
  const put = (k: VoteKey, line: string | null) => { if (line) out[k] = line; };
  put("action", voteLine(as.map((a) => a.action), names, d.action));
  put("semantic_type", voteLine(as.map((a) => typeLabel(a.semantic_type)), names, typeLabel(d.semantic_type)));
  if (as.some((a) => a.action === "target")) {
    const task = (t: Parameters<typeof bandsText>[0], b: Parameters<typeof bandsText>[1]) => bandsText(t, b)?.toLowerCase() ?? null;
    put("target", voteLine(as.map((a) => (a.action === "target" ? task(a.target_task, a.target_bins) : null)), names, task(d.target_task, d.target_bins)));
  }
  for (const k of DIMS) {
    const label = (v: string) => (catalog.labels[k][v] ?? v).split(" (")[0].toLowerCase();
    put(k, voteLine(as.map((a) => { const s = settings(a); return s ? label(s[k]) : null; }), names, label(d[k])));
  }
  put("impute_value", voteLine(as.map((a) => { const s = settings(a); return s?.imputation === "constant" ? s.impute_value : null; }), names, d.impute_value));
  put("add_missing_indicator", voteLine(as.map((a) => { const s = settings(a); return s ? (s.add_missing_indicator ? "flag" : "no flag") : null; }),
    names, d.add_missing_indicator ? "flag" : "no flag"));
  return out;
}

const TeamDecision = forwardRef<DecisionPanelHandle, {
  item: QueueItem; ds: Dataset; catalog: OptionCatalog; data: FieldResponse | undefined; error: unknown; review: FieldReview | undefined;
  readOnly: string | null; onSettled: () => void;
}>(function TeamDecision({ item, ds, catalog, data, error, review, readOnly, onSettled }, ref) {
  if (!data || data.field.id !== item.field_id) return <aside className="decision" aria-label="Team decision"><ErrorAlert error={error} /></aside>;
  return (
    <DecisionPanel ref={ref} data={data} catalog={catalog} familyLabel={familyShort(ds.model_family)} typing={false}
      owner="Team decision" readOnly={readOnly} onShowShortcuts={() => undefined}
      lens="team" votes={votesFor(review, data, catalog)} settleLabel
      onConfirmed={() => { toast({ message: `${item.name}: team decision set` }); onSettled(); }} />
  );
});
