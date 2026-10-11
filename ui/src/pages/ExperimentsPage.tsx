import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { can } from "../can";
import { useMe, useTeam } from "../hooks";
import { useDatasetContext } from "../components/DatasetLayout";
import { ErrorAlert } from "../components/ui";
import { ExperimentWorkspace } from "./ExperimentWorkspace";
import {
  addBlocked, canEdit, filterExperiments, metricLabel, mlflowExperimentUrl, privacyNote, sourceLabel, sourceOptions,
  type ExperimentFilter,
} from "../experiments";
import { baselineLabel, familyShort, fmtAgo, fmtDate, fmtNum } from "../format";
import { fmtChange } from "../recipes";
import { withView } from "../view";
import { toast } from "../toast";
import type { Dataset, Experiment, ExperimentDetail, Me } from "../types";

const FILTERS: { value: ExperimentFilter; label: string }[] = [
  { value: "all", label: "All" }, { value: "mine", label: "Mine" }, { value: "team", label: "Team" },
  { value: "archived", label: "Archived" },
];

export function PrivateBadge({ x }: { x: Experiment }) {
  if (!x.private) return null;
  return <span className="badge accent" title={privacyNote(x) ?? undefined}>🔒 Private</span>;
}

/** How a team experiment differs from its baseline, in a badge or two. */
export function AdjustedBadges({ x }: { x: Experiment }) {
  return (
    <>
      {(x.adjusted_count ?? 0) > 0 && <span className="badge" title="Fields that differ from the baseline">{x.adjusted_count} adjusted</span>}
      {x.targets_changed && <span className="badge warn" title="Its targets differ from the baseline's, so its results aren't comparable with experiments on the baseline's targets">targets changed</span>}
    </>
  );
}

function CreateDialog({ ds, me, onClose }: { ds: Dataset; me: Me; onClose: () => void }) {
  const qc = useQueryClient();
  const nav = useNavigate();
  const versions = useQuery({ queryKey: ["versions", ds.id], queryFn: () => api.versions(ds.id) });
  const team = useTeam(ds.id);
  const options = sourceOptions(versions.data ?? [], team.data, me);
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  useEffect(() => {
    if (!source && options.length) setSource(options[0].value);
  }, [source, options]);
  const create = useMutation({
    mutationFn: () => api.createExperiment(ds.id, { name: name.trim(), source }),
    onSuccess: (x) => {
      qc.invalidateQueries({ queryKey: ["experiments", ds.id] });
      toast({ message: `Created ${x.name}` });
      nav(`/d/${ds.id}/experiments/${x.id}`);
    },
  });
  const loading = versions.isPending || team.isPending;
  const personal = source.startsWith("assessment:");
  return (
    <div className="overlay" onClick={onClose}>
      <form className="dialog" role="dialog" aria-modal="true" aria-label="New experiment" onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.key === "Escape" && onClose()}
        onSubmit={(e) => { e.preventDefault(); if (name.trim() && source) create.mutate(); }}>
        <h2>New experiment</h2>
        <label className="field">Name
          <input autoFocus value={name} maxLength={200} onChange={(e) => setName(e.target.value)} placeholder="e.g. Gradient boosting, no leaky fields" />
        </label>
        <label className="field">Built from
          <select value={source} onChange={(e) => setSource(e.target.value)} disabled={!options.length}>
            {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>
        {!loading && !options.length && (
          <div className="alert warn">Nothing to experiment on yet: no baseline is finalized, and you aren't assessing the current one.</div>
        )}
        {personal && (
          <span className="secondary small-text">
            Exports use your own choices as they stand when you export. Only you see this experiment until {team.data?.recipe.label} reaches team review; its runs are logged to MLflow then.
          </span>
        )}
        <ErrorAlert error={create.error} />
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary" disabled={!name.trim() || !source || create.isPending}>Create</button>
        </div>
      </form>
    </div>
  );
}

export function ExperimentsPage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const [filter, setFilter] = useState<ExperimentFilter>("all");
  const [creating, setCreating] = useState(false);
  const list = useQuery({ queryKey: ["experiments", ds.id], queryFn: () => api.experiments(ds.id) });
  if (!me) return null;
  const shown = filterExperiments(list.data ?? [], filter, me);
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Experiments</h1>
        {can(me, "createExperiment") && <button type="button" className="primary" onClick={() => setCreating(true)}>New experiment</button>}
      </div>
      <p className="secondary" style={{ margin: 0 }}>
        Each experiment holds exports and the models trained on them. Build one from a finalized baseline, or from your own assessment while you assess; that one stays private to you until team review opens.
      </p>
      <section className="card stack" style={{ gap: 10 }}>
        <div className="segmented soft" role="radiogroup" aria-label="Show" style={{ alignSelf: "flex-start" }}>
          {FILTERS.map((f) => (
            <button key={f.value} type="button" role="radio" aria-checked={filter === f.value} className={filter === f.value ? "on" : ""}
              onClick={() => setFilter(f.value)}>{f.label}</button>
          ))}
        </div>
        <ErrorAlert error={list.error} />
        {list.data && (shown.length ? (
          <div className="table-wrap">
            <table className="data experiments">
              <thead>
                <tr><th>Name</th><th>Built from</th><th>Owner</th><th className="num">Exports</th><th className="num">Runs</th><th className="num">Best (validation)</th><th>Created</th></tr>
              </thead>
              <tbody>
                {shown.map((x) => (
                  <tr key={x.id}>
                    <td>
                      <Link to={`/d/${ds.id}/experiments/${x.id}`}><b>{x.name}</b></Link>{" "}
                      <PrivateBadge x={x} /><AdjustedBadges x={x} />{x.archived_at && <span className="badge">archived</span>}
                    </td>
                    <td className="secondary">{sourceLabel(x, me)}</td>
                    <td className="secondary">{x.owner_name ?? <span className="muted">team</span>}</td>
                    <td className="num">{x.export_count ?? 0}</td>
                    <td className="num">{x.run_count ?? 0}</td>
                    <td className="num" title={x.best ? `Run #${x.best.run_id} · ${x.best.target}` : undefined}>
                      {x.best ? `${fmtNum(x.best.value, 3)} ${metricLabel(x.best.metric)}` : <span className="muted">—</span>}
                    </td>
                    <td className="muted small-text" title={fmtDate(x.created_at)}>{fmtAgo(x.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">
            {filter === "archived" ? "No archived experiments." : list.data.length ? "None here." : "No experiments yet. Create one to export and train."}
          </div>
        ))}
      </section>
      {creating && <CreateDialog ds={ds} me={me} onClose={() => setCreating(false)} />}
    </div>
  );
}

function Header({ ds, x, me }: { ds: Dataset; x: Experiment; me: Me }) {
  const qc = useQueryClient();
  const editable = canEdit(x, me);
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(x.name);
  const [notes, setNotes] = useState(x.notes ?? "");
  useEffect(() => { setName(x.name); setNotes(x.notes ?? ""); }, [x.name, x.notes]);
  const patch = useMutation({
    mutationFn: (data: { name?: string; notes?: string | null; archived?: boolean }) => api.patchExperiment(x.id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiment", x.id] });
      qc.invalidateQueries({ queryKey: ["experiments", ds.id] });
      setRenaming(false);
    },
  });
  const mlflow = mlflowExperimentUrl(x.mlflow_experiment_id, x.id);
  const note = privacyNote(x);
  return (
    <section className="card stack" style={{ gap: 10 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="row" style={{ gap: 8 }}>
          <Link to={`/d/${ds.id}/experiments`} className="muted">Experiments</Link>
          <span className="muted">/</span>
          {renaming ? (
            <form className="row" onSubmit={(e) => { e.preventDefault(); if (name.trim()) patch.mutate({ name: name.trim() }); }}>
              <input autoFocus value={name} maxLength={200} aria-label="Experiment name" onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Escape" && setRenaming(false)} />
              <button type="submit" className="small primary" disabled={!name.trim() || patch.isPending}>Save</button>
            </form>
          ) : (
            <h1 style={{ margin: 0 }}>{x.name}</h1>
          )}
          <PrivateBadge x={x} />
          <AdjustedBadges x={x} />
          {x.archived_at && <span className="badge">archived</span>}
        </div>
        <div className="row">
          {x.can_adjust && <Link className="btn small primary" to={withView(`/d/${ds.id}/review`, `experiment:${x.id}`)}>Adjust fields</Link>}
          {mlflow ? <a className="btn small" href={mlflow} target="_blank" rel="noopener">MLflow ↗</a>
            : x.private && <span className="muted small-text">MLflow: logged when review opens</span>}
          {editable && !renaming && <button type="button" className="small" onClick={() => setRenaming(true)}>Rename</button>}
          {editable && (
            <button type="button" className="small" disabled={patch.isPending} onClick={() => patch.mutate({ archived: !x.archived_at })}
              title={x.archived_at ? "Bring it back to the experiment list, so exports and runs can be added again."
                : "Hide it from the experiment list. Nothing is deleted: its exports, runs and MLflow history stay, and Unarchive brings it back. While archived, nothing new can be added."}>
              {x.archived_at ? "Unarchive" : "Archive"}
            </button>
          )}
        </div>
      </div>
      <div className="secondary small-text">
        Built from <b>{sourceLabel(x, me)}</b> · {x.owner_name ? `by ${x.owner_name}` : "team"} · created {fmtAgo(x.created_at)}
        {x.best && <> · best {fmtNum(x.best.value, 3)} {metricLabel(x.best.metric)} on validation (run #{x.best.run_id})</>}
      </div>
      {note && <div className="alert info" style={{ margin: 0 }}>{note}</div>}
      {editable ? (
        <label className="field">Notes
          <textarea rows={2} value={notes} maxLength={10_000} placeholder="What this experiment tries"
            onChange={(e) => setNotes(e.target.value)}
            onBlur={() => notes !== (x.notes ?? "") && patch.mutate({ notes: notes || null })} />
        </label>
      ) : x.notes && <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{x.notes}</p>}
      <ErrorAlert error={patch.error} />
    </section>
  );
}

export function ExperimentPage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const xid = Number(useParams().experimentId);
  const x = useQuery({ queryKey: ["experiment", xid], queryFn: () => api.experiment(xid), enabled: !Number.isNaN(xid) });
  if (x.error) {
    const missing = (x.error as { status?: number }).status === 404;
    return (
      <div className="state-card card">
        {missing ? <h1>Experiment #{xid} doesn't exist, or isn't shared with you</h1> : <ErrorAlert error={x.error} />}
        <div><Link className="btn" to={`/d/${ds.id}/experiments`}>Back to experiments</Link></div>
      </div>
    );
  }
  if (!x.data || !me) return <div className="page" aria-busy="true"><span className="skeleton" style={{ width: 240, height: 22 }} /></div>;
  const blocked = addBlocked(x.data, me);
  return (
    <div className="page">
      <Header ds={ds} x={x.data} me={me} />
      {blocked && <div className="alert info">{blocked}</div>}
      {x.data.assessor_id === null && !x.data.is_default && <AdjustmentsCard ds={ds} x={x.data} />}
      <ExperimentWorkspace ds={ds} x={x.data} canAdd={!blocked} />
    </div>
  );
}

/** A team experiment's changes to its baseline: each field with what changed, and a way back. */
function AdjustmentsCard({ ds, x }: { ds: Dataset; x: ExperimentDetail }) {
  const qc = useQueryClient();
  const a = x.adjustments;
  const revert = useMutation({
    mutationFn: (fieldIds: number[] | null) => api.revertAdjustments(x.id, fieldIds),
    onSuccess: () => {
      for (const k of ["experiment", "experiments", "exportPreview", "fields", "field", "fieldQuery"]) qc.invalidateQueries({ queryKey: [k] });
      toast({ message: "Reverted to the baseline" });
    },
  });
  const ids = useQuery({ queryKey: ["fields", ds.id, "current"], queryFn: () => api.fields(ds.id) });
  const idOf = (name: string) => ids.data?.find((f) => f.name === name)?.id;
  const base = baselineLabel(x.recipe_family, x.recipe_version);
  return (
    <section className="card stack" aria-label="Adjustments">
      <div className="card-head">
        <h2>Adjustments</h2>
        <span className="muted small-text">{a.count === 0 && !a.targets_changed ? `Same as ${base}` : `Changes to ${base}, this experiment only`}</span>
      </div>
      {a.count === 0 && !a.targets_changed ? (
        <span className="secondary">
          Every field is prepared as {base} decided.{x.can_adjust ? " Adjust fields to try other choices: a field's role, type or prep settings, or the targets." : ""}
        </span>
      ) : (
        <table className="data diff">
          <thead><tr><th>Field</th><th>Changed</th>{x.can_adjust && <th aria-label="Revert" />}</tr></thead>
          <tbody>
            {a.targets && (
              <tr><td><b>Targets</b></td><td>{a.targets.before.join(", ") || "none"} → {a.targets.after.join(", ") || "none"}</td>{x.can_adjust && <td />}</tr>
            )}
            {a.fields.map((f) => (
              <tr key={f.name}>
                <td className="mono">{x.can_adjust ? <Link to={withView(`/d/${ds.id}/review/${idOf(f.name) ?? ""}`, `experiment:${x.id}`)}>{f.name}</Link> : f.name}</td>
                <td><ul className="plain">{f.changes.map((c) => <li key={`${c.family}:${c.key}`}>{fmtChange({ ...c, family: null }, familyShort)}</li>)}</ul></td>
                {x.can_adjust && (
                  <td><button type="button" className="ghost small" disabled={revert.isPending || idOf(f.name) === undefined}
                    onClick={() => revert.mutate([idOf(f.name)!])}>Revert</button></td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {x.can_adjust && (a.count > 0 || a.targets_changed) && (
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button type="button" className="small" disabled={revert.isPending}
            onClick={() => window.confirm(`Put every field back to ${base}?`) && revert.mutate(null)}>Revert everything</button>
        </div>
      )}
      <ErrorAlert error={revert.error} />
    </section>
  );
}
