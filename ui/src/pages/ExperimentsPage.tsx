import { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useMe, useTeam } from "../hooks";
import { useCan, useDatasetContext } from "../components/DatasetLayout";
import { Empty, ErrorAlert } from "../components/ui";
import { Banner, Chip, confirmDialog, Dialog, Icon, Segmented } from "../components/kit";
import { ExportsTab, RunsTab, TrainTab, useJobs, useJobToasts } from "./ExperimentWorkspace";
import {
  addBlocked,
  canEdit,
  experimentTab,
  filterExperiments,
  finalizedPrompt,
  metricLabel,
  mlflowExperimentUrl,
  privacyNote,
  sourceLabel,
  sourceOptions,
  type ExperimentFilter,
} from "../experiments";
import { baselineLabel, familyShort, fmtAgo, fmtDate, fmtMetric } from "../format";
import { fmtChange } from "../recipes";
import { withView } from "../view";
import { toast } from "../toast";
import type { Dataset, Experiment, ExperimentDetail, ExportJob, Me, ModelRun, TrainPrefill } from "../types";

const FILTERS: { value: ExperimentFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "mine", label: "Mine" },
  { value: "team", label: "Team" },
  { value: "archived", label: "Archived" },
];

export function PrivateBadge({ x }: { x: Experiment }) {
  if (!x.private) return null;
  return (
    <Chip tone="private" icon="lock" title={privacyNote(x) ?? undefined}>
      Private
    </Chip>
  );
}

/** The run a lead marked as the experiment's chosen model. */
function ChosenBadge({ x }: { x: Experiment }) {
  if (!x.chosen) return null;
  return (
    <Chip tone="good" title={`Run #${x.chosen.run_id}${x.chosen.by ? `, chosen by ${x.chosen.by}` : ""}`}>
      Chosen: {x.chosen.model_label ?? x.chosen.model}
    </Chip>
  );
}

/** How a team experiment differs from its baseline, in a badge or two. */
export function AdjustedBadges({ x }: { x: Experiment }) {
  return (
    <>
      {(x.adjusted_count ?? 0) > 0 && (
        <Chip tone="experiment" title="Fields that differ from the baseline">
          {x.adjusted_count} adjusted
        </Chip>
      )}
      {x.targets_changed && (
        <Chip
          tone="warn"
          title="Its targets differ from the baseline's, so its results aren't comparable with experiments on the baseline's targets"
        >
          Targets changed
        </Chip>
      )}
    </>
  );
}

function CreateDialog({ ds, me, initial, onClose }: { ds: Dataset; me: Me; initial?: string; onClose: () => void }) {
  const qc = useQueryClient();
  const nav = useNavigate();
  const versions = useQuery({ queryKey: ["versions", ds.id], queryFn: () => api.versions(ds.id) });
  const team = useTeam(ds.id);
  const options = sourceOptions(versions.data ?? [], team.data, me);
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  useEffect(() => {
    // The current finalized baseline by default (sourceOptions lists it first among the finalized), then anything else.
    // Wait for the versions: an assessment listed before they load mustn't become the default.
    if (!source && options.length && !versions.isPending) {
      const current = ds.baseline?.phase === "finalized" ? `recipe:${ds.baseline.id}` : null;
      setSource(
        (
          options.find((o) => o.value === initial) ??
          options.find((o) => o.value === current) ??
          options.find((o) => o.value.startsWith("recipe:")) ??
          options[0]
        ).value,
      );
    }
  }, [source, options, initial, versions.isPending, ds.baseline]);
  const create = useMutation({
    mutationFn: () => api.createExperiment(ds.id, { name: name.trim(), source }),
    onSuccess: (x) => {
      qc.invalidateQueries({ queryKey: ["experiments", ds.id] });
      toast({ message: `Created ${x.name}` });
      nav(`/d/${ds.id}/experiments/${x.id}?tab=exports`);
    },
  });
  const loading = versions.isPending || team.isPending;
  const personal = source.startsWith("assessment:");
  const submit = () => {
    if (name.trim() && source) create.mutate();
  };
  return (
    <Dialog
      title="New experiment"
      onClose={onClose}
      actions={
        <>
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="submit"
            form="new-experiment"
            className="primary"
            disabled={!name.trim() || !source || create.isPending}
          >
            Create
          </button>
        </>
      }
    >
      <form
        id="new-experiment"
        className="stack"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <label className="field">
          Name
          <input
            data-autofocus
            value={name}
            maxLength={200}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Gradient boosting, no leaky fields"
          />
        </label>
        <label className="field">
          Built from
          <select value={source} onChange={(e) => setSource(e.target.value)} disabled={!options.length}>
            {options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        {!loading && !options.length && (
          <Banner tone="warn">
            Nothing to experiment on yet: no baseline is finalized, and you aren't assessing the current one.
          </Banner>
        )}
        {personal && (
          <span className="secondary small-text">
            Exports use your own choices as they stand when you export. Only you see this experiment until{" "}
            {team.data?.recipe.label} reaches team review; its runs are logged to MLflow then.
          </span>
        )}
        <ErrorAlert error={create.error} />
      </form>
    </Dialog>
  );
}

/** After finalizing (or while nothing is built on the finalized baseline): the next step, offered. */
function FinalizedCard({
  ds,
  recipeId,
  canCreate,
  onCreate,
  onDismiss,
}: {
  ds: Dataset;
  recipeId: number;
  canCreate: boolean;
  onCreate: () => void;
  onDismiss: (() => void) | null;
}) {
  const versions = useQuery({ queryKey: ["versions", ds.id], queryFn: () => api.versions(ds.id) });
  const v = versions.data?.find((x) => x.id === recipeId);
  const label = v
    ? baselineLabel(v.family, v.version, false, v.name)
    : ds.baseline?.id === recipeId
      ? ds.baseline.label
      : null;
  const short = v ? `v${v.version}` : ds.baseline?.id === recipeId ? `v${ds.baseline.version}` : "it";
  if (!label) return null;
  return (
    <section className="card finalized-card" aria-label={`${label} is final`}>
      <div className="stack exp-gap-1">
        <b>{label} is final</b>
        <span className="secondary">Build an experiment on it to export and train.</span>
      </div>
      <div className="row">
        {canCreate && (
          <button type="button" className="primary" onClick={onCreate}>
            <Icon name="plus" />
            New experiment from {short}
          </button>
        )}
        {onDismiss && (
          <button type="button" className="ghost small exp-icon" aria-label="Dismiss" onClick={onDismiss}>
            <Icon name="x" />
          </button>
        )}
      </div>
    </section>
  );
}

export function ExperimentsPage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const canDo = useCan();
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState<ExperimentFilter>("all");
  const [creating, setCreating] = useState<{ source?: string } | null>(null);
  const list = useQuery({ queryKey: ["experiments", ds.id], queryFn: () => api.experiments(ds.id) });
  if (!me) return null;
  const shown = filterExperiments(list.data ?? [], filter, me);
  const finalizedParam = params.get("finalized");
  const prompt = finalizedPrompt(list.data, ds.baseline, finalizedParam);
  const canCreate = canDo("createExperiment");
  return (
    <div className="page experiments-page">
      <div className="row exp-between">
        <h1>Experiments</h1>
        {canCreate && (
          <button type="button" className="primary" onClick={() => setCreating({})}>
            <Icon name="plus" />
            New experiment
          </button>
        )}
      </div>
      <p className="secondary exp-flush">
        Each experiment holds exports and the models trained on them. Build one from a finalized baseline, or from your
        own assessment while you assess; that one stays private to you until team review opens.
      </p>
      {prompt !== null && (
        <FinalizedCard
          ds={ds}
          recipeId={prompt}
          canCreate={canCreate}
          onCreate={() => setCreating({ source: `recipe:${prompt}` })}
          onDismiss={finalizedParam ? () => setParams({}, { replace: true }) : null}
        />
      )}
      <section className="card stack exp-gap-3" aria-label="Experiment list">
        <Segmented
          label="Show"
          size="small"
          value={filter}
          onChange={setFilter}
          options={FILTERS}
          className="exp-self-start"
        />
        <ErrorAlert error={list.error} />
        {!list.data && !list.error && <span className="skeleton exp-skel-block" />}
        {list.data &&
          (shown.length ? (
            <div className="table-wrap">
              <table className="data experiments">
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Built from</th>
                    <th scope="col">Owner</th>
                    <th scope="col" className="num">
                      Exports
                    </th>
                    <th scope="col" className="num">
                      Runs
                    </th>
                    <th scope="col" className="num">
                      Best (validation)
                    </th>
                    <th scope="col">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((x) => (
                    <tr key={x.id}>
                      <td>
                        <div className="row exp-gap-1">
                          <Link to={`/d/${ds.id}/experiments/${x.id}`} className="exp-link">
                            {x.name}
                          </Link>
                          <PrivateBadge x={x} />
                          <AdjustedBadges x={x} />
                          <ChosenBadge x={x} />
                          {x.archived_at && <Chip>Archived</Chip>}
                        </div>
                      </td>
                      <td className="secondary">{sourceLabel(x, me)}</td>
                      <td className="secondary">{x.owner_name ?? <span className="muted">Team</span>}</td>
                      <td className="num">{x.export_count ?? 0}</td>
                      <td className="num">{x.run_count ?? 0}</td>
                      <td className="num" title={x.best ? `Run #${x.best.run_id} · ${x.best.target}` : undefined}>
                        {x.best ? (
                          <>
                            {fmtMetric(x.best.value)} <span className="muted">{metricLabel(x.best.metric)}</span>
                          </>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td className="muted small-text" title={fmtDate(x.created_at)}>
                        {fmtAgo(x.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty">
              {filter === "archived"
                ? "No archived experiments."
                : list.data.length
                  ? "None here."
                  : "No experiments yet. Create one to export and train."}
            </div>
          ))}
      </section>
      {creating && <CreateDialog ds={ds} me={me} initial={creating.source} onClose={() => setCreating(null)} />}
    </div>
  );
}

/** The experiment's details and notes, and renaming and archiving for its owner or a lead. */
function NotesTab({ ds, x, me, blocked }: { ds: Dataset; x: ExperimentDetail; me: Me; blocked: string | null }) {
  const qc = useQueryClient();
  const editable = canEdit(x, me, ds.status === "refreshing");
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(x.name);
  const [notes, setNotes] = useState(x.notes ?? "");
  useEffect(() => {
    setName(x.name);
    setNotes(x.notes ?? "");
  }, [x.name, x.notes]);
  const patch = useMutation({
    mutationFn: (data: { name?: string; notes?: string | null; archived?: boolean }) => api.patchExperiment(x.id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiment", x.id] });
      qc.invalidateQueries({ queryKey: ["experiments", ds.id] });
      setRenaming(false);
    },
  });
  const archive = async () => {
    if (
      !x.archived_at &&
      !(await confirmDialog({
        title: `Archive ${x.name}?`,
        confirm: "Archive",
        body: "It leaves the experiment list. Nothing is deleted: its exports, runs and MLflow history stay, and Unarchive brings it back. While archived, nothing new can be added.",
      }))
    )
      return;
    patch.mutate({ archived: !x.archived_at });
  };
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  const mlflow = mlflowExperimentUrl(system.data?.mlflow.browseUrl ?? null, x.mlflow_experiment_id, x.id);
  const note = privacyNote(x);
  return (
    <div className="page exp-narrow">
      {blocked && <Banner>{blocked}</Banner>}
      {note && <Banner>{note}</Banner>}
      <section className="card stack" aria-labelledby="notes-title">
        <div className="card-head exp-flush">
          {renaming ? (
            <form
              className="row"
              onSubmit={(e) => {
                e.preventDefault();
                if (name.trim()) patch.mutate({ name: name.trim() });
              }}
            >
              <input
                autoFocus
                value={name}
                maxLength={200}
                aria-label="Experiment name"
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    setRenaming(false);
                    setName(x.name);
                  }
                }}
              />
              <button type="submit" className="small primary" disabled={!name.trim() || patch.isPending}>
                Save
              </button>
              <button
                type="button"
                className="small ghost"
                onClick={() => {
                  setRenaming(false);
                  setName(x.name);
                }}
              >
                Cancel
              </button>
            </form>
          ) : (
            <h2 id="notes-title">{x.name}</h2>
          )}
          {editable && !renaming && (
            <button type="button" className="small" onClick={() => setRenaming(true)}>
              Rename
            </button>
          )}
        </div>
        <dl className="exp-details">
          <dt>Built from</dt>
          <dd>{sourceLabel(x, me)}</dd>
          <dt>Owner</dt>
          <dd>{x.owner_name ?? "Team"}</dd>
          <dt>Created</dt>
          <dd title={fmtDate(x.created_at)}>{fmtAgo(x.created_at)}</dd>
          {x.best && (
            <>
              <dt>Best</dt>
              <dd className="num">
                {fmtMetric(x.best.value)} {metricLabel(x.best.metric)} on validation (run #{x.best.run_id},{" "}
                {x.best.target})
              </dd>
            </>
          )}
          {x.chosen && (
            <>
              <dt>Chosen</dt>
              <dd>
                {x.chosen.model_label ?? x.chosen.model} (run #{x.chosen.run_id})
                {x.chosen.by ? `, by ${x.chosen.by}` : ""}
              </dd>
            </>
          )}
          {x.archived_at && (
            <>
              <dt>Archived</dt>
              <dd title={fmtDate(x.archived_at)}>{fmtAgo(x.archived_at)}</dd>
            </>
          )}
          {(mlflow || x.private) && (
            <>
              <dt>MLflow</dt>
              <dd>
                {mlflow ? (
                  <a href={mlflow} target="_blank" rel="noopener">
                    Open its runs in MLflow <Icon name="external" size={14} label="(opens in a new tab)" />
                  </a>
                ) : (
                  <span className="muted">Logged when review opens</span>
                )}
              </dd>
            </>
          )}
        </dl>
        {editable ? (
          <label className="field">
            Notes
            <textarea
              rows={5}
              value={notes}
              maxLength={10_000}
              placeholder="What this experiment tries"
              onChange={(e) => setNotes(e.target.value)}
              onBlur={() => notes !== (x.notes ?? "") && patch.mutate({ notes: notes || null })}
            />
          </label>
        ) : x.notes ? (
          <p className="exp-flush exp-pre">{x.notes}</p>
        ) : (
          <span className="muted">No notes.</span>
        )}
        <ErrorAlert error={patch.error} />
        {editable && (
          <div className="row exp-manage">
            <span className="secondary small-text exp-grow">
              {x.archived_at
                ? "Unarchive it to add exports and runs again."
                : "Archiving hides it from the list; nothing is deleted."}
            </span>
            <button type="button" className="small" disabled={patch.isPending} onClick={archive}>
              {x.archived_at ? "Unarchive" : "Archive"}
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

/** A team experiment's changes to its baseline: each field with what changed, and a way back. */
function AdjustmentsTab({ ds, x }: { ds: Dataset; x: ExperimentDetail }) {
  const qc = useQueryClient();
  const a = x.adjustments;
  const revert = useMutation({
    mutationFn: (fieldIds: number[] | null) => api.revertAdjustments(x.id, fieldIds),
    onSuccess: () => {
      for (const k of ["experiment", "experiments", "exportPreview", "fields", "field", "fieldQuery"])
        qc.invalidateQueries({ queryKey: [k] });
      toast({ message: "Reverted to the baseline" });
    },
  });
  const ids = useQuery({ queryKey: ["fields", ds.id, "current"], queryFn: () => api.fields(ds.id) });
  const idOf = (name: string) => ids.data?.find((f) => f.name === name)?.id;
  const base = baselineLabel(x.recipe_family, x.recipe_version, false, x.recipe_name);
  const adjust = withView(`/d/${ds.id}/review`, `experiment:${x.id}`);
  if (x.assessor_id !== null || x.is_default) {
    return (
      <div className="page exp-narrow">
        <section className="card">
          <Empty title="No adjustments here">
            {x.assessor_id !== null
              ? "This experiment exports its assessment as it stands, so change fields in that assessment instead."
              : `This is ${base}'s Team experiment and stays as the team agreed. Start your own experiment to adjust fields.`}
          </Empty>
        </section>
      </div>
    );
  }
  const same = a.count === 0 && !a.targets_changed;
  const adjustable = x.can_adjust && ds.status !== "refreshing";
  return (
    <div className="page exp-narrow">
      <section className="card stack" aria-labelledby="adjust-title">
        <div className="card-head exp-flush">
          <div className="stack exp-gap-1">
            <h2 id="adjust-title">Adjustments</h2>
            <span className="muted small-text">
              {same ? `Same as ${base}` : `Changes to ${base}, this experiment only`}
            </span>
          </div>
          {adjustable && (
            <Link className="btn primary" to={adjust}>
              <Icon name="flask" />
              Adjust fields
            </Link>
          )}
        </div>
        {same ? (
          <span className="secondary">
            Every field is prepared as {base} decided.
            {adjustable
              ? " Adjust fields to try other choices: a field's role, type or prep settings, or the targets."
              : ""}
          </span>
        ) : (
          <div className="table-wrap">
            <table className="data diff">
              <thead>
                <tr>
                  <th scope="col">Field</th>
                  <th scope="col">Changed</th>
                  {adjustable && (
                    <th scope="col">
                      <span className="exp-sr">Revert</span>
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {a.targets && (
                  <tr>
                    <td>
                      <b>Targets</b>
                    </td>
                    <td>
                      {a.targets.before.join(", ") || "none"} → {a.targets.after.join(", ") || "none"}
                    </td>
                    {adjustable && <td />}
                  </tr>
                )}
                {a.fields.map((f) => (
                  <tr key={f.name}>
                    <td>
                      {adjustable ? (
                        <Link to={withView(`/d/${ds.id}/review/${idOf(f.name) ?? ""}`, `experiment:${x.id}`)}>
                          {f.name}
                        </Link>
                      ) : (
                        f.name
                      )}
                    </td>
                    <td>
                      <ul className="plain">
                        {f.changes.map((c) => (
                          <li key={`${c.family}:${c.key}`}>{fmtChange({ ...c, family: null }, familyShort)}</li>
                        ))}
                      </ul>
                    </td>
                    {adjustable && (
                      <td>
                        <button
                          type="button"
                          className="ghost small"
                          disabled={revert.isPending || idOf(f.name) === undefined}
                          aria-label={`Revert ${f.name}`}
                          onClick={() => revert.mutate([idOf(f.name)!])}
                        >
                          Revert
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {adjustable && !same && (
          <div className="row exp-end">
            <button
              type="button"
              className="small"
              disabled={revert.isPending}
              onClick={async () => {
                if (
                  await confirmDialog({
                    title: "Revert everything?",
                    body: `Every field goes back to ${base}. Exports already made keep their data.`,
                    confirm: "Revert everything",
                    danger: true,
                  })
                )
                  revert.mutate(null);
              }}
            >
              Revert everything
            </button>
          </div>
        )}
        {!x.can_adjust && (
          <span className="muted small-text">Only {x.owner_name ?? "its owner"} adjusts this experiment.</span>
        )}
        <ErrorAlert error={revert.error} />
      </section>
    </div>
  );
}

export function ExperimentPage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const nav = useNavigate();
  const [params] = useSearchParams();
  const xid = Number(useParams().experimentId);
  const tab = experimentTab(params.get("tab"));
  const x = useQuery({
    queryKey: ["experiment", xid],
    queryFn: () => api.experiment(xid),
    enabled: !Number.isNaN(xid),
  });
  const jobs = useJobs(ds.id, xid);
  const [fresh, setFresh] = useState<number[]>([]);
  const base = `/d/${ds.id}/experiments/${xid}`;
  const blocked = x.data && me ? addBlocked(x.data, me, ds.status === "refreshing") : null;
  const canAdd = Boolean(x.data && me) && !blocked;
  useJobToasts({
    x: x.data ?? ({ id: xid } as Experiment),
    exports: jobs.exports.data,
    runs: jobs.runs.data,
    fresh,
    canAdd,
    base,
  });
  if (x.error) {
    const missing = (x.error as { status?: number }).status === 404;
    return (
      <div className="state-card card">
        {missing ? <h1>Experiment #{xid} doesn't exist, or isn't shared with you</h1> : <ErrorAlert error={x.error} />}
        <div>
          <Link className="btn" to={`/d/${ds.id}/experiments`}>
            Back to experiments
          </Link>
        </div>
      </div>
    );
  }
  if (!x.data || !me)
    return (
      <div className="page" aria-busy="true">
        <span className="skeleton exp-skel-line short" />
      </div>
    );
  const trainAgain = (p: TrainPrefill) => nav(`${base}?tab=train&export=${p.exportId}`, { state: { prefill: p } });
  const onStarted = (runs: ModelRun[], e: ExportJob, label: string) => {
    setFresh(runs.map((r) => r.id));
    toast({
      message:
        runs.length > 1
          ? `Training ${runs.length} ${label} runs on Export #${e.id}`
          : `Training ${label} on Export #${e.id}`,
    });
    nav(`${base}?tab=runs&export=${e.id}`);
    // The new runs are at the top of Runs: land there, not where Train was scrolled to.
    document.querySelector("main")?.scrollTo({ top: 0 });
  };
  const reason = blocked && (
    <div className="exp-page-banner">
      <Banner>{blocked}</Banner>
    </div>
  );
  switch (tab) {
    case "train":
      return canAdd ? (
        <TrainTab exports={jobs.exports.data} onStarted={onStarted} private={x.data.private} />
      ) : (
        <div className="page exp-narrow">
          <Banner>{blocked}</Banner>
        </div>
      );
    case "exports":
      return (
        <>
          {reason}
          <ExportsTab
            ds={ds}
            x={x.data}
            exports={jobs.exports.data}
            error={jobs.exports.error}
            canAdd={canAdd}
            onTrain={(id) => nav(`${base}?tab=train&export=${id}`)}
          />
        </>
      );
    case "adjustments":
      return (
        <>
          {reason}
          <AdjustmentsTab ds={ds} x={x.data} />
        </>
      );
    case "notes":
      return <NotesTab ds={ds} x={x.data} me={me} blocked={blocked} />;
    default:
      return (
        <>
          {reason}
          <RunsTab
            x={x.data}
            exports={jobs.exports.data ?? []}
            runs={jobs.runs.data}
            error={jobs.runs.error ?? jobs.exports.error}
            canAdd={canAdd}
            fresh={fresh}
            onTrainAgain={trainAgain}
          />
        </>
      );
  }
}
