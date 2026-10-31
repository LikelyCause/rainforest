import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useDatasetContext } from "../components/DatasetLayout";
import { Chip, Segmented, StatusBadge } from "../components/kit";
import { Empty, ErrorAlert } from "../components/ui";
import { fmtAgo, fmtDate, fmtMetric } from "../format";
import { metricFor, runColumns, runLabel, whyFailed } from "../runs";
import type { ModelRun } from "../types";

type RunFilter = "all" | "done" | "active" | "failed";

const FILTERS: { value: RunFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "done", label: "Finished" },
  { value: "active", label: "In progress" },
  { value: "failed", label: "Failed" },
];

const active = (r: ModelRun) => r.status === "pending" || r.status === "running";

const shows = (r: ModelRun, f: RunFilter) =>
  f === "all" || (f === "active" ? active(r) : f === "failed" ? r.status === "failed" : r.status === "done");

/** A run's first target and its validation score there: AUC for classes, R² for a number. */
function score(r: ModelRun): { target: string; label: string; value: number | undefined } | null {
  const target = r.config?.order?.[0] ?? r.config?.target ?? Object.keys(r.metrics?.valid ?? {})[0];
  if (!target) return null;
  return {
    target,
    label: runColumns(r.report?.target_kinds?.[target])[0].label,
    value: metricFor(r, "valid", target, "primary"),
  };
}

/** Every run in the dataset's experiments the viewer can see, newest first; a row opens it in its experiment. */
export function RunsPage() {
  const ds = useDatasetContext();
  const nav = useNavigate();
  const [filter, setFilter] = useState<RunFilter>("all");
  const runs = useQuery({
    queryKey: ["runs", ds.id],
    queryFn: () => api.runs(ds.id),
    refetchInterval: (q) => ((q.state.data ?? []).some(active) ? 3000 : false),
  });
  const experiments = useQuery({ queryKey: ["experiments", ds.id], queryFn: () => api.experiments(ds.id) });
  const names = new Map((experiments.data ?? []).map((x) => [x.id, x.name]));
  const all = [...(runs.data ?? [])].sort((a, b) => b.id - a.id);
  const shown = all.filter((r) => shows(r, filter));
  const href = (r: ModelRun) => `/d/${ds.id}/experiments/${r.experiment_id}?tab=runs&export=${r.export_id}&run=${r.id}`;
  return (
    <div className="page runs-page">
      <h1 className="tools-sr">Runs</h1>
      <p className="secondary exp-flush">
        Every model trained on this dataset, across its experiments. A row opens the run in its experiment, where runs
        on the same export are compared side by side.
      </p>
      <section className="card stack exp-gap-3" aria-label="Run list">
        <Segmented
          label="Show"
          size="small"
          value={filter}
          onChange={setFilter}
          options={FILTERS}
          className="exp-self-start"
        />
        <ErrorAlert error={runs.error} />
        {!runs.data && !runs.error && <span className="skeleton exp-skel-block" />}
        {runs.data &&
          (shown.length ? (
            <div className="table-wrap">
              <table className="data runs-all">
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">Experiment</th>
                    <th scope="col">Target</th>
                    <th scope="col" className="num">
                      Validation
                    </th>
                    <th scope="col">Status</th>
                    <th scope="col">Started</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r) => {
                    const s = score(r);
                    return (
                      <tr key={r.id} className="run-row clickable" onClick={() => nav(href(r))}>
                        <td>
                          <div className="row exp-gap-1">
                            <Link to={href(r)} className="exp-link" onClick={(ev) => ev.stopPropagation()}>
                              {runLabel(r, undefined)}
                            </Link>
                            <span className="muted small-text num">#{r.id}</span>
                            {r.chosen_at && (
                              <Chip
                                tone="good"
                                icon="check"
                                title={r.chosen_by_name ? `Chosen by ${r.chosen_by_name}` : undefined}
                              >
                                Chosen
                              </Chip>
                            )}
                          </div>
                        </td>
                        <td className="secondary">
                          {names.get(r.experiment_id) ?? <span className="muted">Experiment #{r.experiment_id}</span>}
                        </td>
                        <td className="secondary">{s?.target ?? <span className="muted">—</span>}</td>
                        <td className="num">
                          {s?.value !== undefined ? (
                            <>
                              {fmtMetric(s.value)} <span className="muted">{s.label}</span>
                            </>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                        <td>
                          <span title={r.status === "failed" ? whyFailed(r.error) : undefined}>
                            <StatusBadge status={r.status} />
                          </span>
                        </td>
                        <td className="muted small-text batch-nowrap" title={fmtDate(r.created_at)}>
                          {fmtAgo(r.created_at)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : all.length ? (
            <div className="empty">None here.</div>
          ) : (
            <Empty
              title="No runs yet"
              action={
                <Link className="btn" to={`/d/${ds.id}/experiments`}>
                  Go to the experiments
                </Link>
              }
            >
              Train a model from an experiment&apos;s Train tab; every run lands here as well as in its experiment.
            </Empty>
          ))}
      </section>
    </div>
  );
}
