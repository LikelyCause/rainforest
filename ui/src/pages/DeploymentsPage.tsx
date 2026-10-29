import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useDatasetContext } from "../components/DatasetLayout";
import { Chip } from "../components/kit";
import { Empty, ErrorAlert } from "../components/ui";
import { fmtAgo, fmtDate, fmtInt } from "../format";
import type { Deployment } from "../types";

export function DeploymentStatus({ d }: { d: Pick<Deployment, "status" | "retired_at"> }) {
  return d.status === "active" ? (
    <Chip tone="good">Active</Chip>
  ) : (
    <Chip title={d.retired_at ? `Retired ${fmtDate(d.retired_at)}` : undefined}>Retired</Chip>
  );
}

/** The dataset's Deployments step: every deployment in an experiment the viewer can see, newest first. */
export function DeploymentsPage() {
  const ds = useDatasetContext();
  const nav = useNavigate();
  const list = useQuery({ queryKey: ["deployments", ds.id], queryFn: () => api.deployments(ds.id) });
  const href = (d: Deployment) => `/d/${ds.id}/deployments/${d.id}`;
  return (
    <div className="page deployments-page">
      <h1>Deployments</h1>
      <p className="secondary exp-flush">
        A deployment scores new records with finished runs, through an API call with a key or a file uploaded here, to
        test them on data they haven&apos;t seen. It isn&apos;t production serving.
      </p>
      <section className="card stack exp-gap-3" aria-label="Deployment list">
        <ErrorAlert error={list.error} />
        {!list.data && !list.error && <span className="skeleton exp-skel-block" />}
        {list.data &&
          (list.data.length ? (
            <div className="table-wrap">
              <table className="data deployments">
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Experiment</th>
                    <th scope="col">Batch</th>
                    <th scope="col" className="num">
                      Models
                    </th>
                    <th scope="col">Status</th>
                    <th scope="col">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {list.data.map((d) => (
                    <tr key={d.id} className="run-row clickable" onClick={() => nav(href(d))}>
                      <td>
                        <Link to={href(d)} className="exp-link" onClick={(ev) => ev.stopPropagation()}>
                          {d.name}
                        </Link>
                      </td>
                      <td className="secondary">
                        {d.experiment_name}
                        {d.experiment_archived_at && <span className="muted"> · archived</span>}
                      </td>
                      <td className="secondary deploy-batch">{d.batch_name ?? <span className="muted">—</span>}</td>
                      <td className="num">{fmtInt(d.models.length)}</td>
                      <td>
                        <DeploymentStatus d={d} />
                      </td>
                      <td className="muted small-text batch-nowrap" title={fmtDate(d.created_at)}>
                        {d.created_by_name ?? "Someone"}, {fmtAgo(d.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty
              title="No deployments yet"
              action={
                <Link className="btn" to={`/d/${ds.id}/experiments`}>
                  Go to the experiments
                </Link>
              }
            >
              Make one with Deploy… on a batch&apos;s page (a one-vs-rest batch whole, or the rows you select) or on a
              finished run&apos;s detail in an experiment&apos;s Runs tab.
            </Empty>
          ))}
      </section>
    </div>
  );
}
