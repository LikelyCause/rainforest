import { createContext, useContext } from "react";
import { Link, Outlet, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { PROCESSING, useDataset } from "../hooks";
import { ErrorAlert } from "./ui";
import type { Dataset } from "../types";

const DatasetContext = createContext<Dataset | null>(null);

/** Pages under /d/:id render only once the dataset is ready, so none of them repeats these states. */
export function useDatasetContext(): Dataset {
  const ds = useContext(DatasetContext);
  if (!ds) throw new Error("useDatasetContext outside DatasetLayout");
  return ds;
}

export function DatasetLayout() {
  const datasetId = Number(useParams().datasetId);
  const ds = useDataset(datasetId);
  const qc = useQueryClient();
  const nav = useNavigate();
  const del = useMutation({
    mutationFn: () => api.deleteDataset(datasetId),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["datasets"] }); nav("/"); },
  });

  if (ds.isLoading) {
    return (
      <div className="state-card" aria-busy="true">
        <span className="skeleton" style={{ width: 220, height: 18 }} />
        <span className="skeleton" />
      </div>
    );
  }
  if (!ds.data || Number.isNaN(datasetId)) {
    const notFound = !ds.error || (ds.error as { status?: number }).status === 404;
    return (
      <div className="state-card card">
        {notFound ? <h1>Dataset #{params(datasetId)} doesn't exist</h1> : <ErrorAlert error={ds.error} title="Couldn't load the dataset." />}
        <div><Link className="btn" to="/">Back to datasets</Link></div>
      </div>
    );
  }
  const d = ds.data;
  if (PROCESSING.has(d.status)) {
    return (
      <div className="state-card card">
        <h1>{d.name}</h1>
        <div className="meter"><span style={{ width: `${Math.max(d.progress, 0.02) * 100}%` }} /></div>
        <span className="secondary">{d.status_message ?? "Processing…"}</span>
        <span className="muted small-text">Review, Bulk edit and Export open when profiling finishes.</span>
      </div>
    );
  }
  if (d.status === "failed") {
    return (
      <div className="state-card card">
        <h1>{d.name}</h1>
        <ErrorAlert error={{ message: `Processing failed: ${d.error?.split("\n")[0] ?? "unknown error"}` }}>
          {d.error && d.error.includes("\n") && <details><summary>Details</summary><pre>{d.error}</pre></details>}
        </ErrorAlert>
        <ErrorAlert error={del.error} />
        <div className="row">
          <button type="button" className="danger" disabled={del.isPending}
            onClick={() => window.confirm(`Delete ${d.name}?`) && del.mutate()}>Delete</button>
          <Link className="btn" to="/">Back to datasets</Link>
        </div>
      </div>
    );
  }
  return (
    <DatasetContext.Provider value={d}>
      <Outlet />
    </DatasetContext.Provider>
  );
}

const params = (id: number) => (Number.isNaN(id) ? "?" : id);
