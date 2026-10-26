import { createContext, useContext, useEffect, useRef } from "react";
import { Link, Outlet, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { can, type Action } from "../can";
import { PROCESSING, useDataset, useMe, usePulls } from "../hooks";
import { pullStarter } from "../pulls";
import { ErrorAlert } from "./ui";
import { Banner, ProgressBar, confirmDialog } from "./kit";
import type { Dataset } from "../types";

const DatasetContext = createContext<Dataset | null>(null);

/** Pages under /d/:id render only once the dataset is ready, so none of them repeats these states. */
export function useDatasetContext(): Dataset {
  const ds = useContext(DatasetContext);
  if (!ds) throw new Error("useDatasetContext outside DatasetLayout");
  return ds;
}

/** `can` for the dataset being viewed: nothing that writes to it while a refresh holds it. */
export function useCan(): (action: Action) => boolean {
  const me = useMe().data;
  const locked = useContext(DatasetContext)?.status === "refreshing";
  return (action) => can(me, action, locked);
}

/** A lead who can act now: not a lead at all, or a lead while a refresh holds the dataset, gets false. */
export const useLead = () => useCan()("finalize");

function RefreshBanner({ ds }: { ds: Dataset }) {
  const pulls = usePulls(ds.id, true, true);
  const starter = pullStarter(pulls.data, ds.last_pull?.id);
  return (
    <div className="page refresh-banner">
      <Banner
        tone="info"
        role="status"
        title={`Refreshing from ${ds.source?.connection ?? "its source"}${starter ? `, started by ${starter}` : ""}.`}
      >
        This dataset is read-only until the pull finishes.
      </Banner>
    </div>
  );
}

export function DatasetLayout() {
  const datasetId = Number(useParams().datasetId);
  const ds = useDataset(datasetId, !Number.isNaN(datasetId));
  const me = useMe().data;
  const qc = useQueryClient();
  const nav = useNavigate();
  const status = ds.data?.status;
  const was = useRef(status);
  useEffect(() => {
    if (was.current === "refreshing" && status && status !== "refreshing") qc.invalidateQueries();
    was.current = status;
  }, [status, qc]);
  const del = useMutation({
    mutationFn: () => api.deleteDataset(datasetId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      nav("/");
    },
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
    const notFound = Number.isNaN(datasetId) || !ds.error || (ds.error as { status?: number }).status === 404;
    return (
      <div className="state-card card">
        {notFound ? (
          <h1>Dataset #{params(datasetId)} doesn't exist</h1>
        ) : (
          <ErrorAlert error={ds.error} title="Couldn't load the dataset." />
        )}
        <div>
          <Link className="btn" to="/">
            Back to datasets
          </Link>
        </div>
      </div>
    );
  }
  const d = ds.data;
  if (PROCESSING.has(d.status)) {
    return (
      <div className="state-card card">
        <h1>{d.name}</h1>
        <ProgressBar value={Math.max(d.progress, 0.02)} max={1} label={`Processing ${d.name}`} color="var(--accent)" />
        <span className="secondary">{d.status_message ?? "Processing…"}</span>
        <span className="muted small-text">
          Fields, Explore, Baseline and Experiments open when profiling finishes.
        </span>
      </div>
    );
  }
  if (d.status === "failed") {
    return (
      <div className="state-card card">
        <h1>{d.name}</h1>
        <ErrorAlert error={{ message: `Processing failed: ${d.error?.split("\n")[0] ?? "unknown error"}` }}>
          {d.error && d.error.includes("\n") && (
            <details>
              <summary>Details</summary>
              <pre>{d.error}</pre>
            </details>
          )}
        </ErrorAlert>
        <ErrorAlert error={del.error} />
        <div className="row">
          {can(me, "deleteDataset") && (
            <button
              type="button"
              className="danger"
              disabled={del.isPending}
              onClick={async () =>
                (await confirmDialog({
                  title: `Delete ${d.name}?`,
                  body: "Its fields, exports, model runs and their files go too. MLflow keeps its own copies.",
                  confirm: "Delete dataset",
                  cancel: "Keep it",
                  danger: true,
                })) && del.mutate()
              }
            >
              Delete
            </button>
          )}
          <Link className="btn" to="/">
            Back to datasets
          </Link>
        </div>
      </div>
    );
  }
  return (
    <DatasetContext.Provider value={d}>
      {d.status === "refreshing" && <RefreshBanner ds={d} />}
      <Outlet />
    </DatasetContext.Provider>
  );
}

const params = (id: number) => (Number.isNaN(id) ? "?" : id);
