import { useMemo, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api";
import { can } from "../../can";
import { useFields, useMe } from "../../hooks";
import { Empty, ErrorAlert } from "../ui";
import { Banner, ProgressBar, StatusBadge } from "../kit";
import { fmtAgo, fmtInt } from "../../format";
import type { Analysis, AnalysisKind, AnalysisParams, Dataset, NotEnoughReason, Skipped } from "../../types";
import { parseExploreParams } from "./params";
import { parseView, type View, viewKey } from "../../view";

/** Whose decisions raw-data views use (?decisions=experiment:<id> | baseline:<id>), else the baseline in progress. */
export function decisionsView(params: URLSearchParams): View {
  const v = params.get("decisions");
  return parseView(v) ? v : null;
}

export { SAMPLES } from "./params";
const isActive = (s?: string) => s === "pending" || s === "running";

/** Toolbar state shared by every view; all of it lives in the URL. */
export function useExploreState(ds: Dataset) {
  const [params, setParams] = useSearchParams();
  const view = decisionsView(params);
  // The targets of the decisions in use: an experiment or another baseline may have its own.
  const viewFields = useFields(ds.id, view !== null, view);
  const targets = view ? (viewFields.data ?? []).filter((f) => f.action === "target").map((f) => f.name) : ds.target_columns;
  const { data, isRaw, kept, target, sample } = parseExploreParams(params, targets);
  const base: AnalysisParams = useMemo(() => ({ data, kept, target, sample }), [data, kept, target, sample]);
  const set = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) v === null ? next.delete(k) : next.set(k, v);
    setParams(next, { replace: true });
  };
  return { params, data, isRaw, kept, target, sample, view, targets, base, set, get: (k: string) => params.get(k) };
}
export type ExploreState = ReturnType<typeof useExploreState>;

/** POSTs when the params change (the API returns a cached row when it can), then polls the row until it settles. */
export function useAnalysis<R>(datasetId: number, kind: AnalysisKind, params: AnalysisParams) {
  const qc = useQueryClient();
  const [search] = useSearchParams();
  const view = params.data === "raw" ? decisionsView(search) : null;
  const key = ["analysis", datasetId, kind, params, viewKey(view)];
  const q = useQuery({
    queryKey: key,
    queryFn: async (): Promise<Analysis> => {
      const prev = qc.getQueryData<Analysis>(key);
      if (prev && isActive(prev.status)) {
        const row = await api.analysis(prev.id);
        // Another view's job replaced this one while we were away; ask again.
        if (!(row.status === "failed" && row.error?.startsWith("Superseded"))) return { ...row, stale: row.stale ?? false };
      }
      return api.createAnalysis(datasetId, { kind, params, view });
    },
    refetchInterval: (query) => (isActive((query.state.data as Analysis | undefined)?.status) ? 1000 : false),
    retry: false,
  });
  const m = useMutation({
    mutationFn: () => api.createAnalysis(datasetId, { kind, params, recompute: true, view }),
    onSuccess: (row) => qc.setQueryData(key, row),
  });
  return {
    query: q,
    row: q.data as Analysis<R> | undefined,
    recompute: () => m.mutate(),
    recomputing: m.isPending,
    recomputeError: m.error,
  };
}
export type AnalysisHandle<R> = ReturnType<typeof useAnalysis<R>>;

const NOT_ENOUGH: Record<NotEnoughReason, { title: string; body: string }> = {
  no_target: { title: "This view needs a target", body: "Choose the column you want to predict; the analysis compares every field against it." },
  too_few_fields: { title: "Not enough usable fields", body: "Fewer than two fields are left after skipping text, ids and constants. Keep more fields, or turn off Kept only." },
  too_few_rows: { title: "Not enough rows", body: "The sample has too few complete rows. Try a larger sample, or check the target's missing values." },
  single_class: { title: "The target has a single class", body: "Every sampled row has the same target value, so there is nothing to compare." },
};

/** The one frame for job-backed views: computing, ready, out of date, failed, or not enough data. */
export function AnalysisState<R>({ a, datasetId, children }: {
  a: AnalysisHandle<R>; datasetId: number; children: (result: R, row: Analysis<R>) => ReactNode;
}) {
  const { row, query } = a;
  const mayRecompute = can(useMe().data, "recompute");
  if (!row) {
    if (query.error) {
      return (
        <ErrorAlert error={query.error} title="Couldn't start the analysis.">
          <div className="tools-retry"><button type="button" className="small" onClick={() => query.refetch()}>Retry</button></div>
        </ErrorAlert>
      );
    }
    return <Computing status="pending" progress={0} message="Starting…" />;
  }
  if (isActive(row.status)) return <Computing status={row.status} progress={row.progress} message={row.status_message} />;
  if (row.status === "failed") {
    const reason = (row.result as { reason?: NotEnoughReason } | null)?.reason;
    if (row.error?.startsWith("Not enough") && reason && NOT_ENOUGH[reason]) {
      const info = NOT_ENOUGH[reason];
      return (
        <div className="card">
          <Empty title={info.title}
            action={reason === "no_target" && mayRecompute ? <Link className="btn" to={`/?target=${datasetId}`}>Choose a target</Link> : undefined}>
            {info.body}
          </Empty>
        </div>
      );
    }
    return (
      <ErrorAlert error={{ message: row.error?.split("\n")[0] ?? "The analysis failed." }} title="Analysis failed.">
        <div className="row tools-retry">
          <button type="button" className="small" disabled={a.recomputing} onClick={a.recompute}>Retry</button>
          {a.recomputeError != null && <span>{(a.recomputeError as Error).message}</span>}
        </div>
      </ErrorAlert>
    );
  }
  const n = (row.result as { n?: number } | null)?.n;
  return (
    <>
      {row.stale ? (
        <Banner tone="warn" title="Out of date."
          action={mayRecompute && <button type="button" className="small" disabled={a.recomputing} onClick={a.recompute}>Recompute</button>}>
          Decisions changed since this ran ({fmtAgo(row.finished_at)}).
        </Banner>
      ) : (
        <div className="analysis-bar">
          <span className="row tools-gap-6"><StatusBadge status="done" /><span className="num">As of {fmtAgo(row.finished_at)}{n != null && <> · {fmtInt(n)} rows</>}</span></span>
          {mayRecompute && (
            <button type="button" className="small ghost" disabled={a.recomputing} onClick={a.recompute}>
              {a.recomputing ? "Starting…" : "Recompute"}
            </button>
          )}
        </div>
      )}
      <ErrorAlert error={a.recomputeError} />
      {row.result != null && children(row.result as R, row)}
    </>
  );
}

function Computing({ status, progress, message }: { status: string; progress: number; message: string | null }) {
  return (
    <div className="card computing" aria-busy="true">
      <div className="row tools-between"><b>Computing…</b><StatusBadge status={status} /></div>
      <ProgressBar value={Math.max(progress, 0.03)} max={1} label="Analysis progress" />
      <span className="secondary small-text">{message ?? "Waiting for a worker"}</span>
    </div>
  );
}

export function SkippedList({ skipped }: { skipped: Skipped[] }) {
  if (!skipped.length) return null;
  return (
    <details className="more">
      <summary><b>{skipped.length}</b> field{skipped.length === 1 ? "" : "s"} skipped</summary>
      <ul className="skipped-list">
        {skipped.map((s) => <li key={s.name}><span className="mono">{s.name}</span> <span className="muted">— {s.reason}</span></li>)}
      </ul>
    </details>
  );
}
