import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { accessOf, type Access } from "./team";
import type { Dataset } from "./types";
import { parseView, type View, viewKey } from "./view";

export const PROCESSING = new Set(["pending", "ingesting", "profiling", "correlating"]);

/** The signed-in user; RequireAuth has loaded it before any page renders. */
export const useMe = () => useQuery({ queryKey: ["me"], queryFn: api.me, staleTime: Infinity, retry: false });

export const useOptions = () => useQuery({ queryKey: ["options"], queryFn: api.options, staleTime: Infinity });

export const useDataset = (id: number, enabled = true) =>
  useQuery({
    queryKey: ["dataset", id],
    queryFn: () => api.dataset(id),
    enabled,
    refetchInterval: (q) => (q.state.data && PROCESSING.has((q.state.data as Dataset).status) ? 1000 : false),
  });

export const useTeam = (id: number, enabled = true) =>
  useQuery({ queryKey: ["team", id], queryFn: () => api.team(id), enabled });

/** What the viewer works on in this dataset now and whether they can change it. Read-only until the team state
 * has loaded, so nobody briefly edits something they can't save. */
export function useAccess(ds: Dataset, view: View = null): Access {
  const me = useMe().data;
  const v = parseView(view);
  const team = useTeam(ds.id, !v && ds.phase === "assessing");
  const x = useExperiment(v?.kind === "experiment" ? v.id : null);
  if (v?.kind === "experiment") {
    if (!x.data) return { source: "experiment", canEdit: false, note: "Loading…" };
    return x.data.can_adjust
      ? {
          source: "experiment",
          canEdit: true,
          note: `Adjusting ${x.data.name}: changes apply to this experiment only.`,
        }
      : {
          source: "experiment",
          canEdit: false,
          note: x.data.is_default
            ? "This is the baseline's default Team experiment; start your own experiment to adjust fields."
            : `Only ${x.data.owner_name ?? "its owner"} adjusts ${x.data.name}.`,
        };
  }
  if (v?.kind === "baseline") return { source: "baseline", canEdit: false, note: "A finalized baseline: read-only." };
  if (!me || (ds.phase === "assessing" && !team.data)) return { source: "team", canEdit: false, note: "Loading…" };
  return accessOf(ds, me, team.data);
}

export const useExperiment = (id: number | null) =>
  useQuery({ queryKey: ["experiment", id], queryFn: () => api.experiment(id!), enabled: id !== null });

export const useFields = (id: number, enabled = true, view: View = null) =>
  useQuery({ queryKey: ["fields", id, viewKey(view)], queryFn: () => api.fields(id, view), enabled });

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** Everything that reads the decisions or the team state goes stale when the phase or an assessment changes. */
export function useInvalidateTeam(datasetId: number) {
  const qc = useQueryClient();
  return () => {
    for (const k of [
      "team",
      "reviewQueue",
      "dataset",
      "fields",
      "fieldQuery",
      "exportPreview",
      "corrMatrix",
      "analysis",
      "versions",
      "baselines",
      "recipeDiff",
      "experiments",
      "exports",
      "runs",
    ]) {
      qc.invalidateQueries({ queryKey: [k, datasetId] });
    }
    qc.invalidateQueries({ queryKey: ["datasets"] });
    qc.invalidateQueries({ queryKey: ["finalizedRecipes"] });
    qc.invalidateQueries({ queryKey: ["field"] });
    qc.invalidateQueries({ queryKey: ["fieldReview"] });
    qc.invalidateQueries({ queryKey: ["experiment"] });
  };
}
