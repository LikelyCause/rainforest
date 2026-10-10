import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { accessOf, type Access } from "./team";
import type { Dataset } from "./types";

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
export function useAccess(ds: Dataset): Access {
  const me = useMe().data;
  const team = useTeam(ds.id, ds.phase === "assessing");
  if (!me || (ds.phase === "assessing" && !team.data)) return { source: "team", canEdit: false, note: "Loading…" };
  return accessOf(ds, me, team.data);
}

export const useFields = (id: number, enabled = true) =>
  useQuery({ queryKey: ["fields", id], queryFn: () => api.fields(id), enabled });

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}
