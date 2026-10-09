import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { Dataset } from "./types";

export const PROCESSING = new Set(["pending", "ingesting", "profiling", "correlating"]);

export const useOptions = () => useQuery({ queryKey: ["options"], queryFn: api.options, staleTime: Infinity });

export const useDataset = (id: number, enabled = true) =>
  useQuery({
    queryKey: ["dataset", id],
    queryFn: () => api.dataset(id),
    enabled,
    refetchInterval: (q) => (q.state.data && PROCESSING.has((q.state.data as Dataset).status) ? 1000 : false),
  });

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
