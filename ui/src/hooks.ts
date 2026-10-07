import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { Dataset } from "./types";

export const PROCESSING = new Set(["pending", "ingesting", "profiling", "correlating"]);

export const useOptions = () => useQuery({ queryKey: ["options"], queryFn: api.options, staleTime: Infinity });

export const useDataset = (id: number) =>
  useQuery({
    queryKey: ["dataset", id],
    queryFn: () => api.dataset(id),
    refetchInterval: (q) => (q.state.data && PROCESSING.has((q.state.data as Dataset).status) ? 1000 : false),
  });

export const useFields = (id: number, enabled = true) =>
  useQuery({ queryKey: ["fields", id], queryFn: () => api.fields(id), enabled });
