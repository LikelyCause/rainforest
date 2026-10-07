import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { PROCESSING, useDataset, useFields, useOptions } from "../hooks";
import { FieldList, type StatusFilter } from "../components/FieldList";
import { FieldDetailView } from "../components/FieldDetailView";
import { DecisionPanel, type DecisionPanelHandle } from "../components/DecisionPanel";
import type { FieldListItem } from "../types";

function useSessionState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => {
    try {
      const raw = sessionStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  const set = useCallback((next: T) => {
    setV(next);
    try { sessionStorage.setItem(key, JSON.stringify(next)); } catch { /* storage unavailable */ }
  }, [key]);
  return [v, set];
}

const isTyping = (el: EventTarget | null) =>
  el instanceof HTMLElement && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName));

export function ReviewPage() {
  const params = useParams();
  const datasetId = Number(params.datasetId);
  const fieldId = params.fieldId ? Number(params.fieldId) : null;
  const nav = useNavigate();
  const qc = useQueryClient();
  const searchRef = useRef<HTMLInputElement>(null);
  const panel = useRef<DecisionPanelHandle>(null);

  const ds = useDataset(datasetId);
  const ready = ds.data?.status === "ready";
  const family = ds.data?.model_family ?? "tree";
  const fields = useFields(datasetId, ready);
  const options = useOptions();

  const [search, setSearch] = useSessionState(`search:${datasetId}`, "");
  const [typeList, setTypeList] = useSessionState<string[]>(`types:${datasetId}`, []);
  const [status, setStatus] = useSessionState<StatusFilter>(`status:${datasetId}`, "all");
  const types = useMemo(() => new Set(typeList), [typeList]);

  const all = fields.data ?? [];
  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    return all.filter((f) =>
      (!q || f.name.toLowerCase().includes(q)) &&
      (types.size === 0 || types.has(f.semantic_type)) &&
      (status === "all" || (status === "unreviewed" ? !f.reviewed : f.action === status)),
    );
  }, [all, search, types, status]);

  // Pick a field when none is in the URL: first unreviewed, else first.
  useEffect(() => {
    if (fieldId === null && all.length) {
      const first = all.find((f) => !f.reviewed) ?? all[0];
      nav(`/d/${datasetId}/review/${first.id}`, { replace: true });
    }
  }, [fieldId, all, datasetId, nav]);

  const go = useCallback((id: number | undefined) => {
    if (id !== undefined) nav(`/d/${datasetId}/review/${id}`);
  }, [datasetId, nav]);

  const list: FieldListItem[] = visible.length ? visible : all;
  const idx = list.findIndex((f) => f.id === fieldId);
  const next = useCallback(() => go(list[idx + 1]?.id ?? list[idx]?.id), [go, list, idx]);
  const prev = useCallback(() => go(list[Math.max(idx - 1, 0)]?.id), [go, list, idx]);
  const nextUnreviewed = useCallback(() => {
    const after = [...list.slice(idx + 1), ...list.slice(0, Math.max(idx, 0))];
    go(after.find((f) => !f.reviewed && f.id !== fieldId)?.id);
  }, [go, list, idx, fieldId]);

  const detail = useQuery({
    queryKey: ["field", fieldId, family],
    queryFn: () => api.field(fieldId!, family),
    enabled: fieldId !== null && ready,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  // Prefetch neighbours so stepping through fields feels instant.
  useEffect(() => {
    if (idx < 0) return;
    for (const f of [list[idx + 1], list[idx - 1], list[idx + 2]]) {
      if (!f) continue;
      qc.prefetchQuery({ queryKey: ["field", f.id, family], queryFn: () => api.field(f.id, family), staleTime: 60_000 });
      qc.prefetchQuery({ queryKey: ["corr", f.id], queryFn: () => api.correlations(f.id, 40), staleTime: Infinity });
    }
  }, [idx, list, qc, family]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTyping(e.target)) return;
      const k = e.key;
      if (k === "j" || k === "ArrowRight" || k === "ArrowDown") { e.preventDefault(); next(); }
      else if (k === "k" || k === "ArrowLeft" || k === "ArrowUp") { e.preventDefault(); prev(); }
      else if (k === "u") nextUnreviewed();
      else if (k === "x") panel.current?.toggleKeep();
      else if (k === "a") panel.current?.acceptSuggestion();
      else if (k === "Enter") { e.preventDefault(); panel.current?.markReviewed(); }
      else if (k === "/") { e.preventDefault(); searchRef.current?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, nextUnreviewed]);

  if (ds.isLoading) return <div className="empty">Loading…</div>;
  if (ds.error || !ds.data) return <div className="empty">Dataset not found. <Link to="/">Back to datasets</Link></div>;
  if (PROCESSING.has(ds.data.status)) {
    return (
      <div className="page" style={{ maxWidth: 560 }}>
        <h1>Processing {ds.data.name}</h1>
        <div className="meter"><span style={{ width: `${Math.max(ds.data.progress, 0.02) * 100}%` }} /></div>
        <p className="secondary">{ds.data.status_message}</p>
      </div>
    );
  }
  if (ds.data.status === "failed") {
    return <div className="page"><div className="alert bad">✕ Processing failed: {ds.data.error?.split("\n")[0]}</div></div>;
  }

  const current = detail.data;
  return (
    <div className="review">
      <FieldList
        fields={all}
        visible={visible}
        activeId={fieldId}
        search={search}
        onSearch={setSearch}
        types={types}
        onToggleType={(t) => setTypeList(types.has(t) ? typeList.filter((x) => x !== t) : [...typeList, t])}
        status={status}
        onStatus={setStatus}
        onSelect={go}
        searchRef={searchRef}
      />
      <section className="detail" aria-busy={detail.isFetching} style={{ opacity: detail.isPlaceholderData ? 0.6 : 1, transition: "opacity 0.12s" }}>
        {detail.error && <div className="alert bad">{(detail.error as Error).message}</div>}
        {current && <FieldDetailView data={current} onOpenField={go} />}
        {!current && !detail.error && <div className="empty">Loading field…</div>}
      </section>
      {current && options.data && !detail.isPlaceholderData ? (
        <DecisionPanel key={`${current.field.id}:${current.family}`} ref={panel} data={current} catalog={options.data} onAdvance={next} />
      ) : (
        <aside className="decision" />
      )}
    </div>
  );
}
