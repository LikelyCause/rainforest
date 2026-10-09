import { useSyncExternalStore } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { showError } from "./toast";
import type { Decision, DecisionInput, FieldListItem, FieldResponse } from "./types";

/*
 * Decision saves live outside React so leaving a field never loses an edit (#29):
 * one serial lane per field (an older payload can't land after a newer one),
 * the latest unsent draft wins, failures surface as a toast with Retry,
 * and pending drafts go out with keepalive when the page is hidden.
 */

interface Lane {
  datasetId: number;
  name: string;
  queued: DecisionInput | null;
  inflight: DecisionInput | null;
  running: Promise<void> | null;
  timer: number | undefined;
}

let qc: QueryClient | null = null;
const lanes = new Map<number, Lane>();
const failed = new Set<number>();
const listeners = new Set<() => void>();
let failedSnapshot: ReadonlySet<number> = new Set();
const emit = () => {
  failedSnapshot = new Set(failed);
  listeners.forEach((l) => l());
};

export function initSaveQueue(client: QueryClient) {
  qc = client;
  window.addEventListener("pagehide", () => {
    for (const [fieldId, lane] of lanes) {
      if (!lane.queued) continue;
      void fetch(`/api/fields/${fieldId}/decision`, {
        method: "PUT", keepalive: true,
        headers: { "content-type": "application/json" }, body: JSON.stringify(lane.queued),
      });
    }
  });
}

function lane(fieldId: number, datasetId: number, name: string): Lane {
  let l = lanes.get(fieldId);
  if (!l) {
    l = { datasetId, name, queued: null, inflight: null, running: null, timer: undefined };
    lanes.set(fieldId, l);
  }
  return l;
}

/** Queue a save; `delay` debounces typing, 0 sends at once. */
export function queueSave(fieldId: number, datasetId: number, name: string, input: DecisionInput, delay = 0): Promise<void> {
  const l = lane(fieldId, datasetId, name);
  l.queued = input;
  window.clearTimeout(l.timer);
  if (delay > 0) {
    l.timer = window.setTimeout(() => void drain(fieldId), delay);
    return Promise.resolve();
  }
  return drain(fieldId);
}

export const flushSave = (fieldId: number) => drain(fieldId);

/** The newest draft not yet confirmed by the server, so a remounted panel starts from it. */
export function pendingDraft(fieldId: number): DecisionInput | null {
  const l = lanes.get(fieldId);
  return l ? l.queued ?? l.inflight : null;
}

async function drain(fieldId: number): Promise<void> {
  const l = lanes.get(fieldId);
  if (!l) return;
  window.clearTimeout(l.timer);
  if (l.running) {
    await l.running;
    return l.queued ? drain(fieldId) : undefined;
  }
  if (!l.queued) return;
  const input = l.queued;
  l.queued = null;
  l.inflight = input;
  l.running = (async () => {
    try {
      const saved = await api.saveDecision(fieldId, input);
      if (failed.delete(fieldId)) emit();
      applySaved(fieldId, l.datasetId, saved);
    } catch (err) {
      failed.add(fieldId);
      emit();
      // Keep the failed draft unless a newer edit already replaced it.
      if (!l.queued) l.queued = input;
      showError(`Save failed for ${l.name}`, err, { label: "Retry", run: () => void drain(fieldId) });
    } finally {
      l.inflight = null;
      l.running = null;
    }
  })();
  await l.running;
  if (l.queued && !failed.has(fieldId)) return drain(fieldId);
}

function applySaved(fieldId: number, datasetId: number, saved: Decision) {
  if (!qc) return;
  const prev = qc.getQueryData<FieldResponse>(["field", fieldId, saved.family]);
  qc.setQueryData<FieldResponse>(["field", fieldId, saved.family], (old) => (old ? { ...old, decision: saved } : old));
  // Shared parts (keep/drop, type, notes) changed for every family; refetch the others when shown.
  qc.invalidateQueries({ queryKey: ["field", fieldId], predicate: (q) => q.queryKey[2] !== saved.family });
  qc.setQueryData<FieldListItem[]>(["fields", datasetId], (list) =>
    list?.map((f) => (f.id === fieldId
      ? { ...f, action: saved.action, semantic_type: saved.semantic_type, reviewed: saved.reviewed, source: saved.source }
      : f)),
  );
  for (const k of ["summary", "exportPreview", "fieldQuery", "corrMatrix", "analysis"]) qc.invalidateQueries({ queryKey: [k, datasetId] });
  const wasTarget = prev?.decision.action === "target";
  if (wasTarget !== (saved.action === "target")) {
    // Target set changed: associations, leakage and the dataset's target list all move (#30).
    qc.invalidateQueries({ queryKey: ["corr"] });
    qc.invalidateQueries({ queryKey: ["field"] });
    for (const k of ["fields", "dataset"]) qc.invalidateQueries({ queryKey: [k, datasetId] });
  }
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

/** Field ids whose last save failed, for the list's "!" mark. */
export const useFailedSaves = () => useSyncExternalStore(subscribe, () => failedSnapshot);
