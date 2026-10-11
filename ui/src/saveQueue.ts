import { useSyncExternalStore } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { showError } from "./toast";
import type { Decision, DecisionInput, FieldListItem, FieldResponse } from "./types";
import { type View, viewKey, viewQuery } from "./view";

/*
 * Decision saves live outside React so leaving a field never loses an edit (#29):
 * one serial lane per field and view (an older payload can't land after a newer one, and an experiment's edit
 * never goes to the baseline),
 * the latest unsent draft wins, failures surface as a toast with Retry,
 * and pending drafts go out with keepalive when the page is hidden.
 */

interface Lane {
  fieldId: number;
  view: View;
  datasetId: number;
  name: string;
  queued: DecisionInput | null;
  inflight: DecisionInput | null;
  running: Promise<void> | null;
  timer: number | undefined;
}

let qc: QueryClient | null = null;
const lanes = new Map<string, Lane>();
const failed = new Set<string>();
const listeners = new Set<() => void>();
let failedSnapshot: ReadonlySet<string> = new Set();
const laneKey = (fieldId: number, view: View) => `${viewKey(view)}|${fieldId}`;
const emit = () => {
  failedSnapshot = new Set(failed);
  listeners.forEach((l) => l());
};

export function initSaveQueue(client: QueryClient) {
  qc = client;
  window.addEventListener("pagehide", () => {
    for (const lane of lanes.values()) {
      if (!lane.queued) continue;
      void fetch(`/api/fields/${lane.fieldId}/decision${viewQuery(lane.view)}`, {
        method: "PUT", keepalive: true,
        headers: { "content-type": "application/json" }, body: JSON.stringify(lane.queued),
      });
    }
  });
}

function lane(fieldId: number, view: View, datasetId: number, name: string): Lane {
  const key = laneKey(fieldId, view);
  let l = lanes.get(key);
  if (!l) {
    l = { fieldId, view, datasetId, name, queued: null, inflight: null, running: null, timer: undefined };
    lanes.set(key, l);
  }
  return l;
}

/** Queue a save; `delay` debounces typing, 0 sends at once. */
export function queueSave(fieldId: number, datasetId: number, name: string, input: DecisionInput, delay = 0,
  view: View = null): Promise<void> {
  const l = lane(fieldId, view, datasetId, name);
  const key = laneKey(fieldId, view);
  l.queued = input;
  window.clearTimeout(l.timer);
  if (delay > 0) {
    l.timer = window.setTimeout(() => void drain(key), delay);
    return Promise.resolve();
  }
  return drain(key);
}

export const flushSave = (fieldId: number, view: View = null) => drain(laneKey(fieldId, view));

/** The newest draft not yet confirmed by the server, so a remounted panel starts from it. */
export function pendingDraft(fieldId: number, view: View = null): DecisionInput | null {
  const l = lanes.get(laneKey(fieldId, view));
  return l ? l.queued ?? l.inflight : null;
}

async function drain(key: string): Promise<void> {
  const l = lanes.get(key);
  if (!l) return;
  window.clearTimeout(l.timer);
  if (l.running) {
    await l.running;
    return l.queued ? drain(key) : undefined;
  }
  if (!l.queued) return;
  const input = l.queued;
  l.queued = null;
  l.inflight = input;
  l.running = (async () => {
    try {
      const saved = await api.saveDecision(l.fieldId, input, l.view);
      if (failed.delete(key)) emit();
      applySaved(l.fieldId, l.datasetId, saved, l.view);
    } catch (err) {
      failed.add(key);
      emit();
      // Keep the failed draft unless a newer edit already replaced it.
      if (!l.queued) l.queued = input;
      showError(`Save failed for ${l.name}`, err, { label: "Retry", run: () => void drain(key) });
    } finally {
      l.inflight = null;
      l.running = null;
    }
  })();
  await l.running;
  if (l.queued && !failed.has(key)) return drain(key);
}

function applySaved(fieldId: number, datasetId: number, saved: Decision, view: View) {
  if (!qc) return;
  const vk = viewKey(view);
  const prev = qc.getQueryData<FieldResponse>(["field", fieldId, vk]);
  qc.setQueryData<FieldResponse>(["field", fieldId, vk], (old) => (old ? { ...old, decision: saved } : old));
  qc.setQueryData<FieldListItem[]>(["fields", datasetId, vk], (list) =>
    list?.map((f) => (f.id === fieldId
      ? { ...f, action: saved.action, semantic_type: saved.semantic_type, reviewed: saved.reviewed, source: saved.source }
      : f)),
  );
  for (const k of ["exportPreview", "fieldQuery", "corrMatrix", "analysis", "team", "reviewQueue"]) {
    qc.invalidateQueries({ queryKey: [k, datasetId] });
  }
  if (!view) qc.invalidateQueries({ queryKey: ["datasets"] });
  if (view) {
    qc.invalidateQueries({ queryKey: ["experiment"] });
    qc.invalidateQueries({ queryKey: ["experiments", datasetId] });
  }
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

/** Field ids whose last save in `view` failed, for the list's "!" mark. */
export function useFailedSaves(view: View = null): ReadonlySet<number> {
  const all = useSyncExternalStore(subscribe, () => failedSnapshot);
  const prefix = `${viewKey(view)}|`;
  return new Set([...all].filter((k) => k.startsWith(prefix)).map((k) => Number(k.slice(prefix.length))));
}
