import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useAccess, useExperiment, useFields, useOptions } from "../hooks";
import { JoinAssessment } from "../components/JoinAssessment";
import { parseView, useView, viewKey, withView, type View } from "../view";
import { useDatasetContext } from "../components/DatasetLayout";
import { FieldList, type StatusFilter, type WorkingSet } from "../components/FieldList";
import { FieldDetailView } from "../components/FieldDetailView";
import { DecisionPanel, type Confirmed, type DecisionPanelHandle, type PanelLens } from "../components/DecisionPanel";
import { Empty, ErrorAlert } from "../components/ui";
import { Banner, Dialog, Icon } from "../components/kit";
import { attentionOf } from "../signals";
import { queueSave, useFailedSaves } from "../saveQueue";
import { baselineLabel, familyShort, fmtInt } from "../format";
import { toast } from "../toast";
import type { Access } from "../team";
import type { Dataset, FieldListItem } from "../types";

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

export function readWorkingSet(key: string | null): WorkingSet | null {
  if (!key) return null;
  try {
    const raw = sessionStorage.getItem(`set:${key}`);
    return raw ? { key, ...(JSON.parse(raw) as Omit<WorkingSet, "key">) } : null;
  } catch {
    return null;
  }
}

const isControl = (el: EventTarget | null) =>
  el instanceof HTMLElement && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName));
const ACTIVATES_ON_ENTER = "button, a, summary, [role=button], [role=option], [role=radio], input, select, textarea";
const UNDO_DEPTH = 20;

const SHORTCUTS: [string[], string][] = [
  [["J", "→"], "next field"], [["K", "←"], "previous field"], [["U"], "next unreviewed"], [["X"], "toggle keep / drop"],
  [["↵"], "confirm and go next"], [["A"], "use the suggestion and confirm"], [["Z"], "undo the last confirm"],
  [["N"], "add a note"], [["/"], "search fields"], [["Esc"], "leave a control, back to review"], [["?"], "show or hide this list"],
];

/** Whose decisions are on screen, and the name the decision panel gives them; matches the lens in the bar above. */
function usePanelLens(ds: Dataset, view: View, access: Access): { lens: PanelLens; owner: string } {
  const v = parseView(view);
  const x = useExperiment(v?.kind === "experiment" ? v.id : null);
  if (v?.kind === "experiment") return { lens: "experiment", owner: x.data?.name ?? "This experiment" };
  if (v?.kind === "baseline") {
    const line = ds.baselines.find((l) => l.latest_finalized_id === v.id || l.id === v.id);
    return { lens: "frozen", owner: line ? baselineLabel(line.family, line.version, true, line.name) : "Finalized baseline" };
  }
  if (access.source === "mine") return { lens: "private", owner: "Your assessment" };
  const b = ds.baseline;
  if (ds.phase === "finalized") return { lens: "frozen", owner: b ? baselineLabel(b.family, b.version, true, b.name) : "Finalized baseline" };
  if (ds.phase === "review" && !b?.setup) return { lens: "team", owner: "Team decision" };
  return { lens: "suggestions", owner: "Suggestions" };
}

type Finish = null | { kind: "end"; left: number } | { kind: "complete" } | { kind: "setdone" };

export function ReviewPage() {
  const ds = useDatasetContext();
  const datasetId = ds.id;
  const params = useParams();
  const fieldId = params.fieldId ? Number(params.fieldId) : null;
  const [query, setQuery] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const searchRef = useRef<HTMLInputElement>(null);
  const panel = useRef<DecisionPanelHandle>(null);
  const undoStack = useRef<Confirmed[]>([]);
  // An undo replays a save into the current view, so it never crosses from one view to another.
  const undoView = useRef<string | null>(null);
  const view = useView();
  const failed = useFailedSaves(view);
  const access = useAccess(ds, view);
  // Finishing an assessment leads to Submit, in the bar above; for everyone else that bar holds the next step.
  const assessor = access.source === "mine";
  const fields = useFields(datasetId, true, view);
  const options = useOptions();
  const { lens, owner } = usePanelLens(ds, view, access);

  const setKey = query.get("set");
  const workingSet = useMemo(() => readWorkingSet(setKey), [setKey]);
  const [search, setSearch] = useSessionState(`search:${datasetId}`, "");
  const [typeList, setTypeList] = useSessionState<string[]>(`types:${datasetId}`, []);
  // In an experiment or a finalized baseline every field counts as reviewed, so the list starts with all of them.
  const [status, setStatus] = useSessionState<StatusFilter>(`status:${datasetId}:${viewKey(view)}`, view || ds.phase === "finalized" ? "all" : "todo");
  const [attention, setAttention] = useSessionState(`attention:${datasetId}`, false);
  const types = useMemo(() => new Set(typeList), [typeList]);
  // Confirmed while "To do" is showing: stays in place (faded) until the filter changes, so K still reaches it.
  const [settled, setSettled] = useState<Set<number>>(new Set());
  useEffect(() => setSettled(new Set()), [search, typeList, status, attention, setKey]);

  const [finish, setFinish] = useState<Finish>(null);
  const [overlay, setOverlay] = useState(false);
  const [typing, setTyping] = useState(false);
  const [panelVersion, setPanelVersion] = useState(0);
  const [drawer, setDrawer] = useState(false);

  const all = fields.data ?? [];
  const scope = useMemo(() => {
    if (!workingSet) return all;
    const byId = new Map(all.map((f) => [f.id, f]));
    return workingSet.ids.map((id) => byId.get(id)).filter((f): f is FieldListItem => Boolean(f));
  }, [all, workingSet]);
  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    return scope.filter((f) => settled.has(f.id) || (
      (!q || f.name.toLowerCase().includes(q)) &&
      (workingSet !== null || (
        (types.size === 0 || types.has(f.semantic_type)) &&
        (!attention || (!f.reviewed && ((attentionOf(f) && attentionOf(f) !== "edited") || failed.has(f.id)))) &&
        (status === "all" || (status === "todo" ? !f.reviewed : f.reviewed && f.action === (status === "kept" ? "keep" : "drop")))
      ))
    ));
  }, [scope, search, types, status, attention, settled, workingSet, failed]);

  // No field in the URL: open the first to-do field of the current scope.
  useEffect(() => {
    if (fieldId === null && scope.length) {
      const first = scope.find((f) => !f.reviewed) ?? scope[0];
      nav({ pathname: `/d/${datasetId}/review/${first.id}`, search: query.toString() }, { replace: true });
    }
  }, [fieldId, scope, datasetId, nav, query]);

  const go = useCallback((id: number | undefined) => {
    if (id === undefined) return;
    setFinish(null);
    setDrawer(false);
    nav({ pathname: `/d/${datasetId}/review/${id}`, search: query.toString() });
  }, [datasetId, nav, query]);

  // Navigation follows the rows on screen only; an empty filter goes nowhere (#54).
  const idx = visible.findIndex((f) => f.id === fieldId);
  const next = useCallback(() => {
    if (idx >= 0 && idx === visible.length - 1) {
      const left = scope.filter((f) => !f.reviewed).length;
      if (left) setFinish({ kind: "end", left });
      return;
    }
    go(visible[idx + 1]?.id);
  }, [go, visible, idx, scope]);
  const prev = useCallback(() => go(visible[Math.max(idx - 1, 0)]?.id), [go, visible, idx]);
  const nextUnreviewed = useCallback(() => {
    const after = visible.slice(idx + 1).find((f) => !f.reviewed);
    if (after) return go(after.id);
    const wrapped = visible.slice(0, Math.max(idx, 0)).find((f) => !f.reviewed);
    if (wrapped) {
      toast({ message: "Wrapped to top" });
      go(wrapped.id);
    }
  }, [go, visible, idx]);

  const detail = useQuery({
    queryKey: ["field", fieldId, viewKey(view)],
    queryFn: () => api.field(fieldId!, view),
    enabled: fieldId !== null,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  // Remembered so the Datasets list can continue where you left off.
  useEffect(() => {
    if (fieldId === null) return;
    try { localStorage.setItem(`rf:last:${datasetId}`, String(fieldId)); } catch { /* storage unavailable */ }
  }, [datasetId, fieldId]);

  // Prefetch neighbours so stepping through fields feels instant.
  useEffect(() => {
    if (idx < 0) return;
    for (const f of [visible[idx + 1], visible[idx - 1], visible[idx + 2]]) {
      if (!f) continue;
      qc.prefetchQuery({ queryKey: ["field", f.id, viewKey(view)], queryFn: () => api.field(f.id, view), staleTime: 60_000 });
      qc.prefetchQuery({ queryKey: ["corr", f.id], queryFn: () => api.correlations(f.id, 40), staleTime: Infinity });
    }
  }, [idx, visible, qc, view]);

  const onConfirmed = useCallback((c: Confirmed) => {
    if (undoView.current !== view) undoStack.current = [];
    undoView.current = view;
    undoStack.current = [...undoStack.current.slice(-(UNDO_DEPTH - 1)), c];
    const verb = c.after.action === "drop" ? "dropped" : c.after.action === "target" ? "target" : "kept";
    toast({
      message: `${c.name} → ${verb} · reviewed`,
      action: { label: "Undo", kbd: "Z", run: () => undoRef.current() },
    });
    if (status === "todo" && !workingSet) setSettled((s) => new Set(s).add(c.fieldId));
    const remaining = scope.filter((f) => !f.reviewed && f.id !== c.fieldId);
    if (remaining.length === 0) {
      setFinish(workingSet ? { kind: "setdone" } : { kind: "complete" });
      return;
    }
    next();
  }, [next, scope, status, workingSet, view]);

  const undo = useCallback(() => {
    const last = undoView.current === view ? undoStack.current.pop() : undefined;
    if (!last) return toast({ message: "Nothing to undo" });
    void queueSave(last.fieldId, datasetId, last.name, last.before, 0, view);
    setPanelVersion((v) => v + 1);
    go(last.fieldId);
    toast({ message: `Undid ${last.name}` });
  }, [datasetId, go, view]);
  const undoRef = useRef(undo);
  undoRef.current = undo;

  // Review mode = focus is not inside a form control; then single keys drive the loop.
  useEffect(() => {
    const update = () => setTyping(isControl(document.activeElement));
    const later = () => window.setTimeout(update, 0);
    document.addEventListener("focusin", update);
    document.addEventListener("focusout", later);
    return () => {
      document.removeEventListener("focusin", update);
      document.removeEventListener("focusout", later);
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      if (k === "Escape") {
        if (overlay) return setOverlay(false);
        if (drawer) return setDrawer(false);
        if (isControl(e.target) || e.target instanceof HTMLButtonElement) (e.target as HTMLElement).blur();
        if (finish?.kind === "end") setFinish(null);
        return;
      }
      if (isControl(e.target)) return;
      if (k === "?") { e.preventDefault(); return setOverlay((o) => !o); }
      if (overlay) return;
      if (k === "Enter") {
        // Enter on a focused button or link runs that element, never a confirm (#27).
        if (e.target instanceof Element && e.target.closest(ACTIVATES_ON_ENTER)) return;
        e.preventDefault();
        if (finish?.kind === "end") return goFirstUnreviewed();
        if (!finish) panel.current?.confirm();
        return;
      }
      if (!access.canEdit && ["x", "a", "z", "n"].includes(k)) return;
      if (finish && !["j", "k", "u", "z", "ArrowRight", "ArrowLeft"].includes(k)) return;
      if (k === "j" || k === "ArrowRight") { e.preventDefault(); next(); }
      else if (k === "k" || k === "ArrowLeft") { e.preventDefault(); if (finish) setFinish(null); else prev(); }
      else if (k === "u") nextUnreviewed();
      else if (k === "x") panel.current?.toggleKeep();
      else if (k === "a") panel.current?.revertAndConfirm();
      else if (k === "z") undo();
      else if (k === "n") { e.preventDefault(); panel.current?.focusNote(); }
      else if (k === "/") {
        e.preventDefault();
        setDrawer(true);
        window.setTimeout(() => searchRef.current?.focus(), 0);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const goFirstUnreviewed = () => go(scope.find((f) => !f.reviewed)?.id);
  const clearFilters = () => { setSearch(""); setTypeList([]); setAttention(false); setStatus("all"); };
  const clearSet = () => {
    const q = new URLSearchParams(query);
    q.delete("set");
    setQuery(q);
  };

  const current = detail.data;
  const position = idx >= 0 ? { index: scope.findIndex((f) => f.id === fieldId), total: scope.length } : null;
  const missingField = detail.error && (detail.error as { status?: number }).status === 404;

  const typeLabel = (t: string) => options.data?.types.find((x) => x.value === t)?.label ?? t.replace("_", " ");
  const panelReady = Boolean(current && options.data && !detail.isPlaceholderData && !finish);
  const confirmable = panelReady && access.canEdit && !current?.locked;

  return (
    <div className={`review${!typing && !overlay && !finish && current ? " mode" : ""}${drawer ? " drawer-open" : ""}`} data-lens={lens}>
      <FieldList view={view} ds={ds}
        fields={scope} visible={visible} settled={settled} activeId={fieldId}
        search={search} onSearch={setSearch}
        types={types} onTypes={setTypeList}
        status={status} onStatus={setStatus}
        attention={attention} onAttention={setAttention}
        workingSet={workingSet} onClearSet={clearSet}
        onClearFilters={clearFilters} onSelect={go} searchRef={searchRef}
        onClose={() => setDrawer(false)}
      />
      <div className="drawer-scrim" onClick={() => setDrawer(false)} aria-hidden="true" />
      <section className="detail" aria-busy={detail.isFetching} aria-label="Field">
        {!view && <JoinAssessment ds={ds} />}
        {detail.isFetching && detail.isPlaceholderData && <div className="loading-bar" aria-hidden="true" />}
        {finish ? (
          <FinishCard finish={finish} scope={scope} datasetId={datasetId} workingSet={workingSet} assessor={assessor}
            onFirstUnreviewed={goFirstUnreviewed} onStay={() => setFinish(null)} onClearSet={clearSet}
            onOpen={go} />
        ) : missingField ? (
          <Empty title={`Field #${fieldId} isn't in this dataset`}
            action={<button type="button" className="primary" onClick={goFirstUnreviewed}>Go to first unreviewed</button>} />
        ) : detail.error ? (
          <ErrorAlert error={detail.error} />
        ) : current ? (
          <FieldDetailView data={current} position={position} stale={detail.isPlaceholderData} onOpenField={go}
            onPrev={prev} onNext={next} onOpenList={() => setDrawer(true)} typeLabel={options.data ? typeLabel : undefined} />
        ) : visible.length === 0 && all.length > 0 ? (
          <Empty title="No fields match these filters" action={<button type="button" onClick={clearFilters}>Clear filters</button>} />
        ) : (
          <div className="stack" aria-hidden="true">
            <span className="skeleton" style={{ width: 240, height: 22 }} />
            <span className="skeleton" style={{ width: 360 }} />
            <span className="skeleton" style={{ height: 220 }} />
          </div>
        )}
      </section>
      {current && options.data && panelReady ? (
        <DecisionPanel key={`${current.field.id}:${current.family}:${viewKey(view)}:${panelVersion}`} ref={panel} data={current}
          catalog={options.data} familyLabel={familyShort(current.family)} typing={typing} view={view}
          onConfirmed={onConfirmed} onShowShortcuts={() => setOverlay(true)}
          readOnly={access.canEdit ? (current.locked ?? null) : access.note}
          owner={owner} lens={lens} onNext={next} onNextTodo={nextUnreviewed} />
      ) : (
        <aside className="decision" data-lens={lens} aria-label="Decision" />
      )}
      <div className="phone-bar">
        <button type="button" className="phone-prev" onClick={prev} aria-label="Previous field"><Icon name="chevron-left" /></button>
        <button type="button" className="phone-list" onClick={() => setDrawer(true)} aria-expanded={drawer}>
          Fields · {position ? `${fmtInt(position.index + 1)} of ${fmtInt(position.total)}` : fmtInt(scope.length)}
        </button>
        {confirmable ? (
          <button type="button" className="primary phone-confirm" onClick={() => panel.current?.confirm()}>Confirm</button>
        ) : (
          <button type="button" className="phone-confirm" onClick={next} aria-label="Next field">Next <Icon name="chevron-right" /></button>
        )}
      </div>
      {overlay && <ShortcutsOverlay onClose={() => setOverlay(false)} />}
    </div>
  );
}

function FinishCard({ finish, scope, datasetId, workingSet, assessor, onFirstUnreviewed, onStay, onClearSet, onOpen }: {
  finish: NonNullable<Finish>; scope: FieldListItem[]; datasetId: number; workingSet: WorkingSet | null; assessor: boolean;
  onFirstUnreviewed: () => void; onStay: () => void; onClearSet: () => void; onOpen: (id: number) => void;
}) {
  const firstTodo = scope.find((f) => !f.reviewed);
  const finishView = useView();
  if (finish.kind === "end") {
    return (
      <div className="finish card">
        <h2>End of the list: {finish.left} field{finish.left === 1 ? "" : "s"} still to review</h2>
        <div className="row">
          <button type="button" className="primary" onClick={onFirstUnreviewed} autoFocus>
            Jump to first unreviewed{firstTodo ? ` · ${firstTodo.name}` : ""} <kbd>↵</kbd>
          </button>
          <button type="button" onClick={onStay}>Stay here <kbd>Esc</kbd></button>
        </div>
      </div>
    );
  }
  if (finish.kind === "setdone" && workingSet) {
    return (
      <div className="finish card">
        <h2>{scope.length} of {scope.length} reviewed in {workingSet.label}.</h2>
        <div className="row">
          <Link className="btn" to={withView(`/d/${datasetId}/bulk`, finishView)}>Back to the table</Link>
          <button type="button" onClick={onClearSet}>Clear set</button>
        </div>
      </div>
    );
  }
  const kept = scope.filter((f) => f.action === "keep").length;
  const dropped = scope.filter((f) => f.action === "drop").length;
  const targets = scope.filter((f) => f.action === "target").length;
  const leaks = scope.filter((f) => f.action !== "target" && attentionOf(f) === "leak");
  const keptLeaks = leaks.filter((f) => f.action === "keep");
  return (
    <div className="finish card">
      <h2>Review complete</h2>
      <div className="counts num">
        <span>Kept <b>{kept}</b></span><span>Dropped <b>{dropped}</b></span><span>Targets <b>{targets}</b></span>
      </div>
      {leaks.length > 0 && (keptLeaks.length ? (
        <Banner tone="bad" title={`${keptLeaks.length} leaky field${keptLeaks.length === 1 ? "" : "s"} kept:`}>
          {keptLeaks.map((f, i) => (
            <span key={f.id}>{i > 0 && ", "}<button type="button" className="link" onClick={() => onOpen(f.id)}>{f.name}</button></span>
          ))}
        </Banner>
      ) : (
        <span className="secondary">Leakage flags: {leaks.length} dropped, 0 kept.</span>
      ))}
      <p className="secondary">
        {assessor ? "Every field is reviewed. Submit your assessment with the button at the top right; until then you can still change any field."
          : "Every field is reviewed. The next step is at the top right."}
      </p>
      <div className="row">
        <button type="button" onClick={onStay}>Back to the field</button>
      </div>
    </div>
  );
}

function ShortcutsOverlay({ onClose }: { onClose: () => void }) {
  return (
    <Dialog title="Keyboard shortcuts" onClose={onClose} labelledBy="shortcuts-title"
      actions={<button type="button" onClick={onClose} data-autofocus>Close</button>}>
      <div className="shortcut-grid">
        {SHORTCUTS.map(([keys, label]) => (
          <div key={label}>{keys.map((k) => <kbd key={k}>{k}</kbd>)}<span>{label}</span></div>
        ))}
      </div>
    </Dialog>
  );
}
