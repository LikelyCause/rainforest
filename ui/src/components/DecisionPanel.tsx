import { forwardRef, Fragment, useEffect, useImperativeHandle, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { flushSave, pendingDraft, queueSave, useFailedSaves } from "../saveQueue";
import { bandCount, bandsText, defaultTask, NUMERIC, paramProblem, parseEdges, retype, taskLabel, toInput, withEncoding, type ParamRule } from "../decision";
import { baselineDiff, revertedInput } from "../experiments";
import type { Action, DecisionInput, Dim, FieldResponse, OptionCatalog, TargetBins, TargetTask } from "../types";
import { parseView, type View } from "../view";
import { useDataset } from "../hooks";
import { Chip, Icon, LensChip, Segmented } from "./kit";
import { InfoTip } from "./ui";

const DIMS: Dim[] = ["imputation", "transform", "scaling", "encoding"];
const DIM_LABEL: Record<Dim, string> = { imputation: "Missing values", transform: "Transform", scaling: "Scaling", encoding: "Encoding" };
const SAVE_DEBOUNCE_MS = 400;

/** Whose decision the panel shows; the same colour as the lens chip in the bar above. */
export type PanelLens = "private" | "team" | "experiment" | "frozen" | "suggestions";
const LENS_WORD: Record<PanelLens, string> = { private: "Private", team: "Shared", experiment: "Experiment", frozen: "Frozen", suggestions: "Read-only" };
const LENS_TITLE: Record<PanelLens, string> = {
  private: "Your assessment", team: "Team decision", experiment: "This experiment", frozen: "Frozen snapshot", suggestions: "Suggestions",
};

/** What each assessor chose for a setting, in team review: shown in small text under that setting. */
export type VoteKey = "action" | "semantic_type" | "target" | "imputation" | "transform" | "scaling" | "encoding" | "impute_value" | "add_missing_indicator";

export interface DecisionPanelHandle {
  confirm: () => void;
  revertAndConfirm: () => void;
  toggleKeep: () => void;
  focusNote: () => void;
}

export interface Confirmed {
  fieldId: number;
  name: string;
  before: DecisionInput;
  after: DecisionInput;
}

interface Props {
  data: FieldResponse;
  catalog: OptionCatalog;
  familyLabel: string;
  typing: boolean;
  onConfirmed: (c: Confirmed) => void;
  onShowShortcuts: () => void;
  /** Why this decision can't be changed here; the panel then shows plain values and this reason. */
  readOnly?: string | null;
  /** Whose decision this is, as the header names it ("Your assessment", an experiment's name). */
  owner?: string;
  /** The decision source (see view.ts); null for the baseline in progress. */
  view?: View;
  /** The header's lens; derived from owner, view and phase when left out. */
  lens?: PanelLens;
  /** Each assessor's choice per setting, in team review. */
  votes?: Partial<Record<VoteKey, string>>;
  /** Team review: the confirm button settles the field ("Settle as keep, category"); a string replaces "Settle". */
  settleLabel?: boolean | string;
  /** Read-only: moves on to the next field (the footer button). */
  onNext?: () => void;
  /** A carried-over field: moves on to the next field still to do. */
  onNextTodo?: () => void;
  /** The footer's status line in place of the assessing one ("Open · 2 comments · field 1 of 6 open", "12 fields
   * differ from Trees v1", "Finalized by … You can view, not change."); a failed save still says so after it. */
  footLine?: ReactNode;
  /** A second, quieter footer line: what the button does next ("Settling moves you to the next open field"). */
  footNote?: string;
  /** Told the decision on screen (keep, drop, target) whenever it changes, for the phone bar's confirm button. */
  onAction?: (a: Action) => void;
}

function retypeWarning(type: string, data: FieldResponse): string | null {
  const f = data.field;
  if (type === f.inferred_type) return null;
  if (type === "binary" && f.n_unique > 2) return `${f.n_unique.toLocaleString()} distinct values — Binary expects 2.`;
  if (type === "constant" && f.n_unique > 1) return `${f.n_unique.toLocaleString()} distinct values — not constant.`;
  if (NUMERIC.has(type) && !NUMERIC.has(f.inferred_type) && !f.stats?.numeric_like && !f.stats?.stored_as_text)
    return "Values weren't recognised as numbers; non-numeric ones become missing.";
  if (type === "categorical" && f.n_unique > 1000) return `${f.n_unique.toLocaleString()} levels — most will be grouped as rare.`;
  return null;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** How a target is predicted, in a few lower-case words for a button: "one vs rest <30", "4 bands". */
function taskWords(task: TargetTask, bins: TargetBins | null | undefined): string {
  if (task !== "bins" || !bins) return taskLabel(task).toLowerCase();
  if ("positive" in bins) return `one vs rest ${bins.positive.join(" / ")}`;
  return "k" in bins ? `${bins.k} bands` : `bands at ${bins.edges.join(", ")}`;
}
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export const DecisionPanel = forwardRef<DecisionPanelHandle, Props>(function DecisionPanel(
  { data, catalog, familyLabel, typing, onConfirmed, onShowShortcuts, readOnly, owner, view = null, lens: lensProp, votes, settleLabel, onNext, onNextTodo,
    footLine, footNote, onAction }, ref,
) {
  const f = data.field;
  const sugg = f.suggestion;
  const family = data.family;
  const server = toInput(data.decision, family);
  const [draft, setDraft] = useState<DecisionInput>(() => pendingDraft(f.id, view) ?? server);
  const draftRef = useRef(draft);
  const before = useRef(server);
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const [noteOpen, setNoteOpen] = useState(Boolean(draft.notes));
  const ds = useDataset(f.dataset_id);
  const saveFailed = useFailedSaves(view).has(f.id);
  useEffect(() => onAction?.(draft.action), [draft.action]);
  const v = parseView(view);
  const lens: PanelLens = lensProp ?? (owner === "Your assessment" ? "private" : v?.kind === "experiment" ? "experiment"
    : v?.kind === "baseline" || ds.data?.phase === "finalized" ? "frozen" : "team");
  const title = owner ?? LENS_TITLE[lens];

  const commit = (next: DecisionInput, delay = SAVE_DEBOUNCE_MS) => {
    if (readOnly) return;
    draftRef.current = next;
    setDraft(next);
    void queueSave(f.id, f.dataset_id, f.name, next, delay, view);
  };
  const update = (patch: Partial<DecisionInput>, delay?: number) => commit({ ...draftRef.current, ...patch }, delay);

  const suggested = toInput(sugg, family);
  const differs = (dim: Dim | "semantic_type" | "action" | "add_missing_indicator") => !same(draft[dim], suggested[dim]);
  const isTarget = draft.action === "target";
  const nonFeature = catalog.non_feature_types.includes(draft.semantic_type);
  const allowed = catalog.allowed[family][draft.semantic_type];
  const notes = catalog.family_notes[family] ?? {};
  const labels = catalog.labels;
  const typeLabel = (t: string) => catalog.types.find((x) => x.value === t)?.label ?? t;
  const optLabel = (dim: Dim, o: string) => labels[dim][o] ?? o;

  const editableDims = draft.action === "keep" ? DIMS.filter((d) => allowed[d].length > 1 || draft[d] !== allowed[d][0]) : [];
  const lockedDims = draft.action === "keep" ? DIMS.filter((d) => !editableDims.includes(d) && notes[d]) : [];
  // How a target is predicted counts as a change too: the suggestion is the type's default task.
  const task = isTarget ? draft.target_task ?? defaultTask(draft.semantic_type, f.n_unique) : null;
  const suggestedTask = defaultTask(sugg.semantic_type, f.n_unique);
  const taskEdited = isTarget && task !== null && (task !== suggestedTask || (task === "bins" && !same(draft.target_bins, suggested.target_bins)));
  const edited = [
    ...(["action", "semantic_type", ...editableDims, ...(NUMERIC.has(draft.semantic_type) && draft.action === "keep" ? ["add_missing_indicator" as const] : [])] as const)
      .filter((k) => differs(k)),
    ...(taskEdited ? ["target_task" as const] : []),
  ];

  const verb = { keep: "keep", drop: "drop", target: "target" }[draft.action];
  const confirmLabel = (() => {
    const how = isTarget && task ? `, ${taskWords(task, draft.target_bins)}` : "";
    if (settleLabel) return `${settleLabel === true ? "Settle" : settleLabel} as ${verb}, ${typeLabel(draft.semantic_type).toLowerCase()}${how}`;
    const unchanged = same({ ...draft, notes: null }, { ...server, notes: null }) && draft.reviewed;
    if (unchanged) return "Next field";
    const parts = edited.filter((k) => k !== "action").map((k) =>
      k === "semantic_type" ? typeLabel(draft.semantic_type).toLowerCase()
        : k === "add_missing_indicator" ? "missing flag"
        : k === "target_task" ? taskWords(task!, draft.target_bins)
        : `${optLabel(k as Dim, draft[k as Dim]).split(" (")[0].toLowerCase()}${k === "imputation" ? " fill" : ""}`);
    const shown = parts.slice(0, 2).join(", ");
    return `Confirm · ${verb}${shown ? `, ${shown}` : ""}${parts.length > 2 ? ` +${parts.length - 2}` : ""}`;
  })();

  const confirm = () => {
    if (readOnly) return;
    const after = { ...draftRef.current, reviewed: true };
    commit(after, 0);
    onConfirmed({ fieldId: f.id, name: f.name, before: before.current, after });
  };
  const revertAndConfirm = () => {
    if (readOnly) return;
    const after = { ...suggested, notes: draftRef.current.notes, reviewed: true };
    commit(after, 0);
    onConfirmed({ fieldId: f.id, name: f.name, before: before.current, after });
  };
  const toggleKeep = () => {
    const d = draftRef.current;
    if (d.action === "target" || (d.action === "drop" && catalog.non_feature_types.includes(d.semantic_type))) return;
    update({ action: d.action === "keep" ? "drop" : "keep" }, 0);
  };
  const focusNote = () => {
    if (readOnly) return;
    setNoteOpen(true);
    window.setTimeout(() => noteRef.current?.focus(), 0);
  };
  useImperativeHandle(ref, () => ({ confirm, revertAndConfirm, toggleKeep, focusNote }));

  // A select keeps keyboard focus after a mouse change, which used to swallow J/K (P2).
  const blurAfter = (fn: (e: ChangeEvent<HTMLSelectElement>) => void) => (e: ChangeEvent<HTMLSelectElement>) => {
    fn(e);
    e.currentTarget.blur();
  };
  const vote = (k: VoteKey) => (votes?.[k] ? <span className="vote">{votes[k]}</span> : null);
  const suggestedLine = (k: Dim | "semantic_type" | "action" | "add_missing_indicator", name: string, text: string) =>
    differs(k) ? (
      <span className="suggested">
        Suggested: {text}
        <button type="button" className="link" aria-label={`Reset ${name.toLowerCase()} to the suggestion, ${text}`}
          onClick={() => update({ [k]: suggested[k] } as Partial<DecisionInput>, 0)}>Reset</button>
      </span>
    ) : null;

  const param = (key: string, title: string, step = 1) => (
    <ParamInput key={`${draft.encoding}-${key}`} id={`p-${key}`} title={title} step={step}
      value={draft.params?.[key] ?? catalog.default_params[draft.encoding]?.[key]}
      rule={catalog.param_rules?.[draft.encoding]?.[key]}
      onChange={(n) => update({ params: { ...draftRef.current.params, [key]: n } })} />
  );
  const paramValue = (key: string) => draft.params?.[key] ?? catalog.default_params[draft.encoding]?.[key];
  const PARAMS: Record<string, [string, string][]> = {
    onehot: [["max_categories", "Max categories"], ["min_frequency", "Min frequency"]],
    ordinal: [["min_frequency", "Min frequency"]],
    tfidf_svd: [["svd_components", "SVD components"], ["max_features", "Max vocabulary"]],
  };

  const warning = retypeWarning(draft.semantic_type, data);
  const carried = Boolean(readOnly && data.locked && readOnly === data.locked);
  const fallbackTask = defaultTask(draft.semantic_type, f.n_unique);
  const fillLabel = ["categorical", "binary"].includes(draft.semantic_type) ? "Missing label" : "Fill value";
  const lockedLine = lockedDims.length > 0 && (
    <div className="locked-line">
      {lockedDims.map((d) => DIM_LABEL[d]).join(", ")}: not used for {familyLabel.toLowerCase()}
      <InfoTip label={<Icon name="info" size={14} />} ariaLabel={`Why ${lockedDims.map((d) => DIM_LABEL[d].toLowerCase()).join(" and ")} aren't used`}
        text={lockedDims.map((d) => notes[d]).join("\n")} />
    </div>
  );
  const why = sugg.reasons.length > 0 && (
    <details className="why" open>
      <summary>Why these suggestions</summary>
      <ul>{sugg.reasons.map((r) => <li key={r}><Reason text={r} /></li>)}</ul>
    </details>
  );
  const experimentBox = data.baseline !== undefined && (
    <ExperimentBox draft={draft} data={data} readOnly={Boolean(readOnly)} typeLabel={typeLabel}
      onRevert={() => data.baseline && commit(revertedInput(draftRef.current, data.baseline), 0)} />
  );

  // In an experiment a drop keeps the field's settings, so a revert brings them back as they were.
  const droppedNote = data.baseline !== undefined && draft.action === "drop" && (
    <p className="dropped-note">Dropped fields aren't exported; their settings are kept in case you revert.</p>
  );

  const header = (
    <div className="decision-head">
      <h2>{title}</h2>
      <LensChip lens={lens}>{LENS_WORD[lens]}</LensChip>
      <span className="spacer" />
      <Chip title="Keep / drop, type and notes are shared; the settings below are for this model">{familyLabel}</Chip>
    </div>
  );

  if (readOnly) {
    // Read-only, a setting that departs from the suggestion still says what was suggested (no Reset).
    const was = (k: Dim | "semantic_type" | "action" | "add_missing_indicator", text: string) =>
      differs(k) ? <span className="suggested">Suggested: {text}</span> : null;
    const bins = draft.target_bins;
    return (
      <aside className="decision" aria-label={`${title}, read-only`} data-lens={lens}>
        <div className="decision-body">
          {header}
          {carried && (
            <div className="carried-box">
              <Icon name="lock" />
              <p><b>Carried over from the version before.</b> <span className="secondary">Only reopened fields are assessed in this version.</span></p>
            </div>
          )}
          {experimentBox}
          <dl className="plain-values num">
            <div className="setting-row"><dt>Decision</dt><dd>{cap(draft.action)}{was("action", cap(sugg.action))}{vote("action")}</dd></div>
            {isTarget && task && (
              <div className="setting-row">
                <dt>Predict as</dt>
                <dd>
                  {bandsText(task, bins)}
                  {taskEdited && suggestedTask && <span className="suggested">Default for this type: {taskLabel(suggestedTask).toLowerCase()}</span>}
                  {vote("target")}
                  {task === "bins" && bins && !("positive" in bins) && <BandPreview fieldId={f.id} bins={bins} />}
                </dd>
              </div>
            )}
            <div className="setting-row"><dt>Type</dt><dd>{typeLabel(draft.semantic_type)}{was("semantic_type", typeLabel(sugg.semantic_type))}{vote("semantic_type")}</dd></div>
            {editableDims.map((dim) => (
              <Fragment key={dim}>
                <div className="setting-row"><dt>{DIM_LABEL[dim]}</dt><dd>{optLabel(dim, draft[dim])}{was(dim, optLabel(dim, sugg[dim]).split(" (")[0])}{vote(dim)}</dd></div>
                {dim === "imputation" && draft.imputation === "constant" && (
                  <div className="setting-row"><dt>{fillLabel}</dt><dd className="mono">{draft.impute_value ?? "—"}{vote("impute_value")}</dd></div>
                )}
                {dim === "encoding" && (PARAMS[draft.encoding] ?? []).map(([k, l]) => (
                  <div key={k} className="setting-row"><dt>{l}</dt><dd>{String(paramValue(k) ?? "—")}</dd></div>
                ))}
              </Fragment>
            ))}
            {draft.action === "keep" && NUMERIC.has(draft.semantic_type) && (
              <div className="setting-row"><dt>Missing flag</dt><dd className={draft.add_missing_indicator ? "" : "off"}>{draft.add_missing_indicator ? "On" : "Off"}{was("add_missing_indicator", suggested.add_missing_indicator ? "on" : "off")}{vote("add_missing_indicator")}</dd></div>
            )}
          </dl>
          {droppedNote}
          {lockedLine}
          {draft.notes && (
            <div className="plain-note">
              <span className="field-label">Note</span>
              <p>{draft.notes}</p>
            </div>
          )}
          {why}
        </div>
        <div className="decision-foot">
          <span className="foot-line">{footLine ?? readOnly}</span>
          {footNote && <span className="foot-line foot-note">{footNote}</span>}
          {carried && onNextTodo ? (
            <button type="button" className="primary large confirm" onClick={onNextTodo}><span>Next to-do field</span><kbd>U</kbd></button>
          ) : onNext && (
            <button type="button" className="large confirm" onClick={onNext}><span>Next field</span><kbd>J</kbd></button>
          )}
        </div>
      </aside>
    );
  }

  return (
    <aside className="decision" aria-label={title} data-lens={lens}>
      <div className="decision-body">
        {header}

        <div className={`setting action-setting${differs("action") ? " edited" : ""}`}>
          <Segmented full label="Keep, drop or target" value={draft.action}
            options={(["keep", "drop", "target"] as const).map((a) => ({
              value: a, label: cap(a), disabled: a === "keep" && nonFeature && !isTarget,
              title: a === "keep" && nonFeature ? `${draft.semantic_type} fields can't be model features` : a === "target" ? "Use as a target column" : "X toggles keep / drop",
            }))}
            onChange={(a) => update({ action: a }, 0)} />
          {suggestedLine("action", "Keep, drop or target", sugg.action)}
          {vote("action")}
        </div>

        {experimentBox}

        {isTarget && (() => {
          const task = draft.target_task ?? fallbackTask;
          const classField = draft.semantic_type === "binary" || draft.semantic_type === "categorical";
          const levels = f.charts.distribution?.kind === "bars" ? f.charts.distribution.labels : [];
          const allowedTask = (t: TargetTask) => (t === "classes"
            ? draft.semantic_type !== "numeric_continuous" && (!NUMERIC.has(draft.semantic_type) || (f.n_unique ?? 0) <= catalog.target_rule.max_levels)
            : t === "bins" ? NUMERIC.has(draft.semantic_type) || (classField && levels.length > 1) : NUMERIC.has(draft.semantic_type));
          const hint = (t: TargetTask) => (allowedTask(t) ? undefined
            : t === "classes" ? (draft.semantic_type === "numeric_continuous" ? "A continuous field is predicted as a number or as bands"
              : `More than ${catalog.target_rule.max_levels} distinct values`)
            : "Only numeric fields can be predicted as a number or as bands");
          return (
            <div className={`setting predict-as${task !== fallbackTask ? " edited" : ""}`}>
              <span className="setting-label" id="d-predict">Predict as</span>
              <Segmented full label="Predict as" value={task ?? ("" as TargetTask)}
                options={(["classes", "regression", "bins"] as const).map((t) => ({
                  value: t, label: t === "bins" && classField ? "One vs rest" : taskLabel(t), disabled: !allowedTask(t), title: hint(t),
                }))}
                onChange={(t) => update({ target_task: t, target_bins: t !== "bins" ? null
                  : classField ? (draftRef.current.target_bins && "positive" in draftRef.current.target_bins ? draftRef.current.target_bins : { positive: [levels[levels.length - 1]] })
                  : draftRef.current.target_bins ?? { k: catalog.target_rule.default_k } }, 0)} />
              {task !== fallbackTask && fallbackTask && <span className="suggested">Default for this type: {taskLabel(fallbackTask).toLowerCase()}</span>}
              {vote("target")}
              {task === "bins" && classField && (
                <PositivePicker levels={levels} positive={draft.target_bins && "positive" in draft.target_bins ? draft.target_bins.positive : []}
                  readOnly={false} onChange={(positive) => update({ target_task: "bins", target_bins: { positive } }, 0)} />
              )}
              {task === "bins" && !classField && (
                <BandSettings fieldId={f.id} bins={draft.target_bins ?? { k: catalog.target_rule.default_k }} readOnly={false}
                  onChange={(b) => update({ target_task: "bins", target_bins: b }, 0)} />
              )}
            </div>
          );
        })()}

        <div className="settings">
          <div className={`setting${differs("semantic_type") ? " edited" : ""}`}>
            <div className="setting-row">
              <label htmlFor="d-type">Type</label>
              <select id="d-type" value={draft.semantic_type}
                onChange={blurAfter((e) => commit(retype(draftRef.current, e.target.value, catalog, sugg), 0))}>
                {catalog.types.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            {warning && (
              <span className="inline-warn">
                {warning}{" "}
                <button type="button" className="link" onClick={() => commit(retype(draftRef.current, f.inferred_type, catalog, sugg), 0)}>
                  Undo retype
                </button>
              </span>
            )}
            {!warning && suggestedLine("semantic_type", "Type", typeLabel(sugg.semantic_type))}
            {vote("semantic_type")}
          </div>

          {editableDims.map((dim) => (
            <div key={dim} className={`setting${differs(dim) ? " edited" : ""}`}>
              {allowed[dim].length > 1 ? (
                <div className="setting-row">
                  <label htmlFor={`d-${dim}`}>{DIM_LABEL[dim]}</label>
                  <select id={`d-${dim}`} value={draft[dim]}
                    onChange={blurAfter((e) => dim === "encoding"
                      ? commit(withEncoding(draftRef.current, e.target.value, catalog), 0)
                      : update({ [dim]: e.target.value } as Partial<DecisionInput>, 0))}>
                    {allowed[dim].map((o) => <option key={o} value={o}>{optLabel(dim, o)}</option>)}
                  </select>
                </div>
              ) : (
                <div className="setting-row">
                  <span>{DIM_LABEL[dim]}</span>
                  <span className="secondary fixed-value">
                    {optLabel(dim, draft[dim])} (fixed for {familyLabel.toLowerCase()})
                    {notes[dim] && <InfoTip label={<Icon name="info" size={14} />} ariaLabel={`Why ${DIM_LABEL[dim].toLowerCase()} is fixed`} text={notes[dim]!} />}
                  </span>
                </div>
              )}
              {suggestedLine(dim, DIM_LABEL[dim], optLabel(dim, sugg[dim]).split(" (")[0])}
              {vote(dim)}
              {dim === "imputation" && draft.imputation === "constant" && (
                <>
                  <div className="setting-row">
                    <label htmlFor="d-fill">{fillLabel}</label>
                    <input id="d-fill" value={draft.impute_value ?? ""} inputMode={NUMERIC.has(draft.semantic_type) ? "decimal" : undefined}
                      onChange={(e) => update({ impute_value: e.target.value })} />
                  </div>
                  {vote("impute_value")}
                </>
              )}
              {dim === "encoding" && draft.encoding === "onehot" && <>{param("max_categories", "Max categories")}{param("min_frequency", "Min frequency", 0.005)}</>}
              {dim === "encoding" && draft.encoding === "ordinal" && param("min_frequency", "Min frequency", 0.001)}
              {dim === "encoding" && draft.encoding === "tfidf_svd" && <>{param("svd_components", "SVD components")}{param("max_features", "Max vocabulary", 1000)}</>}
            </div>
          ))}

          {draft.action === "keep" && NUMERIC.has(draft.semantic_type) && (
            <div className={`setting${differs("add_missing_indicator") ? " edited" : ""}`}>
              <div className="setting-row">
                <label htmlFor="d-flag">Missing flag</label>
                <label className="check">
                  <input id="d-flag" type="checkbox" checked={draft.add_missing_indicator}
                    onChange={(e) => update({ add_missing_indicator: e.target.checked }, 0)} />
                  Add an indicator column
                </label>
              </div>
              {suggestedLine("add_missing_indicator", "Missing flag", suggested.add_missing_indicator ? "on" : "off")}
              {vote("add_missing_indicator")}
            </div>
          )}

          {lockedLine}
        </div>

        {droppedNote}
        {why}

        {noteOpen ? (
          <label className="field note-field">Note
            <textarea ref={noteRef} value={draft.notes ?? ""} placeholder="Why this choice?"
              onChange={(e) => update({ notes: e.target.value || null })} />
          </label>
        ) : (
          <button type="button" className="ghost small add-note" onClick={focusNote} aria-keyshortcuts="N">
            <Icon name="plus" size={14} /> Add a note <kbd>N</kbd>
          </button>
        )}
      </div>

      <div className="decision-foot">
        <span className={`foot-line${saveFailed && !typing ? " error" : ""}`}>
          {typing ? "Esc to return to review" : <>
            {footLine !== undefined ? <>{footLine}{saveFailed && " · Not saved"}</> : [
              draft.reviewed && server.reviewed ? "Reviewed" : "Not reviewed yet",
              edited.length ? `${edited.length} change${edited.length === 1 ? "" : "s"} from the suggestion` : "matches the suggestion",
              saveFailed ? "Not saved" : "saved as you go",
            ].join(" · ")}
            {/* The last save failed: send the kept draft again, as the toast's Retry does. */}
            {saveFailed && <>{" · "}<button type="button" className="link" onClick={() => void flushSave(f.id, view)}>Retry</button></>}
          </>}
        </span>
        {footNote && !typing && <span className="foot-line foot-note">{footNote}</span>}
        <button type="button" className="primary large confirm" onClick={confirm} aria-keyshortcuts="Enter">
          <span>{confirmLabel}</span><kbd>↵</kbd>
        </button>
        {edited.length > 0 && (
          <button type="button" className="ghost small revert-all" onClick={() => commit({ ...suggested, notes: draftRef.current.notes, reviewed: draftRef.current.reviewed }, 0)}>
            <Icon name="reset" size={14} /> Revert all to the suggestion
          </button>
        )}
        {!settleLabel && (
          <div className="key-hints">
            <span><kbd>X</kbd> keep or drop</span>
            <span><kbd>A</kbd> accept suggestion</span>
            <span><kbd>Z</kbd> undo</span>
            <button type="button" className="link" onClick={onShowShortcuts}><kbd>?</kbd> all keys</button>
          </div>
        )}
      </div>
    </aside>
  );
});

/** A number field that can be cleared and retyped: only valid values are saved, and leaving it while it's
 * empty or out of range puts the saved value back. */
function ParamInput({ id, title, step, value, rule, onChange }: {
  id: string; title: string; step: number; value: number | undefined; rule: ParamRule | undefined;
  onChange: (n: number) => void;
}) {
  const [text, setText] = useState(value === undefined ? "" : String(value));
  useEffect(() => {
    if (value !== undefined && Number(text) !== value) setText(String(value));
    // Deliberately narrow deps: only an outside change (revert, retype) resyncs.
  }, [value]);
  const problem = paramProblem(text, rule);
  return (
    <div className="setting-row">
      <label htmlFor={id}>{title}</label>
      <input id={id} type="number" step={step} min={rule?.min} max={rule?.max} value={text}
        aria-invalid={problem !== null} title={problem ?? undefined}
        onChange={(e) => {
          setText(e.target.value);
          if (paramProblem(e.target.value, rule) === null) onChange(Number(e.target.value));
        }}
        onBlur={() => problem && setText(value === undefined ? "" : String(value))} />
    </div>
  );
}

/** "Setting: explanation" with the setting in bold; reasons from before the labels show as they are. */
function Reason({ text }: { text: string }) {
  const m = /^([A-Z][\w /-]{1,24}): (.*)$/.exec(text);
  return m ? <><b>{m[1]}</b>: {m[2]}</> : <>{text}</>;
}

/** In an experiment: what its baseline says about this field, and a way back to it. */
function ExperimentBox({ draft, data, readOnly, typeLabel, onRevert }: {
  draft: DecisionInput; data: FieldResponse; readOnly: boolean; typeLabel: (t: string) => string; onRevert: () => void;
}) {
  if (!data.baseline) return <div className="experiment-box"><span className="grow">Not in the baseline.</span></div>;
  const changed = baselineDiff(draft, data.baseline);
  const b = data.baseline;
  return (
    <div className={`experiment-box${changed.length ? " changed" : ""}`}>
      <span className="grow">
        {changed.length === 0 ? "Same as the baseline" : <>
          Baseline says <b>{b.action}</b>{b.semantic_type !== draft.semantic_type && <>, <b>{typeLabel(b.semantic_type).toLowerCase()}</b></>}
          <span className="small-text secondary"> · changed here: {changed.join(", ")}</span>
        </>}
        {!readOnly && <span className="small-text secondary block">Changes apply to this experiment only; the baseline stays as the team agreed it.</span>}
      </span>
      {changed.length > 0 && !readOnly && <button type="button" className="small" onClick={onRevert}>Revert</button>}
    </div>
  );
}

/** Equal-sized bands or chosen cut points, with how the field's rows fall into them. Cut points are saved once they
 * parse; until then the last valid setting stays. */
/** Which classes count as positive when a class target is predicted one-vs-rest; the rest form one class. */
function PositivePicker({ levels, positive, readOnly, onChange }: {
  levels: string[]; positive: string[]; readOnly: boolean; onChange: (p: string[]) => void;
}) {
  const toggle = (l: string) => {
    const next = positive.includes(l) ? positive.filter((x) => x !== l) : [...positive, l];
    if (next.length && next.length < levels.length) onChange(next);
  };
  return (
    <div className="stack positive-picker">
      <span className="muted small-text">Predict these against all the others:</span>
      {levels.map((l) => (
        <label key={l} className="check small-text">
          <input type="checkbox" checked={positive.includes(l)} disabled={readOnly} onChange={() => toggle(l)} /> {l}
        </label>
      ))}
    </div>
  );
}

function BandSettings({ fieldId, bins, readOnly, onChange }: {
  fieldId: number; bins: TargetBins; readOnly: boolean; onChange: (b: TargetBins) => void;
}) {
  const [mode, setMode] = useState<"k" | "edges">("k" in bins ? "k" : "edges");
  const [text, setText] = useState("edges" in bins ? bins.edges.join(", ") : "");
  // Typed as text, so "12" can pass through "1" on the way.
  const [kText, setKText] = useState(String("k" in bins ? bins.k : 5));
  const parsed = parseEdges(text);
  const pick = (m: "k" | "edges") => {
    setMode(m);
    if (m === "k") onChange({ k: "k" in bins ? bins.k : 5 });
    else if (typeof parsed !== "string") onChange({ edges: parsed });
  };
  return (
    <div className="stack band-settings">
      <Segmented full size="small" label="Bands" value={mode}
        options={(["k", "edges"] as const).map((m) => ({ value: m, label: m === "k" ? "Equal-sized" : "Cut points", disabled: readOnly }))}
        onChange={pick} />
      {mode === "k" ? (
        <div className="setting-row">
          <label htmlFor="d-bands-k">Bands</label>
          <input id="d-bands-k" type="number" min={2} max={20} step={1} value={kText} disabled={readOnly}
            onChange={(e) => { setKText(e.target.value); const k = bandCount(e.target.value); if (k !== null) onChange({ k }); }}
            onBlur={() => setKText(String("k" in bins ? bins.k : 5))} />
        </div>
      ) : (
        <div className="setting-row">
          <label htmlFor="d-bands-edges">Cut at</label>
          <input id="d-bands-edges" value={text} placeholder="10, 25, 80" disabled={readOnly}
            onChange={(e) => { setText(e.target.value); const p = parseEdges(e.target.value); if (typeof p !== "string") onChange({ edges: p }); }} />
        </div>
      )}
      {mode === "edges" && text.trim() !== "" && typeof parsed === "string" && <div className="inline-warn">{parsed}</div>}
      {mode === "edges" && !("edges" in bins) ? <span className="muted small-text">Type cut points to use them.</span>
        : <BandPreview fieldId={fieldId} bins={bins} />}
      <span className="muted small-text">Equal-sized bands are cut again on each export's training rows.</span>
    </div>
  );
}

/** How the field's rows fall into the bands: for editors under the band settings, for readers under "Predict as". */
function BandPreview({ fieldId, bins }: { fieldId: number; bins: TargetBins }) {
  const preview = useQuery({
    queryKey: ["bands", fieldId, JSON.stringify(bins)],
    queryFn: () => api.bands(fieldId, "positive" in bins ? { k: 5 } : bins),
    placeholderData: keepPreviousData,
  });
  const max = Math.max(1, ...(preview.data?.counts ?? [1]));
  if (preview.error) return <div className="inline-warn">{(preview.error as Error).message}</div>;
  if (!preview.data) return null;
  return (
    <ul className="band-preview" aria-label="Rows per band">
      {preview.data.labels.map((l, i) => (
        <li key={l}>
          <span className="mono">{l}</span>
          <span className="band-bar" style={{ width: `${Math.round((preview.data!.counts[i] / max) * 100)}%` }} />
          <span className="num">{preview.data!.counts[i].toLocaleString()}</span>
        </li>
      ))}
    </ul>
  );
}
