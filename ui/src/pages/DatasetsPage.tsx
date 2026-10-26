import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
  type RefObject,
} from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, uploadDataset } from "../api";
import { PROCESSING, pollInterval, refreshesDone, useFields, useMe } from "../hooks";
import { pullState } from "../pulls";
import { eligibleTarget } from "../baselines";
import { can, lockedTitle, refreshBlocker } from "../can";
import { metricLabel } from "../experiments";
import { defaultTask, taskLabel } from "../decision";
import { fmtAgo, fmtDate, fmtInt, fmtNum, fmtPct } from "../format";
import { sourceSummary } from "../sources";
import { ConnectDialog } from "../components/ConnectDialog";
import { ErrorAlert, Empty, InfoTip } from "../components/ui";
import {
  Banner,
  Chip,
  Icon,
  ProgressBar,
  Segmented,
  StageBar,
  StatusBadge,
  STAGE_LABEL,
  confirmDialog,
  stageOf,
  useDismiss,
  type ChipTone,
} from "../components/kit";
import { showError, toast } from "../toast";
import type { Dataset, Phase } from "../types";

const ACCEPT = /\.(csv|parquet|pq|txt)$/i;
const PHONE = "(max-width: 759px)";
/** "Waiting on you" cards shown before "Show all N". */
const WAITING_CAP = 6;

const STATUS_LABEL: Record<string, string> = {
  pending: "Queued",
  ingesting: "Loading",
  profiling: "Profiling",
  correlating: "Correlating",
};
/** Processing runs in three steps after the upload: load, profile, correlate. */
const STATUS_STEP: Record<string, number> = { ingesting: 1, profiling: 2, correlating: 3 };
const STEPS = 3;
const PHASE_TONE: Record<Phase, ChipTone> = { assessing: "private", review: "team", finalized: "good" };

function lastVisited(id: number): string | null {
  try {
    return localStorage.getItem(`rf:last:${id}`);
  } catch {
    return null;
  }
}

/** Where opening a dataset goes: the field last open, else the dataset's own landing for its phase. */
function openPath(ds: Dataset): string {
  const last = lastVisited(ds.id);
  return last ? `/d/${ds.id}/review/${last}` : `/d/${ds.id}`;
}

/** Phone width: the table gives way to a card list. */
function usePhone(): boolean {
  return useSyncExternalStore(
    (l) => {
      const m = window.matchMedia(PHONE);
      m.addEventListener("change", l);
      return () => m.removeEventListener("change", l);
    },
    () => window.matchMedia(PHONE).matches,
  );
}

/** Fixed position under `anchor`, so a popover isn't clipped by the table's scroll box.
 *  `popover` lets a left-aligned one be clamped by its real width; on phone it sits at the 16px gutter. */
function useAnchored(
  anchor: RefObject<HTMLElement>,
  open: boolean,
  align: "left" | "right",
  popover?: RefObject<HTMLElement>,
) {
  const [style, setStyle] = useState<React.CSSProperties>({ visibility: "hidden" });
  const place = useCallback(() => {
    const r = anchor.current?.getBoundingClientRect();
    if (!r) return;
    const width = popover?.current?.offsetWidth || 320;
    const left = window.matchMedia(PHONE).matches ? 16 : Math.max(8, Math.min(r.left, window.innerWidth - width - 8));
    const below = window.innerHeight - r.bottom - 12;
    const above = r.top - 12;
    // Opens upwards when there's more room above than the little left below.
    const up = below < 360 && above > below;
    setStyle({
      position: "fixed",
      overflowY: "auto",
      ...(up
        ? { top: "auto", bottom: window.innerHeight - r.top + 4, maxHeight: above }
        : { top: r.bottom + 4, bottom: "auto", maxHeight: Math.max(160, below) }),
      ...(align === "left"
        ? { left, right: "auto" }
        : { right: Math.max(8, window.innerWidth - r.right), left: "auto" }),
    });
  }, [anchor, align, popover]);
  useLayoutEffect(() => {
    if (!open) return;
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, place]);
  return style;
}

/** Uploads start the moment a file is picked or dropped; name and targets are set afterwards (P11). */
function useUpload() {
  const qc = useQueryClient();
  const [progress, setProgress] = useState<{ name: string; frac: number } | null>(null);
  const start = async (file: File | undefined) => {
    if (!file) return;
    if (!ACCEPT.test(file.name)) {
      showError("Can't upload that file", { message: "Choose a .csv or .parquet file" });
      return;
    }
    setProgress({ name: file.name, frac: 0 });
    try {
      const ds = await uploadDataset(file, "", "", (frac) => setProgress({ name: file.name, frac }));
      qc.invalidateQueries({ queryKey: ["datasets"] });
      return ds;
    } catch (e) {
      showError(`Upload of ${file.name} failed`, e);
    } finally {
      setProgress(null);
    }
  };
  return { progress, start };
}

/** `reveal`: opened from a link (?target=), so bring its row into view first; the page may start far above it. */
function TargetPicker({
  ds,
  anchor,
  onClose,
  reveal,
}: {
  ds: Dataset;
  anchor: RefObject<HTMLElement>;
  onClose: () => void;
  reveal?: boolean;
}) {
  const qc = useQueryClient();
  const fields = useFields(ds.id);
  const [q, setQ] = useState("");
  const [chosen, setChosen] = useState<Set<string>>(new Set(ds.target_columns));
  const save = useMutation({
    mutationFn: (names: string[]) => api.patchDataset(ds.id, { target_columns: names }),
    onSuccess: () => {
      for (const k of [
        "datasets",
        "dataset",
        "fields",
        "field",
        "corr",
        "corrMatrix",
        "analysis",
        "fieldQuery",
        "exportPreview",
      ]) {
        qc.invalidateQueries({ queryKey: [k] });
      }
      onClose();
    },
  });
  const ref = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);
  useDismiss(ref, true, onClose);
  // Runs before useAnchored places the popover, so it is placed at the row's scrolled-to position.
  useLayoutEffect(() => {
    if (reveal) anchor.current?.scrollIntoView({ block: "center" });
  }, [reveal, anchor]);
  const style = useAnchored(anchor, true, "left", ref);
  // The popover starts hidden until placed, so autoFocus would miss: focus the search once it shows.
  const shownOnce = style.visibility !== "hidden";
  useEffect(() => {
    if (shownOnce) search.current?.focus({ preventScroll: true });
  }, [shownOnce]);
  const shown = (fields.data ?? []).filter((f) => f.name.toLowerCase().includes(q.toLowerCase())).slice(0, 200);
  const toggle = (n: string) => {
    const next = new Set(chosen);
    next.has(n) ? next.delete(n) : next.add(n);
    setChosen(next);
  };
  return (
    <div
      ref={ref}
      className="popover combobox"
      role="dialog"
      aria-label={`Targets for ${ds.name}`}
      style={style}
      onClick={(e) => e.stopPropagation()}
    >
      <input
        ref={search}
        type="search"
        placeholder="Search columns"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        aria-label="Search columns"
      />
      <label className="check">
        <span className="row home-gap-8">
          <input type="radio" checked={chosen.size === 0} onChange={() => setChosen(new Set())} />
          No target (unsupervised)
        </span>
      </label>
      <div className="combo-list">
        {fields.isLoading && <span className="muted small-text home-pad-6">Loading columns…</span>}
        {shown.map((f) => {
          const blocker = eligibleTarget(f);
          const task = defaultTask(f.semantic_type, f.n_unique);
          // A current target that no longer qualifies stays enabled so it can be unticked.
          const disabled = blocker !== null && !chosen.has(f.name);
          return (
            <label
              key={f.id}
              className={`check${blocker ? " blocked" : ""}${disabled ? " disabled" : ""}`}
              title={blocker ?? undefined}
            >
              <span className="row home-pick-row">
                <input
                  type="checkbox"
                  checked={chosen.has(f.name)}
                  disabled={disabled}
                  onChange={() => toggle(f.name)}
                  aria-describedby={blocker ? `tb-${f.id}` : undefined}
                />
                <span className="home-pick-name">{f.name}</span>
              </span>
              <span
                className="type-tag"
                title={task ? `Predicted as ${taskLabel(task).toLowerCase()} unless changed in Review` : undefined}
              >
                {f.semantic_type.replace("_", " ")}
                {!blocker && task ? ` · ${taskLabel(task).toLowerCase()}` : ""}
              </span>
              {blocker && (
                <span id={`tb-${f.id}`} className="muted small-text home-pick-why">
                  {blocker}
                </span>
              )}
            </label>
          );
        })}
      </div>
      <ErrorAlert error={save.error} />
      <div className="row home-pick-foot">
        <span className="muted small-text">{chosen.size} selected · associations are recomputed</span>
        <button
          type="button"
          className="primary small"
          disabled={save.isPending}
          onClick={() => save.mutate([...chosen])}
        >
          Save
        </button>
      </div>
    </div>
  );
}

function RowMenu({
  ds,
  canDelete,
  onRename,
  onDelete,
  onTargets,
  onRefresh,
}: {
  ds: Dataset;
  canDelete: boolean;
  onRename: () => void;
  onDelete: () => void;
  onTargets?: () => void;
  /** Given to a lead on a connected dataset; `blocked` says why it can't run now. */
  onRefresh?: { run: () => void; blocked: string | null };
}) {
  const nav = useNavigate();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const style = useAnchored(button, open, "right");
  const locked = lockedTitle(ds);
  const item = (label: string, run: () => void, extra?: { disabled?: boolean; title?: string; danger?: boolean }) => (
    <button
      type="button"
      role="menuitem"
      className={`ghost menu-item${extra?.danger ? " danger-text" : ""}`}
      disabled={extra?.disabled}
      title={extra?.title}
      onClick={() => {
        setOpen(false);
        run();
      }}
    >
      {label}
    </button>
  );
  return (
    <div ref={ref} className="home-menu" onClick={(e) => e.stopPropagation()}>
      <button
        ref={button}
        type="button"
        className="ghost small icon-only"
        aria-label={`Actions for ${ds.name}`}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <Icon name="more" />
      </button>
      {open && (
        <div className="popover" role="menu" style={style}>
          {item("Rename", onRename, { disabled: locked !== null, title: locked ?? undefined })}
          {onTargets && ds.status === "ready" && item("Targets…", onTargets)}
          {onRefresh &&
            item("Refresh…", onRefresh.run, {
              disabled: onRefresh.blocked !== null,
              title: onRefresh.blocked ?? "Pull the collections again",
            })}
          {canDelete &&
            ds.status === "ready" &&
            ds.phase === "assessing" &&
            item("Assessors…", () => nav(`/d/${ds.id}/baseline`))}
          {item("Delete…", onDelete, {
            danger: true,
            disabled: !canDelete || PROCESSING.has(ds.status) || locked !== null,
            title: !canDelete
              ? "Only a lead can delete a dataset"
              : PROCESSING.has(ds.status)
                ? "Wait for processing to finish"
                : (locked ?? undefined),
          })}
        </div>
      )}
    </div>
  );
}

/** Shown only when the deployment configures at least one connection. */
function ConnectMenu({ onCreated }: { onCreated: (ds: Dataset) => void }) {
  const [open, setOpen] = useState(false);
  const [dialog, setDialog] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const style = useAnchored(button, open, "left");
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources, staleTime: 60_000 });
  if (!sources.data?.length) return null;
  return (
    <div ref={ref} className="home-menu">
      <button ref={button} type="button" aria-expanded={open} aria-haspopup="menu" onClick={() => setOpen(!open)}>
        Connect
        <Icon name="chevron-down" />
      </button>
      {open && (
        <div className="popover" role="menu" style={style}>
          <button
            type="button"
            role="menuitem"
            className="ghost menu-item"
            onClick={() => {
              setOpen(false);
              setDialog(true);
            }}
          >
            MongoDB collection…
          </button>
        </div>
      )}
      {dialog && <ConnectDialog sources={sources.data} onClose={() => setDialog(false)} onCreated={onCreated} />}
    </div>
  );
}

/** "vehicles · curated, mining" and how the newest pull went; nothing for an upload or a sample. */
function SourceBadge({ ds }: { ds: Dataset }) {
  if (!ds.source) return null;
  const pulled = pullState(ds.last_pull);
  return (
    <span className="home-source small-text">
      <Chip tone="accent" title={`${ds.source.database} on ${ds.source.connection}`}>
        connected
      </Chip>
      {ds.status === "refreshing" && <Chip tone="warn">refreshing</Chip>}
      <span className="secondary">
        {sourceSummary(ds.source)}
        {pulled && ` · ${pulled}`}
      </span>
    </span>
  );
}

function SampleMenu({ onCreated }: { onCreated: (ds: Dataset) => void }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState("50000");
  const [fields, setFields] = useState("150");
  const [kind, setKind] = useState("mixed");
  const [numeric, setNumeric] = useState(false);
  const [tab, setTab] = useState<"library" | "generate">("library");
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  const limits = useQuery({
    queryKey: ["sample-limits"],
    queryFn: api.sampleLimits,
    enabled: open,
    staleTime: Infinity,
  });
  // rows × fields is capped server-side to bound generation memory.
  const width = Math.max(Number(fields) || 30, 30) + (numeric ? 1 : 0);
  const maxRows = Math.min(2000000, Math.floor((limits.data?.max_cells ?? Infinity) / width));
  const overBudget = Number(rows) > maxRows;
  const m = useMutation({
    mutationFn: () =>
      api.createSample({
        rows: Number(rows),
        fields: Number(fields),
        seed: 42,
        target_kind: kind,
        numeric_target: numeric,
      }),
    onSuccess: (ds) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      setOpen(false);
      onCreated(ds);
    },
  });
  const tabs = (
    <Segmented
      full
      value={tab}
      onChange={setTab}
      label="Sample kind"
      options={[
        { value: "library", label: "Real datasets" },
        { value: "generate", label: "Generate" },
      ]}
    />
  );
  return (
    <div ref={ref} className="home-menu">
      <button type="button" aria-expanded={open} aria-haspopup="dialog" onClick={() => setOpen(!open)}>
        Add a sample
        <Icon name="chevron-down" />
      </button>
      {open && tab === "library" && (
        <div className="popover stack home-sample" role="dialog" aria-label="Add a sample">
          {tabs}
          <LibraryList
            onCreated={(ds) => {
              setOpen(false);
              onCreated(ds);
            }}
          />
        </div>
      )}
      {open && tab === "generate" && (
        <form
          className="popover stack home-sample"
          role="dialog"
          aria-label="Add a sample"
          onSubmit={(e) => {
            e.preventDefault();
            m.mutate();
          }}
        >
          {tabs}
          <span className="secondary small-text">
            Mixed numeric, categorical, text and date fields with missingness, skew, IDs, numbers-as-text, a constant
            column and one leaky feature.
          </span>
          <div className="row home-sample-size">
            <label className="field home-grow">
              Rows
              <input
                type="number"
                min={1000}
                max={maxRows}
                step={1000}
                value={rows}
                onChange={(e) => setRows(e.target.value)}
              />
              <span className="muted home-hint">
                1,000 – {fmtInt(maxRows)}
                {maxRows < 2000000 ? ` at ${fields} fields` : ""}
              </span>
            </label>
            <label className="field home-sample-fields">
              Fields
              <input type="number" min={30} max={400} value={fields} onChange={(e) => setFields(e.target.value)} />
              <span className="muted home-hint">30 – 400</span>
            </label>
          </div>
          <label className="field">
            Targets
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="mixed">column_a binary, column_b 3-class</option>
              <option value="binary">both binary</option>
              <option value="multiclass">both multiclass</option>
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={numeric} onChange={(e) => setNumeric(e.target.checked)} />
            Add a numeric target (column_c), predicted as a number
          </label>
          {overBudget && (
            <span className="small-text" role="alert">
              Too large: rows × fields is capped at {fmtInt(limits.data?.max_cells)}.{" "}
              <button type="button" className="link" onClick={() => setRows(String(Math.floor(maxRows / 1000) * 1000))}>
                Use {fmtInt(Math.floor(maxRows / 1000) * 1000)} rows
              </button>
            </span>
          )}
          <ErrorAlert error={m.error} />
          <button type="submit" className="primary" disabled={m.isPending || overBudget}>
            Generate
          </button>
        </form>
      )}
    </div>
  );
}

/** Real datasets that ship with the app; adding one processes a copy like an upload, with its targets set. */
function LibraryList({ onCreated }: { onCreated: (ds: Dataset) => void }) {
  const qc = useQueryClient();
  const lib = useQuery({ queryKey: ["library"], queryFn: api.library, staleTime: Infinity });
  const add = useMutation({
    mutationFn: (key: string) => api.addFromLibrary(key),
    onSuccess: (ds) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      onCreated(ds);
    },
  });
  return (
    <div className="stack home-library">
      {lib.isLoading && <span className="muted small-text">Loading…</span>}
      {(lib.data ?? []).map((d) => (
        <div key={d.key} className="library-item stack home-gap-2">
          <div className="row home-spread">
            <b>{d.name}</b>
            <button type="button" className="small primary" disabled={add.isPending} onClick={() => add.mutate(d.key)}>
              Add
            </button>
          </div>
          <span className="muted small-text num">
            {fmtInt(d.rows)} rows × {d.columns} columns · target{d.targets.length === 1 ? "" : "s"}{" "}
            {d.targets.join(", ")}
          </span>
          <span className="secondary small-text">{d.description}</span>
          <a className="small-text" href={d.source_url} target="_blank" rel="noopener" title={d.citation}>
            UCI ML Repository · CC BY 4.0 <Icon name="external" size={12} />
          </a>
        </div>
      ))}
      <ErrorAlert error={lib.error ?? add.error} />
    </div>
  );
}

function FirstRun({ onBrowse, onDemo, busy }: { onBrowse: () => void; onDemo: () => void; busy: boolean }) {
  return (
    <section className="first-run card">
      <button type="button" className="dropzone first-drop" onClick={onBrowse}>
        <Icon name="upload" size={20} />
        <b>Drop a CSV or Parquet file</b>
        <span>
          or <span className="home-accent">browse</span>
        </span>
        <span className="muted small-text">Targets are picked after upload</span>
      </button>
      <div className="stack home-first-side">
        <h2>No data handy?</h2>
        <div>
          <button type="button" className="primary" disabled={busy} onClick={onDemo}>
            Open the 50k × 150 demo
          </button>
        </div>
        <span className="secondary small-text">
          A generated dataset with a leaky feature, ID columns, numbers stored as text and missing values, for
          Rainforest to catch.
        </span>
        <span className="muted small-text">
          Or choose a real dataset, or a generated one's size and targets, under <b>Add a sample</b> above.
        </span>
      </div>
    </section>
  );
}

type Edge = "private" | "team" | "good" | "plain";
interface Waiting {
  key: string;
  ds: Dataset;
  edge: Edge;
  chip: ReactNode;
  title: ReactNode;
  body: ReactNode;
  to: string;
  action: string;
  primary?: boolean;
  progress?: { value: number; max: number };
}

/** " · 1 of 3 submitted": who is in on the version being assessed (counts only, never choices). */
function submittedNote(ds: Dataset, short = false): string {
  const a = ds.assessors;
  if (!a || stageOf(ds) !== "assessing") return "";
  return ` · ${fmtInt(a.submitted)} of ${fmtInt(a.total)} ${short ? "in" : "submitted"}`;
}

/** What the viewer can move forward, from what the datasets list says. */
function waitingOn(rows: Dataset[], me: ReturnType<typeof useMe>["data"]): Waiting[] {
  const out: Waiting[] = [];
  for (const ds of rows) {
    if (ds.status !== "ready" || !ds.baseline) continue;
    const stage = stageOf(ds);
    const total = ds.reopened ?? ds.field_count ?? ds.column_count ?? 0;
    if (stage === "assessing" && ds.my_assessment && !ds.my_assessment.submitted && total > 0) {
      const done = Math.min(ds.my_assessment.reviewed, total);
      out.push({
        key: `assess-${ds.id}`,
        ds,
        edge: "private",
        chip: (
          <Chip tone="private" icon="user">
            Private
          </Chip>
        ),
        title: "Finish your assessment",
        to: `/d/${ds.id}/review`,
        action: done ? "Continue assessing" : "Start assessing",
        primary: true,
        progress: { value: done, max: total },
        body: (
          <>
            {fmtInt(done)} of {fmtInt(total)}
            {ds.reopened != null ? " reopened" : ""} fields · {fmtInt(total - done)} left{submittedNote(ds)}
          </>
        ),
      });
    }
    if (stage === "setup" && can(me, "startBaseline")) {
      out.push({
        key: `setup-${ds.id}`,
        ds,
        edge: "plain",
        chip: <Chip>Needs a baseline</Chip>,
        title: "Set up the first baseline",
        to: `/d/${ds.id}/baseline`,
        action: "Set up",
        body: (
          <>
            {fmtInt(ds.row_count)} rows · {fmtInt(ds.column_count)} fields. Choose the model type, targets and
            assessors.
          </>
        ),
      });
    }
    if (stage === "review" && can(me, "finalize")) {
      const open = ds.review_open;
      out.push({
        key: `review-${ds.id}`,
        ds,
        edge: "team",
        chip: (
          <Chip tone="team" icon="users">
            Team review
          </Chip>
        ),
        title:
          open === 0 ? (
            "Every field is settled"
          ) : open != null ? (
            <>
              Settle <b className="num">{fmtInt(open)}</b> open field{open === 1 ? "" : "s"}
            </>
          ) : (
            "Settle open fields"
          ),
        to: `/d/${ds.id}/team`,
        action: open === 0 ? "Review and finalize" : "Settle fields",
        body:
          open === 0 ? (
            <>Every disagreement has a team decision. Finalize when you're ready.</>
          ) : (
            <>Every assessment is in. Compare them side by side, settle where they differ, then finalize.</>
          ),
      });
    } else if (stage === "review" && can(me, "comment")) {
      // Everyone else hears that review opened: they can follow it and comment on the open fields.
      const open = ds.review_open;
      out.push({
        key: `review-${ds.id}`,
        ds,
        edge: "team",
        chip: (
          <Chip tone="team" icon="users">
            Team review
          </Chip>
        ),
        title: "Team review is open",
        to: `/d/${ds.id}/team`,
        action: "See the review",
        body: (
          <>
            Every assessment is in
            {open ? (
              <>
                ; {fmtInt(open)} field{open === 1 ? "" : "s"} still open
              </>
            ) : null}
            . A lead settles where assessors differ; you can comment.
          </>
        ),
      });
    }
    const best = ds.best_run;
    if (best && !best.chosen && can(me, "chooseModel")) {
      out.push({
        key: `choose-${ds.id}`,
        ds,
        edge: "good",
        chip: (
          <Chip tone="good">
            {fmtInt(ds.run_count ?? 0)} run{ds.run_count === 1 ? "" : "s"}
          </Chip>
        ),
        title: "Mark a chosen model",
        to: `/d/${ds.id}/experiments/${best.experiment_id}?tab=runs&run=${best.run_id}`,
        action: "Compare runs",
        body: (
          <>
            Best on validation: {best.model_label}
            {best.value !== null ? (
              <>
                , {metricLabel(best.metric ?? "")} {fmtNum(best.value, 3)}
              </>
            ) : null}
            . No model chosen yet.
          </>
        ),
      });
    }
  }
  // Your own assessment first, then what only a lead can move.
  const rank = (w: Waiting) => ["assess", "review", "setup", "choose"].indexOf(w.key.split("-")[0]);
  return out.sort((a, b) => rank(a) - rank(b));
}

function WaitingCard({ w, phone }: { w: Waiting; phone: boolean }) {
  const where = (
    <>
      {w.ds.name} · {w.ds.baseline?.label}
    </>
  );
  if (phone) {
    return (
      <Link to={w.to} className={`card home-wait phone edge-${w.edge}`}>
        <span className="home-wait-head">
          <b>{w.title}</b>
          <Icon name="chevron-right" className="muted" />
        </span>
        <span className="secondary">{w.ds.baseline?.setup ? w.ds.name : where}</span>
        {w.progress && (
          <ProgressBar
            value={w.progress.value}
            max={w.progress.max}
            color="var(--lens-private)"
            label={`Your assessment: ${w.ds.name}`}
          />
        )}
        {w.progress && (
          <span className="num muted small-text">
            {fmtInt(w.progress.value)} of {fmtInt(w.progress.max)} · {fmtInt(w.progress.max - w.progress.value)} left
            {submittedNote(w.ds)}
          </span>
        )}
      </Link>
    );
  }
  return (
    <article className={`card home-wait edge-${w.edge}`}>
      <div className="row home-wait-where">
        {w.chip}
        <span className="secondary">{w.ds.baseline?.setup ? w.ds.name : where}</span>
      </div>
      <div className="stack home-gap-6">
        <h3>{w.title}</h3>
        {w.progress && (
          <ProgressBar
            value={w.progress.value}
            max={w.progress.max}
            color="var(--lens-private)"
            label={`Your assessment: ${w.ds.name}`}
          />
        )}
        <p className="num secondary">{w.body}</p>
      </div>
      <div>
        <Link className={`btn${w.primary ? " primary" : ""}`} to={w.to}>
          {w.action}
          <Icon name="arrow-right" />
        </Link>
      </div>
    </article>
  );
}

/** Each target with how it is predicted ("classes", "number", "bands"), or "suggested" while nobody has decided. */
function TargetNames({ ds }: { ds: Dataset }) {
  const kinds = ds.target_kinds ?? {};
  return (
    <span className="home-target-names">
      {ds.target_columns.map((t) => {
        const kind = kinds[t];
        const note = ds.target_kinds_suggested ? "suggested" : kind;
        return (
          <span
            key={t}
            className="home-target-name"
            title={kind ? `${t}: predicted as ${kind}${ds.target_kinds_suggested ? " (the suggestion)" : ""}` : t}
          >
            {t}
            {note && <span className="muted"> · {note}</span>}
          </span>
        );
      })}
    </span>
  );
}

/** Stage bar with the stage and the current baseline; one chip per baseline when there are several. */
function StageCell({ ds }: { ds: Dataset }) {
  const stage = stageOf(ds);
  return (
    <div className="stack home-stage">
      <StageBar stage={stage} />
      {stage === "setup" || !ds.baseline ? (
        <span title="No baseline yet: a lead sets one up on the Baseline tab (model type, targets and assessors)">
          <b>Needs a baseline</b>
        </span>
      ) : (
        <span>
          <b>{STAGE_LABEL[stage!]}</b> · {ds.baseline.label}
        </span>
      )}
      {ds.baselines.length > 1 && (
        <span className="row home-chips">
          {ds.baselines.map((b) => (
            <Chip key={b.family} tone={PHASE_TONE[b.phase]} title={`${b.label}: ${STAGE_LABEL[b.phase].toLowerCase()}`}>
              {b.label} · {STAGE_LABEL[b.phase]}
            </Chip>
          ))}
        </span>
      )}
    </div>
  );
}

/** The viewer's own progress while they assess, else the team decision's. */
function AssessmentCell({ ds }: { ds: Dataset }) {
  const stage = stageOf(ds);
  const mine = ds.my_assessment;
  // An assessor of a later version only assesses the reopened fields: count those, and say so, not "of" the dataset.
  const partial = mine != null && ds.reopened != null;
  const total = (partial ? ds.reopened : ds.field_count) ?? ds.column_count ?? 0;
  if (stage === null || stage === "setup" || total === 0) return <span className="muted">—</span>;
  const title = `${ds.kept_count ?? 0} kept · ${ds.dropped_count ?? 0} dropped · ${ds.target_count ?? 0} targets`;
  const scope = partial ? " reopened" : "";
  if (stage === "assessing" && mine && !mine.submitted) {
    return (
      <div className="stack home-assess" title={title}>
        <ProgressBar
          value={mine.reviewed}
          max={total}
          color="var(--lens-private)"
          label={`Your assessment of ${ds.name}`}
        />
        <span className="num small-text secondary">
          You {fmtInt(mine.reviewed)}/{fmtInt(total)}
          {scope}
          {submittedNote(ds)}
        </span>
      </div>
    );
  }
  // In team review the team decision's progress counts (open disagreements aren't settled), not the viewer's own.
  const reviewed = (stage === "review" && !partial ? ds.reviewed_count : (mine?.reviewed ?? ds.reviewed_count)) ?? 0;
  if (stage === "finalized" || reviewed >= total) {
    const word = stage === "finalized" || stage === "review" ? "settled" : "assessed";
    return (
      <span title={title} className="row home-gap-6">
        <Chip tone="good" icon="check">
          {partial ? `${fmtInt(total)} reopened · ${word}` : `${fmtInt(total)} of ${fmtInt(total)} ${word}`}
        </Chip>
        {stage === "assessing" && mine?.submitted && (
          <Chip tone="good" title="You submitted your assessment">
            Submitted
          </Chip>
        )}
        {stage === "assessing" && ds.assessors && (
          <span className="num small-text secondary">{submittedNote(ds).slice(3)}</span>
        )}
      </span>
    );
  }
  return (
    <div className="stack home-assess" title={title}>
      <ProgressBar
        value={reviewed}
        max={total}
        color={stage === "review" ? "var(--lens-team)" : "var(--good)"}
        label={`Assessment of ${ds.name}`}
      />
      <span className="num small-text secondary">
        {fmtInt(reviewed)} of {fmtInt(total)}
        {scope}
        {mine?.submitted ? " · you submitted" : ""}
        {submittedNote(ds)}
      </span>
    </div>
  );
}

/** The run that stands for the dataset: its chosen model, else its best on validation; else the latest run's state. */
function ResultCell({ ds, compact }: { ds: Dataset; compact?: boolean }) {
  const best = ds.best_run;
  const run = ds.last_run;
  const exp = ds.last_export ? `Latest export v${ds.last_export.id} · ${ds.last_export.status}` : undefined;
  if (best && best.value !== null) {
    const score = (
      <span className="num home-score">
        {fmtNum(best.value, 3)} <span>{metricLabel(best.metric ?? "")}</span>
      </span>
    );
    if (compact) return score;
    return (
      <div
        className="stack home-gap-2"
        title={`${best.chosen ? "Chosen model" : "Best on validation"} · validation score`}
      >
        <span className="row home-gap-6">
          {score}
          {best.chosen && <Chip tone="good">Chosen</Chip>}
        </span>
        <Link
          className="small-text"
          to={`/d/${ds.id}/experiments/${best.experiment_id}?tab=runs&run=${best.run_id}`}
          onClick={(e) => e.stopPropagation()}
        >
          {best.model_label} · {best.experiment_name}
        </Link>
      </div>
    );
  }
  if (!run) {
    if (compact || !ds.last_export) return compact ? null : <span className="muted">—</span>;
    return (
      <Link className="small-text" to={`/d/${ds.id}/experiments`} title={exp} onClick={(e) => e.stopPropagation()}>
        Export v{ds.last_export.id}
      </Link>
    );
  }
  if (compact) return null;
  return (
    <div className="stack home-gap-2 home-start" title={exp}>
      <StatusBadge status={run.status} />
      <Link className="small-text" to={`/d/${ds.id}/experiments`} onClick={(e) => e.stopPropagation()}>
        {run.model_label ?? run.model}
      </Link>
    </div>
  );
}

function Processing({ ds, compact }: { ds: Dataset; compact?: boolean }) {
  const label = STATUS_LABEL[ds.status] ?? ds.status;
  const step = STATUS_STEP[ds.status];
  return (
    <div className={`home-processing${compact ? " compact" : ""}`}>
      {!compact && <span className="home-dot" aria-hidden="true" />}
      {!compact && <b>{label}</b>}
      <ProgressBar value={ds.progress} max={1} label={`${label} ${ds.name}`} className="home-proc-bar" />
      <span className="num secondary">
        {compact ? `${label} · ` : ""}
        {step ? `Step ${step} of ${STEPS} · ` : ""}
        {ds.status_message ?? label} · {fmtPct(ds.progress, 0)}
      </span>
      {!compact && <span className="muted">Opens when profiling finishes</span>}
    </div>
  );
}

function Failed({ ds, open, onToggle }: { ds: Dataset; open: boolean; onToggle: () => void }) {
  return (
    <div className="home-failed">
      <StatusBadge status="failed" />
      <span className="home-error">{ds.error?.split("\n")[0] ?? "Processing failed"}</span>
      {ds.error && (
        <button
          type="button"
          className="link small-text"
          aria-expanded={open}
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
        >
          {open ? "Hide details" : "Details"}
        </button>
      )}
    </div>
  );
}

export function DatasetsPage() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const phone = usePhone();
  const fileInput = useRef<HTMLInputElement>(null);
  const upload = useUpload();
  const me = useMe();
  const mayUpload = useRef(false);
  const [dragging, setDragging] = useState(false);
  const [q, setQ] = useState("");
  // ?target=<dataset id> opens that dataset's target picker (linked from Explore).
  const [targetFor, setTargetFor] = useState<number | null>(
    () => Number(new URLSearchParams(window.location.search).get("target")) || null,
  );
  const linkedTarget = useRef(targetFor);
  const [allWaiting, setAllWaiting] = useState(false);
  const [renaming, setRenaming] = useState<number | null>(null);
  const [fresh, setFresh] = useState<number | null>(null);
  const [details, setDetails] = useState<number | null>(null);
  const targetAnchor = useRef<HTMLElement>(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["datasets"],
    queryFn: api.datasets,
    refetchInterval: (q) => pollInterval(((q.state.data as Dataset[] | undefined) ?? []).map((d) => d.status)),
  });
  const demo = useMutation({
    mutationFn: () => api.createSample({ rows: 50_000, fields: 150, seed: 42, target_kind: "mixed" }),
    onSuccess: (ds) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      highlight(ds.id);
    },
  });
  const rename = useMutation({
    mutationFn: ({ id, name }: { id: number; name: string }) => api.patchDataset(id, { name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      qc.invalidateQueries({ queryKey: ["dataset"] });
      setRenaming(null);
    },
    onError: (e) => showError("Couldn't rename", e),
  });

  const highlight = (id: number) => {
    setFresh(id);
    window.setTimeout(() => setFresh((f) => (f === id ? null : f)), 1500);
  };
  const uploaded = (ds: Dataset) => {
    highlight(ds.id);
    if (me.data?.role === "lead") {
      toast({
        message: `${ds.name} uploaded`,
        action: { label: "Set up its baseline…", run: () => nav(`/d/${ds.id}/baseline`) },
      });
    }
  };
  const remove = async (ds: Dataset) => {
    const parts = [
      ds.field_count !== undefined && `${fmtInt(ds.field_count)} field decisions`,
      ds.export_count ? `${ds.export_count} export${ds.export_count === 1 ? "" : "s"}` : null,
      ds.run_count ? `${ds.run_count} model run${ds.run_count === 1 ? "" : "s"}` : null,
      "the raw table",
    ].filter(Boolean);
    const ok = await confirmDialog({
      title: `Delete ${ds.name}?`,
      confirm: "Delete dataset",
      cancel: "Keep it",
      danger: true,
      body: (
        <p>
          This removes {parts.join(", ")}. Its baselines, experiments and runs go too. MLflow keeps its own copies. It
          can't be undone.
        </p>
      ),
    });
    if (!ok) return;
    try {
      await api.deleteDataset(ds.id);
      qc.invalidateQueries({ queryKey: ["datasets"] });
      toast({ message: `Deleted ${ds.name}` });
    } catch (e) {
      showError(`Couldn't delete ${ds.name}`, e);
    }
  };

  const refresh = async (ds: Dataset) => {
    if (!ds.source) return;
    const pulled = ds.last_pull?.finished_at;
    const outcome = ds.last_pull?.status === "failed" ? "failed" : "finished";
    const ok = await confirmDialog({
      title: `Refresh ${ds.name}?`,
      confirm: "Refresh",
      body: (
        <div className="stack bl-gap-8">
          <p>
            Pulls {ds.source.collections.map((c) => c.label).join(", ")} again from {ds.source.connection}
            {pulled ? `; the last pull ${outcome} ${fmtAgo(pulled)} (${fmtDate(pulled)})` : ""}. While it runs the
            dataset is read-only.
          </p>
          <ul>
            <li>The raw data is replaced with the new pull.</li>
            <li>Fields are profiled again.</li>
            <li>Explore results are marked out of date.</li>
            <li>Decisions are kept for the fields that remain.</li>
            <li>New and drifted fields are listed so you can start a new baseline version with them reopened.</li>
          </ul>
        </div>
      ),
    });
    if (!ok) return;
    try {
      await api.refreshDataset(ds.id);
      qc.invalidateQueries({ queryKey: ["datasets"] });
      qc.invalidateQueries({ queryKey: ["dataset", ds.id] });
      qc.invalidateQueries({ queryKey: ["pulls", ds.id] });
      toast({ message: `Refreshing ${ds.name}`, detail: "It is read-only until the pull finishes." });
    } catch (e) {
      showError(`Couldn't refresh ${ds.name}`, e);
    }
  };

  // A refresh replaces the fields, pulls, team state and baselines; the dataset's own page invalidates them when it
  // sees the refresh end, but a refresh seen ending here would leave them cached for the next visit.
  const refreshing = useRef<Set<number>>(new Set());
  useEffect(() => {
    if (refreshesDone(refreshing.current, data ?? []).length > 0) qc.invalidateQueries();
  }, [data, qc]);

  // Tell the user when a dataset they're waiting on becomes ready.
  const wasProcessing = useRef<Set<number>>(new Set());
  useEffect(() => {
    for (const ds of data ?? []) {
      if (PROCESSING.has(ds.status)) wasProcessing.current.add(ds.id);
      else if (wasProcessing.current.delete(ds.id) && ds.status === "ready") {
        highlight(ds.id);
        toast({
          message: `${ds.name} is ready`,
          detail: `${fmtInt(ds.column_count)} fields`,
          tone: "good",
          duration: 8000,
          action: { label: "Open it", run: () => nav(`/d/${ds.id}`) },
        });
      }
    }
  }, [data, nav]);

  // A file dropped anywhere on the page uploads; without this the browser would navigate to it.
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const enter = (e: DragEvent) => {
      if (hasFiles(e)) {
        depth++;
        setDragging(true);
      }
    };
    const leave = (e: DragEvent) => {
      if (hasFiles(e) && --depth <= 0) {
        depth = 0;
        setDragging(false);
      }
    };
    const over = (e: DragEvent) => hasFiles(e) && e.preventDefault();
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      if (!mayUpload.current) return;
      void upload.start(e.dataTransfer?.files[0]).then((ds) => ds && uploaded(ds));
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []);

  const browse = () => fileInput.current?.click();
  const editor = can(me.data, "upload");
  const canDelete = can(me.data, "deleteDataset");
  mayUpload.current = editor;
  const rows = useMemo(() => data ?? [], [data]);
  const shown = useMemo(() => rows.filter((d) => d.name.toLowerCase().includes(q.trim().toLowerCase())), [rows, q]);
  const waiting = useMemo(() => waitingOn(rows, me.data), [rows, me.data]);
  const processing = rows.filter((d) => PROCESSING.has(d.status)).length;
  const failed = rows.filter((d) => d.status === "failed").length;
  const closeTarget = useCallback(() => {
    linkedTarget.current = null;
    setTargetFor(null);
  }, []);

  const nameInput = (ds: Dataset) => (
    <input
      defaultValue={ds.name}
      autoFocus
      aria-label="Dataset name"
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Enter") rename.mutate({ id: ds.id, name: e.currentTarget.value.trim() || ds.name });
        if (e.key === "Escape") setRenaming(null);
      }}
      onBlur={(e) =>
        e.currentTarget.value.trim() !== ds.name
          ? rename.mutate({ id: ds.id, name: e.currentTarget.value.trim() || ds.name })
          : setRenaming(null)
      }
    />
  );
  const warnings = (ds: Dataset) =>
    ds.warnings.length > 0 && (
      <InfoTip
        className="badge warn"
        ariaLabel={`${ds.warnings.length} warning${ds.warnings.length === 1 ? "" : "s"}`}
        label={
          <>
            <Icon name="alert" size={13} /> {ds.warnings.length}
          </>
        }
        text={ds.warnings.join("\n")}
      />
    );
  const menu = (ds: Dataset, withTargets: boolean) =>
    editor && (
      <RowMenu
        ds={ds}
        canDelete={canDelete}
        onRename={() => setRenaming(ds.id)}
        onDelete={() => void remove(ds)}
        onTargets={withTargets && ds.status === "ready" ? () => setTargetFor(ds.id) : undefined}
        onRefresh={
          ds.source && me.data?.role === "lead"
            ? { run: () => void refresh(ds), blocked: refreshBlocker(me.data, ds) }
            : undefined
        }
      />
    );
  const errorDetails = (ds: Dataset) =>
    details === ds.id &&
    ds.error && (
      <Banner tone="bad" title="Processing failed">
        <pre>{ds.error}</pre>
      </Banner>
    );

  const summary = [
    `${rows.length} dataset${rows.length === 1 ? "" : "s"}`,
    processing > 0 && `${processing} processing`,
    failed > 0 && `${failed} failed`,
    waiting.length > 0 && `${waiting.length} thing${waiting.length === 1 ? "" : "s"} waiting on you`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="page home">
      <input
        ref={fileInput}
        type="file"
        accept=".csv,.parquet,.pq,.txt"
        hidden
        onChange={(e) => {
          void upload.start(e.target.files?.[0]).then((ds) => ds && uploaded(ds));
          e.target.value = "";
        }}
      />
      <div className="home-head">
        <div className="stack home-gap-4">
          <h1>Datasets</h1>
          {data && rows.length > 0 && <p className="secondary num">{summary}</p>}
        </div>
        {editor && (
          <div className="row home-actions">
            <SampleMenu onCreated={(ds) => highlight(ds.id)} />
            <ConnectMenu onCreated={(ds) => highlight(ds.id)} />
            <button type="button" className="primary" onClick={browse} disabled={upload.progress !== null}>
              <Icon name="upload" />
              Upload CSV or Parquet
            </button>
          </div>
        )}
      </div>
      <ErrorAlert error={error} />
      {isLoading && (
        <div className="stack">
          <span className="skeleton" />
          <span className="skeleton" />
        </div>
      )}
      {data &&
        rows.length === 0 &&
        !upload.progress &&
        (editor ? (
          <FirstRun onBrowse={browse} onDemo={() => demo.mutate()} busy={demo.isPending} />
        ) : (
          <div className="card">
            <Empty title="No datasets yet">A team member adds the first one by uploading a file.</Empty>
          </div>
        ))}
      <ErrorAlert error={demo.error} />

      {waiting.length > 0 && (
        <section className="stack home-section" aria-labelledby="home-waiting">
          <h2 id="home-waiting">Waiting on you{phone ? <span className="muted"> · {waiting.length}</span> : null}</h2>
          <div id="home-wait-cards" className={phone ? "stack home-wait-list" : "home-wait-grid"}>
            {(allWaiting ? waiting : waiting.slice(0, WAITING_CAP)).map((w) => (
              <WaitingCard key={w.key} w={w} phone={phone} />
            ))}
          </div>
          {/* Past a few cards, the rest fold away so the table (uploads, processing, failures) stays near the top. */}
          {waiting.length > WAITING_CAP && (
            <div>
              <button
                type="button"
                className="ghost"
                aria-expanded={allWaiting}
                aria-controls="home-wait-cards"
                onClick={() => setAllWaiting(!allWaiting)}
              >
                {allWaiting ? (
                  "Show fewer"
                ) : (
                  <>
                    Show all {waiting.length}
                    <Icon name="chevron-down" />
                  </>
                )}
              </button>
            </div>
          )}
        </section>
      )}

      {(rows.length > 0 || upload.progress) && (
        <section className="stack home-section" aria-labelledby="home-all">
          <div className="home-all-head">
            <h2 id="home-all">All datasets{phone ? <span className="muted"> · {rows.length}</span> : null}</h2>
            <label className="home-search">
              <Icon name="search" />
              <input
                type="search"
                placeholder="Find a dataset"
                aria-label="Find a dataset"
                value={q}
                onChange={(e) => setQ(e.target.value)}
              />
            </label>
          </div>

          {phone ? (
            <ul className="card home-list" aria-label="All datasets">
              {upload.progress && (
                <li className="home-item fresh">
                  <b>{upload.progress.name}</b>
                  <div className="home-processing compact">
                    <ProgressBar
                      value={upload.progress.frac}
                      max={1}
                      label={`Uploading ${upload.progress.name}`}
                      className="home-proc-bar"
                    />
                    <span className="num secondary">Uploading {fmtPct(upload.progress.frac, 0)}</span>
                  </div>
                </li>
              )}
              {shown.map((ds) => {
                const ready = ds.status === "ready" || ds.status === "refreshing";
                const stage = stageOf(ds);
                const body = (
                  <>
                    <span className="home-item-head">
                      {renaming === ds.id ? nameInput(ds) : <b className="home-item-name">{ds.name}</b>}
                      <ResultCell ds={ds} compact />
                    </span>
                    {ready && (
                      <span className="home-item-stage">
                        <StageBar stage={stage} />
                        {stage === "setup" || !ds.baseline
                          ? "Needs a baseline"
                          : `${STAGE_LABEL[stage!]} · ${ds.baseline.label}${submittedNote(ds, true)}`}
                        {warnings(ds)}
                      </span>
                    )}
                    <SourceBadge ds={ds} />
                    {PROCESSING.has(ds.status) && <Processing ds={ds} compact />}
                    {ds.status === "failed" && (
                      <Failed
                        ds={ds}
                        open={details === ds.id}
                        onToggle={() => setDetails(details === ds.id ? null : ds.id)}
                      />
                    )}
                  </>
                );
                return (
                  <li key={ds.id} className={`home-item${fresh === ds.id ? " fresh" : ""}`}>
                    <div className="home-item-row">
                      {ready && renaming !== ds.id ? (
                        <Link to={openPath(ds)} className="home-item-main">
                          {body}
                        </Link>
                      ) : (
                        <div className="home-item-main">{body}</div>
                      )}
                      <span ref={targetFor === ds.id ? targetAnchor : undefined} className="home-item-menu">
                        {menu(ds, true)}
                      </span>
                    </div>
                    {targetFor === ds.id && ds.status === "ready" && editor && (
                      <TargetPicker
                        ds={ds}
                        anchor={targetAnchor}
                        onClose={closeTarget}
                        reveal={linkedTarget.current === ds.id}
                      />
                    )}
                    {errorDetails(ds)}
                  </li>
                );
              })}
              {shown.length === 0 && !upload.progress && <li className="home-item muted">No dataset matches “{q}”.</li>}
            </ul>
          ) : (
            <div className="card home-table-box">
              <table className="data home-table">
                <thead>
                  <tr>
                    <th scope="col">Dataset</th>
                    <th scope="col">Stage</th>
                    <th scope="col">Target</th>
                    <th scope="col">Assessment</th>
                    <th scope="col" title="The chosen model, else the best run on the validation split">
                      Best result
                    </th>
                    <th scope="col" className="num">
                      Updated
                    </th>
                    <th scope="col">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {upload.progress && (
                    <tr className="fresh">
                      <td>
                        <b>{upload.progress.name}</b>
                      </td>
                      <td colSpan={6}>
                        <div className="home-processing">
                          <span className="home-dot" aria-hidden="true" />
                          <b>Uploading</b>
                          <ProgressBar
                            value={upload.progress.frac}
                            max={1}
                            label={`Uploading ${upload.progress.name}`}
                            className="home-proc-bar"
                          />
                          <span className="num secondary">{fmtPct(upload.progress.frac, 0)}</span>
                        </div>
                      </td>
                    </tr>
                  )}
                  {shown.map((ds) => {
                    const ready = ds.status === "ready" || ds.status === "refreshing";
                    const busy = PROCESSING.has(ds.status);
                    const locked = lockedTitle(ds);
                    return (
                      <Fragment key={ds.id}>
                        <tr
                          className={`${ready ? "clickable" : ""}${fresh === ds.id ? " fresh" : ""}`}
                          onClick={(e) =>
                            ready &&
                            renaming !== ds.id &&
                            !(e.target as HTMLElement).closest("a, button, input, details, .popover") &&
                            nav(openPath(ds))
                          }
                        >
                          <td className="home-name">
                            {renaming === ds.id ? (
                              nameInput(ds)
                            ) : ready ? (
                              <Link to={openPath(ds)} title={ds.name}>
                                {ds.name}
                              </Link>
                            ) : (
                              <b title={ds.name}>{ds.name}</b>
                            )}
                            <span className="num muted small-text">
                              {ds.row_count != null
                                ? `${fmtInt(ds.row_count)} rows · ${fmtInt(ds.column_count)} fields`
                                : (ds.source_filename ?? "")}
                            </span>
                            <SourceBadge ds={ds} />
                          </td>
                          {busy || ds.status === "failed" ? (
                            <td colSpan={4}>
                              {busy ? (
                                <Processing ds={ds} />
                              ) : (
                                <Failed
                                  ds={ds}
                                  open={details === ds.id}
                                  onToggle={() => setDetails(details === ds.id ? null : ds.id)}
                                />
                              )}
                            </td>
                          ) : (
                            <>
                              <td>
                                <div className="row home-stage-row">
                                  <StageCell ds={ds} />
                                  {warnings(ds)}
                                </div>
                              </td>
                              <td className="home-target">
                                <span
                                  ref={targetFor === ds.id ? targetAnchor : undefined}
                                  className="home-target-anchor"
                                >
                                  {!editor &&
                                    (ds.target_columns.length ? (
                                      <TargetNames ds={ds} />
                                    ) : (
                                      <span className="muted">None</span>
                                    ))}
                                  {editor &&
                                    (ds.target_columns.length ? (
                                      <button
                                        type="button"
                                        className="ghost target-chips"
                                        title={locked ?? "Change targets"}
                                        disabled={locked !== null}
                                        aria-expanded={targetFor === ds.id}
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setTargetFor(targetFor === ds.id ? null : ds.id);
                                        }}
                                      >
                                        <TargetNames ds={ds} />
                                      </button>
                                    ) : (
                                      <button
                                        type="button"
                                        className="chip attention on"
                                        title={locked ?? undefined}
                                        disabled={locked !== null}
                                        aria-expanded={targetFor === ds.id}
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setTargetFor(targetFor === ds.id ? null : ds.id);
                                        }}
                                      >
                                        Choose target
                                        <Icon name="chevron-down" size={14} />
                                      </button>
                                    ))}
                                </span>
                                {targetFor === ds.id && editor && locked === null && (
                                  <TargetPicker
                                    ds={ds}
                                    anchor={targetAnchor}
                                    onClose={closeTarget}
                                    reveal={linkedTarget.current === ds.id}
                                  />
                                )}
                              </td>
                              <td>
                                <AssessmentCell ds={ds} />
                              </td>
                              <td>
                                <ResultCell ds={ds} />
                              </td>
                            </>
                          )}
                          <td className="num muted small-text" title={fmtDate(ds.updated_at)}>
                            {fmtAgo(ds.updated_at)}
                          </td>
                          <td className="home-actions-cell">{menu(ds, false)}</td>
                        </tr>
                        {details === ds.id && ds.error && (
                          <tr>
                            <td colSpan={7}>{errorDetails(ds)}</td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                  {shown.length === 0 && !upload.progress && (
                    <tr>
                      <td colSpan={7} className="muted">
                        No dataset matches “{q}”.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {dragging && editor && (
        <div className="drop-overlay" aria-hidden="true">
          <span>Drop to upload: CSV or Parquet</span>
        </div>
      )}
    </div>
  );
}
