import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useDebounced, useFields } from "../hooks";
import { Empty, ErrorAlert, InfoTip } from "../components/ui";
import { Banner, Chip, Icon, ProgressBar, Segmented, StatusBadge, useDismiss } from "../components/kit";
import { MlflowLink, RunDetail } from "../components/RunDetail";
import { TrainPanel, type TrainRequest } from "../components/TrainPanel";
import { baselineLabel, familyShort, fmtAgo, fmtDate, fmtInt, fmtMetric, fmtNum, fmtPct } from "../format";
import { toast } from "../toast";
import { batchSummary, bestRuns, isActive, metricFor, runColumns, runLabel, spreadFor, starTitle, whyFailed, type RunMetric, type Split } from "../runs";
import { lastRowRules, NO_RULES, rowsSentence, sourceLabel, splitPreview, type RowRules } from "../experiments";
import { describeSettings } from "../settings";
import type { Dataset, Experiment, ExportJob, ModelRun, SettingValue, TrainPrefill } from "../types";

const FAMILY_EXPLAINER: Record<string, string> = {
  tree: "Missing values stay as NaN for the trees to route; no scaling or log transforms; mid/high-cardinality categoricals become compact ordinal codes.",
  linear: "Every numeric is imputed (with missing flags), log-transformed when skewed and standardized; categoricals are one-hot encoded.",
  neural: "Like linear, with quantile-normal scaling for skewed inputs so the network sees well-conditioned features.",
};
const HOW_IT_WORKS =
  "Your decisions become a scikit-learn ColumnTransformer. It is fitted on training rows only, then applied in chunks to every row. " +
  "The output is all-numeric Parquet with a __split column and label-encoded targets, ready for a ClassifierChain.";
const MIN_TRAIN = 0.5;

/** A small anchored popover that closes on outside click or Escape. */
function Pop({ label, children, primary, align = "right" }: {
  label: ReactNode; children: (close: () => void) => ReactNode; primary?: boolean; align?: "left" | "right";
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(ref, open, close);
  return (
    <div ref={ref} className="exp-pop-anchor">
      <button type="button" className={`small${primary ? " primary" : ""}`} aria-expanded={open} onClick={() => setOpen(!open)}>{label}</button>
      {open && <div className={`popover stack exp-pop${align === "left" ? " left" : ""}`}>{children(close)}</div>}
    </div>
  );
}

let measurer: CanvasRenderingContext2D | null = null;
/** How wide a label renders in the element's font (7px a character where canvas can't say). */
function textWidth(text: string, el: HTMLElement | null): number {
  measurer ??= document.createElement("canvas").getContext("2d");
  if (!measurer || !el) return text.length * 7;
  measurer.font = getComputedStyle(el).font;
  return measurer.measureText(text).width;
}

/** The split's shares, with each wide-enough segment's row count (`about` when the counts are an estimate). */
function SplitBar({ valid, test, counts, about }: {
  valid: number; test: number; counts: { train: number; valid: number; test: number } | null; about: boolean;
}) {
  const train = 1 - valid - test;
  const bad = train < MIN_TRAIN;
  // Text on the amber and grey segments takes the page colour, so it reads in both themes.
  // A segment too narrow for its label (a 15% share on a phone) shows it on a line under the bar instead.
  const bar = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = bar.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const segs = [
    { label: "train", share: Math.max(train, 0), color: bad ? "var(--bad)" : "var(--accent-fill)" },
    { label: "valid", share: valid, color: "var(--warn)" },
    { label: "test", share: test, color: "var(--text-secondary)" },
  ].filter((s) => s.share > 0).map((s) => {
    const n = counts?.[s.label as "train" | "valid" | "test"];
    const text = `${s.label} ${fmtPct(s.share, 0)}${n !== undefined && s.share >= 0.3 ? ` · ${about ? "about " : ""}${fmtInt(n)}` : ""}`;
    // The text's width plus the segment's padding; before the bar is measured, the old 10% rule.
    const fits = width ? s.share * width >= textWidth(text, bar.current) + 16 : s.share >= 0.1;
    return { ...s, text, fits };
  });
  return (
    <div className="split">
      <div ref={bar} className="split-bar" role="img" aria-label={`Train ${fmtPct(train, 0)}, validation ${fmtPct(valid, 0)}, test ${fmtPct(test, 0)}`}>
        {segs.map(({ label, share, color, text, fits }) => (
          <span key={label} style={{ width: `${share * 100}%`, background: color, color: label === "train" ? undefined : "var(--bg)" }} title={`${label} ${fmtPct(share, 0)}`}>
            {fits && text}
          </span>
        ))}
      </div>
      {width > 0 && segs.some((s) => !s.fits) && (
        <div className="split-legend small-text secondary" aria-hidden="true">
          {segs.filter((s) => !s.fits).map(({ label, share, color }) => (
            <span key={label}><i style={{ background: color }} />{label} {fmtPct(share, 0)}</span>
          ))}
        </div>
      )}
    </div>
  );
}

const rulesBody = (r: RowRules) => ({ ...r, exclude: r.exclude.filter((e) => e.field && e.values.length) });

/** Which rows go into the export and how the split keeps related rows together. */
function RowRulesEditor({ rules, onChange, from, names }: { rules: RowRules; onChange: (r: RowRules) => void; from: number | null; names: string[] }) {
  const set = (patch: Partial<RowRules>) => onChange({ ...rules, ...patch });
  const pick = (value: string | null, onPick: (v: string | null) => void, none: string, label: string) => (
    <select aria-label={label} value={value ?? ""} onChange={(e) => onPick(e.target.value || null)}>
      <option value="">{none}</option>
      {names.map((n) => <option key={n} value={n}>{n}</option>)}
    </select>
  );
  const active = rules.dedupe || rules.exclude.length > 0 || rules.first_per !== null || rules.group_by !== null;
  return (
    <details className="advanced" open={active || undefined}>
      <summary className="small-text secondary">Rows{active ? (from ? ` · same rules as Export #${from}` : " · rules set") : ""}</summary>
      <div className="stack exp-rules">
        <label className="check small-text">
          <input type="checkbox" checked={rules.dedupe} onChange={(e) => set({ dedupe: e.target.checked })} /> Leave out exact duplicate rows
        </label>
        <div className="row small-text">
          One row per {pick(rules.first_per?.key ?? null, (k) => set({ first_per: k ? { key: k, order: rules.first_per?.order ?? null } : null }), "every row", "One row per")}
          {rules.first_per && <>first by {pick(rules.first_per.order, (o) => set({ first_per: { key: rules.first_per!.key, order: o } }), "file order", "First by")}</>}
        </div>
        {rules.exclude.map((r, i) => (
          <div key={i} className="row small-text">
            Leave out rows where {pick(r.field || null, (f) => set({ exclude: rules.exclude.map((x, j) => (j === i ? { ...x, field: f ?? "" } : x)) }), "choose a field", "Field")}
            is <input type="text" placeholder="values, comma-separated" defaultValue={r.values.join(", ")} className="exp-values-input"
              onBlur={(e) => set({ exclude: rules.exclude.map((x, j) => (j === i ? { ...x, values: e.target.value.split(",").map((v) => v.trim()).filter(Boolean) } : x)) })} />
            <button type="button" className="link small-text" onClick={() => set({ exclude: rules.exclude.filter((_, j) => j !== i) })}>remove</button>
          </div>
        ))}
        <button type="button" className="link small-text exp-self-start"
          onClick={() => set({ exclude: [...rules.exclude, { field: "", values: [] }] })}>+ Leave out rows by value</button>
        <div className="row small-text">
          Keep rows together by {pick(rules.group_by, (g) => set({ group_by: g }), "no field (split rows freely)", "Keep rows together by")}
          <span className="muted">each value's rows stay in one split and one CV fold, e.g. a patient's visits</span>
        </div>
      </div>
    </details>
  );
}

function BuildPanel({ datasetId, family, rows, x }: { datasetId: number; family: string; rows: number | null; x: Experiment }) {
  const qc = useQueryClient();
  const personal = x.assessor_id !== null;
  const fields = useFields(datasetId, personal);
  const preview = useQuery({ queryKey: ["exportPreview", datasetId, family, x.id], queryFn: () => api.exportPreview(datasetId, family, null, x.id) });
  const exports = useQuery({ queryKey: ["exports", datasetId, x.id], queryFn: () => api.exports(datasetId, x.id) });
  const [valid, setValid] = useState("15");
  const [test, setTest] = useState("15");
  const [seed, setSeed] = useState(42);
  const [fitRows, setFitRows] = useState<number | null>(null);
  const [fitText, setFitText] = useState<string | null>(null);
  const [rules, setRules] = useState<RowRules>(NO_RULES);
  // Start from the rules the dataset's last export used, until someone changes them here.
  const allExports = useQuery({ queryKey: ["exports", datasetId, "all"], queryFn: () => api.exports(datasetId) });
  const [rulesFrom, setRulesFrom] = useState<number | null>(null);
  const touched = useRef(false);
  useEffect(() => {
    const last = touched.current ? null : lastRowRules(allExports.data ?? []);
    if (last && last.from !== rulesFrom) { setRules(last.rules); setRulesFrom(last.from); }
  }, [allExports.data]);
  const changeRules = (r: RowRules) => { touched.current = true; setRulesFrom(null); setRules(r); };
  const v = Number(valid) / 100, t = Number(test) / 100;
  const train = 1 - v - t;
  const splitError = !Number.isFinite(v) || !Number.isFinite(t) || v < 0 || t < 0 ? "Shares must be numbers of 0% or more."
    : v >= 0.5 || t >= 0.5 ? "Validation and test must each stay under 50%."
    : train < MIN_TRAIN ? `Train would be ${fmtPct(train, 0)}; keep it at ${MIN_TRAIN * 100}% or more.` : null;
  // What the row rules leave to split, counted on the server by the export's own code once typing settles.
  // The shares and seed go too, so the server splits the rows exactly as the export will (groups included).
  const ask = JSON.stringify({ ...rulesBody(rules), ...(splitError ? {} : { valid_size: v, test_size: t, seed }) });
  const askKey = useDebounced(ask, 400);
  const counted = useQuery({
    queryKey: ["exportRows", datasetId, x.id, askKey], queryFn: () => api.exportRows(datasetId, JSON.parse(askKey), x.id),
    placeholderData: keepPreviousData, staleTime: 60_000,
  });
  const grouped = Boolean(rules.group_by);
  const { counts, about } = splitError ? { counts: null, about: false } : splitPreview(counted.data, rows, v, t, grouped);
  const counting = counted.isFetching || askKey !== ask;
  const maxFit = rows ?? 500_000;
  const fit = Math.min(fitRows ?? maxFit, 5_000_000);

  const start = useMutation({
    mutationFn: (cfg: { valid_size: number; test_size: number; seed: number; fit_sample_rows: number } & Partial<RowRules>) =>
      api.startExport(datasetId, { ...rulesBody(rules), ...cfg, model_family: family, experiment_id: x.id }),
    onSuccess: (e) => {
      qc.invalidateQueries({ queryKey: ["exports", datasetId] });
      qc.invalidateQueries({ queryKey: ["experiment", x.id] });
      toast({ message: `Export #${e.id} started` });
    },
  });
  const p = preview.data;
  const lastSame = exports.data?.find((e) => e.status === "done" && e.model_family === family);
  const firstTodo = fields.data?.find((f) => !f.reviewed);
  const reExport = () => lastSame && start.mutate({
    valid_size: lastSame.config.valid_size, test_size: lastSame.config.test_size,
    seed: lastSame.config.seed, fit_sample_rows: lastSame.config.fit_sample_rows,
    dedupe: Boolean(lastSame.config.dedupe), exclude: lastSame.config.exclude ?? [], first_per: lastSame.config.first_per ?? null,
    group_by: lastSame.config.group_by ?? null,
  });

  return (
    <section className="card build-panel">
      <div className="stack build-col">
        <div className="row exp-between">
          <h2>What will be exported <InfoTip className="muted exp-normal" label={<Icon name="info" size={14} label="How the export works" />} text={`${HOW_IT_WORKS}\n\n${familyShort(family)}: ${FAMILY_EXPLAINER[family] ?? ""}`} /></h2>
          <span className="secondary small-text" title={personal ? "Your assessment as it stands when you export; later edits don't change an export already made." : "The version's frozen snapshot."}>
            from {sourceLabel(x)}{personal ? ", as it stands now" : ""}
          </span>
        </div>
        <ErrorAlert error={preview.error} />
        {p && (
          <>
            <div className="secondary num">
              <b>{p.kept}</b> kept · <b>{p.dropped}</b> dropped · targets {p.targets.length ? p.targets.map((x) => <span key={x} className="exp-target">{x}</span>) : <span className="muted">none</span>} · <b>{p.customized}</b> customized
            </div>
            <div className="chips">
              {Object.entries(p.kept_by_type).map(([type, n]) => <span key={type} className="badge"><span className="mono">{type.replace("_", " ")}</span> {n}</span>)}
            </div>
            {p.unreviewed > 0 && (
              <Banner tone="warn" action={personal && firstTodo ? <Link to={`/d/${datasetId}/review/${firstTodo.id}`}>Review next</Link> : undefined}>
                {p.unreviewed} unreviewed: their suggested settings will be used.
              </Banner>
            )}
            {p.warnings.filter((w) => !w.includes("not reviewed")).map((w) => <Banner key={w} tone="warn">{w}</Banner>)}
            {p.blocking.map((b) => <ErrorAlert key={b} error={{ message: b }} />)}
            {p.last_export && (p.changed_since_export ? (
              <Banner tone="warn" action={
                <button type="button" className="small" disabled={start.isPending || p.blocking.length > 0} onClick={reExport}>Re-export with same split</button>}>
                {p.changed_since_export} field{p.changed_since_export === 1 ? "" : "s"} decided differently than in Export #{p.last_export.id}
              </Banner>
            ) : (
              <Chip tone="good" icon="check" className="exp-self-start">Same decisions as Export #{p.last_export.id}</Chip>
            ))}
          </>
        )}
      </div>
      <div className="stack build-col">
        <h2>Split</h2>
        <SplitBar valid={Math.max(v, 0) || 0} test={Math.max(t, 0) || 0} counts={counts} about={about} />
        {counted.data && (
          <p className="small-text secondary num split-rows" aria-live="polite">
            {rowsSentence(counted.data)}.
            {counts && <> {about ? "About t" : "T"}rain {fmtInt(counts.train)} · valid {fmtInt(counts.valid)} · test {fmtInt(counts.test)}{grouped && counted.data.groups !== null
              ? `, with each of the ${fmtInt(counted.data.groups)} values of ${rules.group_by} in one split` : ""}.</>}
            {counting && <span className="muted"> Counting…</span>}
          </p>
        )}
        <ErrorAlert error={counted.error} />
        <div className="row exp-split-inputs">
          <label className="field">Validation
            <span className="value-pct"><input type="number" min={0} max={45} step={5} value={valid} onChange={(e) => setValid(e.target.value)} className="exp-pct-input" /> %</span>
          </label>
          <label className="field">Test
            <span className="value-pct"><input type="number" min={0} max={45} step={5} value={test} onChange={(e) => setTest(e.target.value)} className="exp-pct-input" /> %</span>
          </label>
        </div>
        {splitError && <span className="small-text exp-bad" role="alert">{splitError}</span>}
        <RowRulesEditor rules={rules} onChange={changeRules} from={rulesFrom} names={(fields.data ?? []).map((f) => f.name)} />
        <details className="advanced">
          <summary className="small-text secondary">Advanced</summary>
          <div className="row exp-advanced-row">
            <label className="field">Random seed<input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="exp-seed-input" /></label>
            <label className="field">Max rows to fit on
              <input type="number" min={1000} max={maxFit} step={10_000} value={fitText ?? fit} className="exp-fit-input"
                onChange={(e) => setFitText(e.target.value)}
                onBlur={() => { if (fitText !== null) { setFitRows(Math.max(1000, Math.min(Number(fitText) || maxFit, maxFit))); setFitText(null); } }} />
            </label>
          </div>
        </details>
        <div className="row exp-export-go">
          <span className="muted small-text">Prepared as {baselineLabel(x.recipe_family, x.recipe_version, false, x.recipe_name)} decided</span>
          <button type="button" className="primary" disabled={!p || p.blocking.length > 0 || Boolean(splitError) || start.isPending}
            onClick={() => start.mutate({ valid_size: v, test_size: t, seed, fit_sample_rows: Math.max(1000, Math.min(fitText !== null ? Number(fitText) || maxFit : fit, maxFit)) })}>
            Export for {familyShort(family).toLowerCase()} models
          </button>
        </div>
        <ErrorAlert error={start.error} />
      </div>
    </section>
  );
}

function CopyCommand({ command }: { command: string }) {
  const [state, setState] = useState<"idle" | "copied" | "manual">("idle");
  const ref = useRef<HTMLElement>(null);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setState("copied");
    } catch {
      // Clipboard needs a secure context; over plain-http LAN access, select the text instead (#53).
      const sel = window.getSelection();
      if (ref.current && sel) { const r = document.createRange(); r.selectNodeContents(ref.current); sel.removeAllRanges(); sel.addRange(r); }
      setState("manual");
    }
  };
  return (
    <div className="stack exp-gap-2">
      <code ref={ref} className="cmd">{command}</code>
      <div className="row">
        <button type="button" className="small" onClick={copy}>{state === "copied" ? <><Icon name="check" size={14} />Copied</> : "Copy"}</button>
        {state === "manual" && <span className="muted small-text">Selected: press <kbd>⌘C</kbd> to copy</span>}
      </div>
    </div>
  );
}


/** Downloads the export's script: a Python file that reads the original data the way Rainforest did and rebuilds this
 * export with its decisions, then checks its result against the export. Fetched rather than linked, so a refusal (a
 * sample made before Rainforest recorded how) shows here. */
function ScriptDownload({ exportId }: { exportId: number }) {
  const [state, setState] = useState<"idle" | "busy" | Error>("idle");
  const save = async () => {
    setState("busy");
    try {
      const { name, blob } = await api.exportScript(exportId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      setState("idle");
    } catch (err) {
      setState(err as Error);
    }
  };
  return (
    <div className="stack exp-gap-2">
      <div className="row">
        <button type="button" className="small" disabled={state === "busy"} onClick={save}>
          <Icon name="download" size={14} />{state === "busy" ? "Writing script…" : "Python script (.py)"}
        </button>
      </div>
      <span className="muted small-text">
        The decisions as code you can edit. Run it on the original data and it rebuilds this export, then checks it
        matches.
      </span>
      {state instanceof Error && <ErrorAlert error={state} />}
    </div>
  );
}

function featuresByType(e: ExportJob): string {
  const by = e.summary?.features_by_type as Record<string, { fields: number; features: number }> | undefined;
  if (!by) return "";
  return Object.entries(by).map(([t, x]) => `${t.replace("_", " ")}: ${x.fields} → ${x.features}`).join("\n");
}

function ExportRow({ e, canAdd, onTrain }: { e: ExportJob; canAdd: boolean; onTrain: (exportId: number) => void }) {
  const s = e.summary ?? {};
  const by = s.features_by_type as Record<string, { fields: number }> | undefined;
  const inputs = by ? Object.values(by).reduce((a, x) => a + x.fields, 0) : null;
  return (
    <div className="export-row">
      <div className="stack exp-gap-1 exp-grow">
        <div className="row">
          <b>Export #{e.id}</b>
          <Chip>{familyShort(e.model_family)}</Chip>
          {e.recipe_version != null && <Chip title="Built from this baseline version's snapshot">{baselineLabel(e.recipe_family ?? e.model_family, e.recipe_version, true, e.recipe_name)}</Chip>}
          <StatusBadge status={e.status} />
          <span className="muted small-text num" title={fmtDate(e.created_at)}>
            {fmtAgo(e.created_at)}
            {e.n_rows !== null && ` · ${fmtInt(e.n_rows)} rows · `}
            {e.n_features !== null && (
              <InfoTip text={featuresByType(e) || "Text, date and one-hot fields expand into several features each."}
                label={`${inputs !== null ? `${fmtInt(inputs)} fields → ` : ""}${fmtInt(e.n_features)} features`} />
            )}
          </span>
        </div>
        {isActive(e.status) && (
          <div className="row export-progress">
            <ProgressBar value={Math.max(e.progress, 0.02)} max={1} label={`Export #${e.id}: ${e.status_message ?? e.status}`} />
            <span className="muted small-text">{e.status_message}</span>
          </div>
        )}
        {e.status === "done" && s.split_counts && (
          <span className="muted small-text num">
            train {fmtInt(s.split_counts.train)} · valid {fmtInt(s.split_counts.valid)} · test {fmtInt(s.split_counts.test)}
            {s.split_info?.stratified ? ` · stratified on ${s.split_info.stratified_on?.join(" × ") ?? "the targets"}` : ""}
            {s.split_info?.grouped_on ? ` · ${s.split_info.grouped_on} kept together` : ""}
            {s.split_info?.rows_removed && Object.keys(s.split_info.rows_removed).length > 0
              ? ` · left out: ${Object.entries(s.split_info.rows_removed).map(([k, n]) => `${fmtInt(n as number)} ${k}`).join(", ")}` : ""}
            {s.contains_nan ? " · contains NaN (tree models only)" : ""}
          </span>
        )}
        {e.status === "failed" && e.error && (
          <Banner tone="bad"><details><summary>{e.error.split("\n")[0]}</summary><pre>{e.error}</pre></details></Banner>
        )}
      </div>
      {e.status === "done" && (
        <div className="row export-actions">
          {canAdd && <button type="button" className="small primary" onClick={() => onTrain(e.id)}>Train on it</button>}
          <Pop label={<>Files<Icon name="chevron-down" size={14} /></>}>{() => (
            <>
              <a className="btn" href={`/api/exports/${e.id}/files/data`}>Dataset (.parquet)</a>
              <a className="btn" href={`/api/exports/${e.id}/files/preprocessor`}>Pipeline (.joblib)</a>
              <a className="btn" href={`/api/exports/${e.id}/files/manifest`}>Manifest (.json)</a>
            </>
          )}</Pop>
          <Pop label="Use from code">{() => (
            <>
              <span className="secondary small-text">Rebuild it in Python, outside the app</span>
              <ScriptDownload exportId={e.id} />
              <span className="secondary small-text">Train from a shell</span>
              <CopyCommand command={s.train_command ?? `docker compose exec api python -m app.modeling.train --export-id ${e.id}`} />
              {s.targets && (
                <span className="muted small-text">
                  {s.targets.map((t: string) => (s.target_kinds?.[t] === "regression" ? `${t}: number`
                    : `${t}: ${s.target_classes?.[t]?.map((c: string, i: number) => `${i}=${c}`).join(", ")}`)).join(" · ")}
                </span>
              )}
            </>
          )}</Pop>
        </div>
      )}
    </div>
  );
}

/** An experiment's exports and runs, polled while any job runs. The page holds these so polling and the
 * finished-job toasts carry on whichever tab is open. */
export function useJobs(datasetId: number, experimentId: number) {
  const exports = useQuery({
    queryKey: ["exports", datasetId, experimentId],
    queryFn: () => api.exports(datasetId, experimentId),
    refetchInterval: (q) => ((q.state.data as ExportJob[] | undefined)?.some((e) => isActive(e.status)) ? 1000 : false),
  });
  const runs = useQuery({
    queryKey: ["runs", datasetId, experimentId],
    queryFn: () => api.runs(datasetId, experimentId),
    refetchInterval: (q) => ((q.state.data as ModelRun[] | undefined)?.some((r) => isActive(r.status)) ? 1500 : false),
  });
  return { exports, runs };
}

/** Says when a job finishes: an export that was running ("Export N is ready · Train on it"), and a batch of runs this
 * page started ("3 runs finished · best R² 0.941 · Compare"). */
export function useJobToasts({ x, exports, runs, fresh, canAdd, base }: {
  x: Experiment; exports: ExportJob[] | undefined; runs: ModelRun[] | undefined; fresh: number[]; canAdd: boolean; base: string;
}) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const seen = useRef(new Map<string, string>());
  const told = useRef<string>("");
  useEffect(() => {
    let changed = false;
    let exportDone = false;
    for (const e of exports ?? []) {
      const before = seen.current.get(`e${e.id}`);
      if (before && isActive(before) && !isActive(e.status)) {
        changed = true;
        exportDone = true;
        if (e.status === "done") {
          toast({ tone: "good", message: `Export #${e.id} is ready`, duration: 10_000,
            action: canAdd ? { label: "Train on it", run: () => nav(`${base}?tab=train&export=${e.id}`) } : undefined });
        } else {
          toast({ tone: "bad", message: `Export #${e.id} failed`, detail: e.error?.split("\n")[0], action: { label: "See why", run: () => nav(`${base}?tab=exports`) } });
        }
      }
      seen.current.set(`e${e.id}`, e.status);
    }
    for (const r of runs ?? []) {
      const before = seen.current.get(`r${r.id}`);
      if (before && isActive(before) && !isActive(r.status)) changed = true;
      seen.current.set(`r${r.id}`, r.status);
    }
    if (changed) qc.invalidateQueries({ queryKey: ["experiment", x.id] });
    // A finished export is the newest to compare decisions with (and to take row rules from).
    if (exportDone) {
      qc.invalidateQueries({ queryKey: ["exportPreview"] });
      qc.invalidateQueries({ queryKey: ["exportRows"] });
      qc.invalidateQueries({ queryKey: ["exports"] });
    }
    // A batch this page started: say so once all of it is through.
    const batch = (runs ?? []).filter((r) => fresh.includes(r.id));
    const key = fresh.join(",");
    if (key && told.current !== key && batch.length === fresh.length && batch.every((r) => !isActive(r.status))) {
      told.current = key;
      const e = (exports ?? []).find((x) => x.id === batch[0].export_id);
      const target = batch[0].report?.order?.[0] ?? e?.summary?.targets?.[0];
      toast({ tone: batch.some((r) => r.status === "done") ? "good" : "bad", duration: 10_000,
        message: batchSummary(batch, target, target ? e?.summary?.target_kinds?.[target] : undefined),
        action: { label: "Compare", run: () => nav(`${base}?tab=runs&export=${batch[0].export_id}`) } });
    }
  }, [exports, runs, fresh, canAdd, base, nav, qc, x.id]);
}

/** The runs table for one export, with the run picked open in an aside beside it (below it on a narrow screen). */
export function RunsTab({ x, exports, runs, error, canAdd, fresh, onTrainAgain }: {
  x: Experiment; exports: ExportJob[]; runs: ModelRun[] | undefined; error: unknown; canAdd: boolean; fresh: number[];
  onTrainAgain: ((p: TrainPrefill) => void) | null;
}) {
  const [params, setParams] = useSearchParams();
  const [split, setSplit] = useState<Split>("test");
  const aside = useRef<HTMLElement>(null);
  const all = useMemo(() => [...(runs ?? [])].sort((a, b) => b.id - a.id), [runs]);
  const withRuns = useMemo(() => exports.filter((e) => all.some((r) => r.export_id === e.id)).sort((a, b) => b.id - a.id), [exports, all]);
  const selectedId = Number(params.get("run")) || null;
  const selected = all.find((r) => r.id === selectedId) ?? null;
  const asked = Number(params.get("export")) || null;
  const e = withRuns.find((x) => x.id === asked) ?? withRuns.find((x) => x.id === selected?.export_id) ?? withRuns.find((x) => x.id === all[0]?.export_id) ?? null;
  const shown = e ? all.filter((r) => r.export_id === e.id) : [];
  const set = (patch: Record<string, string | null>) => setParams((p) => {
    const next = new URLSearchParams(p);
    for (const [k, v] of Object.entries(patch)) (v === null ? next.delete(k) : next.set(k, v));
    return next;
  }, { replace: true });
  const select = (id: number | null) => set({ run: id === null ? null : String(id) });
  // Closing the aside puts focus back on the run's row, where it was opened from.
  const close = () => {
    const id = selected?.id;
    select(null);
    if (id) requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-run-open="${id}"]`)?.focus());
  };
  const closeRef = useRef(close);
  closeRef.current = close;
  // Esc closes it wherever focus is, unless a dialog or popover takes the Esc, or you are typing.
  useEffect(() => {
    if (!selectedId) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== "Escape" || ev.defaultPrevented) return;
      if (document.querySelector('[aria-modal="true"], .popover')) return;
      if ((ev.target as HTMLElement | null)?.closest?.("textarea, input, select")) return;
      closeRef.current();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [selectedId]);
  // Open a run as soon as the one run you just started finishes.
  const prev = useRef(new Map<number, string>());
  useEffect(() => {
    for (const r of all) {
      const before = prev.current.get(r.id);
      if (fresh.length === 1 && r.id === fresh[0] && before && isActive(before) && !isActive(r.status)) select(r.id);
      prev.current.set(r.id, r.status);
    }
  }, [all, fresh]);
  // On a narrow screen the aside sits below the table: bring it into view.
  useEffect(() => {
    if (selected && window.matchMedia("(max-width: 1099px)").matches) aside.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selected?.id]);

  if (runs && !all.length) {
    const ready = exports.some((x) => x.status === "done");
    return (
      <div className="page">
        <ErrorAlert error={error} />
        <section className="card">
          <Empty title="No runs yet"
            action={canAdd ? (ready ? <Link className="btn primary" to="?tab=train"><Icon name="play" />Train a model</Link>
              : <Link className="btn primary" to="?tab=exports">Make an export</Link>) : undefined}>
            {ready ? "Train a model on a finished export to compare runs here." : "Make an export first: it prepares the data the models train on."}
          </Empty>
        </section>
      </div>
    );
  }
  const targets: string[] = e?.summary?.targets ?? [];
  const kinds: Record<string, string> = e?.summary?.target_kinds ?? {};
  // The batch just started, counted while any of it still trains.
  const batch = all.filter((r) => fresh.includes(r.id));
  const batchDone = batch.filter((r) => !isActive(r.status)).length;
  const batchOn = fresh.length > 1 && batch.some((r) => isActive(r.status)) ? batch[0].export_id : null;
  return (
    <div className={`exp-runs${selected ? " with-aside" : ""}`}>
      <div className="exp-runs-main stack">
        <ErrorAlert error={error} />
        {!runs && <span className="skeleton exp-skel-line" />}
        {e && (
          <>
            <div className="runs-toolbar">
              <label className="exp-export-pick">
                <span className="exp-sr">Export</span>
                <select value={e.id} aria-label="Export" onChange={(ev) => set({ export: ev.target.value, run: null })}>
                  {withRuns.map((x) => {
                    const sc = x.summary?.split_counts;
                    return <option key={x.id} value={x.id}>Export #{x.id}{sc ? ` · ${fmtInt(sc.train)} / ${fmtInt(sc.valid)} / ${fmtInt(sc.test)} rows` : ""} · {plural(all.filter((r) => r.export_id === x.id).length, "run")}</option>;
                  })}
                </select>
              </label>
              <Segmented label="Scores on" value={split} onChange={setSplit} options={[
                { value: "valid", label: "Validation", title: "Scores on the validation split, where the star is picked" },
                { value: "test", label: "Test", title: "Scores on the held-out test split" },
                { value: "cv", label: "Cross-validation", title: "Cross-validation on the training rows: mean ± spread over the folds (— for runs without it)" },
              ]} />
              <span className="spacer" />
              {batchOn !== null && (
                <span className="batch-progress small-text num" role="status">
                  <ProgressBar value={batchDone} max={fresh.length} label={`Runs just started: ${batchDone} of ${fresh.length} done`} />
                  <span><b>{batchDone} of {fresh.length}</b> done{batchOn !== e.id ? ` on Export #${batchOn}` : ""}</span>
                </span>
              )}
              <span className="star-legend small-text"><Icon name="star" size={14} className="star" />best on validation, whichever split is shown</span>
            </div>
            <RunsTable e={e} runs={shown} targets={targets} kinds={kinds} split={split} fresh={fresh} selectedId={selected?.id ?? null} onSelect={select} />
            <p className="muted small-text exp-flush">
              {x.private ? "Runs are logged to MLflow once this baseline version reaches review. " : "Every run is logged to MLflow with its full preparation. "}
              Select a run to see its detail{selected ? " here" : ""}.
            </p>
          </>
        )}
      </div>
      {selected && (
        <aside ref={aside} id="run-detail" className="run-aside" aria-labelledby={`run-${selected.id}-title`}>
          <RunDetail key={selected.id} run={selected} split={split}
            targets={selected.report?.order ?? selected.config?.order ?? exports.find((x) => x.id === selected.export_id)?.summary?.targets ?? []}
            kinds={exports.find((x) => x.id === selected.export_id)?.summary?.target_kinds}
            onTrainAgain={canAdd ? onTrainAgain : null} onClose={close} />
        </aside>
      )}
    </div>
  );
}

/** One export's runs: its targets are its columns, and the star compares only runs on its features. */
function RunsTable({ e, runs, targets, kinds, split, fresh, selectedId, onSelect }: {
  e: ExportJob; runs: ModelRun[]; targets: string[]; kinds: Record<string, string>; split: Split; fresh: number[];
  selectedId: number | null; onSelect: (id: number | null) => void;
}) {
  const best = useMemo(() => bestRuns(runs, targets), [runs, targets]);
  const models = useQuery({ queryKey: ["exportModels", e.id], queryFn: () => api.modelsForExport(e.id), staleTime: 60_000 });
  const settingsOf = (model: string) => models.data?.find((m) => m.key === model)?.settings;
  const multi = targets.length > 1;
  const cols = (["primary", "pr", "secondary"] as const).flatMap((m) => targets.flatMap((t) =>
    runColumns(kinds[t]).filter((c) => c.metric === m).map((c) => ({ t, m, label: c.label }))));
  // A finished run opens on its results; a failed one on why it failed.
  const opens = (r: ModelRun) => r.status === "done" || r.status === "failed";
  const cell = (r: ModelRun, t: string, m: RunMetric) => {
    const v = metricFor(r, split, t, m);
    if (v === undefined) return <td className="num muted">—</td>;
    const isBest = best[`${t}:${m}`] === r.id;
    return (
      <td className={`num${isBest ? " best" : ""}`} title={isBest ? starTitle(r, split, t, m) : undefined}>
        {fmtMetric(v)}{split === "cv" && spreadFor(r, t, m) !== undefined && <span className="muted small-text"> ± {fmtMetric(spreadFor(r, t, m))}</span>}
        {isBest && <Icon name="star" size={13} className="star" label="best on validation" />}
      </td>
    );
  };
  const adjusted = adjustedNote(e);
  return (
    <div className="card table-wrap runs-card">
      {adjusted && <div className="runs-note muted small-text">Export #{e.id}{adjusted}</div>}
      <table className="data runs">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col" className="run-params-col">Changed settings</th>
            {multi && <th scope="col">Chain order</th>}
            {cols.map((c) => <th key={c.t + c.m} scope="col" className="num">{multi ? `${c.t} ` : ""}{c.label}</th>)}
            <th scope="col" className="num">Fit</th>
            <th scope="col">Device</th>
            <th scope="col">MLflow</th>
            <th scope="col"><span className="exp-sr">Status</span></th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const order = r.report?.order ?? r.config?.order ?? targets;
            const label = runLabel(r, models.data?.find((m) => m.key === r.model));
            const on = selectedId === r.id;
            const params = (r.config?.params ?? {}) as Record<string, SettingValue>;
            const changes = describeSettings(params, settingsOf(r.model));
            return (
              <tr key={r.id} className={`run-row${on ? " selected" : ""}${fresh.includes(r.id) ? " fresh" : ""}${opens(r) ? "" : " inert"}`}
                onClick={() => opens(r) && onSelect(on ? null : r.id)}>
                <td className="run-model">
                  <div className="row exp-nowrap">
                    {opens(r) ? (
                      <button type="button" className="run-open" data-run-open={r.id} aria-pressed={on} aria-controls={on ? "run-detail" : undefined}
                        onClick={(ev) => { ev.stopPropagation(); onSelect(on ? null : r.id); }}>{label}</button>
                    ) : <span className="secondary">{label}</span>}
                    {r.chosen_at && <Chip tone="good" title={r.chosen_by_name ? `Chosen by ${r.chosen_by_name}` : undefined}>Chosen</Chip>}
                    {r.note && <span title={r.note}><Icon name="comment" size={14} className="muted" label="Has a note" /></span>}
                  </div>
                  <div className="muted small-text run-params-inline">{changes}</div>
                </td>
                <td className={`run-params run-params-col ${changes === "defaults" ? "muted" : "secondary"}`}>{changes}</td>
                {multi && <td className="secondary">{order.join(" → ")}</td>}
                {isActive(r.status) ? (
                  <td colSpan={cols.length + 1}>
                    <span className="run-progress">
                      <span className="meter indeterminate" role="progressbar" aria-label={`${label}: ${r.status_message ?? (r.status === "pending" ? "queued" : "training")}`}><span /></span>
                      <span className="secondary small-text">{r.status_message ?? (r.status === "pending" ? "Queued" : "Training")}</span>
                    </span>
                  </td>
                ) : (
                  <>
                    {cols.map((c) => <Fragment key={c.t + c.m}>{cell(r, c.t, c.m)}</Fragment>)}
                    <td className="num secondary">{r.report?.fit_seconds !== undefined ? `${fmtNum(r.report.fit_seconds, 2)} s` : "—"}</td>
                  </>
                )}
                <td>{(r.report?.device ?? r.config?.device) === "cuda" ? <Chip tone="accent">GPU</Chip> : <span className="muted small-text">CPU</span>}</td>
                <td><MlflowLink run={r} /></td>
                <td>{r.status !== "done" && <span title={r.status === "failed" ? whyFailed(r.error) : undefined}><StatusBadge status={r.status} /></span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Make an export from the experiment's source, and the exports made so far. */
export function ExportsTab({ ds, x, exports, error, canAdd, onTrain }: {
  ds: Dataset; x: Experiment; exports: ExportJob[] | undefined; error: unknown; canAdd: boolean; onTrain: (exportId: number) => void;
}) {
  return (
    <div className="page exports-tab">
      {canAdd && <BuildPanel datasetId={ds.id} family={x.recipe_family} rows={ds.row_count} x={x} />}
      <section className="card stack exports-list" aria-labelledby="exports-title">
        <div className="card-head"><h2 id="exports-title">Exports</h2></div>
        <ErrorAlert error={error} />
        {exports?.length === 0 && <div className="empty">No exports yet.{canAdd ? " Make one above, then train models on it." : ""}</div>}
        {[...(exports ?? [])].sort((a, b) => b.id - a.id).map((e) => <ExportRow key={e.id} e={e} canAdd={canAdd} onTrain={onTrain} />)}
      </section>
    </div>
  );
}

/** Train on one of the experiment's finished exports. `?export=` picks the export; Train again brings a run's
 * settings in the navigation state. */
export function TrainTab({ exports, onStarted, private: personal = false }: {
  exports: ExportJob[] | undefined; onStarted: (runs: ModelRun[], e: ExportJob, label: string) => void; private?: boolean;
}) {
  const loc = useLocation();
  const [params] = useSearchParams();
  const prefill = (loc.state as { prefill?: TrainPrefill } | null)?.prefill;
  const asked = Number(params.get("export")) || null;
  const request = useMemo<TrainRequest | null>(() => (prefill ? { exportId: prefill.exportId, prefill, nonce: Date.now() }
    : asked ? { exportId: asked, nonce: Date.now() } : null), [loc.key]);
  if (!exports) return <div className="page" aria-busy="true"><span className="skeleton exp-skel-line short" /></div>;
  if (!exports.some((e) => e.status === "done")) {
    const running = exports.some((e) => isActive(e.status));
    return (
      <div className="page">
        <section className="card">
          <Empty title={running ? "The export is still running" : "Nothing to train on yet"}
            action={<Link className="btn primary" to="?tab=exports">{running ? "See the export" : "Make an export"}</Link>}>
            {running ? "Train opens on it as soon as it finishes." : "Models train on an export: the data prepared as this experiment's source decided."}
          </Empty>
        </section>
      </div>
    );
  }
  return <div className="page exp-wide"><TrainPanel exports={exports} request={request} onStarted={onStarted} private={personal} /></div>;
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** " · 3 fields adjusted · targets changed", from what the export recorded. */
function adjustedNote(e: ExportJob): string {
  const a = (e.config as { adjustments?: { count?: number; targets_changed?: boolean } } | null)?.adjustments;
  if (!a) return "";
  return `${a.count ? ` · ${a.count} field${a.count === 1 ? "" : "s"} adjusted` : ""}${a.targets_changed ? " · targets changed" : ""}`;
}
