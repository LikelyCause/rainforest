import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { can } from "../can";
import { useMe } from "../hooks";
import { describeSettings } from "../settings";
import { showError } from "../toast";
import { whyFailed, type Split } from "../runs";
import { Banner, Chip, Icon, Segmented } from "./kit";
import { histogramOption, rocOption, scatterOption } from "../charts";
import { fmtInt, fmtMetric, fmtNum, fmtPct } from "../format";
import { useChartTheme } from "../theme";
import type { CrossValidation, HistogramChart, ModelRun, RegressionEvaluation, RocCurve, SettingValue, TargetEvaluation, TrainPrefill } from "../types";
import { Chart } from "./Chart";


/** Background for a cell holding `share` of its actual row: the diagonal in the accent colour (recall), the rest in
 * the error colour, so mistakes stand out from near-empty cells. */
export function cellShade(share: number, diag: boolean): { background: string; color: string } {
  const pct = Math.round(Math.min(1, Math.max(0, share)) * 100);
  return {
    background: `color-mix(in srgb, var(${diag ? "--series-1" : "--bad"}) ${diag ? pct : Math.round(pct * 0.8)}%, var(--surface-1))`,
    color: diag && share > 0.55 ? "#ffffff" : "var(--text-primary)",
  };
}

/** Rows = actual, columns = predicted. */
function ConfusionMatrix({ ev, accuracy }: { ev: TargetEvaluation; accuracy?: number }) {
  const [mode, setMode] = useState<"count" | "row">("count");
  const cm = ev.confusion_matrix;
  const rowTotals = cm.map((r) => r.reduce((a, b) => a + b, 0));
  const total = rowTotals.reduce((a, b) => a + b, 0);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="muted small-text">
          Confusion matrix · {fmtInt(total)} rows{accuracy !== undefined ? ` · accuracy ${fmtMetric(accuracy)}` : ""}
        </span>
        <Segmented label="Show as" pressed size="small" value={mode} onChange={setMode}
          options={[{ value: "count", label: "Counts" }, { value: "row", label: "Row %" }]} />
      </div>
      <div style={{ overflowX: "auto" }}>
        <table className="cm" role="grid">
          <thead>
            <tr>
              <th className="cm-corner">actual ↓<br />predicted →</th>
              {ev.labels.map((l) => <th key={l} scope="col">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={ev.labels[i]}>
                <th scope="row">{ev.labels[i]}</th>
                {row.map((v, j) => {
                  const share = rowTotals[i] ? v / rowTotals[i] : 0;
                  return (
                    <td key={j} className={i === j ? "diag" : undefined} style={cellShade(share, i === j)}
                      title={`actual ${ev.labels[i]}, predicted ${ev.labels[j]}: ${fmtInt(v)} (${fmtPct(share)} of actual ${ev.labels[i]})`}>
                      {mode === "count" ? fmtInt(v) : fmtPct(share)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RocChart({ curves, binary }: { curves: RocCurve[]; binary: boolean }) {
  const t = useChartTheme();
  const option = useMemo(() => rocOption(t, curves), [t, curves]);
  return (
    <div className="stack" style={{ gap: 4 }}>
      <span className="muted small-text">
        ROC curve{binary ? ` · positive class ${curves[0]?.label}` : "s · each class against the rest"}
      </span>
      <Chart option={option} height={binary ? 240 : 270} ariaLabel={binary ? "ROC curve" : "ROC curves, one per class"} />
    </div>
  );
}

/** Headline tiles: two rows of four. Precision, recall and F1 are the positive class's for a binary target and the
 * macro average otherwise. */
export function headlineTiles(m: Record<string, number>, binary: boolean): [string, number | undefined, string?][] {
  const avg = binary ? undefined : "Macro average over classes";
  return [
    ["Accuracy", m.accuracy],
    ["Bal. acc.", m.balanced_accuracy, "Balanced accuracy: the mean of each class's recall"],
    ["ROC AUC", m.roc_auc ?? m.roc_auc_ovr_macro, binary ? undefined : "One-vs-rest, macro-averaged over classes"],
    ["PR AUC", binary ? m.pr_auc : m.pr_auc_macro, binary ? "Average precision of the positive class; a rare class's baseline is its share of rows"
      : "Average precision, one-vs-rest, macro-averaged over classes"],
    ["Log loss", m.log_loss],
    ["Precision", binary ? m.precision : m.precision_macro, avg ?? "Positive class"],
    ["Recall", binary ? m.recall : m.recall_macro, avg ?? "Positive class"],
    ["F1", binary ? m.f1 : m.f1_macro, avg ?? "Positive class"],
    ["Specificity", binary ? m.specificity : undefined, binary ? undefined : "Binary targets only"],
  ];
}

/** A number target's tiles: R² first (the headline), then the errors in the target's own units. */
export function regressionTiles(m: Record<string, number>): [string, number | undefined, string?][] {
  return [
    ["R²", m.r2, "Share of the variance explained; 1 is perfect, 0 is no better than predicting the mean"],
    ["RMSE", m.rmse, "Root mean squared error, in the target's units"],
    ["MAE", m.mae, "Mean absolute error, in the target's units"],
    ["MAPE", m.mape, m.mape === undefined ? "Not defined: some actual values are 0" : "Mean absolute percentage error (0.1 = 10%)"],
  ];
}

type RegressionTab = "scatter" | "errors" | "bands" | "features";
export function regressionTabs(ev: RegressionEvaluation, features = false): { key: RegressionTab; label: string }[] {
  return [
    { key: "scatter", label: "Predicted vs actual" },
    { key: "errors", label: "Errors" },
    ...(ev.bands ? [{ key: "bands" as const, label: "Ranges" }] : []),
    ...(features ? [{ key: "features" as const, label: "Top features" }] : []),
  ];
}

/** A number scored as classes: which range it fell in against which range was predicted. */
function RangesView({ ev, m }: { ev: RegressionEvaluation; m: Record<string, number> }) {
  const b = ev.bands!;
  const asClasses = { labels: b.labels, confusion_matrix: b.confusion_matrix, per_class: b.per_class } as unknown as TargetEvaluation;
  const macro = (k: "precision" | "recall" | "f1") => b.per_class.reduce((a, c) => a + c[k], 0) / Math.max(b.per_class.length, 1);
  return (
    <div className="stack" style={{ gap: 10 }}>
      <span className="muted small-text">
        Actual and predicted values put in {b.labels.length} ranges of roughly equal count, cut on the training rows (tied
        values can make fewer), then scored as classes: how often the model predicts the right range.
      </span>
      <div className="stats metric-tiles">
        {([["Range accuracy", m.band_accuracy], ["Precision", macro("precision")], ["Recall", macro("recall")], ["F1", m.band_f1_macro ?? macro("f1")]] as [string, number | undefined][])
          .map(([label, v]) => (
            <div key={label} className="stat" title={label === "Range accuracy" ? undefined : "Macro average over ranges"}>
              <div className="label">{label}</div>
              <div className={`value${v === undefined ? " muted" : ""}`}>{fmtMetric(v)}</div>
            </div>
          ))}
      </div>
      <ConfusionMatrix ev={asClasses} accuracy={m.band_accuracy} />
      <table className="data class-metrics">
        <thead>
          <tr><th>Range</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr>
        </thead>
        <tbody>
          {b.per_class.map((c) => (
            <tr key={c.label}>
              <td className="mono">{c.label}</td>
              <td className="num">{fmtMetric(c.precision)}</td>
              <td className="num">{fmtMetric(c.recall)}</td>
              <td className="num">{fmtMetric(c.f1)}</td>
              <td className="num">{fmtInt(c.support)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type Features = { feature: string; source: string; importance: number }[];

function RegressionCard({ target, m, ev, features }: { target: string; m: Record<string, number>; ev?: RegressionEvaluation; features?: Features }) {
  const t = useChartTheme();
  const [tab, setTab] = useState<RegressionTab>("scatter");
  const steady = useSteadyHeight();
  const scatter = useMemo(() => (ev ? scatterOption(t, ev.scatter.actual, ev.scatter.predicted) : null), [t, ev]);
  const residuals = useMemo(() => (ev ? histogramOption(t, { kind: "histogram", edges: ev.residuals.edges, counts: ev.residuals.counts } as HistogramChart,
    { xLabel: "Error (predicted − actual)" }) : null), [t, ev]);
  const tabs = ev ? regressionTabs(ev, Boolean(features?.length)) : [];
  const shown = tabs.some((x) => x.key === tab) ? tab : "scatter";
  return (
    <section className="card stack target-card">
      <div className="card-head" style={{ marginBottom: 0 }}>
        <h3><Icon name="target" size={14} /> {target}</h3>
        <span className="muted small-text">predicted as a number</span>
      </div>
      <div className="stats metric-tiles">
        {regressionTiles(m).map(([label, v, hint]) => (
          <div key={label} className="stat" title={hint}>
            <div className="label">{label}</div>
            <div className={`value${v === undefined ? " muted" : ""}`}>{fmtMetric(v)}</div>
          </div>
        ))}
      </div>
      {ev && scatter && residuals ? (
        <>
          <ResultTabs tabs={tabs} value={shown} onChange={(k) => { steady.hold(); setTab(k); }} label={`Results for ${target}`} />
          <div ref={steady.box} className="stack" style={{ gap: 8 }}>
          {shown === "scatter" && (
            <div className="stack" style={{ gap: 4 }}>
              <span className="muted small-text">
                {fmtInt(ev.n)} rows{ev.n > ev.scatter.actual.length ? `, showing ${fmtInt(ev.scatter.actual.length)}` : ""} · on the dashed line the prediction is exact
              </span>
              <Chart option={scatter} height={300} ariaLabel="Predicted against actual values" />
            </div>
          )}
          {shown === "errors" && (
            <div className="stack" style={{ gap: 4 }}>
              <span className="muted small-text">How far off each prediction was: left of 0 too low, right of 0 too high</span>
              <Chart option={residuals} height={260} ariaLabel="Histogram of prediction errors" />
            </div>
          )}
          {shown === "bands" && <RangesView ev={ev} m={m} />}
          {shown === "features" && features && <TopFeatures rows={features} />}
          </div>
          {!ev.bands && <span className="muted small-text">This run predates range scores; train it again to see them.</span>}
        </>
      ) : (
        <>
          <div className="muted small-text">No evaluation details for this split.</div>
          {features?.length ? <TopFeatures rows={features} /> : null}
        </>
      )}
    </section>
  );
}

type ClassTab = "matrix" | "roc" | "classes" | "threshold" | "features";
export function classTabs(threshold = false, features = false): { key: ClassTab; label: string }[] {
  return [{ key: "matrix", label: "Confusion" }, { key: "roc", label: "ROC curves" }, { key: "classes", label: "Per class" },
    ...(threshold ? [{ key: "threshold" as const, label: "Threshold" }] : []),
    ...(features ? [{ key: "features" as const, label: "Top features" }] : [])];
}

/** What moving a binary target's cut-off from 0.5 would trade: precision against recall, at each threshold. */
function ThresholdView({ rows, positive }: { rows: NonNullable<TargetEvaluation["thresholds"]>; positive: string | null }) {
  const [t, setT] = useState(0.5);
  const at = rows.reduce((a, r) => (Math.abs(r.threshold - t) < Math.abs(a.threshold - t) ? r : a));
  const best = rows.reduce((a, r) => (r.f1 > a.f1 ? r : a));
  return (
    <div className="stack" style={{ gap: 8 }}>
      <label className="field">
        <span>Flag as <b>{positive ?? "positive"}</b> when its probability is at least <b>{fmtNum(at.threshold)}</b></span>
        <input type="range" min={0.05} max={0.95} step={0.05} value={t} onChange={(e) => setT(Number(e.target.value))} aria-label="Threshold" />
      </label>
      <div className="stats metric-tiles">
        {([["Precision", at.precision], ["Recall", at.recall], ["F1", at.f1], ["Rows flagged", at.flagged]] as const).map(([label, v]) => (
          <div key={label} className="stat"><div className="label">{label}</div><div className="value">{label === "Rows flagged" ? fmtPct(v, 1) : fmtMetric(v)}</div></div>
        ))}
      </div>
      <span className="muted small-text">
        Best F1 is {fmtMetric(best.f1)} at {fmtNum(best.threshold)}{best.threshold !== 0.5 ? " (the scores above use 0.5)" : ""}.
        Scores here are for this split; pick a threshold on validation, then check it on test.
      </span>
    </div>
  );
}

/** Keeps a box from shrinking when its content is swapped, so a page scrolled to the bottom doesn't jump up when a
 * shorter tab or split replaces a taller one. */
function useSteadyHeight() {
  const box = useRef<HTMLDivElement>(null);
  const hold = () => { if (box.current) box.current.style.minHeight = `${box.current.offsetHeight}px`; };
  return { box, hold };
}

/** A card's views, one at a time, under its tiles. */
function ResultTabs<K extends string>({ tabs, value, onChange, label }: {
  tabs: { key: K; label: string }[]; value: K; onChange: (k: K) => void; label: string;
}) {
  return (
    <div className="view-tabs" role="tablist" aria-label={label}>
      {tabs.map((x) => (
        <button key={x.key} type="button" role="tab" aria-selected={value === x.key} className={value === x.key ? "on" : ""}
          onClick={() => onChange(x.key)}>{x.label}</button>
      ))}
    </div>
  );
}

function TargetCard({ target, m, ev, features }: { target: string; m: Record<string, number>; ev?: TargetEvaluation; features?: Features }) {
  const binary = ev ? ev.labels.length === 2 : "precision" in m;
  const total = ev?.per_class.reduce((a, c) => a + c.support, 0);
  const tiles = headlineTiles(m, binary);
  const [tab, setTab] = useState<ClassTab>("matrix");
  const steady = useSteadyHeight();
  return (
    <section className="card stack target-card">
      <div className="card-head" style={{ marginBottom: 0 }}>
        <h3><Icon name="target" size={14} /> {target}</h3>
        {binary && ev?.positive_label && <span className="muted small-text">positive class: <b>{ev.positive_label}</b></span>}
      </div>
      <div className="stats metric-tiles">
        {tiles.map(([label, v, hint]) => (
          <div key={label} className="stat" title={hint}>
            <div className="label">{label}</div>
            <div className={`value${v === undefined ? " muted" : ""}`}>{fmtMetric(v)}</div>
          </div>
        ))}
      </div>
      <ResultTabs tabs={classTabs(Boolean(ev?.thresholds), Boolean(features?.length))} value={tab} onChange={(k) => { steady.hold(); setTab(k); }} label={`Results for ${target}`} />
      <div ref={steady.box} className="stack" style={{ gap: 8 }}>
      {tab === "threshold" && ev?.thresholds && <ThresholdView rows={ev.thresholds} positive={ev.positive_label} />}
      {tab === "features" && features && <TopFeatures rows={features} />}
      {tab === "matrix" && (ev ? <ConfusionMatrix ev={ev} accuracy={m.accuracy} /> : (
        <div className="muted small-text">This run predates confusion matrices; train it again to see them.</div>
      ))}
      {tab === "roc" && (ev?.roc?.length ? <RocChart curves={ev.roc} binary={binary} /> : (
        <div className="muted small-text">
          {ev?.roc ? "No ROC curve: this split has only one class." : "This run predates ROC curves; train it again to see them."}
        </div>
      ))}
      {tab === "classes" && (
        <table className="data class-metrics">
          <thead>
            <tr><th>Class</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr>
          </thead>
          {ev && (
            <tbody>
              {ev.per_class.map((c) => (
                <tr key={c.label}>
                  <td>{c.label}{binary && c.label === ev.positive_label && <span className="muted small-text"> · positive</span>}</td>
                  <td className="num">{fmtMetric(c.precision)}</td>
                  <td className="num">{fmtMetric(c.recall)}</td>
                  <td className="num">{fmtMetric(c.f1)}</td>
                  <td className="num">{fmtInt(c.support)}</td>
                </tr>
              ))}
            </tbody>
          )}
          <tfoot>
            {(["macro", "weighted"] as const).map((a) => (
              <tr key={a}>
                <td>{a === "macro" ? "Macro avg" : "Weighted avg"}</td>
                <td className="num">{fmtMetric(m[`precision_${a}`])}</td>
                <td className="num">{fmtMetric(m[`recall_${a}`])}</td>
                <td className="num">{fmtMetric(m[`f1_${a}`])}</td>
                <td className="num">{total !== undefined ? fmtInt(total) : ""}</td>
              </tr>
            ))}
          </tfoot>
        </table>
      )}
      </div>
    </section>
  );
}

/** A target's cross-validation: each headline metric as mean ± spread over the folds, the folds on hover. */
function CvCard({ target, cv }: { target: string; cv: CrossValidation }) {
  const s = cv.targets[target];
  const pick = (f: (v: { mean: number; std: number; folds: (number | null)[] }) => number | null) =>
    Object.fromEntries(Object.entries(s).map(([k, v]) => [k, f(v) ?? undefined])) as Record<string, number>;
  const number = "r2" in s || "rmse" in s;
  const tiles = (m: Record<string, number>) => (number ? regressionTiles(m) : headlineTiles(m, "precision" in s));
  const mean = tiles(pick((v) => v.mean));
  const std = tiles(pick((v) => v.std));
  const folds = Array.from({ length: cv.folds }, (_, i) => tiles(pick((v) => v.folds[i])));
  return (
    <section className="card stack target-card">
      <div className="card-head" style={{ marginBottom: 0 }}>
        <h3><Icon name="target" size={14} /> {target}</h3>
        <span className="muted small-text">
          {cv.folds}-fold cross-validation on the training rows{cv.stratified_on ? ` · stratified on ${cv.stratified_on}` : ""}
          {cv.seconds !== undefined ? ` · ${fmtNum(cv.seconds)} s` : ""}
        </span>
      </div>
      <div className="stats metric-tiles">
        {mean.map(([label, v, hint], j) => (
          <div key={label} className="stat" title={[hint, v === undefined ? null : `Per fold: ${folds.map((f) => fmtMetric(f[j][1])).join(" · ")}`].filter(Boolean).join("\n")}>
            <div className="label">{label}</div>
            <div className={`value${v === undefined ? " muted" : ""}`}>
              {fmtMetric(v)}{v !== undefined && std[j][1] !== undefined && <span className="muted small-text"> ± {fmtMetric(std[j][1])}</span>}
            </div>
          </div>
        ))}
      </div>
      <details className="cv-folds" open>
        <summary className="small-text">Per fold</summary>
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th>Fold</th>{mean.map(([label, v]) => (v === undefined ? null : <th key={label} className="num">{label}</th>))}</tr></thead>
            <tbody>
              {folds.map((f, i) => (
                <tr key={i}><td>{i + 1}</td>{mean.map(([label, v], j) => (v === undefined ? null : <td key={label} className="num">{fmtMetric(f[j][1])}</td>))}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}

/** The run's note, and for a lead, marking it as the experiment's chosen model (with the note saying why). */
function RunChoice({ run }: { run: ModelRun }) {
  const qc = useQueryClient();
  const me = useMe().data;
  const lead = can(me, "chooseModel");
  const canNote = lead || (me != null && me.role !== "viewer" && me.id === run.created_by);
  const [note, setNote] = useState(run.note ?? "");
  const save = useMutation({
    mutationFn: (body: { note?: string; chosen?: boolean }) => api.updateRun(run.id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs", run.dataset_id] });
      qc.invalidateQueries({ queryKey: ["experiment", run.experiment_id] });
      qc.invalidateQueries({ queryKey: ["experiments", run.dataset_id] });
    },
    onError: (e) => showError("Couldn't save", e),
  });
  const chosen = Boolean(run.chosen_at);
  const edited = note.trim() !== (run.note ?? "");
  if (!canNote && !chosen && !run.note) return null;
  return (
    <div className={`run-choice${chosen ? " chosen" : ""}`}>
      {chosen && <Icon name="check" className="run-choice-icon" />}
      <div className="exp-grow stack" style={{ gap: 6 }}>
        <b>{chosen ? `Chosen model${run.chosen_by_name ? ` · ${run.chosen_by_name}` : ""}` : "Note"}</b>
        {canNote ? (
          <textarea aria-label="Note" rows={2} placeholder="Why this run: what changed, what it showed" value={note}
            onChange={(e) => setNote(e.target.value)} onBlur={() => { if (edited) save.mutate({ note }); }} />
        ) : run.note && <p className="secondary" style={{ margin: 0, whiteSpace: "pre-wrap" }}>“{run.note}”</p>}
      </div>
      {lead && run.status === "done" && (
        <button type="button" className={chosen ? "ghost small" : "small"} disabled={save.isPending}
          onClick={() => save.mutate({ chosen: !chosen, ...(edited ? { note } : {}) })}>
          {chosen ? "Unmark" : "Mark as chosen"}
        </button>
      )}
    </div>
  );
}

/** Where the run is in MLflow: a link, or why it isn't there (yet). */
export function MlflowLink({ run, button }: { run: ModelRun; button?: boolean }) {
  if (run.mlflow_url) {
    return (
      <a className={button ? "btn small" : undefined} href={run.mlflow_url} target="_blank" rel="noopener" onClick={(ev) => ev.stopPropagation()}>
        MLflow <Icon name="external" size={14} label="(opens in a new tab)" />
      </a>
    );
  }
  if (run.report?.mlflow_pending) {
    return <Chip title="This experiment is private while its baseline version is being assessed, and MLflow can't keep runs private. The run is logged when review opens.">MLflow after review</Chip>;
  }
  if (run.report?.mlflow_error) return <Chip tone="warn" title={`Not logged to MLflow: ${run.report.mlflow_error}`}>Not in MLflow</Chip>;
  return button ? null : <span className="muted small-text">—</span>;
}

/** "strength_mpa as a number", "a → b as classes": what the run predicts. */
function predicts(run: ModelRun, targets: string[]): string {
  const kind = (t: string) => {
    const k = run.report?.target_kinds?.[t];
    const m = run.metrics?.test?.[t] ?? run.metrics?.valid?.[t];
    return k === "regression" || (m && ("r2" in m || "rmse" in m)) ? "a number" : "classes";
  };
  const kinds = new Set(targets.map(kind));
  return kinds.size === 1 ? `${targets.join(" → ")} as ${[...kinds][0]}` : targets.map((t) => `${t} as ${kind(t)}`).join(", ");
}

/** One run in full, for the aside beside the runs table: what it is, the chosen-model box, its scores per target with
 * their charts, and a way to train again with its settings. */
export function RunDetail({ run, targets, onTrainAgain = null, onClose, split: initialSplit = "test" }: {
  run: ModelRun; targets: string[]; onTrainAgain?: ((p: TrainPrefill) => void) | null; onClose?: () => void; split?: Split;
}) {
  const cv = run.report?.cross_validation;
  const [split, setSplit] = useState<Split>(initialSplit === "cv" && !(cv && run.metrics?.cv) ? "test" : initialSplit);
  const steady = useSteadyHeight();
  const models = useQuery({ queryKey: ["exportModels", run.export_id], queryFn: () => api.modelsForExport(run.export_id), staleTime: 60_000 });
  const option = models.data?.find((m) => m.key === run.model);
  const settings = option?.settings ?? null;
  const params = (run.config?.params ?? {}) as Record<string, SettingValue>;
  const extra = [run.config?.seed !== undefined && run.config.seed !== 42 ? `seed ${run.config.seed}` : null,
    run.config?.cv !== undefined && run.config.cv !== 5 ? (run.config.cv ? `chain folds ${run.config.cv}` : "chain folds off") : null,
    !run.config?.cv_folds ? "no cross-validation" : run.config.cv_folds !== 5 ? `${run.config.cv_folds}-fold cross-validation` : null]
    .filter(Boolean);
  const settingsText = [describeSettings(params, settings), ...extra].filter((x) => x !== "defaults" || !extra.length).join(" · ");
  const metrics = run.metrics?.[split];
  const top = run.report?.top_features ?? {};
  const label = run.report?.model_label ?? option?.label ?? run.model;
  const trainAgain = onTrainAgain && settings ? () => onTrainAgain({
    exportId: run.export_id, model: run.model, order: run.report?.order ?? run.config?.order ?? [],
    values: { ...params, ...(run.config?.seed !== undefined ? { seed: run.config.seed } : {}),
      ...(run.config?.cv !== undefined ? { cv: run.config.cv } : {}) },
    cvFolds: run.config?.cv_folds ?? 0,
  }) : null;

  const head = (
    <div className="run-head">
      <div className="stack" style={{ gap: 2, minWidth: 0 }}>
        <h2 id={`run-${run.id}-title`}>{label}</h2>
        <span className="secondary small-text">
          Run {run.id} · <span title="Settings changed from the defaults">{settingsText}</span> · predicts {predicts(run, targets)}
        </span>
      </div>
      <div className="row run-actions">
        {trainAgain && <button type="button" className="small" onClick={trainAgain} title="Open Train with this run's export, model, chain order and settings">Train again</button>}
        <MlflowLink run={run} button />
        {run.status === "done" && <a className="btn small" href={`/api/runs/${run.id}/files/model`} title="The fitted model (.joblib)"><Icon name="download" size={14} />Model</a>}
        {onClose && <button type="button" className="ghost small exp-icon" aria-label="Close run detail" onClick={onClose}><Icon name="x" /></button>}
      </div>
    </div>
  );

  if (run.status === "failed") {
    return (
      <div className="stack run-detail-body">
        {head}
        <Banner tone="bad" title="Failed.">
          {whyFailed(run.error)}
          {run.error && run.error.includes("\n") && (
            <details><summary className="small-text">Details</summary><pre style={{ whiteSpace: "pre-wrap" }}>{run.error}</pre></details>
          )}
        </Banner>
      </div>
    );
  }
  if (run.status !== "done") {
    return <div className="stack run-detail-body">{head}<span className="muted">{run.status_message ?? "Training"}…</span></div>;
  }
  return (
    <div className="stack run-detail-body">
      {head}
      <RunChoice run={run} />
      {!run.mlflow_url && run.report?.mlflow_error && (
        <Banner tone="warn">Not logged to MLflow: {run.report.mlflow_error} (log it later with make mlflow-backfill)</Banner>
      )}
      <div className="row" style={{ justifyContent: "space-between" }}>
        <Segmented label="Scores on" size="small" value={split} onChange={(s) => { steady.hold(); setSplit(s); }}
          options={(["test", "valid", ...(cv && run.metrics?.cv ? ["cv"] : [])] as Split[]).map((s) => ({
            value: s, label: s === "valid" ? "Validation" : s === "cv" ? "Cross-validation" : "Test" }))} />
        <span className="muted small-text">
          {targets.some((t) => metrics?.[t] && !("r2" in metrics[t] || "rmse" in metrics[t]))
            ? "Class targets use the most likely class (0.5 threshold for binary targets)" : ""}
        </span>
      </div>
      {cv?.note && <div className="muted small-text">{cv.note}</div>}
      <div ref={steady.box} className="stack run-targets">
      {split === "cv" && cv && metrics ? (
        targets.filter((t) => cv.targets[t]).map((t) => <CvCard key={t} target={t} cv={cv} />)
      ) : metrics ? (
        targets.filter((t) => metrics[t]).map((t) => {
          const ev = run.report?.evaluation?.[split]?.[t];
          return ev?.kind === "regression" || "r2" in metrics[t] || "rmse" in metrics[t]
            ? <RegressionCard key={t} target={t} m={metrics[t]} ev={ev?.kind === "regression" ? ev : undefined} features={top[t]} />
            : <TargetCard key={t} target={t} m={metrics[t]} ev={ev as TargetEvaluation | undefined} features={top[t]} />;
        })
      ) : (
        <div className="muted">No {split === "valid" ? "validation" : split} split in this export.</div>
      )}
      </div>
    </div>
  );
}

/** The features a target's model leaned on most (gain or impurity importance, scaled to the largest). */
function TopFeatures({ rows }: { rows: Features }) {
  const max = Math.max(...rows.map((r) => r.importance), 1e-12);
  return (
    <div className="stack top-features" style={{ gap: 3 }}>
      <span className="muted small-text">What the model relied on most, scaled to the largest</span>
      {rows.slice(0, 15).map((r) => (
        <div key={r.feature} className="row small-text" style={{ gap: 8, flexWrap: "nowrap" }} title={r.source !== r.feature ? `from ${r.source}` : undefined}>
          <span className="mono tf-name">{r.feature}</span>
          <span className="tf-track" role="img" aria-label={`${r.feature}: ${fmtNum((100 * r.importance) / max, 0)}% of the largest`}>
            <span style={{ width: `${(100 * r.importance) / max}%` }} />
          </span>
        </div>
      ))}
    </div>
  );
}
