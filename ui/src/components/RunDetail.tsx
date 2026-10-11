import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { describeSettings } from "../settings";
import { histogramOption, rocOption, scatterOption } from "../charts";
import { fmtInt, fmtMetric, fmtPct } from "../format";
import { useChartTheme } from "../theme";
import type { HistogramChart, ModelRun, RegressionEvaluation, RocCurve, SettingValue, TargetEvaluation, TrainPrefill } from "../types";
import { Chart } from "./Chart";

type Split = "test" | "valid";

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
        <div className="segmented" role="group" aria-label="Cell values">
          <button className={`small${mode === "count" ? " on" : ""}`} onClick={() => setMode("count")}>counts</button>
          <button className={`small${mode === "row" ? " on" : ""}`} onClick={() => setMode("row")}>row %</button>
        </div>
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

export function regressionTabs(ev: RegressionEvaluation): { key: "scatter" | "errors" | "bands"; label: string }[] {
  return [
    { key: "scatter", label: "Predicted vs actual" },
    { key: "errors", label: "Errors" },
    ...(ev.bands ? [{ key: "bands" as const, label: "Ranges" }] : []),
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
        Actual and predicted values put in {b.labels.length} equal-sized ranges cut on the training rows, then scored as
        classes: how often the model predicts the right range.
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

function RegressionCard({ target, m, ev }: { target: string; m: Record<string, number>; ev?: RegressionEvaluation }) {
  const t = useChartTheme();
  const [tab, setTab] = useState<"scatter" | "errors" | "bands">("scatter");
  const scatter = useMemo(() => (ev ? scatterOption(t, ev.scatter.actual, ev.scatter.predicted) : null), [t, ev]);
  const residuals = useMemo(() => (ev ? histogramOption(t, { kind: "histogram", edges: ev.residuals.edges, counts: ev.residuals.counts } as HistogramChart,
    { xLabel: "Error (predicted − actual)" }) : null), [t, ev]);
  const tabs = ev ? regressionTabs(ev) : [];
  const shown = tabs.some((x) => x.key === tab) ? tab : "scatter";
  return (
    <section className="card stack target-card">
      <div className="card-head" style={{ marginBottom: 0 }}>
        <h3>◎ {target}</h3>
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
          <div className="segmented soft" role="tablist" aria-label={`Results for ${target}`}>
            {tabs.map((x) => (
              <button key={x.key} type="button" role="tab" aria-selected={shown === x.key} className={shown === x.key ? "on" : ""}
                onClick={() => setTab(x.key)}>{x.label}</button>
            ))}
          </div>
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
          {!ev.bands && <span className="muted small-text">This run predates range scores; train it again to see them.</span>}
        </>
      ) : <div className="muted small-text">No evaluation details for this split.</div>}
    </section>
  );
}

function TargetCard({ target, m, ev }: { target: string; m: Record<string, number>; ev?: TargetEvaluation }) {
  const binary = ev ? ev.labels.length === 2 : "precision" in m;
  const total = ev?.per_class.reduce((a, c) => a + c.support, 0);
  const tiles = headlineTiles(m, binary);
  return (
    <section className="card stack target-card">
      <div className="card-head" style={{ marginBottom: 0 }}>
        <h3>◎ {target}</h3>
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
      {ev ? <ConfusionMatrix ev={ev} accuracy={m.accuracy} /> : (
        <div className="muted small-text">This run predates confusion matrices — train it again to see them.</div>
      )}
      {ev?.roc?.length ? <RocChart curves={ev.roc} binary={binary} /> : ev && (
        <div className="muted small-text">
          {ev.roc ? "No ROC curve: this split has only one class." : "This run predates ROC curves — train it again to see them."}
        </div>
      )}
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
    </section>
  );
}

/** The run's changed settings, and a way to train again with them. */
function RunSettings({ run, onTrainAgain }: { run: ModelRun; onTrainAgain: ((p: TrainPrefill) => void) | null }) {
  const models = useQuery({ queryKey: ["exportModels", run.export_id], queryFn: () => api.modelsForExport(run.export_id), staleTime: 60_000 });
  const settings = models.data?.find((m) => m.key === run.model)?.settings ?? null;
  const params = (run.config?.params ?? {}) as Record<string, SettingValue>;
  const extra = [run.config?.seed !== undefined && run.config.seed !== 42 ? `seed ${run.config.seed}` : null,
    run.config?.cv !== undefined && run.config.cv !== 5 ? (run.config.cv ? `chain folds ${run.config.cv}` : "chain folds off") : null]
    .filter(Boolean);
  const text = [describeSettings(params, settings), ...extra].filter((x) => x !== "defaults" || !extra.length).join(" · ");
  return (
    <div className="row run-settings" style={{ gap: 10 }}>
      <span className="secondary small-text">Settings: <span className="mono">{text}</span></span>
      {onTrainAgain && settings && (
        <button type="button" className="small" onClick={() => onTrainAgain({
          exportId: run.export_id, model: run.model, order: run.report?.order ?? run.config?.order ?? [],
          values: { ...params, ...(run.config?.seed !== undefined ? { seed: run.config.seed } : {}),
            ...(run.config?.cv !== undefined ? { cv: run.config.cv } : {}) },
        })}>Train again with these settings</button>
      )}
    </div>
  );
}

export function RunDetail({ run, targets, onTrainAgain = null }: {
  run: ModelRun; targets: string[]; onTrainAgain?: ((p: TrainPrefill) => void) | null;
}) {
  const [split, setSplit] = useState<Split>("test");
  const metrics = run.metrics?.[split];
  if (run.status !== "done") {
    return run.error ? <pre className="alert bad" style={{ whiteSpace: "pre-wrap" }}>{run.error}</pre> : null;
  }
  return (
    <div className="stack" style={{ gap: 12, padding: "8px 4px" }}>
      <RunSettings run={run} onTrainAgain={onTrainAgain} />
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="secondary small-text">
          {targets.some((t) => metrics?.[t] && !("r2" in metrics[t] || "rmse" in metrics[t]))
            ? "Class targets use the most likely class (0.5 threshold for binary targets)" : ""}
        </span>
        <div className="row" style={{ gap: 12 }}>
          <a className="small-text" href={`/api/runs/${run.id}/files/model`}>Download model ↓</a>
          <div className="segmented" role="group" aria-label="Split">
            {(["test", "valid"] as Split[]).map((s) => (
              <button key={s} className={`small${split === s ? " on" : ""}`} onClick={() => setSplit(s)}>{s === "valid" ? "validation" : "test"}</button>
            ))}
          </div>
        </div>
      </div>
      {!run.mlflow_url && run.report?.mlflow_error && (
        <div className="alert warn">Not logged to MLflow: {run.report.mlflow_error} (log it later with make mlflow-backfill)</div>
      )}
      {metrics ? (
        <div className="grid-2">
          {targets.filter((t) => metrics[t]).map((t) => {
            const ev = run.report?.evaluation?.[split]?.[t];
            return ev?.kind === "regression" || "r2" in metrics[t] || "rmse" in metrics[t]
              ? <RegressionCard key={t} target={t} m={metrics[t]} ev={ev?.kind === "regression" ? ev : undefined} />
              : <TargetCard key={t} target={t} m={metrics[t]} ev={ev as TargetEvaluation | undefined} />;
          })}
        </div>
      ) : (
        <div className="muted">No {split} split in this export.</div>
      )}
    </div>
  );
}
