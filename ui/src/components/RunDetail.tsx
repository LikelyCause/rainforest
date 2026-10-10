import { useState } from "react";
import { fmtInt, fmtMetric, fmtPct } from "../format";
import type { ModelRun, TargetEvaluation } from "../types";

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

function TargetCard({ target, m, ev }: { target: string; m: Record<string, number>; ev?: TargetEvaluation }) {
  const binary = ev ? ev.labels.length === 2 : "precision" in m;
  const total = ev?.per_class.reduce((a, c) => a + c.support, 0);
  const tiles: [string, number | undefined, string?][] = [
    ["Accuracy", m.accuracy],
    ["Bal. acc.", m.balanced_accuracy, "Balanced accuracy: the mean of each class's recall"],
    ["ROC AUC", m.roc_auc ?? m.roc_auc_ovr_macro, binary ? undefined : "One-vs-rest, macro-averaged over classes"],
    ["Log loss", m.log_loss],
    ["Specificity", binary ? m.specificity : undefined, binary ? undefined : "Binary targets only"],
  ];
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

export function RunDetail({ run, targets }: { run: ModelRun; targets: string[] }) {
  const [split, setSplit] = useState<Split>("test");
  const metrics = run.metrics?.[split];
  if (run.status !== "done") {
    return run.error ? <pre className="alert bad" style={{ whiteSpace: "pre-wrap" }}>{run.error}</pre> : null;
  }
  return (
    <div className="stack" style={{ gap: 12, padding: "8px 4px" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="secondary small-text">Predictions use the most likely class (0.5 threshold for binary targets)</span>
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
          {targets.filter((t) => metrics[t]).map((t) => (
            <TargetCard key={t} target={t} m={metrics[t]} ev={run.report?.evaluation?.[split]?.[t]} />
          ))}
        </div>
      ) : (
        <div className="muted">No {split} split in this export.</div>
      )}
    </div>
  );
}
