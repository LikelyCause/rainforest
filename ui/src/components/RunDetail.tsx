import { useState } from "react";
import { fmtInt, fmtPct } from "../format";
import type { ModelRun, TargetEvaluation } from "../types";

type Split = "test" | "valid";

const f3 = (v: number | undefined) => (v === undefined ? "—" : v.toFixed(3));

/** Rows = actual, columns = predicted. Shade = share of the actual row, so the diagonal reads as recall. */
function ConfusionMatrix({ ev }: { ev: TargetEvaluation }) {
  const [mode, setMode] = useState<"count" | "row">("count");
  const cm = ev.confusion_matrix;
  const rowTotals = cm.map((r) => r.reduce((a, b) => a + b, 0));
  const total = rowTotals.reduce((a, b) => a + b, 0);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="muted" style={{ fontSize: 12 }}>Confusion matrix · {fmtInt(total)} rows · rows = actual, columns = predicted</span>
        <div className="segmented" role="group" aria-label="Cell values">
          <button className={`small${mode === "count" ? " on" : ""}`} onClick={() => setMode("count")}>counts</button>
          <button className={`small${mode === "row" ? " on" : ""}`} onClick={() => setMode("row")}>row %</button>
        </div>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table className="cm" role="grid">
          <thead>
            <tr>
              <th className="cm-corner">actual ↓ / predicted →</th>
              {ev.labels.map((l) => <th key={l} scope="col">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={ev.labels[i]}>
                <th scope="row">{ev.labels[i]}</th>
                {row.map((v, j) => {
                  const share = rowTotals[i] ? v / rowTotals[i] : 0;
                  const strong = share > 0.55;
                  return (
                    <td
                      key={j}
                      className={i === j ? "diag" : undefined}
                      style={{
                        background: `color-mix(in srgb, var(--series-1) ${Math.round(share * 100)}%, var(--surface-1))`,
                        color: strong ? "#ffffff" : "var(--text-primary)",
                      }}
                      title={`actual ${ev.labels[i]}, predicted ${ev.labels[j]}: ${fmtInt(v)} (${fmtPct(share)} of actual ${ev.labels[i]})`}
                    >
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
  const auc = m.roc_auc ?? m.roc_auc_ovr_macro;
  return (
    <section className="card stack" style={{ boxShadow: "none" }}>
      <div className="card-head" style={{ marginBottom: 0 }}>
        <h3>◎ {target}</h3>
        {binary && ev?.positive_label && <span className="muted" style={{ fontSize: 12 }}>positive class: <b>{ev.positive_label}</b></span>}
      </div>
      <div className="stats">
        <div className="stat"><div className="label">Accuracy</div><div className="value">{f3(m.accuracy)}</div></div>
        <div className="stat"><div className="label">Balanced acc.</div><div className="value">{f3(m.balanced_accuracy)}</div></div>
        <div className="stat"><div className="label">{binary ? "ROC AUC" : "ROC AUC (OvR)"}</div><div className="value">{f3(auc)}</div></div>
        <div className="stat"><div className="label">Log loss</div><div className="value">{f3(m.log_loss)}</div></div>
        {binary && <div className="stat"><div className="label">Specificity</div><div className="value">{f3(m.specificity)}</div></div>}
      </div>
      <table className="data">
        <thead>
          <tr>
            <th />
            {binary && <th className="num">Positive class</th>}
            <th className="num">Macro avg</th>
            <th className="num">Weighted avg</th>
          </tr>
        </thead>
        <tbody>
          {(["precision", "recall", "f1"] as const).map((k) => (
            <tr key={k}>
              <td>{k === "f1" ? "F1" : k[0].toUpperCase() + k.slice(1)}</td>
              {binary && <td className="num">{f3(m[k])}</td>}
              <td className="num">{f3(m[`${k}_macro`])}</td>
              <td className="num">{f3(m[`${k}_weighted`])}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {ev ? (
        <>
          <ConfusionMatrix ev={ev} />
          <table className="data">
            <thead>
              <tr><th>Class</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr>
            </thead>
            <tbody>
              {ev.per_class.map((c) => (
                <tr key={c.label}>
                  <td>{c.label}</td>
                  <td className="num">{f3(c.precision)}</td>
                  <td className="num">{f3(c.recall)}</td>
                  <td className="num">{f3(c.f1)}</td>
                  <td className="num">{fmtInt(c.support)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <div className="muted" style={{ fontSize: 12.5 }}>This run predates confusion matrices — train it again to see them.</div>
      )}
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
        <span className="secondary" style={{ fontSize: 13 }}>
          {run.report?.model_label ?? run.model} · chain {(run.report?.order ?? targets).join(" → ")} · predictions use the most likely class
          (0.5 threshold for binary targets)
        </span>
        <div className="segmented" role="group" aria-label="Split">
          {(["test", "valid"] as Split[]).map((s) => (
            <button key={s} className={`small${split === s ? " on" : ""}`} onClick={() => setSplit(s)}>{s === "valid" ? "validation" : "test"}</button>
          ))}
        </div>
      </div>
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
