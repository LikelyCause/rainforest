import { useMemo, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { api } from "../api";
import { Chart } from "./Chart";
import { barsOption, correlationOption, histogramOption } from "../charts";
import { useChartTheme } from "../theme";
import { fmtInt, fmtNum, fmtPct, METHOD_LABEL } from "../format";
import type { BoxSummary, FieldResponse } from "../types";

const NUMERIC = new Set(["numeric_continuous", "numeric_discrete"]);

function Stat({ label, value, tone }: { label: string; value: string; tone?: "warn" | "bad" }) {
  return (
    <div className={`stat${tone ? ` ${tone}` : ""}`}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function QuantileStrip({ box }: { box: BoxSummary }) {
  const { p01, p25, p50, p75, p99 } = box;
  if ([p01, p25, p50, p75, p99].some((v) => v === null) || p99 === p01) return null;
  const lo = p01!, hi = p99!;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="quantile-strip" aria-label="Quantile summary from p1 to p99"
      title={`p1 ${fmtNum(p01)} · p25 ${fmtNum(p25)} · median ${fmtNum(p50)} · p75 ${fmtNum(p75)} · p99 ${fmtNum(p99)}`}>
      <div className="track" />
      <div className="whisker" style={{ left: 0, right: 0 }} />
      <div className="box" style={{ left: x(p25!), width: `calc(${x(p75!)} - ${x(p25!)})` }} />
      <div className="median" style={{ left: x(p50!) }} />
      {([["p1", p01], ["median", p50], ["p99", p99]] as const).map(([k, v]) => (
        <span key={k} className="tick" title={k}
          style={{ left: x(v!), transform: k === "p1" ? "none" : k === "p99" ? "translateX(-100%)" : undefined }}>
          {k} {fmtNum(v)}
        </span>
      ))}
    </div>
  );
}

interface Props {
  data: FieldResponse;
  onOpenField: (id: number) => void;
}

export function FieldDetailView({ data, onOpenField }: Props) {
  const t = useChartTheme();
  const f = data.field;
  const decisionType = data.decision.semantic_type;
  const [logScale, setLogScale] = useState(false);
  const [hideTargets, setHideTargets] = useState(false);

  const corr = useQuery({
    queryKey: ["corr", f.id],
    queryFn: () => api.correlations(f.id, 40),
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  });
  const corrRows = useMemo(
    () => (corr.data ?? []).filter((r) => !hideTargets || !r.is_target).slice(0, 20),
    [corr.data, hideTargets],
  );

  const dist = f.charts.distribution;
  const useLog = logScale && f.charts.log_distribution;
  const distOption = useMemo(() => {
    if (!dist) return null;
    if (useLog) return histogramOption(t, f.charts.log_distribution!, { xLabel: "log1p(value)" });
    if (dist.kind === "bars") return barsOption(t, dist, { total: f.row_count });
    return histogramOption(t, dist, { time: dist.kind === "time_histogram", xLabel: dist.xlabel });
  }, [t, dist, useLog, f.charts.log_distribution, f.row_count]);
  const corrOption = useMemo(() => correlationOption(t, corrRows), [t, corrRows]);
  const tokensOption = useMemo(() => (f.charts.tokens ? barsOption(t, f.charts.tokens) : null), [t, f.charts.tokens]);
  const weekdayOption = useMemo(() => (f.charts.weekday ? barsOption(t, f.charts.weekday) : null), [t, f.charts.weekday]);

  const isNum = NUMERIC.has(f.inferred_type);
  const nullTone = f.null_pct > 0.5 ? "bad" : f.null_pct > 0.05 ? "warn" : undefined;
  const s = f.stats;

  return (
    <>
      <div className="detail-head">
        <div style={{ minWidth: 0 }}>
          <div className="row" style={{ marginBottom: 6 }}>
            <span className="badge accent" title="Inferred semantic type">{f.inferred_type.replace("_", " ")}</span>
            {f.inference_confidence !== null && (
              <span className={`badge ${f.inference_confidence >= 0.85 ? "good" : f.inference_confidence >= 0.7 ? "" : "warn"}`}>
                {fmtPct(f.inference_confidence, 0)} confident
              </span>
            )}
            <span className="badge mono" title="Physical dtype">{f.physical_dtype}</span>
            {f.is_target && <span className="badge accent">◎ target</span>}
            {f.leakage_flag && <span className="badge bad">⚠ possible leakage</span>}
            {s.stored_as_text && <span className="badge warn">numbers stored as text</span>}
          </div>
          <h1>{f.name}</h1>
          <ul className="reasons">{f.inference_reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          {decisionType !== f.inferred_type && (
            <div className="alert info" style={{ marginTop: 8 }}>
              You've set this field to <b>{decisionType.replace("_", " ")}</b>. Statistics below were computed for the inferred type.
            </div>
          )}
        </div>
        <span className="muted num" style={{ whiteSpace: "nowrap" }}>#{f.ordinal + 1}</span>
      </div>

      <div className="stats">
        <Stat label="Rows" value={fmtInt(f.row_count)} />
        <Stat label="Missing" value={`${fmtPct(f.null_pct)}`} tone={nullTone} />
        <Stat label="Distinct" value={fmtInt(f.n_unique)} />
        <Stat label="Distinct %" value={fmtPct(f.unique_pct)} />
        {isNum && (
          <>
            <Stat label="Mean" value={fmtNum(f.mean)} />
            <Stat label="Std dev" value={fmtNum(f.std)} />
            <Stat label="Min" value={fmtNum(f.min_value)} />
            <Stat label="Median" value={fmtNum(f.median)} />
            <Stat label="Max" value={fmtNum(f.max_value)} />
            <Stat label="Skewness" value={fmtNum(f.skewness, 2)} tone={Math.abs(f.skewness ?? 0) > 2 ? "warn" : undefined} />
            <Stat label="Kurtosis" value={fmtNum(f.kurtosis, 2)} />
            <Stat label="Zeros" value={fmtPct(f.zeros_pct)} />
            <Stat label="IQR outliers" value={fmtPct(f.outlier_pct)} tone={(f.outlier_pct ?? 0) > 0.05 ? "warn" : undefined} />
            <Stat label="Negative" value={fmtPct(s.negative_pct)} />
          </>
        )}
        {(f.inferred_type === "categorical" || f.inferred_type === "binary") && (
          <>
            <Stat label="Top value" value={f.top_value ?? "—"} />
            <Stat label="Top value %" value={fmtPct(f.top_value_pct)} />
            <Stat label="Rare levels (<1%)" value={fmtInt(s.rare_levels)} tone={s.rare_levels > 20 ? "warn" : undefined} />
            <Stat label="Entropy (norm.)" value={fmtNum(s.normalized_entropy, 2)} />
          </>
        )}
        {f.inferred_type === "text" && (
          <>
            <Stat label="Avg chars" value={fmtNum(f.avg_length, 1)} />
            <Stat label="Avg words" value={fmtNum(s.words?.mean, 1)} />
            <Stat label="Max chars" value={fmtInt(s.length?.max)} />
            <Stat label="Empty strings" value={fmtPct(s.empty_pct)} />
            <Stat label="Vocabulary (sample)" value={fmtInt(s.vocabulary_sampled)} />
          </>
        )}
        {f.inferred_type === "datetime" && (
          <>
            <Stat label="Earliest" value={s.min?.slice(0, 10) ?? "—"} />
            <Stat label="Latest" value={s.max?.slice(0, 10) ?? "—"} />
            <Stat label="Span (days)" value={fmtInt(Math.round(s.span_days ?? 0))} />
            <Stat label="Has time" value={s.has_time_component ? "yes" : "no"} />
          </>
        )}
        {f.max_target_assoc !== null && !f.is_target && (
          <Stat label="Max target assoc." value={fmtNum(f.max_target_assoc, 2)} tone={f.max_target_assoc > 0.95 ? "bad" : undefined} />
        )}
      </div>

      {data.target_associations.length > 0 && !f.is_target && (
        <div className="row" style={{ gap: 12 }}>
          {data.target_associations.map((a) => (
            <span key={a.field_id} className="secondary" style={{ fontSize: 13 }}>
              ◎ <b>{a.name}</b>: {METHOD_LABEL[a.method]} <span className="num">{a.value.toFixed(3)}</span>
            </span>
          ))}
        </div>
      )}

      <section className="card">
        <div className="card-head">
          <h3>Distribution</h3>
          <div className="row">
            {dist && "other" in dist && (dist.other ?? 0) > 0 && <span className="muted" style={{ fontSize: 12 }}>top {dist.labels.length} shown · {fmtInt(dist.other)} rows in other levels</span>}
            {dist && dist.kind !== "bars" && ((dist.below ?? 0) + (dist.above ?? 0)) > 0 && !useLog && (
              <span className="muted" style={{ fontSize: 12 }}>p0.5–p99.5 range · {fmtInt((dist.below ?? 0) + (dist.above ?? 0))} sampled rows outside</span>
            )}
            {f.charts.log_distribution && (
              <div className="segmented" role="group" aria-label="Scale">
                <button className={`small${!logScale ? " on" : ""}`} onClick={() => setLogScale(false)}>linear</button>
                <button className={`small${logScale ? " on" : ""}`} onClick={() => setLogScale(true)}>log1p</button>
              </div>
            )}
          </div>
        </div>
        {distOption ? (
          <Chart option={distOption} ariaLabel={`Distribution of ${f.name}`}
            height={dist?.kind === "bars" ? Math.max(120, Math.min(20, (dist as any).labels.length) * 24 + 16) : undefined} />
        ) : (
          <div className="empty">No distribution for {f.inferred_type} fields.</div>
        )}
        {isNum && f.charts.box && <QuantileStrip box={f.charts.box} />}
      </section>

      {(tokensOption || weekdayOption) && (
        <section className="card">
          <div className="card-head"><h3>{tokensOption ? "Top tokens (sample)" : "Day of week"}</h3></div>
          <Chart option={(tokensOption ?? weekdayOption)!} ariaLabel={tokensOption ? "Top tokens" : "Day of week"}
            height={Math.max(120, (f.charts.tokens?.labels.length ?? 7) * 22 + 16)} />
        </section>
      )}

      <section className="card">
        <div className="card-head">
          <h3>Top {corrRows.length} associated fields</h3>
          <label className="check" style={{ fontSize: 12.5 }}>
            <input type="checkbox" checked={hideTargets} onChange={(e) => setHideTargets(e.target.checked)} /> hide targets
          </label>
        </div>
        {corrRows.length > 0 ? (
          <>
            <Chart option={corrOption} ariaLabel={`Fields most associated with ${f.name}`}
              height={corrRows.length * 22 + 40} onBarClick={(i) => onOpenField(corrRows[i].field_id)} />
            <p className="muted" style={{ fontSize: 12, margin: "8px 0 0" }}>
              Spearman ρ for numeric pairs (signed), Cramér's V for categorical pairs, correlation ratio η for mixed pairs —
              all on a {fmtInt(corr.data?.[0]?.n_obs)}-row sample. ◎ marks a target. Click a bar to jump to that field.
            </p>
          </>
        ) : (
          <div className="empty">{corr.isLoading ? "Loading…" : `No associations computed for ${f.inferred_type} fields.`}</div>
        )}
      </section>

      {s.examples?.length > 0 && (
        <section className="card">
          <div className="card-head"><h3>Example values</h3></div>
          <div className="row" style={{ gap: 6 }}>
            {s.examples.map((v: string, i: number) => (
              <code key={i} className="badge" style={{ height: "auto", padding: "2px 8px", maxWidth: "100%", overflow: "hidden", textOverflow: "ellipsis" }}>{v}</code>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
