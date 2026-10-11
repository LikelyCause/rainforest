import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Chart } from "./Chart";
import { barsOption, correlationOption, histogramOption } from "../charts";
import { useChartTheme } from "../theme";
import { fmtInt, fmtNum, fmtPct } from "../format";
import { stripLabels } from "../quantiles";
import { fieldSignals } from "../signals";
import type { BoxSummary, FieldResponse } from "../types";
import { ErrorAlert, InfoTip } from "./ui";
import { Icon, Segmented } from "./kit";

const NUMERIC = new Set(["numeric_continuous", "numeric_discrete"]);
const BAR_ROWS = 10;
const ASSOC_ROWS = 10;

function QuantileStrip({ box }: { box: BoxSummary }) {
  const { p01, p25, p50, p75, p99 } = box;
  if ([p01, p25, p50, p75, p99].some((v) => v === null) || p99 === p01) return null;
  const lo = p01!, hi = p99!;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="quantile-strip" role="img" aria-label={`Quantiles: p1 ${fmtNum(p01)}, p25 ${fmtNum(p25)}, median ${fmtNum(p50)}, p75 ${fmtNum(p75)}, p99 ${fmtNum(p99)}`}
      title={`p1 ${fmtNum(p01)} · p25 ${fmtNum(p25)} · median ${fmtNum(p50)} · p75 ${fmtNum(p75)} · p99 ${fmtNum(p99)}`}>
      <div className="track" />
      <div className="whisker" style={{ left: 0, right: 0 }} />
      <div className="box" style={{ left: x(p25!), width: `calc(${x(p75!)} - ${x(p25!)})` }} />
      <div className="median" style={{ left: x(p50!) }} />
      {stripLabels({ p01: lo, p50: p50!, p99: hi }).map((l) => (
        <span key={l.key} className={`tick ${l.row}`} title={l.key}
          style={{ left: `${l.at * 100}%`, transform: l.anchor === "start" ? "none" : l.anchor === "end" ? "translateX(-100%)" : undefined }}>
          {l.text}
        </span>
      ))}
      <span className="middle-half num">middle half {fmtNum(p25)} to {fmtNum(p75)}</span>
    </div>
  );
}

function readOpen(): boolean {
  try { return localStorage.getItem("rf:moreStats") === "1"; } catch { return false; }
}

interface Props {
  data: FieldResponse;
  position: { index: number; total: number } | null;
  stale: boolean;
  onOpenField: (id: number) => void;
  onPrev?: () => void;
  onNext?: () => void;
  /** Opens the field list where it is a drawer (narrow screens). */
  onOpenList?: () => void;
  /** The catalog's label for a semantic type ("Number, continuous"). */
  typeLabel?: (t: string) => string;
}

export function FieldDetailView({ data, position, stale, onOpenField, onPrev, onNext, onOpenList, typeLabel }: Props) {
  const t = useChartTheme();
  const f = data.field;
  const s = f.stats;
  const decisionType = data.decision.semantic_type;
  const [logScale, setLogScale] = useState(false);
  const [moreBars, setMoreBars] = useState(false);
  const [moreAssoc, setMoreAssoc] = useState(false);
  const [statsOpen, setStatsOpen] = useState(readOpen);

  // No placeholder: showing the previous field's associations under this field's name was #57.
  const corr = useQuery({
    queryKey: ["corr", f.id],
    queryFn: () => api.correlations(f.id, 40),
    staleTime: Infinity,
  });
  const corrRows = useMemo(() => {
    const rows = corr.data ?? [];
    const targets = rows.filter((r) => r.is_target);
    const others = rows.filter((r) => !r.is_target).slice(0, moreAssoc ? 20 : ASSOC_ROWS);
    return [...targets, ...others];
  }, [corr.data, moreAssoc]);

  const dist = f.charts.distribution;
  const useLog = logScale && f.charts.log_distribution;
  const barCount = dist?.kind === "bars" ? dist.labels.length : 0;
  const distOption = useMemo(() => {
    if (!dist) return null;
    if (useLog) return histogramOption(t, f.charts.log_distribution!, { xLabel: "log1p(value)" });
    if (dist.kind === "bars") {
      const n = moreBars ? 20 : BAR_ROWS;
      return barsOption(t, { ...dist, labels: dist.labels.slice(0, n), counts: dist.counts.slice(0, n) }, { total: f.row_count });
    }
    return histogramOption(t, dist, { time: dist.kind === "time_histogram", xLabel: dist.xlabel });
  }, [t, dist, useLog, moreBars, f.charts.log_distribution, f.row_count]);
  const corrOption = useMemo(() => correlationOption(t, corrRows), [t, corrRows]);
  const extra = f.charts.tokens ?? f.charts.weekday;
  const extraOption = useMemo(() => (extra ? barsOption(t, { ...extra, labels: extra.labels.slice(0, BAR_ROWS), counts: extra.counts.slice(0, BAR_ROWS) }) : null), [t, extra]);

  const signals = fieldSignals(f, data.target_associations, decisionType);
  const isNum = NUMERIC.has(f.inferred_type);
  const isLabelled = f.inferred_type === "categorical" || f.inferred_type === "binary";
  const distHeight = dist?.kind === "bars" ? Math.max(BAR_ROWS, Math.min(moreBars ? 20 : BAR_ROWS, barCount)) * 22 + 8 : 220;
  const outside = dist && dist.kind !== "bars" ? (dist.below ?? 0) + (dist.above ?? 0) : 0;
  const typeName = typeLabel?.(f.inferred_type) ?? f.inferred_type.replace("_", " ");

  const toggleStats = (open: boolean) => {
    setStatsOpen(open);
    try { localStorage.setItem("rf:moreStats", open ? "1" : "0"); } catch { /* storage unavailable */ }
  };

  return (
    <>
      <header className="field-head">
        <div className="field-title">
          <h1>{f.name}</h1>
          <div className="field-meta num">
            {f.is_target && <span className="target-word"><Icon name="target" size={14} /> Target · </span>}
            {typeName}
            {f.inference_confidence !== null && <>, <InfoTip text={f.inference_reasons.join("\n")} label={`${fmtPct(f.inference_confidence, 0)} sure`}
              ariaLabel={`${fmtPct(f.inference_confidence, 0)} sure of the type: why`} /></>}
            {" · "}<span className="mono">{f.physical_dtype}</span> · {fmtInt(f.row_count)} rows
            {s.stored_as_text && " · numbers stored as text"}
            {position && <> · field {fmtInt(position.index + 1)} of {fmtInt(position.total)}</>}
          </div>
        </div>
        <div className="field-nav">
          {onOpenList && (
            <button type="button" className="small list-button" onClick={onOpenList}>
              Fields{position ? ` · ${position.index + 1} of ${position.total}` : ""}
            </button>
          )}
          {onPrev && <button type="button" className="small" onClick={onPrev} aria-label="Previous field" aria-keyshortcuts="K">Previous <kbd>K</kbd></button>}
          {onNext && <button type="button" className="small" onClick={onNext} aria-label="Next field" aria-keyshortcuts="J">Next <kbd>J</kbd></button>}
        </div>
      </header>

      <dl className="stat-strip num" aria-label="Signals">
        {signals.map((x) => (
          <div key={x.key} className={`stat-cell${x.tone ? ` ${x.tone}` : ""}`} title={x.rule}>
            <dt>{x.key === "target" && x.label === "Target" ? "Target association" : x.label}</dt>
            <dd>{x.short ?? x.value}{x.detail && <span className="stat-detail">{x.detail}</span>}</dd>
          </div>
        ))}
        {isNum && f.mean !== null && (
          <div className="stat-cell"><dt>Mean ± std</dt><dd>{fmtNum(f.mean)} <span className="stat-soft">± {fmtNum(f.std)}</span></dd></div>
        )}
        {isNum && f.min_value !== null && (
          <div className="stat-cell"><dt>Range</dt><dd>{fmtNum(f.min_value)} to {fmtNum(f.max_value)}</dd></div>
        )}
        {isLabelled && f.top_value !== null && (
          <div className="stat-cell" title={f.top_value ?? ""}><dt>Top value</dt><dd><span className="stat-text">{f.top_value}</span> <span className="stat-soft">{fmtPct(f.top_value_pct, 0)}</span></dd></div>
        )}
      </dl>

      <div className={`chart-cards${corrRows.length || corr.isLoading ? " pair" : ""}`}>
        <section className={`card chart-card${stale ? " dim" : ""}`}>
          <div className="card-head">
            <h2>Distribution</h2>
            <div className="row" style={{ gap: 6 }}>
              {dist?.kind === "bars" && barCount > BAR_ROWS && (
                <button type="button" className="link small-text" onClick={() => setMoreBars(!moreBars)}>
                  top {moreBars ? Math.min(20, barCount) : BAR_ROWS} of {fmtInt(f.n_unique)} · show {moreBars ? "fewer" : "20"}
                </button>
              )}
              {outside > 0 && !useLog && <span className="muted small-text num" title="Histogram covers p0.5–p99.5">{fmtInt(outside)} rows outside p0.5–p99.5</span>}
              {f.charts.log_distribution && (
                <Segmented size="small" pressed label="Scale" value={logScale ? "log" : "linear"}
                  options={[{ value: "linear", label: "Linear" }, { value: "log", label: "log1p" }]}
                  onChange={(v) => setLogScale(v === "log")} />
              )}
            </div>
          </div>
          <div className="chart-body">
            {distOption
              ? <Chart key={f.id} option={distOption} ariaLabel={`Distribution of ${f.name}`} height={distHeight} />
              : <div className="empty">No distribution for {f.inferred_type} fields.</div>}
            {isNum && f.charts.box && !useLog && <QuantileStrip box={f.charts.box} />}
          </div>
        </section>

        {(corrRows.length > 0 || corr.isLoading || corr.error) && (
          <section className={`card chart-card${stale || corr.isLoading ? " dim" : ""}`}>
            <div className="card-head">
              <h2 className="row" style={{ gap: 6 }}>
                Strongest associations
                <InfoTip label={<Icon name="info" size={14} />} ariaLabel="About these associations"
                  text={`Spearman ρ for numeric pairs (signed), Cramér's V for categorical pairs, correlation ratio η for mixed pairs, on a ${fmtInt(corr.data?.[0]?.n_obs)}-row sample. ◎ marks a target. Click a bar to open that field.`} />
              </h2>
              <button type="button" className="link small-text" onClick={() => setMoreAssoc(!moreAssoc)}>
                {moreAssoc ? "Show 10" : "Show 20"}
              </button>
            </div>
            <div className="chart-body">
              {corr.error ? <ErrorAlert error={corr.error} /> : (
                <Chart key={f.id} option={corrOption} ariaLabel={`Fields most associated with ${f.name}`}
                  height={Math.max(220, corrRows.length * 20 + 32)} onBarClick={(i) => onOpenField(corrRows[i].field_id)} />
              )}
              {!corr.error && corrRows.some((r) => r.value < 0) && <p className="muted small-text">Bars run from 0; blue goes up together, red goes opposite.</p>}
            </div>
          </section>
        )}
      </div>

      {extraOption && (
        <section className={`card chart-card${stale ? " dim" : ""}`}>
          <div className="card-head"><h2>{f.charts.tokens ? "Top tokens (sample)" : "Day of week"}</h2></div>
          <Chart key={f.id} option={extraOption} ariaLabel={f.charts.tokens ? "Top tokens" : "Day of week"}
            height={Math.min(BAR_ROWS, extra!.labels.length) * 22 + 8} />
        </section>
      )}

      <details className="more" open={statsOpen} onToggle={(e) => toggleStats((e.target as HTMLDetailsElement).open)}>
        <summary>
          <b>More stats</b>
          <span className="muted num small-text">
            {isNum ? `mean ${fmtNum(f.mean)} · std ${fmtNum(f.std)} · min ${fmtNum(f.min_value)} · max ${fmtNum(f.max_value)}`
              : `${fmtInt(f.n_unique)} distinct · top ${f.top_value ?? "—"}`}
          </span>
        </summary>
        <div className="kv">
          <div><span>Distinct</span><b>{fmtInt(f.n_unique)} ({fmtPct(f.unique_pct)})</b></div>
          <div><span>Missing</span><b>{fmtInt(f.null_count)}</b></div>
          {isNum && <>
            <div><span>Mean</span><b>{fmtNum(f.mean)}</b></div>
            <div><span>Std dev</span><b>{fmtNum(f.std)}</b></div>
            <div><span>Min</span><b>{fmtNum(f.min_value)}</b></div>
            <div><span>Median</span><b>{fmtNum(f.median)}</b></div>
            <div><span>Max</span><b>{fmtNum(f.max_value)}</b></div>
            <div><span>Kurtosis</span><b>{fmtNum(f.kurtosis, 2)}</b></div>
            <div><span>Zeros</span><b>{fmtPct(f.zeros_pct)}</b></div>
            <div><span>Negative</span><b>{fmtPct(s.negative_pct)}</b></div>
          </>}
          {(f.inferred_type === "categorical" || f.inferred_type === "binary") && <>
            <div><span>Top value</span><b title={f.top_value ?? ""}>{f.top_value ?? "—"}</b></div>
            <div><span>Top value share</span><b>{fmtPct(f.top_value_pct)}</b></div>
            <div><span>Rare levels (&lt;1%)</span><b>{fmtInt(s.rare_levels)}</b></div>
            <div><span>Entropy (norm.)</span><b>{fmtNum(s.normalized_entropy, 2)}</b></div>
          </>}
          {f.inferred_type === "text" && <>
            <div><span>Avg chars</span><b>{fmtNum(f.avg_length, 1)}</b></div>
            <div><span>Avg words</span><b>{fmtNum(s.words?.mean, 1)}</b></div>
            <div><span>Max chars</span><b>{fmtInt(s.length?.max)}</b></div>
            <div><span>Empty strings</span><b>{fmtPct(s.empty_pct)}</b></div>
            <div><span>Vocabulary (sample)</span><b>{fmtInt(s.vocabulary_sampled)}</b></div>
          </>}
          {f.inferred_type === "datetime" && <>
            <div><span>Earliest</span><b>{s.min?.slice(0, 10) ?? "—"}</b></div>
            <div><span>Latest</span><b>{s.max?.slice(0, 10) ?? "—"}</b></div>
            <div><span>Span (days)</span><b>{fmtInt(Math.round(s.span_days ?? 0))}</b></div>
            <div><span>Has time</span><b>{s.has_time_component ? "yes" : "no"}</b></div>
          </>}
        </div>
        {s.examples?.length > 0 && (
          <div className="examples">
            {s.examples.map((v: string, i: number) => <code key={i}>{v}</code>)}
          </div>
        )}
      </details>
    </>
  );
}
