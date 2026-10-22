import { useMemo, useState, useSyncExternalStore, type ReactNode } from "react";
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
  const lo = p01!,
    hi = p99!;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div
      className="quantile-strip"
      role="img"
      aria-label={`Quantiles: p1 ${fmtNum(p01)}, p25 ${fmtNum(p25)}, median ${fmtNum(p50)}, p75 ${fmtNum(p75)}, p99 ${fmtNum(p99)}`}
      title={`p1 ${fmtNum(p01)} · p25 ${fmtNum(p25)} · median ${fmtNum(p50)} · p75 ${fmtNum(p75)} · p99 ${fmtNum(p99)}`}
    >
      <div className="track" />
      <div className="whisker" />
      <div className="box" style={{ left: x(p25!), width: `calc(${x(p75!)} - ${x(p25!)})` }} />
      <div className="median" style={{ left: x(p50!) }} />
      {stripLabels({ p01: lo, p50: p50!, p99: hi }).map((l) => (
        <span
          key={l.key}
          className={`tick ${l.row}`}
          title={l.key}
          style={{
            left: `${l.at * 100}%`,
            transform: l.anchor === "start" ? "none" : l.anchor === "end" ? "translateX(-100%)" : undefined,
          }}
        >
          {l.text}
        </span>
      ))}
      <span className="middle-half num">
        middle half {fmtNum(p25)} to {fmtNum(p75)}
      </span>
    </div>
  );
}

function readOpen(): boolean {
  try {
    return localStorage.getItem("rf:moreStats") === "1";
  } catch {
    return false;
  }
}

const NARROW = "(max-width: 759px)";
function useNarrow(): boolean {
  return useSyncExternalStore(
    (l) => {
      const m = window.matchMedia(NARROW);
      m.addEventListener("change", l);
      return () => m.removeEventListener("change", l);
    },
    () => window.matchMedia(NARROW).matches,
  );
}

/** One labelled number in More stats; the less obvious ones explain themselves on hover or tap. */
function Stat({ label, tip, title, children }: { label: string; tip?: string; title?: string; children: ReactNode }) {
  return (
    <div className="stat-row">
      <dt>{tip ? <InfoTip label={label} text={tip} ariaLabel={`${label}: what this means`} /> : label}</dt>
      <dd title={title}>{children}</dd>
    </div>
  );
}

function StatGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="stat-group" aria-label={title}>
      <h3>{title}</h3>
      <dl>{children}</dl>
    </section>
  );
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
      return barsOption(
        t,
        { ...dist, labels: dist.labels.slice(0, n), counts: dist.counts.slice(0, n) },
        { total: f.row_count },
      );
    }
    return histogramOption(t, dist, { time: dist.kind === "time_histogram", xLabel: dist.xlabel });
  }, [t, dist, useLog, moreBars, f.charts.log_distribution, f.row_count]);
  const narrow = useNarrow();
  // The card spans the column, so names get more room than the shared helper's 26 characters; echarts trims to the width.
  const corrOption = useMemo(() => {
    const o = correlationOption(t, corrRows) as { yAxis: { axisLabel: object } };
    const anyNegative = corrRows.some((r) => r.value < 0);
    return {
      ...o,
      yAxis: {
        ...o.yAxis,
        data: corrRows.map((r) => (r.is_target ? `◎ ${r.name}` : r.name)),
        // The margin keeps a long negative bar's value label off the names.
        axisLabel: {
          ...o.yAxis.axisLabel,
          width: narrow ? 130 : 300,
          overflow: "truncate",
          margin: anyNegative ? 40 : 8,
        },
      },
    };
  }, [t, corrRows, narrow]);
  const extra = f.charts.tokens ?? f.charts.weekday;
  const extraOption = useMemo(
    () =>
      extra
        ? barsOption(t, { ...extra, labels: extra.labels.slice(0, BAR_ROWS), counts: extra.counts.slice(0, BAR_ROWS) })
        : null,
    [t, extra],
  );

  const signals = fieldSignals(f, data.target_associations, decisionType);
  const isNum = NUMERIC.has(f.inferred_type);
  const isLabelled = f.inferred_type === "categorical" || f.inferred_type === "binary";
  const distHeight =
    dist?.kind === "bars" ? Math.max(BAR_ROWS, Math.min(moreBars ? 20 : BAR_ROWS, barCount)) * 22 + 8 : 220;
  const outside = dist && dist.kind !== "bars" ? (dist.below ?? 0) + (dist.above ?? 0) : 0;
  const typeName = typeLabel?.(f.inferred_type) ?? f.inferred_type.replace("_", " ");
  const isText = f.inferred_type === "text";
  // Free text runs long: fewer examples, a line each.
  const examples: string[] = (s.examples ?? []).slice(0, isText ? 4 : 8);
  const topLevels: { value: string; count: number; pct: number }[] = isLabelled ? (s.top_values ?? []).slice(0, 6) : [];
  const missing = (
    <>
      {fmtInt(f.null_count)} <span className="stat-soft">({fmtPct(f.null_pct)})</span>
    </>
  );

  const toggleStats = (open: boolean) => {
    setStatsOpen(open);
    try {
      localStorage.setItem("rf:moreStats", open ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  };

  return (
    <>
      <header className="field-head">
        <div className="field-title">
          <h1>{f.name}</h1>
          <div className="field-meta num">
            {f.is_target && (
              <>
                <span className="target-word">
                  <Icon name="target" size={14} /> Target
                </span>
                {" · "}
              </>
            )}
            {typeName}
            {f.inference_confidence !== null && (
              <>
                ,{" "}
                <InfoTip
                  text={f.inference_reasons.join("\n")}
                  label={`${fmtPct(f.inference_confidence, 0)} sure`}
                  ariaLabel={`${fmtPct(f.inference_confidence, 0)} sure of the type: why`}
                />
              </>
            )}
            {" · "}
            <span className="mono">{f.physical_dtype}</span> · {fmtInt(f.row_count)} rows
            {s.stored_as_text && " · numbers stored as text"}
            {position && (
              <>
                {" "}
                · field {fmtInt(position.index + 1)} of {fmtInt(position.total)}
              </>
            )}
          </div>
        </div>
        <div className="field-nav">
          {onOpenList && (
            <button type="button" className="small list-button" onClick={onOpenList}>
              Fields{position ? ` · ${position.index + 1} of ${position.total}` : ""}
            </button>
          )}
          {onPrev && (
            <button type="button" className="small" onClick={onPrev} aria-label="Previous field" aria-keyshortcuts="K">
              Previous <kbd>K</kbd>
            </button>
          )}
          {onNext && (
            <button type="button" className="small" onClick={onNext} aria-label="Next field" aria-keyshortcuts="J">
              Next <kbd>J</kbd>
            </button>
          )}
        </div>
      </header>

      <dl className="stat-strip num" aria-label="Signals">
        {signals.map((x) => (
          <div key={x.key} className={`stat-cell${x.tone ? ` ${x.tone}` : ""}`} title={x.rule}>
            <dt>{x.key === "target" && x.label === "Target" ? "Target association" : x.label}</dt>
            <dd>
              {x.short ?? x.value}
              {x.detail && <span className="stat-detail">{x.detail}</span>}
            </dd>
          </div>
        ))}
        {isNum && f.mean !== null && (
          <div className="stat-cell">
            <dt>Mean ± std</dt>
            <dd>
              {fmtNum(f.mean)} <span className="stat-soft">± {fmtNum(f.std)}</span>
            </dd>
          </div>
        )}
        {isNum && f.min_value !== null && (
          <div className="stat-cell">
            <dt>Range</dt>
            <dd>
              {fmtNum(f.min_value)} to {fmtNum(f.max_value)}
            </dd>
          </div>
        )}
        {isLabelled && f.top_value !== null && (
          <div className="stat-cell" title={f.top_value ?? ""}>
            <dt>Top value</dt>
            <dd>
              <span className="stat-text">{f.top_value}</span>{" "}
              <span className="stat-soft">{fmtPct(f.top_value_pct, 0)}</span>
            </dd>
          </div>
        )}
      </dl>

      {(examples.length > 0 || topLevels.length > 0) && (
        <section className={`card sample-values${stale ? " dim" : ""}`} aria-labelledby="sample-values-title">
          <div className="sample-head">
            <h2 id="sample-values-title">Sample values</h2>
            <span className="muted small-text">The first few distinct values, as stored in the file</span>
          </div>
          {examples.length > 0 && (
            <ul className={`sample-chips${isText ? " long" : ""}`} aria-label="Example values">
              {examples.map((v, i) => (
                <li key={i}>
                  <code title={v}>{v === "" ? "(empty)" : v}</code>
                </li>
              ))}
            </ul>
          )}
          {topLevels.length > 0 && (
            <div className="sample-levels">
              <span className="small-text muted">
                Most common{f.null_count > 0 ? ", as a share of rows with a value" : ""}
              </span>
              <ul className="sample-chips" aria-label="Most common values with their share of rows">
                {topLevels.map((l) => (
                  <li key={l.value}>
                    <code title={`${l.value}: ${fmtInt(l.count)} rows`}>{l.value}</code>{" "}
                    <span className="num small-text secondary">{fmtPct(l.pct, 0)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      <div className="chart-cards stacked">
        <section className={`card chart-card${stale ? " dim" : ""}`}>
          <div className="card-head">
            <h2>Distribution</h2>
            <div className="row card-tools">
              {dist?.kind === "bars" && barCount > BAR_ROWS && (
                <button type="button" className="link small-text" onClick={() => setMoreBars(!moreBars)}>
                  top {moreBars ? Math.min(20, barCount) : BAR_ROWS} of {fmtInt(f.n_unique)} · show{" "}
                  {moreBars ? "fewer" : "20"}
                </button>
              )}
              {outside > 0 && !useLog && (
                <span className="muted small-text num" title="Histogram covers p0.5–p99.5">
                  {fmtInt(outside)} rows outside p0.5–p99.5
                </span>
              )}
              {f.charts.log_distribution && (
                <Segmented
                  size="small"
                  pressed
                  label="Scale"
                  value={logScale ? "log" : "linear"}
                  options={[
                    { value: "linear", label: "Linear" },
                    { value: "log", label: "log1p" },
                  ]}
                  onChange={(v) => setLogScale(v === "log")}
                />
              )}
            </div>
          </div>
          <div className="chart-body">
            {distOption ? (
              <Chart key={f.id} option={distOption} ariaLabel={`Distribution of ${f.name}`} height={distHeight} />
            ) : (
              <div className="empty">No distribution for {f.inferred_type} fields.</div>
            )}
            {isNum && f.charts.box && !useLog && <QuantileStrip box={f.charts.box} />}
          </div>
        </section>

        {(corrRows.length > 0 || corr.isLoading || corr.error) && (
          <section className={`card chart-card${stale || corr.isLoading ? " dim" : ""}`}>
            <div className="card-head">
              <h2 className="row card-tools">
                Strongest associations
                <InfoTip
                  label={<Icon name="info" size={14} />}
                  ariaLabel="About these associations"
                  text={`Spearman ρ for numeric pairs (signed), Cramér's V for categorical pairs, correlation ratio η for mixed pairs, on a ${fmtInt(corr.data?.[0]?.n_obs)}-row sample. ◎ marks a target. Click a bar to open that field.`}
                />
              </h2>
              <button type="button" className="link small-text" onClick={() => setMoreAssoc(!moreAssoc)}>
                {moreAssoc ? "Show 10" : "Show 20"}
              </button>
            </div>
            <div className="chart-body">
              {corr.error ? (
                <ErrorAlert error={corr.error} />
              ) : (
                <Chart
                  key={f.id}
                  option={corrOption}
                  ariaLabel={`Fields most associated with ${f.name}`}
                  height={Math.max(120, corrRows.length * 20 + 40)}
                  onBarClick={(i) => onOpenField(corrRows[i].field_id)}
                />
              )}
              {!corr.error && corrRows.some((r) => r.value < 0) && (
                <p className="muted small-text">Bars run from 0; blue goes up together, red goes opposite.</p>
              )}
            </div>
          </section>
        )}
      </div>

      {extraOption && (
        <section className={`card chart-card${stale ? " dim" : ""}`}>
          <div className="card-head">
            <h2>{f.charts.tokens ? "Top tokens (sample)" : "Day of week"}</h2>
          </div>
          <Chart
            key={f.id}
            option={extraOption}
            ariaLabel={f.charts.tokens ? "Top tokens" : "Day of week"}
            height={Math.min(BAR_ROWS, extra!.labels.length) * 22 + 8}
          />
        </section>
      )}

      <details className="more" open={statsOpen} onToggle={(e) => toggleStats((e.target as HTMLDetailsElement).open)}>
        <summary>
          <b>More stats</b>
          <span className="muted num small-text">
            {isNum
              ? `min ${fmtNum(f.min_value)} · median ${fmtNum(f.median)} · max ${fmtNum(f.max_value)}`
              : `${fmtInt(f.n_unique)} distinct · ${fmtInt(f.null_count)} missing`}
          </span>
        </summary>
        <div className="stat-groups num">
          {isNum && (
            <>
              <StatGroup title="Spread">
                <Stat label="Min">{fmtNum(f.min_value)}</Stat>
                <Stat label="Median">{fmtNum(f.median)}</Stat>
                <Stat label="Max">{fmtNum(f.max_value)}</Stat>
                <Stat
                  label="Mean ± std"
                  tip="The average, and the standard deviation: how far a typical value sits from the average."
                >
                  {fmtNum(f.mean)} <span className="stat-soft">± {fmtNum(f.std)}</span>
                </Stat>
                {s.quantiles?.p25 != null && (
                  <Stat
                    label="Middle half"
                    tip="Half of the values fall between these two (the 25th and 75th percentiles)."
                  >
                    {fmtNum(s.quantiles.p25)} to {fmtNum(s.quantiles.p75)}
                  </Stat>
                )}
              </StatGroup>
              <StatGroup title="Shape">
                <Stat
                  label="Skew"
                  tip="Lopsidedness. 0 is symmetric; positive means a long tail of large values, negative a long tail of small ones. Beyond ±2 is strongly skewed."
                >
                  {fmtNum(f.skewness, 2)}
                </Stat>
                <Stat
                  label="Kurtosis"
                  tip="How heavy the tails are compared with a bell curve (which scores 0). Large positive values mean more extreme values than a bell curve would have."
                >
                  {fmtNum(f.kurtosis, 2)}
                </Stat>
                <Stat label="Outliers" tip="Share of values more than 1.5 × the middle half's width beyond it.">
                  {fmtPct(f.outlier_pct)}
                </Stat>
                <Stat label="Zeros">{fmtPct(f.zeros_pct)}</Stat>
                <Stat label="Negative">{fmtPct(s.negative_pct)}</Stat>
              </StatGroup>
            </>
          )}
          {isLabelled && (
            <StatGroup title="Levels">
              <Stat label="Top value" title={f.top_value ?? ""}>
                <span className="stat-text">{f.top_value ?? "—"}</span>
              </Stat>
              <Stat label="Top value share">{fmtPct(f.top_value_pct)}</Stat>
              <Stat
                label="Rare levels"
                tip="Values that each appear in fewer than 1% of rows. Many rare levels are hard to learn from; grouping them often helps."
              >
                {fmtInt(s.rare_levels)}
              </Stat>
              <Stat
                label="Evenness"
                tip="Normalized entropy, from 0 to 1. Near 1 the values are spread evenly across the levels; near 0 one value dominates."
              >
                {fmtNum(s.normalized_entropy, 2)}
              </Stat>
            </StatGroup>
          )}
          {isText && (
            <>
              <StatGroup title="Length">
                <Stat label="Avg characters">{fmtNum(f.avg_length, 1)}</Stat>
                <Stat label="Max characters">{fmtInt(s.length?.max)}</Stat>
                <Stat label="Avg words">{fmtNum(s.words?.mean, 1)}</Stat>
              </StatGroup>
              <StatGroup title="Content">
                <Stat label="Empty strings">{fmtPct(s.empty_pct)}</Stat>
                <Stat
                  label="Vocabulary"
                  tip="Distinct words found in a sample of the rows, ignoring case and common words like 'the'."
                >
                  {fmtInt(s.vocabulary_sampled)}
                </Stat>
              </StatGroup>
            </>
          )}
          {f.inferred_type === "datetime" && (
            <StatGroup title="Range">
              <Stat label="Earliest">{s.min?.slice(0, 10) ?? "—"}</Stat>
              <Stat label="Latest">{s.max?.slice(0, 10) ?? "—"}</Stat>
              <Stat label="Span">{fmtInt(Math.round(s.span_days ?? 0))} days</Stat>
              <Stat label="Has time of day">{s.has_time_component ? "Yes" : "No"}</Stat>
            </StatGroup>
          )}
          <StatGroup title="Counts">
            <Stat
              label="Distinct"
              tip="How many different values the field holds, and that as a share of the rows that aren't missing."
            >
              {fmtInt(f.n_unique)} <span className="stat-soft">({fmtPct(f.unique_pct)})</span>
            </Stat>
            <Stat label="Missing">{missing}</Stat>
            <Stat label="Rows">{fmtInt(f.row_count)}</Stat>
          </StatGroup>
        </div>
      </details>
    </>
  );
}
