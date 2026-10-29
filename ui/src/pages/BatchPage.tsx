import { Fragment, useMemo, useState, type ReactNode } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import {
  batchProgress,
  brierPair,
  describeSampler,
  describeSpace,
  KIND_LABEL,
  LOWER_IS_BETTER,
  METRIC_NAME,
  oneVsRestValues,
  searchChartData,
  sortOvrRows,
  variedSettings,
  weakReasons,
  type OvrSortKey,
} from "../batches";
import { searchChartOption, thresholdCurveOption } from "../charts";
import { BatchDialog, Skips } from "../components/BatchDialog";
import { Chart } from "../components/Chart";
import { useDatasetContext } from "../components/DatasetLayout";
import { Chip, confirmDialog, Icon, ProgressBar, StatusBadge } from "../components/kit";
import { ErrorAlert } from "../components/ui";
import { addBlocked } from "../experiments";
import { fmtAgo, fmtDate, fmtInt, fmtMetric, fmtNum } from "../format";
import { useMe } from "../hooks";
import { isActive, whyFailed } from "../runs";
import { describeSettings, settingText } from "../settings";
import { useChartTheme } from "../theme";
import { showError, toast } from "../toast";
import type { BatchDetail, BatchRow, BatchStarted, ExportJob, LeaderboardEntry, ModelSetting } from "../types";

type Dir = "asc" | "desc";
const opens = (status: string) => status === "done" || status === "failed";

/** The batch's runs that a selection may hold: finished ones. The deploy dialog takes the selection from here. */
function SelectionBar({ selected, onClear }: { selected: number[]; onClear: () => void }) {
  if (!selected.length) return null;
  return (
    <div className="bulk-bar batch-selection" role="region" aria-label="Selected runs">
      <span className="grow">
        <b className="num">{fmtInt(selected.length)}</b> run{selected.length === 1 ? "" : "s"} selected
      </span>
      <button type="button" className="small ghost" onClick={onClear}>
        Clear
      </button>
    </div>
  );
}

function SortHeader({
  k,
  sort,
  onSort,
  children,
  title,
  num = true,
}: {
  k: OvrSortKey;
  sort: { key: OvrSortKey | null; dir: Dir };
  onSort: (k: OvrSortKey) => void;
  children: ReactNode;
  title?: string;
  num?: boolean;
}) {
  const on = sort.key === k;
  return (
    <th
      scope="col"
      className={`sortable${num ? " num" : ""}`}
      aria-sort={on ? (sort.dir === "asc" ? "ascending" : "descending") : undefined}
      title={title}
    >
      <button type="button" className="batch-sort" onClick={() => onSort(k)}>
        {children}
        <span className={`batch-sort-mark${on ? "" : " off"}`} aria-hidden="true">
          {on && sort.dir === "asc" ? "▲" : "▼"}
        </span>
      </button>
    </th>
  );
}

/** A one-vs-rest run's threshold curve and calibration, read from its full report when the row is opened. */
function CurvePanel({ row }: { row: BatchRow }) {
  const t = useChartTheme();
  const run = useQuery({ queryKey: ["run", row.run_id], queryFn: () => api.run(row.run_id), staleTime: 60_000 });
  const curve = run.data?.report?.threshold_curve;
  const option = useMemo(
    () => (curve?.valid ? thresholdCurveOption(t, curve, row.threshold ?? null) : null),
    [t, curve, row.threshold],
  );
  const cal = row.calibration;
  return (
    <div className="batch-curve">
      <div className="batch-curve-chart">
        {run.isLoading && <span className="skeleton exp-skel-line" />}
        <ErrorAlert error={run.error} />
        {option ? (
          <Chart
            option={option}
            height={240}
            ariaLabel={`Precision, recall and F1 of ${row.positive_class} by threshold on validation, with the chosen threshold ${fmtNum(row.threshold)}`}
          />
        ) : (
          run.data && <span className="muted small-text">This run has no threshold curve.</span>
        )}
      </div>
      <dl className="batch-from batch-curve-facts num">
        <dt>Threshold</dt>
        <dd>
          {fmtNum(row.threshold, 2)}, the highest F1 on validation
          {row.threshold_note && <div className="batch-note-text">{row.threshold_note}</div>}
        </dd>
        {cal && (
          <>
            <dt>Calibration</dt>
            <dd>
              {cal.method === "isotonic" ? "Isotonic regression" : "Sigmoid (Platt)"}, fit on validation (
              {fmtInt(cal.valid_positives)} positives, {fmtInt(cal.valid_negatives)} others)
            </dd>
            {cal.test && (
              <>
                <dt>Brier on test</dt>
                <dd>
                  {fmtMetric(cal.test.brier_before)} → {fmtMetric(cal.test.brier_after)}
                </dd>
              </>
            )}
            <dt>Brier on validation</dt>
            <dd>
              {fmtMetric(cal.brier_before)} → {fmtMetric(cal.brier_after)}{" "}
              <span className="muted">(in-sample: the calibrator was fit there)</span>
            </dd>
          </>
        )}
      </dl>
    </div>
  );
}

function OvrTable({
  b,
  runHref,
  selected,
  onSelect,
}: {
  b: BatchDetail;
  runHref: (runId: number) => string;
  selected: number[];
  onSelect: (ids: number[]) => void;
}) {
  const nav = useNavigate();
  const [sort, setSort] = useState<{ key: OvrSortKey | null; dir: Dir }>({ key: null, dir: "desc" });
  const [open, setOpen] = useState<number | null>(null);
  const rows = sortOvrRows(b.rows, sort.key, sort.dir);
  const onSort = (k: OvrSortKey) =>
    setSort((s) =>
      s.key === k ? { key: k, dir: s.dir === "asc" ? "desc" : "asc" } : { key: k, dir: k === "class" ? "asc" : "desc" },
    );
  const done = b.rows.filter((r) => r.status === "done").map((r) => r.run_id);
  const allOn = done.length > 0 && done.every((id) => selected.includes(id));
  const toggle = (id: number, on: boolean) => onSelect(on ? [...selected, id] : selected.filter((x) => x !== id));
  const head = { sort, onSort };
  const COLS = 16;
  const rule = b.weak_rule;
  return (
    <section className="card table-wrap batches-card batch-results" aria-labelledby="ovr-title">
      <div className="batches-head row exp-between">
        <h2 id="ovr-title">
          One model per class <span className="muted num">{fmtInt(b.rows.length)}</span>
        </h2>
        <span className="muted small-text">Select a row to open its run · sort on any column</span>
      </div>
      <div className="exp-scroll-x">
        <table className="data batch-ovr">
          <thead>
            <tr className="batch-group-row">
              <th scope="colgroup" colSpan={2} />
              <th scope="colgroup" colSpan={3} className="batch-group">
                Positives
              </th>
              <th scope="colgroup" colSpan={3} className="batch-group">
                Validation
              </th>
              <th scope="colgroup" colSpan={3} className="batch-group">
                Test
              </th>
              <th scope="colgroup" colSpan={5} />
            </tr>
            <tr>
              <th scope="col" className="batch-check">
                <input
                  type="checkbox"
                  aria-label="Select every finished run"
                  checked={allOn}
                  disabled={!done.length}
                  onChange={(ev) => onSelect(ev.target.checked ? done : [])}
                />
              </th>
              <SortHeader k="class" num={false} {...head}>
                Class
              </SortHeader>
              <SortHeader k="positives.train" {...head} title="Rows of the class in the training split">
                Train
              </SortHeader>
              <SortHeader k="positives.valid" {...head}>
                Valid
              </SortHeader>
              <SortHeader k="positives.test" {...head}>
                Test
              </SortHeader>
              {(["valid", "test"] as const).map((s) => (
                <Fragment key={s}>
                  <SortHeader k={`${s}.pr_auc`} {...head}>
                    PR AUC
                  </SortHeader>
                  <SortHeader k={`${s}.roc_auc`} {...head}>
                    ROC AUC
                  </SortHeader>
                  <SortHeader k={`${s}.f1`} {...head} title="F1 at the run's threshold">
                    F1
                  </SortHeader>
                </Fragment>
              ))}
              <SortHeader k="threshold" {...head} title="The threshold with the highest F1 on validation">
                Threshold
              </SortHeader>
              <th scope="col" title="How the scores were calibrated, and the Brier score before → after on test">
                Calibration
              </th>
              <SortHeader k="weak" num={false} {...head}>
                Weak
              </SortHeader>
              <th scope="col">
                <span className="exp-sr">Status</span>
              </th>
              <th scope="col">
                <span className="exp-sr">Curve</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const brier = brierPair(r.calibration);
              const reasons = weakReasons(r, rule);
              const metric = (v: number | null | undefined) => <td className="num">{fmtMetric(v)}</td>;
              const expanded = open === r.run_id;
              return (
                <Fragment key={r.run_id}>
                  <tr
                    className={`run-row${opens(r.status) ? " clickable" : " inert"}${r.weak ? " weak" : ""}${selected.includes(r.run_id) ? " selected" : ""}`}
                    onClick={() => opens(r.status) && nav(runHref(r.run_id))}
                  >
                    <td className="batch-check" onClick={(ev) => ev.stopPropagation()}>
                      <input
                        type="checkbox"
                        aria-label={`Select ${r.positive_class}`}
                        disabled={r.status !== "done"}
                        checked={selected.includes(r.run_id)}
                        onChange={(ev) => toggle(r.run_id, ev.target.checked)}
                      />
                    </td>
                    <td className="batch-class-cell">
                      {opens(r.status) ? (
                        <Link
                          to={runHref(r.run_id)}
                          className="batch-class-name"
                          onClick={(ev) => ev.stopPropagation()}
                          title={r.label ?? undefined}
                        >
                          {r.positive_class}
                        </Link>
                      ) : (
                        <span className="batch-class-name secondary">{r.positive_class}</span>
                      )}
                      {r.retried && <span className="muted small-text"> · retried</span>}
                      {r.threshold_note && (
                        <div className="batch-note-text small-text">
                          <Icon name="alert" size={13} /> {r.threshold_note}
                        </div>
                      )}
                    </td>
                    <td className="num secondary">{fmtInt(r.positives?.train)}</td>
                    <td className="num secondary">{fmtInt(r.positives?.valid)}</td>
                    <td className="num secondary">{fmtInt(r.positives?.test)}</td>
                    {r.status === "done" ? (
                      <>
                        {metric(r.valid?.pr_auc)}
                        {metric(r.valid?.roc_auc)}
                        {metric(r.valid?.f1)}
                        {metric(r.test?.pr_auc)}
                        {metric(r.test?.roc_auc)}
                        {metric(r.test?.f1)}
                        <td className="num">{fmtNum(r.threshold, 2)}</td>
                        <td
                          className="small-text secondary batch-nowrap"
                          title={
                            brier
                              ? `Brier score on ${brier.split === "test" ? "test" : "validation, in-sample"}: ${fmtMetric(brier.before)} before calibration, ${fmtMetric(brier.after)} after`
                              : undefined
                          }
                        >
                          {r.calibration ? (r.calibration.method === "isotonic" ? "Isotonic" : "Sigmoid") : "—"}
                          {brier && (
                            <span className="num">
                              {" "}
                              · {fmtMetric(brier.before)} → {fmtMetric(brier.after)}
                              {brier.split === "valid" ? "*" : ""}
                            </span>
                          )}
                        </td>
                        <td>
                          {r.weak ? (
                            <span title={reasons.join("\n")}>
                              <Chip tone="warn">Weak</Chip>
                            </span>
                          ) : (
                            <span className="muted small-text">—</span>
                          )}
                        </td>
                        <td />
                      </>
                    ) : (
                      <>
                        <td colSpan={COLS - 7}>
                          <span className="run-progress">
                            {isActive(r.status) && (
                              <span className="meter indeterminate" aria-hidden="true">
                                <span />
                              </span>
                            )}
                            <span className="secondary small-text">
                              {r.status === "failed"
                                ? whyFailed(r.error)
                                : (r.status_message ?? (r.status === "pending" ? "Waiting in its batch" : "Training"))}
                            </span>
                          </span>
                        </td>
                        <td>
                          <StatusBadge status={r.status} />
                        </td>
                      </>
                    )}
                    <td onClick={(ev) => ev.stopPropagation()}>
                      {r.status === "done" && (
                        <button
                          type="button"
                          className="small ghost batch-nowrap"
                          aria-expanded={expanded}
                          aria-controls={`curve-${r.run_id}`}
                          onClick={() => setOpen(expanded ? null : r.run_id)}
                        >
                          Curve
                          <Icon name={expanded ? "chevron-down" : "chevron-right"} size={14} />
                        </button>
                      )}
                    </td>
                  </tr>
                  {expanded && (
                    <tr id={`curve-${r.run_id}`} className="batch-curve-row">
                      <td colSpan={COLS}>
                        <CurvePanel row={r} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="batch-foot muted small-text">
        <p>
          Each model is calibrated on the validation split and its threshold is chosen there too, so the validation F1
          is mildly optimistic; the test columns give the honest number.
          {b.rows.some((r) => r.calibration && !r.calibration.test) &&
            " * A Brier pair on validation is in-sample: this export has no test rows."}
        </p>
        <p>
          <b>Weak:</b> validation PR AUC below {rule.pr_auc} (OVR_WEAK_PR_AUC) or fewer than {rule.test_positives}{" "}
          positives in test.
        </p>
      </div>
    </section>
  );
}

function SearchChart({
  b,
  catalog,
  metric,
}: {
  b: BatchDetail;
  catalog: ModelSetting[] | null;
  metric: { label: string; lowerIsBetter: boolean };
}) {
  const t = useChartTheme();
  const entries = useMemo(() => b.leaderboard ?? [], [b.leaderboard]);
  const space = useMemo(() => b.spec.space ?? {}, [b.spec.space]);
  const varied = variedSettings(entries, Object.keys(space));
  const [x, setX] = useState<string | null>(null);
  const [y, setY] = useState<string | null>(null);
  const ax = x && varied.includes(x) ? x : (varied[0] ?? null);
  const ay = y && y !== ax && varied.includes(y) ? y : null;
  const data = useMemo(
    () => (ax ? searchChartData(entries, ax, ay, space, catalog) : null),
    [entries, ax, ay, space, catalog],
  );
  const option = useMemo(
    () => (data?.points.length ? searchChartOption(t, data, metric, (s) => describeSettings(s, catalog)) : null),
    [t, data, metric, catalog],
  );
  if (!ax) return null;
  const labelOf = (n: string) => catalog?.find((s) => s.name === n)?.label ?? n;
  return (
    <section className="card batch-chart" aria-labelledby="chart-title">
      <div className="row exp-between batch-chart-head">
        <h2 id="chart-title">
          {metric.label} by setting <span className="muted small-text">on validation</span>
        </h2>
        <div className="row exp-gap-2 batch-chart-axes">
          <label className="row exp-gap-1 small-text">
            Across
            <select value={ax} onChange={(ev) => setX(ev.target.value)}>
              {varied.map((n) => (
                <option key={n} value={n}>
                  {labelOf(n)}
                </option>
              ))}
            </select>
          </label>
          {varied.length > 1 && (
            <label className="row exp-gap-1 small-text">
              Against
              <select value={ay ?? ""} onChange={(ev) => setY(ev.target.value || null)}>
                <option value="">{metric.label}</option>
                {varied
                  .filter((n) => n !== ax)
                  .map((n) => (
                    <option key={n} value={n}>
                      {labelOf(n)}
                    </option>
                  ))}
              </select>
            </label>
          )}
        </div>
      </div>
      {option ? (
        <Chart
          option={option}
          height={320}
          ariaLabel={
            ay
              ? `${metric.label} of each combination by ${labelOf(ax)} and ${labelOf(ay)}`
              : `${metric.label} of each combination by ${labelOf(ax)}`
          }
        />
      ) : (
        <span className="muted small-text">The chart fills in as combinations finish.</span>
      )}
      <span className="muted small-text">
        {ay
          ? `Each combination's colour is its ${metric.label}, darker for better.`
          : "The best combination is ringed. Hover a point for its settings."}
      </span>
    </section>
  );
}

function Leaderboard({
  b,
  catalog,
  metric,
  runHref,
  selected,
  onSelect,
  onUse,
}: {
  b: BatchDetail;
  catalog: ModelSetting[] | null;
  metric: { label: string; lowerIsBetter: boolean };
  runHref: (runId: number) => string;
  selected: number[];
  onSelect: (ids: number[]) => void;
  onUse: ((e: LeaderboardEntry) => void) | null;
}) {
  const nav = useNavigate();
  const entries = b.leaderboard ?? [];
  const names = Object.keys(b.spec.space ?? {});
  const classes = b.spec.search_classes ?? [];
  const byId = new Map(b.rows.map((r) => [r.run_id, r]));
  const setting = (n: string) => catalog?.find((s) => s.name === n);
  const runsOf = (e: LeaderboardEntry) => e.run_ids.map((id) => byId.get(id)).filter((r): r is BatchRow => !!r);
  const finished = (e: LeaderboardEntry) => runsOf(e).every((r) => r.status === "done");
  const isOn = (e: LeaderboardEntry) => e.run_ids.every((id) => selected.includes(id));
  const toggle = (e: LeaderboardEntry, on: boolean) =>
    onSelect(on ? [...new Set([...selected, ...e.run_ids])] : selected.filter((id) => !e.run_ids.includes(id)));
  const top = entries.find((e) => e.rank === 1);
  return (
    <section className="card table-wrap batches-card batch-results" aria-labelledby="board-title">
      <div className="batches-head row exp-between">
        <h2 id="board-title">
          Leaderboard <span className="muted num">{fmtInt(entries.length)}</span>
        </h2>
        <span className="row exp-gap-2">
          <span className="muted small-text">
            Ranked on {metric.label} on validation{metric.lowerIsBetter ? ", lower is better" : ""}
            {classes.length ? `, the mean over ${classes.length} classes` : ""}
          </span>
          {onUse && top && (
            <button type="button" className="small" onClick={() => onUse(top)}>
              Use the top settings for a one-vs-rest batch…
            </button>
          )}
        </span>
      </div>
      <div className="exp-scroll-x">
        <table className="data batch-board">
          <thead>
            <tr>
              <th scope="col" className="batch-check">
                <span className="exp-sr">Select</span>
              </th>
              <th scope="col" className="num">
                Rank
              </th>
              {names.map((n) => (
                <th key={n} scope="col" className="num">
                  {setting(n)?.label ?? n}
                </th>
              ))}
              <th scope="col" className="num">
                {metric.label}
              </th>
              {classes.map((c) => (
                <th key={c} scope="col" className="num" title={`${metric.label} of the ${c} vs rest run`}>
                  {c}
                </th>
              ))}
              <th scope="col" className="num" title="Spread over the cross-validation folds">
                CV ±
              </th>
              <th scope="col" className="num">
                Fit
              </th>
              <th scope="col">
                <span className="exp-sr">Status and actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => {
              const runs = runsOf(e);
              const single = runs.length === 1 ? runs[0] : null;
              const byClass = new Map(runs.map((r) => [r.positive_class, r]));
              const pending = runs.filter((r) => isActive(r.status)).length;
              const failed = runs.find((r) => r.status === "failed");
              const cancelled = runs.some((r) => r.status === "cancelled");
              const clickable = single && opens(single.status);
              return (
                <tr
                  key={e.run_ids.join("-")}
                  className={`run-row${clickable ? " clickable" : ""}${e.rank === 1 ? " batch-top" : ""}${isOn(e) ? " selected" : ""}`}
                  onClick={() => clickable && nav(runHref(single.run_id))}
                >
                  <td className="batch-check" onClick={(ev) => ev.stopPropagation()}>
                    <input
                      type="checkbox"
                      aria-label={`Select rank ${e.rank ?? "unranked"}: ${describeSettings(e.settings, catalog)}`}
                      disabled={!finished(e)}
                      checked={isOn(e)}
                      onChange={(ev) => toggle(e, ev.target.checked)}
                    />
                  </td>
                  <td className="num">
                    {clickable ? (
                      <Link to={runHref(single.run_id)} onClick={(ev) => ev.stopPropagation()}>
                        {e.rank ?? "—"}
                      </Link>
                    ) : (
                      (e.rank ?? "—")
                    )}
                  </td>
                  {names.map((n) => {
                    const v = e.settings[n];
                    const shown = v === undefined ? "—" : settingText(setting(n), v);
                    const exact = v === undefined ? shown : settingText(setting(n), v, true);
                    return (
                      <td key={n} className="num" title={exact !== shown ? exact : undefined}>
                        {shown}
                      </td>
                    );
                  })}
                  <td className="num batch-value">{fmtMetric(e.value)}</td>
                  {classes.map((c) => {
                    const r = byClass.get(c);
                    return (
                      <td key={c} className="num secondary">
                        {r && opens(r.status) ? (
                          <Link to={runHref(r.run_id)} title={r.label ?? undefined}>
                            {fmtMetric(e.classes[c])}
                          </Link>
                        ) : (
                          fmtMetric(e.classes[c])
                        )}
                      </td>
                    );
                  })}
                  <td className="num secondary">{e.cv_std === null ? "—" : `± ${fmtMetric(e.cv_std)}`}</td>
                  <td className="num secondary">{e.fit_seconds ? `${fmtNum(e.fit_seconds, 2)} s` : "—"}</td>
                  <td className="batch-nowrap" onClick={(ev) => ev.stopPropagation()}>
                    {failed ? (
                      <span title={whyFailed(failed.error)}>
                        <StatusBadge status="failed" />
                      </span>
                    ) : pending ? (
                      <StatusBadge status={runs.some((r) => r.status === "running") ? "running" : "pending"}>
                        {runs.some((r) => r.status === "running") ? "Training" : "Waiting"}
                      </StatusBadge>
                    ) : cancelled ? (
                      <StatusBadge status="cancelled" />
                    ) : (
                      onUse &&
                      e.rank !== null && (
                        <button
                          type="button"
                          className="small ghost"
                          title="Use these settings for a one-vs-rest batch"
                          aria-label={`Use the settings of rank ${e.rank} for a one-vs-rest batch`}
                          onClick={() => onUse(e)}
                        >
                          Per class…
                        </button>
                      )
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/** One batch: what it trains, how far it has got (and Cancel), the classes it skipped, and its results: a row per
 * class for one-vs-rest, a leaderboard and chart for a search. */
export function BatchPage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const nav = useNavigate();
  const qc = useQueryClient();
  const params = useParams();
  const xid = Number(params.experimentId);
  const bid = Number(params.batchId);
  const x = useQuery({ queryKey: ["experiment", xid], queryFn: () => api.experiment(xid), enabled: !!xid });
  const batch = useQuery({
    queryKey: ["batch", bid],
    queryFn: () => api.batch(bid),
    enabled: !!bid,
    // A cancelled batch can still have a run training; keep polling until it ends too.
    refetchInterval: (q) => {
      const d = q.state.data as BatchDetail | undefined;
      return d && (d.status === "running" || d.counts.running + d.counts.pending > 0) ? 2000 : false;
    },
  });
  const b = batch.data;
  const exports = useQuery({
    queryKey: ["exports", ds.id, xid],
    queryFn: () => api.exports(ds.id, xid),
    enabled: !!b,
  });
  const models = useQuery({
    queryKey: ["exportModels", b?.export_id],
    queryFn: () => api.modelsForExport(b!.export_id),
    enabled: !!b,
    staleTime: 60_000,
  });
  const [selected, setSelected] = useState<number[]>([]);
  const [using, setUsing] = useState<LeaderboardEntry | null>(null);
  const cancel = useMutation({
    mutationFn: () => api.cancelBatch(bid),
    onSuccess: (d) => {
      qc.setQueryData(["batch", bid], d);
      qc.invalidateQueries({ queryKey: ["batches", xid] });
      qc.invalidateQueries({ queryKey: ["runs", ds.id] });
    },
    onError: (err) => showError("The batch couldn't be cancelled", err),
  });

  const back = `/d/${ds.id}/experiments/${xid}?tab=runs`;
  if (batch.error) {
    const missing = (batch.error as { status?: number }).status === 404;
    return (
      <div className="state-card card">
        {missing ? <h1>Batch #{bid} doesn't exist, or isn't shared with you</h1> : <ErrorAlert error={batch.error} />}
        <div>
          <Link className="btn" to={back}>
            Back to the runs
          </Link>
        </div>
      </div>
    );
  }
  if (!b)
    return (
      <div className="page" aria-busy="true">
        <span className="skeleton exp-skel-line short" />
      </div>
    );
  if (b.experiment_id !== xid)
    return <Navigate to={`/d/${b.dataset_id}/experiments/${b.experiment_id}/batches/${b.id}`} replace />;

  const option = models.data?.find((m) => m.key === b.spec.model);
  const catalog = option?.settings ?? null;
  const e: ExportJob | undefined = exports.data?.find((x) => x.id === b.export_id);
  const blocked = x.data && me ? addBlocked(x.data, me, ds.status === "refreshing") : "";
  const p = batchProgress(b);
  const runHref = (runId: number) => `/d/${ds.id}/experiments/${xid}?tab=runs&export=${b.export_id}&run=${runId}`;
  const ranking = b.spec.ranking ?? "";
  const metric = { label: METRIC_NAME[ranking] ?? ranking, lowerIsBetter: LOWER_IS_BETTER.has(ranking) };
  const kinds: Record<string, string> = e?.summary?.target_kinds ?? {};
  const hasClassTarget = Object.values(kinds).some((k) => k && k !== "regression");
  const canUse = b.kind === "search" && !blocked && hasClassTarget && Boolean(e && option);
  const settingsLine = describeSettings(b.spec.params, catalog);
  const paramsExact = describeSettings(b.spec.params, catalog, true);
  const space = b.spec.space ?? {};

  const onCancel = async () => {
    const waiting = b.counts.pending;
    const ok = await confirmDialog({
      title: "Cancel this batch?",
      body: `${fmtInt(waiting)} run${waiting === 1 ? "" : "s"} that haven't started will be cancelled. A run already training finishes, and the finished runs stay.`,
      confirm: "Cancel the batch",
      cancel: "Keep it running",
      danger: true,
    });
    if (ok) cancel.mutate();
  };
  const onStarted = (started: BatchStarted) => {
    setUsing(null);
    qc.invalidateQueries({ queryKey: ["runs", ds.id] });
    qc.invalidateQueries({ queryKey: ["batches", xid] });
    qc.invalidateQueries({ queryKey: ["experiment", xid] });
    toast({
      message: `Started ${started.batch.name}: ${started.run_ids.length} run${started.run_ids.length === 1 ? "" : "s"} on Export #${b.export_id}`,
      detail: started.skipped.length
        ? `${started.skipped.length} class${started.skipped.length === 1 ? "" : "es"} skipped`
        : undefined,
    });
    nav(`/d/${ds.id}/experiments/${xid}/batches/${started.batch.id}`);
  };

  return (
    <div className="page batch-page">
      <div className="stack exp-gap-1">
        <Link to={back} className="secondary small-text batch-back">
          <Icon name="arrow-left" size={14} />
          Runs
        </Link>
        <div className="row batch-title">
          <h1>{b.name}</h1>
          <Chip>
            {KIND_LABEL[b.kind]} #{b.id}
          </Chip>
          <StatusBadge status={b.status} />
        </div>
        <span className="secondary small-text" title={fmtDate(b.created_at)}>
          Export #{b.export_id} · started by {b.created_by_name ?? "someone"} {fmtAgo(b.created_at)}
          {b.finished_at ? ` · ended ${fmtAgo(b.finished_at)}` : ""}
        </span>
      </div>

      <div className="batch-top-grid">
        <section className="card stack batch-progress-card" aria-labelledby="progress-title">
          <h2 id="progress-title">Progress</h2>
          <div className="batch-runs num">
            <ProgressBar value={p.ended} max={p.total} label={`${b.name}: ${p.text}`} />
            <span role="status">{p.text}</span>
          </div>
          {b.status === "running" && (
            <div className="row exp-gap-2">
              <button
                type="button"
                className="small"
                disabled={Boolean(blocked) || cancel.isPending || b.counts.pending === 0}
                title={
                  blocked ||
                  (b.counts.pending === 0 ? "No run is waiting; the one training finishes on its own" : undefined)
                }
                onClick={onCancel}
              >
                {cancel.isPending ? "Cancelling…" : "Cancel the batch"}
              </button>
              <span className="muted small-text">Runs train one at a time; the waiting ones are cancelled.</span>
            </div>
          )}
          {b.status === "cancelled" && (
            <span className="muted small-text">Cancelled: its runs that hadn't started were cancelled.</span>
          )}
          {b.skipped.length > 0 && <Skips skipped={b.skipped} />}
        </section>

        <section className="card stack" aria-labelledby="spec-title">
          <h2 id="spec-title">What it trains</h2>
          <dl className="batch-from num">
            <dt>Model</dt>
            <dd>
              {option?.label ?? b.spec.model} on {b.spec.device === "cuda" ? "the GPU" : "the CPU"}
              {b.spec.balance ? ", classes balanced" : ""}
            </dd>
            {b.spec.target && (
              <>
                <dt>Target</dt>
                <dd>
                  {b.spec.target}
                  {b.kind === "one_vs_rest"
                    ? ` · ${fmtInt(b.spec.classes?.length ?? 0)} classes, each against the rest`
                    : ` · each combination on ${(b.spec.search_classes ?? []).join(", ")}`}
                </dd>
              </>
            )}
            <dt>Settings</dt>
            <dd title={paramsExact !== settingsLine ? paramsExact : undefined}>{settingsLine}</dd>
            {b.kind === "search" && (
              <>
                <dt>Searched</dt>
                <dd>
                  <ul className="batch-space-list">
                    {describeSpace(space, catalog).map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </dd>
                <dt>Combinations</dt>
                <dd>{describeSampler(b.spec.sampler ?? null, space)}</dd>
                <dt>Ranked on</dt>
                <dd>
                  {metric.label} on validation{b.spec.ranked_target ? ` (${b.spec.ranked_target})` : ""}
                </dd>
              </>
            )}
            <dt>Scoring</dt>
            <dd>{b.spec.cv_folds ? `Validation, test and ${b.spec.cv_folds} folds` : "Validation and test"}</dd>
          </dl>
        </section>
      </div>

      {b.kind === "one_vs_rest" ? (
        <OvrTable b={b} runHref={runHref} selected={selected} onSelect={setSelected} />
      ) : (
        <>
          <Leaderboard
            b={b}
            catalog={catalog}
            metric={metric}
            runHref={runHref}
            selected={selected}
            onSelect={setSelected}
            onUse={canUse ? setUsing : null}
          />
          <SearchChart b={b} catalog={catalog} metric={metric} />
        </>
      )}
      <SelectionBar selected={selected} onClear={() => setSelected([])} />

      {using && e && option && (
        <BatchDialog
          kind="one_vs_rest"
          e={e}
          model={b.spec.model}
          modelName={option.label}
          catalog={catalog}
          values={oneVsRestValues(b.spec, using.settings)}
          device={b.spec.device}
          cvFolds={b.spec.cv_folds}
          sweep={[]}
          target={b.spec.target}
          origin={`The settings of rank ${using.rank} in ${KIND_LABEL[b.kind]} #${b.id}, on its model and export.`}
          onClose={() => setUsing(null)}
          onStarted={onStarted}
        />
      )}
    </div>
  );
}
