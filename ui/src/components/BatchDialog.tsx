import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import {
  batchRequest,
  buildSpace,
  capProblem,
  classPlan,
  defaultRanking,
  LOWER_IS_BETTER,
  METRIC_NAME,
  rankingChoices,
  runCount,
  weightingClash,
  type AxisDraft,
} from "../batches";
import { fmtInt } from "../format";
import { changed, describeSettings, isVisible, runFields, suitsTargets } from "../settings";
import type {
  BatchKind,
  BatchStarted,
  ClassCount,
  Device,
  ExportJob,
  ModelSetting,
  SettingValue,
  Skip,
} from "../types";
import { Dialog, Icon, Segmented } from "./kit";
import { SweepValues } from "./SweepValues";
import { ErrorAlert } from "./ui";

// The models whose one-vs-rest runs the API weights when asked to balance: the boosters by negatives / positives, the
// others with balanced class weights. MLP has no class weighting.
const BALANCING_MODELS = ["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest", "logistic_regression"];

const plural = (n: number, word: string) => `${fmtInt(n)} ${word}${n === 1 ? "" : word.endsWith("s") ? "es" : "s"}`;
const isNumberSetting = (s: ModelSetting) => s.kind === "int" || s.kind === "float" || s.kind === "int_or_none";
const newAxis = (s: ModelSetting, values: string[] = []): AxisDraft => ({
  name: s.name,
  mode: "list",
  values,
  low: "",
  high: "",
  log: s.log,
  integer: s.kind !== "float",
});

/** Every class not trained, grouped by why, so a long target's skips stay readable. */
export function Skips({ skipped }: { skipped: Skip[] }) {
  if (!skipped.length) return null;
  const byReason = new Map<string, string[]>();
  for (const s of skipped) byReason.set(s.reason, [...(byReason.get(s.reason) ?? []), s.class]);
  return (
    <div className="batch-skips small-text" aria-live="polite">
      <b>Not trained: {plural(skipped.length, "class")}</b>
      <ul>
        {[...byReason].map(([reason, classes]) => (
          <li key={reason}>
            <span className="secondary">{reason.charAt(0).toUpperCase() + reason.slice(1)}:</span> {classes.join(", ")}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ClassTable({
  target,
  classes,
  ticked,
  reasons,
  onTick,
}: {
  target: string;
  classes: ClassCount[];
  ticked: string[] | null;
  reasons: Map<string, string>;
  onTick: (t: string[] | null) => void;
}) {
  const all = classes.map((c) => c.class);
  const on = (c: string) => ticked === null || ticked.includes(c);
  const toggle = (c: string, yes: boolean) => {
    const next = yes ? [...(ticked ?? all), c] : (ticked ?? all).filter((x) => x !== c);
    onTick(next.length === all.length ? null : all.filter((x) => next.includes(x)));
  };
  return (
    <div className="stack exp-gap-2">
      <div className="row exp-between">
        <span className="field-label">Classes of {target}</span>
        <span className="row exp-gap-2">
          <button type="button" className="small ghost" disabled={ticked === null} onClick={() => onTick(null)}>
            Tick all
          </button>
          <button type="button" className="small ghost" disabled={ticked?.length === 0} onClick={() => onTick([])}>
            Untick all
          </button>
        </span>
      </div>
      <div className="batch-classes table-wrap">
        <table className="data batch-class-table">
          <thead>
            <tr>
              <th scope="col">
                <span className="exp-sr">Train</span>
              </th>
              <th scope="col">Class</th>
              <th scope="col" className="num" title="Rows of the class in the training split">
                Train
              </th>
              <th scope="col" className="num">
                Valid
              </th>
              <th scope="col" className="num">
                Test
              </th>
              <th scope="col">
                <span className="exp-sr">Plan</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {classes.map((c) => {
              const reason = reasons.get(c.class);
              return (
                <tr key={c.class} className={reason ? "skipped" : undefined}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Train ${c.class}`}
                      checked={on(c.class)}
                      onChange={(ev) => toggle(c.class, ev.target.checked)}
                    />
                  </td>
                  <td className="batch-class-name">{c.class}</td>
                  <td className="num">{fmtInt(c.train)}</td>
                  <td className="num">{fmtInt(c.valid)}</td>
                  <td className="num">{fmtInt(c.test)}</td>
                  <td className={`small-text ${reason ? "muted" : "secondary"}`}>
                    {reason ? reason.charAt(0).toUpperCase() + reason.slice(1) : "Trains"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SpaceEditor({
  catalog,
  choices,
  axes,
  onChange,
}: {
  catalog: ModelSetting[];
  choices: ModelSetting[];
  axes: AxisDraft[];
  onChange: (a: AxisDraft[]) => void;
}) {
  const set = (i: number, a: AxisDraft | null) =>
    onChange(a ? axes.map((x, j) => (j === i ? a : x)) : axes.filter((_, j) => j !== i));
  const unused = choices.filter((c) => !axes.some((a) => a.name === c.name));
  return (
    <fieldset className="field bare batch-space">
      <legend className="field-label">Settings to search</legend>
      {axes.map((a, i) => {
        const s = catalog.find((x) => x.name === a.name)!;
        const error = buildSpace([a], catalog).error;
        const started = a.mode === "list" ? a.values.length > 0 : Boolean(a.low.trim() || a.high.trim());
        return (
          // Keyed by setting, so a half-typed value stays with its setting when another axis is removed or changed.
          <div key={a.name} className="batch-axis stack exp-gap-1">
            <div className="row batch-axis-row">
              <select
                aria-label={`Setting ${i + 1}`}
                value={a.name}
                onChange={(ev) => set(i, newAxis(catalog.find((x) => x.name === ev.target.value)!))}
              >
                {choices.map((c) => (
                  <option
                    key={c.name}
                    value={c.name}
                    disabled={c.name !== a.name && axes.some((x) => x.name === c.name)}
                  >
                    {c.label}
                  </option>
                ))}
              </select>
              <Segmented
                label={`${s.label}: values or a range`}
                size="small"
                value={a.mode}
                onChange={(mode) => set(i, { ...a, mode })}
                options={[
                  { value: "list", label: "Values" },
                  {
                    value: "range",
                    label: "Range",
                    disabled: !isNumberSetting(s),
                    title: isNumberSetting(s) ? undefined : `${s.label} takes a list of values`,
                  },
                ]}
              />
              <button
                type="button"
                className="ghost small"
                aria-label={`Remove ${s.label}`}
                onClick={() => set(i, null)}
              >
                Remove
              </button>
            </div>
            {a.mode === "list" ? (
              <SweepValues s={s} values={a.values} onChange={(values) => set(i, { ...a, values })} />
            ) : (
              <div className="row batch-range">
                <input
                  aria-label={`${s.label}: low`}
                  inputMode="decimal"
                  placeholder={s.min !== null ? `from ${s.min}` : "low"}
                  value={a.low}
                  onChange={(ev) => set(i, { ...a, low: ev.target.value })}
                />
                <span className="muted">to</span>
                <input
                  aria-label={`${s.label}: high`}
                  inputMode="decimal"
                  placeholder={s.max !== null ? `up to ${s.max}` : "high"}
                  value={a.high}
                  onChange={(ev) => set(i, { ...a, high: ev.target.value })}
                />
                <label className="check small-text">
                  <input type="checkbox" checked={a.log} onChange={(ev) => set(i, { ...a, log: ev.target.checked })} />
                  Log scale
                </label>
                <label
                  className="check small-text"
                  title={s.kind !== "float" ? `${s.label} is a whole number` : undefined}
                >
                  <input
                    type="checkbox"
                    checked={a.integer || s.kind !== "float"}
                    disabled={s.kind !== "float"}
                    onChange={(ev) => set(i, { ...a, integer: ev.target.checked })}
                  />
                  Whole numbers
                </label>
              </div>
            )}
            {error && started && (
              <span className="small-text exp-bad" role="alert">
                {error.slice(s.label.length + 2)}
              </span>
            )}
          </div>
        );
      })}
      {unused.length > 0 && (
        <div>
          <button type="button" className="small" onClick={() => onChange([...axes, newAxis(unused[0])])}>
            <Icon name="plus" />
            {axes.length ? "Add a setting" : "Choose a setting to search"}
          </button>
        </div>
      )}
    </fieldset>
  );
}

/** Starts a batch on the Train panel's export, model and settings: one model per class of a class target, or a search
 * over settings. The preview counts the runs against the server's cap and names every class that won't train, by the
 * rules the API plans with. */
export function BatchDialog({
  kind,
  e,
  model,
  modelName,
  catalog,
  values,
  device,
  cvFolds,
  sweep,
  target: initialTarget = null,
  order = [],
  origin,
  onClose,
  onStarted,
}: {
  kind: BatchKind;
  e: ExportJob;
  model: string;
  /** The model's label for class targets, as the API names batches. */
  modelName: string;
  catalog: ModelSetting[] | null;
  values: Record<string, SettingValue>;
  device: Device;
  cvFolds: number;
  /** The panel's "Try several values", to start the search from. */
  sweep: { name: string; values: string[] }[];
  onClose: () => void;
  onStarted: (started: BatchStarted) => void;
  /** The class target to start on, when not the export's first. */
  target?: string | null;
  /** The targets ticked on the Train panel, in chain order: what a search on the whole target trains. */
  order?: string[];
  /** Where the settings come from, when not the Train panel. */
  origin?: string;
}) {
  const targets: string[] = e.summary?.targets ?? [];
  const targetKinds: Record<string, string> = e.summary?.target_kinds ?? {};
  const classTargets = targets.filter((t) => targetKinds[t] && targetKinds[t] !== "regression");
  const [target, setTarget] = useState<string | null>(
    initialTarget && classTargets.includes(initialTarget) ? initialTarget : (classTargets[0] ?? null),
  );
  const [scope, setScope] = useState<"whole" | "ovr">(kind === "one_vs_rest" ? "ovr" : "whole");
  const ovr = kind === "one_vs_rest" || scope === "ovr";
  const panelOrder = order.filter((t) => targets.includes(t));
  const trained = panelOrder.length ? panelOrder : targets;
  const planTarget = ovr ? target : null;
  const plan = useQuery({
    queryKey: ["batchPlan", e.id, planTarget],
    queryFn: () => api.batchPlan(e.id, planTarget),
    staleTime: 60_000,
  });
  // null: every class (one-vs-rest), or the plan's default classes (a one-vs-rest search).
  const [ticked, setTicked] = useState<string[] | null>(null);
  const [tuneOn, setTuneOn] = useState<string[] | null>(null);
  const [balanceOn, setBalanceOn] = useState(true);
  const [name, setName] = useState("");
  const kinds = ovr ? ["binary"] : trained.map((t) => targetKinds[t]).filter(Boolean);
  const choices = (catalog ?? []).filter(
    (s) => s.group !== "run" && isVisible(s, values, catalog ?? []) && suitsTargets(s, kinds),
  );
  const [axes, setAxes] = useState<AxisDraft[]>(() =>
    sweep.flatMap((a) => {
      const s = catalog?.find((x) => x.name === a.name);
      return s ? [newAxis(s, a.values)] : [];
    }),
  );
  const [sampler, setSampler] = useState<"grid" | "random">("grid");
  const [draws, setDraws] = useState("20");
  const [seed, setSeed] = useState("0");
  const [ranking, setRanking] = useState<string | null>(null);

  const params = catalog ? changed(values, catalog, kinds) : {};
  const run = catalog ? runFields(values, catalog) : { seed: 42, cv: 5 };
  const p = plan.data;
  const classes = p?.classes ?? [];
  const ovrPlan = classPlan(classes, kind === "one_vs_rest" ? ticked : null);
  const reasonOf = new Map(ovrPlan.skipped.map((s) => [s.class, s.reason]));
  const ineligible = new Map(classes.flatMap((c) => (c.reason ? [[c.class, c.reason] as const] : [])));
  const chosen = (tuneOn ?? p?.search_classes ?? []).filter((c) => !ineligible.has(c));
  const searched = new Set(axes.map((a) => a.name));
  // A searched setting's own value is never used, so it isn't listed among the settings every run takes.
  const fixed = Object.fromEntries(Object.entries(params).filter(([k]) => kind !== "search" || !searched.has(k)));
  const unusedSweep = kind === "one_vs_rest" && sweep.some((a) => a.values.length > 0);

  const space = kind === "search" && catalog ? buildSpace(axes, catalog) : { space: {}, error: null };
  const count =
    kind === "one_vs_rest"
      ? runCount({ kind, classes: ovrPlan.trained.length })
      : runCount({
          kind,
          space: space.space,
          sampler: { kind: sampler, n: Number(draws) },
          classes: ovr ? chosen.length : undefined,
        });
  const cap = p ? capProblem(count.runs, p.max_runs) : null;
  const rankKind = ovr ? "binary" : targetKinds[trained[0]];
  const rankings = rankingChoices(rankKind);
  const rank = ranking && rankings.includes(ranking) ? ranking : defaultRanking(rankKind);
  const balanceOffered = ovr && BALANCING_MODELS.includes(model);
  const clash = weightingClash(params, kind === "search" ? axes.map((a) => a.name) : []);
  const balance = balanceOffered && balanceOn && clash.length === 0;
  const labelOf = (n: string) => catalog?.find((s) => s.name === n)?.label ?? n;

  const blocker = plan.isLoading
    ? ovr
      ? "Loading the export's classes…"
      : "Loading…"
    : plan.error
      ? ovr
        ? "The export's classes couldn't be read"
        : "The batch limits couldn't be read"
      : ovr && !target
        ? "This export has no class target"
        : kind === "one_vs_rest" && !ovrPlan.trained.length
          ? `No class of ${target} can be trained`
          : kind === "search" && !catalog
            ? "This model's settings aren't in the app yet, so there is nothing to search"
            : kind === "search" && space.error
              ? space.error
              : kind === "search" && ovr && !chosen.length
                ? "Choose at least one class to tune on"
                : kind === "search" && sampler === "random" && !/^-?\d+$/.test(seed.trim())
                  ? "The seed must be a whole number"
                  : (count.error ?? cap);

  const body = () =>
    batchRequest({
      kind,
      name,
      model,
      device,
      params,
      seed: run.seed,
      cv: run.cv,
      cvFolds,
      ovr,
      target,
      order: panelOrder,
      balance,
      ticked,
      space: space.space,
      sampler: { kind: sampler, n: Number(draws), seed: Number(seed.trim()) },
      ranking: rank,
      searchClasses: chosen,
    });
  const start = useMutation({ mutationFn: () => api.startBatch(e.id, body()), onSuccess: onStarted });
  // A refusal is about the form as it was sent; once the form changes it no longer applies.
  const edit =
    <T,>(set: (v: T) => void) =>
    (v: T) => {
      if (start.isError) start.reset();
      set(v);
    };
  const defaultName =
    kind === "one_vs_rest"
      ? `${modelName} · one model per class of ${target ?? "…"}`
      : `${modelName} · search over ${Object.keys(space.space).join(", ") || "…"}${ovr ? ` on ${chosen.length} classes of ${target}` : ""}`;
  const splitText = ovr
    ? ""
    : trained.length > 1
      ? ` Every run trains ${trained.join(" → ")}, ranked on ${trained[0]}.`
      : targets.length > 1
        ? ` Every run trains ${trained[0]}, the target ticked on Train.`
        : "";

  return (
    <Dialog
      title={kind === "one_vs_rest" ? "Train one model per class" : "Search settings"}
      wide
      onClose={onClose}
      labelledBy="batch-title"
      actions={
        <>
          {/* In the actions row, above the buttons, so a refusal is in view however far the body is scrolled. */}
          {start.error != null && (
            <div className="batch-error">
              <ErrorAlert error={start.error} />
            </div>
          )}
          {blocker && (
            <span id="batch-why" className={`small-text batch-why${cap ? " exp-bad" : " muted"}`}>
              {blocker}
            </span>
          )}
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            disabled={blocker !== null || start.isPending}
            aria-describedby={blocker ? "batch-why" : undefined}
            onClick={() => start.mutate()}
          >
            <Icon name="play" />
            {start.isPending
              ? "Starting…"
              : count.error || space.error
                ? "Start"
                : `Start ${plural(count.runs, "run")}`}
          </button>
        </>
      }
    >
      <div className="batch-body stack">
        <p className="muted small-text exp-flush">
          {kind === "one_vs_rest"
            ? "A binary model for each class against the rest of the target, each calibrated and thresholded on validation."
            : "Runs with different settings, ranked on the validation split."}{" "}
          The runs train one at a time, so other exports and runs queue between them.
        </p>
        <dl className="batch-from num">
          <dt>Model</dt>
          <dd>
            {modelName} on Export #{e.id}
          </dd>
          <dt>Settings</dt>
          <dd title={catalog ? describeSettings(fixed, catalog, true) : undefined}>
            {catalog ? describeSettings(fixed, catalog) : "defaults"}
          </dd>
          <dt>Scoring</dt>
          <dd>{cvFolds ? `Validation, test and ${cvFolds} folds` : "Validation and test"}</dd>
          <dt>Device</dt>
          <dd>{device === "cuda" ? "GPU" : "CPU"}</dd>
        </dl>
        <span className="muted small-text">{origin ?? "Change the model and its settings in the Train panel."}</span>
        {unusedSweep && (
          <span className="small-text batch-note" role="note">
            The values to try in the panel aren't used; each run takes the panel's value
          </span>
        )}

        {kind === "search" && classTargets.length > 0 && (
          <div className="field">
            <span>Train</span>
            <Segmented
              label="Train"
              value={scope}
              onChange={edit(setScope)}
              options={[
                { value: "whole", label: trained.length > 1 ? "The ticked targets" : "The whole target" },
                { value: "ovr", label: "One model per class" },
              ]}
            />
            <span className="muted small-text">
              {ovr
                ? "Each combination is trained once per chosen class and ranked on the mean of their scores."
                : `One run per combination.${splitText}`}
            </span>
          </div>
        )}
        {ovr && classTargets.length > 1 && (
          <label className="field">
            Target
            <select
              value={target ?? ""}
              onChange={(ev) => {
                edit(setTarget)(ev.target.value);
                setTicked(null);
                setTuneOn(null);
              }}
            >
              {classTargets.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
        )}
        <ErrorAlert error={plan.error} />

        {kind === "one_vs_rest" && p?.classes && target && (
          <ClassTable target={target} classes={classes} ticked={ticked} reasons={reasonOf} onTick={edit(setTicked)} />
        )}
        {kind === "search" && ovr && p?.classes && (
          <fieldset className="field bare">
            <legend className="field-label">Classes to tune on</legend>
            <div className="check-grid">
              {classes.map((c) => {
                const why = ineligible.get(c.class);
                return (
                  <label key={c.class} className="check" title={why}>
                    <input
                      type="checkbox"
                      disabled={Boolean(why)}
                      checked={!why && chosen.includes(c.class)}
                      onChange={(ev) =>
                        edit(setTuneOn)(
                          ev.target.checked
                            ? classes.map((x) => x.class).filter((x) => x === c.class || chosen.includes(x))
                            : chosen.filter((x) => x !== c.class),
                        )
                      }
                    />
                    <span className="batch-class-name">{c.class}</span>
                    <span className="muted small-text num">{why ? "can't train" : fmtInt(c.train)}</span>
                  </label>
                );
              })}
            </div>
            <span className="muted small-text">
              By default the most frequent, the median and the rarest class that train; the number is each class's
              training rows.
            </span>
          </fieldset>
        )}

        {kind === "search" && catalog && (
          <>
            <SpaceEditor catalog={catalog} choices={choices} axes={axes} onChange={edit(setAxes)} />
            <div className="row batch-sampler">
              <div className="field">
                <span>Combinations</span>
                <Segmented
                  label="Combinations"
                  value={sampler}
                  onChange={edit(setSampler)}
                  options={[
                    { value: "grid", label: "Grid" },
                    { value: "random", label: "Random" },
                  ]}
                />
              </div>
              {sampler === "random" && (
                <>
                  <label className="field">
                    Draw
                    <input
                      type="number"
                      min={1}
                      inputMode="numeric"
                      value={draws}
                      onChange={(ev) => edit(setDraws)(ev.target.value)}
                    />
                  </label>
                  <label className="field">
                    Seed
                    <input inputMode="numeric" value={seed} onChange={(ev) => edit(setSeed)(ev.target.value)} />
                  </label>
                </>
              )}
            </div>
            <span className="muted small-text">
              {sampler === "grid"
                ? "Every combination of the values; ranges need random sampling."
                : "Distinct combinations drawn from the values and ranges, the same ones for the same seed. A combination the setting checks refuse is drawn again."}
            </span>
            <label className="field">
              Rank on
              <select value={rank} onChange={(ev) => edit(setRanking)(ev.target.value)}>
                {rankings.map((m) => (
                  <option key={m} value={m}>
                    {METRIC_NAME[m] ?? m}
                    {m === defaultRanking(rankKind) ? " (default)" : ""}
                  </option>
                ))}
              </select>
              <span className="muted small-text">
                On the validation split{LOWER_IS_BETTER.has(rank) ? "; lower is better" : ""}.
              </span>
            </label>
          </>
        )}

        {balanceOffered && (
          <label
            className="check"
            title={clash.length ? undefined : "Weights each run's positives by negatives / positives"}
          >
            <input
              type="checkbox"
              checked={balance}
              disabled={clash.length > 0}
              onChange={(ev) => edit(setBalanceOn)(ev.target.checked)}
            />
            Balance the classes
            <span className="muted small-text">
              {clash.length
                ? `Off: ${clash.map(labelOf).join(" and ")} ${clash.length === 1 ? "is" : "are"} set${kind === "search" ? " or searched" : ""}`
                : "each run weights its class by its rarity in training"}
            </span>
          </label>
        )}

        <label className="field">
          Name
          <input
            value={name}
            placeholder={defaultName}
            maxLength={200}
            onChange={(ev) => edit(setName)(ev.target.value)}
          />
        </label>

        <div className="batch-preview" aria-live="polite">
          <div>
            <b className="num">{count.error || space.error ? "—" : plural(count.runs, "run")}</b>
            {p && <span className="muted small-text num"> · at most {fmtInt(p.max_runs)} in one batch</span>}
            {kind === "search" && ovr && !space.error && !count.error && (
              <span className="muted small-text num">
                {" "}
                · {plural(count.runs / Math.max(chosen.length, 1), "combination")} × {plural(chosen.length, "class")}
              </span>
            )}
          </div>
          {kind === "one_vs_rest" && <Skips skipped={ovrPlan.skipped} />}
        </div>
      </div>
    </Dialog>
  );
}
