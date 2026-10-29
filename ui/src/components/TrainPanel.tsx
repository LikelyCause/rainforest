import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { fmtInt, fmtPct } from "../format";
import {
  changed,
  describeSettings,
  fmtSetting,
  isVisible,
  prefillValues,
  problems,
  restore,
  runFields,
  suitsTargets,
} from "../settings";
import { combinations, parseSweep, SWEEP_MAX } from "../sweep";
import { modelDescription, modelLabel, pickExport } from "../runs";
import type { BatchKind, BatchStarted, ExportJob, ModelRun, ModelSetting, SettingValue, TrainPrefill } from "../types";
import { BatchDialog } from "./BatchDialog";
import { Icon, Segmented } from "./kit";
import { ModelSettings } from "./ModelSettings";
import { SweepValues } from "./SweepValues";
import { ErrorAlert } from "./ui";

const settingsKey = (exportId: number, model: string) => `rf.trainSettings.${exportId}.${model}`;
function remembered(exportId: number, model: string): unknown {
  try {
    return JSON.parse(localStorage.getItem(settingsKey(exportId, model)) ?? "null");
  } catch {
    return null;
  }
}

/** Which export the panel trains on and, for Train again, the run to start from. `nonce` makes a repeat request new. */
export interface TrainRequest {
  exportId: number;
  prefill?: TrainPrefill;
  nonce: number;
}

type Axis = { name: string; values: string[] };

/** Up to two settings, each with values to try; Train then starts one run per combination. */
function Sweep({
  catalog,
  values,
  kinds,
  axes,
  onChange,
}: {
  catalog: ModelSetting[];
  values: Record<string, SettingValue>;
  kinds: string[];
  axes: Axis[];
  onChange: (a: Axis[]) => void;
}) {
  const choices = catalog.filter((s) => s.group !== "run" && isVisible(s, values, catalog) && suitsTargets(s, kinds));
  const set = (i: number, a: Axis | null) =>
    onChange(a ? axes.map((x, j) => (j === i ? a : x)) : axes.filter((_, j) => j !== i));
  const add = () =>
    onChange([...axes, { name: choices.find((c) => !axes.some((x) => x.name === c.name))!.name, values: [] }]);
  const canAdd = axes.length < 2 && choices.length > axes.length;
  return (
    <section className="card stack sweep" aria-labelledby="sweep-title">
      <div className="row exp-between">
        <h2 id="sweep-title">Try several values</h2>
        {canAdd && axes.length > 0 && (
          <button type="button" className="small ghost" onClick={add}>
            <Icon name="plus" />
            Add a second setting
          </button>
        )}
      </div>
      {axes.map((a, i) => {
        const s = catalog.find((x) => x.name === a.name);
        const parsed = s && a.values.length ? parseSweep(s, a.values.join(", ")) : null;
        return (
          <div key={i} className="stack exp-gap-6">
            <div className="row sweep-row">
              <select
                aria-label={`Setting to vary${axes.length > 1 ? ` ${i + 1}` : ""}`}
                value={a.name}
                onChange={(ev) => set(i, { name: ev.target.value, values: [] })}
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
              {s && <SweepValues key={a.name} s={s} values={a.values} onChange={(v) => set(i, { ...a, values: v })} />}
              <button
                type="button"
                className="ghost small"
                aria-label={`Remove ${s?.label ?? a.name}`}
                onClick={() => set(i, null)}
              >
                Remove
              </button>
            </div>
            {parsed?.error && (
              <span className="small-text exp-bad" role="alert">
                {parsed.error}
              </span>
            )}
          </div>
        );
      })}
      {axes.length === 0
        ? canAdd && (
            <div>
              <button type="button" className="small" onClick={add}>
                <Icon name="plus" />
                Choose a setting to vary
              </button>
            </div>
          )
        : null}
      <p className="muted small-text exp-flush">
        {axes.length === 0
          ? "Compare values of one or two settings: one run per value, or per combination of two. "
          : axes.length === 1
            ? "One run per value; everything else as set above. "
            : "One run per combination; everything else as set above. "}
        Up to {SWEEP_MAX} runs.
      </p>
    </section>
  );
}

function TargetOrder({
  targets,
  order,
  onChange,
}: {
  targets: string[];
  order: string[];
  onChange: (o: string[]) => void;
}) {
  const rows = [...order, ...targets.filter((t) => !order.includes(t))];
  const move = (t: string) => {
    const i = order.indexOf(t);
    onChange([...order.slice(0, i - 1), t, order[i - 1], ...order.slice(i + 1)]);
  };
  return (
    <fieldset className="field bare target-order">
      <legend>Targets</legend>
      {rows.map((t) => {
        const i = order.indexOf(t);
        return (
          <div key={t} className="row">
            <label className="check">
              <input
                type="checkbox"
                checked={i >= 0}
                onChange={(ev) => onChange(ev.target.checked ? [...order, t] : order.filter((x) => x !== t))}
              />
              {i >= 0 && order.length > 1 ? `${i + 1}. ` : ""}
              {t}
            </label>
            {i > 0 && (
              <button
                type="button"
                className="link small-text"
                onClick={() => move(t)}
                aria-label={`Move ${t} earlier`}
              >
                ↑ earlier
              </button>
            )}
          </div>
        );
      })}
      <span className="muted small-text">
        {order.length === 0
          ? "Choose at least one target."
          : order.length === 1
            ? "One target: a single model, no chain."
            : `Chain: ${order.join(" → ")}. Each model also sees the predictions of the ones before it.`}
      </span>
    </fieldset>
  );
}

/** Train a model on one of the experiment's exports: what to train (the export, the model, cross-validation, where it
 * trains, which targets in which order), the model's settings, values to sweep, and a summary with the Train button. */
export function TrainPanel({
  exports,
  request,
  onStarted,
  onBatchStarted,
  private: personal = false,
}: {
  exports: ExportJob[];
  request: TrainRequest | null;
  onStarted: (runs: ModelRun[], e: ExportJob, label: string) => void;
  onBatchStarted: (started: BatchStarted, e: ExportJob) => void;
  /** A private experiment's runs reach MLflow only once its baseline version is in team review. */
  private?: boolean;
}) {
  const qc = useQueryClient();
  const done = exports.filter((e) => e.status === "done");
  const [exportId, setExportId] = useState<number | null>(null);
  const e = pickExport(done, exportId);
  const targets: string[] = e?.summary?.targets ?? [];
  const models = useQuery({
    queryKey: ["exportModels", e?.id],
    queryFn: () => api.modelsForExport(e!.id),
    staleTime: 60_000,
    enabled: !!e,
  });
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  const [model, setModel] = useState("");
  const [gpuOn, setGpuOn] = useState(true);
  const [cvFolds, setCvFolds] = useState(5);
  const [order, setOrder] = useState<string[]>(targets);
  const [valuesByModel, setValuesByModel] = useState<Record<string, Record<string, SettingValue>>>({});
  const [dropped, setDropped] = useState<string[]>([]);
  const [showAll, setShowAll] = useState(false);
  const [pending, setPending] = useState<TrainPrefill | null>(null);
  const [axes, setAxes] = useState<Axis[]>([]);
  const [batchKind, setBatchKind] = useState<BatchKind | null>(null);

  // A request picks the export; Train again also brings the run's model, order and settings once the export's
  // models have loaded.
  useEffect(() => {
    if (!request) return;
    setExportId(request.exportId);
    setPending(request.prefill ?? null);
    if (!request.prefill) setDropped([]);
  }, [request]);
  // Once shown, the panel stays on its export: a newer one finishing mustn't move the run being set up.
  useEffect(() => {
    if (exportId === null && e) setExportId(e.id);
  }, [exportId, e?.id]);
  // Each export starts with all its targets; Train again's order (applied below) comes after.
  useEffect(() => {
    if (e) setOrder(targets);
  }, [e?.id]);
  useEffect(() => {
    if (!pending || !e || pending.exportId !== e.id || !models.data) return;
    const catalog = models.data.find((m) => m.key === pending.model)?.settings;
    setModel(pending.model);
    if (pending.cvFolds !== undefined) setCvFolds(pending.cvFolds);
    setOrder(pending.order.filter((t) => targets.includes(t)));
    if (catalog) {
      const { values: v, dropped: d } = prefillValues(pending.values, catalog);
      setValuesByModel((old) => ({ ...old, [`${e.id}:${pending.model}`]: v }));
      try {
        localStorage.setItem(settingsKey(e.id, pending.model), JSON.stringify(v));
      } catch {
        /* not remembered */
      }
      setDropped(d);
    }
    setPending(null);
  }, [pending, models.data, e?.id]);
  useEffect(() => {
    if (!models.data || pending) return;
    const current = models.data.find((m) => m.key === model);
    if (!current?.compatible) {
      const own = models.data.find((m) => m.compatible && m.family === e?.model_family);
      setModel((own ?? models.data.find((m) => m.compatible))?.key ?? "");
    }
  }, [models.data, model, e?.model_family, pending]);

  const selected = models.data?.find((m) => m.key === model);
  const catalog = selected?.settings ?? null;
  const valueKey = e ? `${e.id}:${model}` : "";
  const values = valuesByModel[valueKey] ?? (catalog && e ? restore(remembered(e.id, model), catalog) : {});
  const setValues = (v: Record<string, SettingValue>) => {
    if (!e) return;
    setValuesByModel((old) => ({ ...old, [valueKey]: v }));
    try {
      localStorage.setItem(settingsKey(e.id, model), JSON.stringify(v));
    } catch {
      /* private mode: not remembered */
    }
  };
  const kinds: string[] = order.map((t) => e?.summary?.target_kinds?.[t]).filter(Boolean);
  const settingProblems = catalog ? Object.keys(problems(values, catalog, kinds)).length : 0;
  const gpu = system.data?.gpu;
  const gpuChoice = Boolean(gpu?.available && selected?.devices?.includes("cuda"));
  const useGpu = gpuChoice && gpuOn;
  const parsedAxes = catalog
    ? axes.map((a) => ({
        name: a.name,
        ...parseSweep(
          catalog.find((s) => s.name === a.name)!,
          a.values.join(", "),
        ),
      }))
    : [];
  const combos = catalog && axes.length ? combinations(values, parsedAxes) : [];
  const comboProblem = combos.map((c) => Object.entries(problems(c, catalog!, kinds))[0]).find(Boolean);
  const startOne = (v: Record<string, SettingValue>) =>
    api.startRun(e!.id, {
      model,
      order,
      device: useGpu ? "cuda" : "cpu",
      cv_folds: cvFolds,
      // A sweep's own settings are sent even at their default, so every run of it says what it tried.
      ...(catalog
        ? {
            params: {
              ...changed(v, catalog, kinds),
              ...Object.fromEntries(parsedAxes.map((x) => [x.name, v[x.name]])),
            },
            ...runFields(v, catalog),
          }
        : { cv: 5 }),
    });
  const start = useMutation({
    // A sweep's runs start one after another so a refusal stops the rest instead of leaving a partial grid unexplained.
    mutationFn: async () => {
      const runs: ModelRun[] = [];
      for (const v of combos.length ? combos : [values]) runs.push(await startOne(v));
      return runs;
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["runs", e!.dataset_id] });
      qc.invalidateQueries({ queryKey: ["experiment", e!.experiment_id] });
    },
    onSuccess: (runs) => {
      qc.setQueryData<ModelRun[]>(["runs", e!.dataset_id, e!.experiment_id], (old) => [
        ...[...runs].reverse(),
        ...(old ?? []).filter((r) => !runs.some((x) => x.id === r.id)),
      ]);
      onStarted(runs, e!, selected ? modelLabel(selected, kinds) : model);
    },
  });
  if (!e) return null;
  const shown = models.data?.filter((m) => !m.planned && (showAll || m.compatible)) ?? [];
  const planned = models.data?.filter((m) => m.planned) ?? [];
  const choose = (next: { exportId?: number; model?: string }) => {
    start.reset();
    setDropped([]);
    if (next.model !== undefined) setAxes([]);
    if (next.exportId !== undefined) setExportId(next.exportId);
    if (next.model !== undefined) setModel(next.model);
  };
  // A batch picks its own target and values, so only the model and its settings must be ready.
  const batchBlocked = models.isLoading
    ? "Loading models…"
    : !model
      ? "Choose a model"
      : !selected?.compatible
        ? "This model can't train on this export"
        : settingProblems > 0
          ? "Fix the settings marked in red"
          : null;
  const blocked =
    batchBlocked ??
    (order.length === 0
      ? "Choose at least one target"
      : parsedAxes.some((a) => a.error)
        ? "Finish the values to try"
        : combos.length > SWEEP_MAX
          ? `${combos.length} combinations; keep it to ${SWEEP_MAX}`
          : comboProblem
            ? `A combination won't train: ${comboProblem[0]} ${comboProblem[1]}`
            : null);
  const classTarget = targets.some((t) => e.summary?.target_kinds?.[t] && e.summary.target_kinds[t] !== "regression");

  const split = e.summary?.split_counts as { train: number; valid: number; test: number } | undefined;
  const info = e.summary?.split_info as { stratified_on?: string[] | null; grouped_on?: string | null } | undefined;
  const cfg = e.config ?? {};
  const splitText =
    cfg.valid_size !== undefined
      ? `Split ${fmtPct(1 - cfg.valid_size - cfg.test_size, 0)} / ${fmtPct(cfg.valid_size, 0)} / ${fmtPct(cfg.test_size, 0)}`
      : null;
  const runChanged = catalog
    ? catalog.filter((s) => s.group === "run" && s.name in values && values[s.name] !== s.default)
    : [];
  // A swept setting's own value is never used, so it isn't listed as changed.
  const sweptNames = new Set(axes.map((a) => a.name));
  const changedValues = {
    ...changed(values, catalog ?? [], kinds),
    ...Object.fromEntries(runChanged.map((s) => [s.name, values[s.name]])),
  };
  const changedText = catalog
    ? describeSettings(Object.fromEntries(Object.entries(changedValues).filter(([k]) => !sweptNames.has(k))), catalog)
    : "defaults";
  const swept = parsedAxes
    .filter((a) => !a.error)
    .map((a) => {
      const s = catalog!.find((x) => x.name === a.name)!;
      return `${s.label}, swept over ${a.values.map((v) => fmtSetting(s, v)).join(", ")}`;
    });
  const count = combos.length || 1;

  return (
    <div className="train-layout">
      <div className="stack train-main">
        <section className="card stack" aria-labelledby="train-what">
          <h2 id="train-what">What to train</h2>
          <div className="train-controls">
            <label className="field">
              Export
              <select value={e.id} onChange={(ev) => choose({ exportId: Number(ev.target.value) })}>
                {done.map((x) => (
                  <option key={x.id} value={x.id}>
                    Export #{x.id} · {fmtInt(x.n_rows)} rows · {(x.summary?.targets ?? []).join(", ")}
                  </option>
                ))}
              </select>
              <span className="muted small-text">
                {[
                  splitText,
                  info?.stratified_on?.length ? `stratified on ${info.stratified_on.join(" × ")}` : null,
                  info?.grouped_on ? `${info.grouped_on} kept together` : null,
                ]
                  .filter(Boolean)
                  .join(", ")}
              </span>
            </label>
            <label className="field">
              Model
              <select value={model} onChange={(ev) => choose({ model: ev.target.value })}>
                {shown.map((m) => (
                  <option key={m.key} value={m.key} disabled={!m.compatible}>
                    {modelLabel(m, kinds)}
                    {m.compatible ? "" : " — not compatible"}
                  </option>
                ))}
                {planned.length > 0 && (
                  <optgroup label="Planned">
                    {planned.map((m) => (
                      <option key={m.key} value={m.key} disabled>
                        {m.label} (planned)
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
              <span className="muted small-text">
                {selected?.compatible === false && selected.reason
                  ? selected.reason
                  : selected && modelDescription(selected, kinds)}
              </span>
            </label>
            <label className="field">
              Cross-validation
              <select value={cvFolds} onChange={(ev) => setCvFolds(Number(ev.target.value))}>
                <option value={0}>Off</option>
                {Array.from({ length: 19 }, (_, i) => i + 2).map((k) => (
                  <option key={k} value={k}>
                    {k} folds
                  </option>
                ))}
              </select>
              <span className="muted small-text">
                {cvFolds
                  ? `Also scores ${cvFolds} folds of the training rows; about ${cvFolds + 1}× the time`
                  : "Validation and test scores only"}
                {cvFolds && info?.grouped_on ? `. Each ${info.grouped_on} stays in one fold.` : ""}
              </span>
            </label>
            {gpuChoice && gpu && (
              <div className="field">
                <span id="train-device">Train on</span>
                <Segmented
                  label="Train on"
                  full
                  value={gpuOn ? "gpu" : "cpu"}
                  onChange={(v) => setGpuOn(v === "gpu")}
                  options={[
                    { value: "cpu", label: "CPU" },
                    { value: "gpu", label: "GPU" },
                  ]}
                />
                <span className="muted small-text num">
                  {gpu.name}
                  {gpu.memory_mb ? ` · ${fmtInt(Math.round(gpu.memory_mb / 1024))} GB` : ""}
                </span>
              </div>
            )}
          </div>
          {models.data?.some((m) => !m.compatible) && (
            <label className="check small-text">
              <input type="checkbox" checked={showAll} onChange={(ev) => setShowAll(ev.target.checked)} /> Show
              incompatible models
            </label>
          )}
          {targets.length > 1 && <TargetOrder targets={targets} order={order} onChange={setOrder} />}
        </section>
        {catalog && selected?.compatible ? (
          <>
            <ModelSettings settings={catalog} values={values} onChange={setValues} kinds={kinds} />
            <Sweep catalog={catalog} values={values} kinds={kinds} axes={axes} onChange={setAxes} />
          </>
        ) : (
          selected?.compatible && (
            <section className="card">
              <span className="muted small-text">
                {modelLabel(selected, kinds)}'s settings aren't in the app yet; it trains with its defaults.
              </span>
            </section>
          )
        )}
      </div>

      <aside className="card stack train-summary" aria-labelledby="train-ready">
        <h2 id="train-ready">{blocked ? "Not ready yet" : "Ready to train"}</h2>
        <dl className="num">
          <dt>Model</dt>
          <dd>{selected ? modelLabel(selected, kinds) : "—"}</dd>
          <dt>On</dt>
          <dd>
            Export #{e.id}
            {split ? ` · ${fmtInt(split.train)} training rows` : ""}
          </dd>
          {order.length > 1 && (
            <>
              <dt>Chain</dt>
              <dd>{order.join(" → ")}</dd>
            </>
          )}
          <dt>Changed</dt>
          <dd>
            {[changedText === "defaults" && swept.length ? null : changedText, ...swept].filter(Boolean).join(" · ")}
          </dd>
          <dt>Scoring</dt>
          <dd>{cvFolds ? `Validation, test and ${cvFolds} folds` : "Validation and test"}</dd>
          {gpuChoice && (
            <>
              <dt>Device</dt>
              <dd>{useGpu ? "GPU" : "CPU"}</dd>
            </>
          )}
          <dt>Runs</dt>
          <dd>
            <b>{count}</b>
          </dd>
        </dl>
        {blocked && (
          <span
            className={`small-text${settingProblems ? " exp-bad" : ""}`}
            role={settingProblems ? "alert" : undefined}
          >
            {blocked}
          </span>
        )}
        <button
          type="button"
          className="primary large"
          disabled={Boolean(blocked) || start.isPending}
          onClick={() => start.mutate()}
        >
          <Icon name="play" />
          {start.isPending
            ? "Starting…"
            : combos.length > 1
              ? `Train ${combos.length} runs`
              : order.length > 1
                ? "Train chain"
                : "Train"}
        </button>
        <div className="train-batch stack exp-gap-2" role="group" aria-label="Batches">
          {classTarget && (
            <button
              type="button"
              disabled={Boolean(batchBlocked)}
              title={batchBlocked ?? "A binary model for each class of a target, against the rest"}
              onClick={() => setBatchKind("one_vs_rest")}
            >
              Train one model per class…
            </button>
          )}
          <button
            type="button"
            disabled={Boolean(batchBlocked) || !catalog}
            title={
              batchBlocked ??
              (catalog
                ? "Many runs over ranges or lists of settings, ranked"
                : "This model's settings aren't in the app yet")
            }
            onClick={() => setBatchKind("search")}
          >
            Search settings…
          </button>
        </div>
        {dropped.length > 0 && (
          <span className="muted small-text">
            Not carried over (no longer settings of this model): {dropped.join(", ")}
          </span>
        )}
        <ErrorAlert error={start.error} />
        <p className="muted small-text exp-flush">
          You'll go to Runs, where {count > 1 ? "they appear" : "it appears"} at the top as{" "}
          {count > 1 ? "they train" : "it trains"}.
          {system.data?.mlflow.enabled &&
            ` ${count > 1 ? "Each one is" : "It's"} logged to MLflow${personal ? " once review opens" : ""}.`}
        </p>
      </aside>
      {batchKind && selected && (
        <BatchDialog
          kind={batchKind}
          e={e}
          model={model}
          modelName={selected.label}
          catalog={catalog}
          values={values}
          device={useGpu ? "cuda" : "cpu"}
          cvFolds={cvFolds}
          sweep={axes}
          onClose={() => setBatchKind(null)}
          onStarted={(started) => {
            setBatchKind(null);
            qc.invalidateQueries({ queryKey: ["runs", e.dataset_id] });
            qc.invalidateQueries({ queryKey: ["batches", e.experiment_id] });
            qc.invalidateQueries({ queryKey: ["experiment", e.experiment_id] });
            onBatchStarted(started, e);
          }}
        />
      )}
    </div>
  );
}
