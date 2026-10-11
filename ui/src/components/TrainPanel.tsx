import { forwardRef, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { fmtInt } from "../format";
import { changed, isVisible, prefillValues, problems, restore, runFields, suitsTargets } from "../settings";
import { combinations, parseSweep, SWEEP_MAX } from "../sweep";
import { pickExport } from "../runs";
import type { ExportJob, ModelRun, ModelSetting, SettingValue, TrainPrefill } from "../types";
import { ModelSettings } from "./ModelSettings";
import { ErrorAlert } from "./ui";

const settingsKey = (exportId: number, model: string) => `rf.trainSettings.${exportId}.${model}`;
function remembered(exportId: number, model: string): unknown {
  try { return JSON.parse(localStorage.getItem(settingsKey(exportId, model)) ?? "null"); } catch { return null; }
}

/** Which export the panel trains on and, for Train again, the run to start from. `nonce` makes a repeat request new. */
export interface TrainRequest { exportId: number; prefill?: TrainPrefill; nonce: number }

/** Which of the export's targets a run trains, and in what chain order: the ticked ones, top to bottom. */
/** Up to two settings, each with values to try; Train then starts one run per combination. */
function Sweep({ catalog, values, kinds, axes, onChange }: {
  catalog: ModelSetting[]; values: Record<string, SettingValue>; kinds: string[];
  axes: { name: string; text: string }[]; onChange: (a: { name: string; text: string }[]) => void;
}) {
  const choices = catalog.filter((s) => s.group !== "run" && isVisible(s, values, catalog) && suitsTargets(s, kinds));
  const set = (i: number, a: { name: string; text: string } | null) =>
    onChange(a ? axes.map((x, j) => (j === i ? a : x)) : axes.filter((_, j) => j !== i));
  return (
    <div className="stack sweep">
      {axes.map((a, i) => {
        const s = catalog.find((x) => x.name === a.name);
        const error = s && a.text.trim() ? parseSweep(s, a.text).error : null;
        return (
          <div key={i} className="row">
            <select aria-label={`Setting ${i + 1}`} value={a.name} onChange={(ev) => set(i, { name: ev.target.value, text: "" })}>
              {choices.map((c) => <option key={c.name} value={c.name} disabled={c.name !== a.name && axes.some((x) => x.name === c.name)}>{c.label}</option>)}
            </select>
            <input aria-label={`Values for ${s?.label ?? a.name}`} value={a.text} onChange={(ev) => set(i, { ...a, text: ev.target.value })}
              placeholder={s?.options ? s.options.map((o) => o.label).slice(0, 3).join(", ") : s?.kind === "float" ? "0.01, 0.05, 0.1" : "15, 31, 63"} />
            <button type="button" className="ghost small" onClick={() => set(i, null)}>Remove</button>
            {error && <span className="small-text" style={{ color: "var(--bad)" }}>{error}</span>}
          </div>
        );
      })}
      {axes.length < 2 && choices.length > axes.length && (
        <div><button type="button" className="ghost small"
          onClick={() => onChange([...axes, { name: choices.find((c) => !axes.some((x) => x.name === c.name))!.name, text: "" }])}>
          {axes.length ? "Add a second setting" : "Try several values"}</button></div>
      )}
    </div>
  );
}

function TargetOrder({ targets, order, onChange }: { targets: string[]; order: string[]; onChange: (o: string[]) => void }) {
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
          <div key={t} className="row" style={{ gap: 8 }}>
            <label className="check">
              <input type="checkbox" checked={i >= 0}
                onChange={(ev) => onChange(ev.target.checked ? [...order, t] : order.filter((x) => x !== t))} />
              {i >= 0 && order.length > 1 ? `${i + 1}. ` : ""}{t}
            </label>
            {i > 0 && <button type="button" className="link small-text" onClick={() => move(t)} aria-label={`Move ${t} earlier`}>↑ earlier</button>}
          </div>
        );
      })}
      <span className="muted small-text">
        {order.length === 0 ? "Choose at least one target."
          : order.length === 1 ? "One target: a single model, no chain."
          : `Chain: ${order.join(" → ")}. Each model also sees the predictions of the ones before it.`}
      </span>
    </fieldset>
  );
}

/** Train a model on one of the experiment's exports: the export, the model, where it trains, which targets in which
 * order, and the model's settings, laid out in full. */
export const TrainPanel = forwardRef<HTMLElement, {
  exports: ExportJob[]; request: TrainRequest | null; onStarted: (r: ModelRun, count: number) => void;
}>(function TrainPanel({ exports, request, onStarted }, ref) {
  const qc = useQueryClient();
  const done = exports.filter((e) => e.status === "done");
  const [exportId, setExportId] = useState<number | null>(null);
  const e = pickExport(done, exportId);
  const targets: string[] = e?.summary?.targets ?? [];
  const models = useQuery({ queryKey: ["exportModels", e?.id], queryFn: () => api.modelsForExport(e!.id), staleTime: 60_000, enabled: !!e });
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  const [model, setModel] = useState("");
  const [gpuOn, setGpuOn] = useState(true);
  const [cvFolds, setCvFolds] = useState(5);
  const [order, setOrder] = useState<string[]>(targets);
  const [valuesByModel, setValuesByModel] = useState<Record<string, Record<string, SettingValue>>>({});
  const [dropped, setDropped] = useState<string[]>([]);
  const [showAll, setShowAll] = useState(false);
  const [pending, setPending] = useState<TrainPrefill | null>(null);
  const [axes, setAxes] = useState<{ name: string; text: string }[]>([]);

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
      try { localStorage.setItem(settingsKey(e.id, pending.model), JSON.stringify(v)); } catch { /* not remembered */ }
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
    try { localStorage.setItem(settingsKey(e.id, model), JSON.stringify(v)); } catch { /* private mode: not remembered */ }
  };
  const kinds: string[] = order.map((t) => e?.summary?.target_kinds?.[t]).filter(Boolean);
  const settingProblems = catalog ? Object.keys(problems(values, catalog, kinds)).length : 0;
  const gpu = system.data?.gpu;
  const gpuChoice = Boolean(gpu?.available && selected?.devices?.includes("cuda"));
  const useGpu = gpuChoice && gpuOn;
  const parsedAxes = catalog ? axes.map((a) => ({ name: a.name, ...parseSweep(catalog.find((s) => s.name === a.name)!, a.text) })) : [];
  const combos = catalog && axes.length ? combinations(values, parsedAxes) : [];
  const comboProblem = combos.map((c) => Object.entries(problems(c, catalog!, kinds))[0]).find(Boolean);
  const startOne = (v: Record<string, SettingValue>) => api.startRun(e!.id, {
    model, order, device: useGpu ? "cuda" : "cpu", cv_folds: cvFolds,
    ...(catalog ? { params: changed(v, catalog, kinds), ...runFields(v, catalog) } : { cv: 5 }),
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
      qc.setQueryData<ModelRun[]>(["runs", e!.dataset_id, e!.experiment_id], (old) => [...[...runs].reverse(), ...(old ?? []).filter((r) => !runs.some((x) => x.id === r.id))]);
      onStarted(runs[runs.length - 1], runs.length);
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
  const blocked = models.isLoading ? "Loading models…" : !model ? "Choose a model"
    : !selected?.compatible ? "This model can't train on this export"
    : order.length === 0 ? "Choose at least one target" : settingProblems > 0 ? "Fix the settings marked in red"
    : parsedAxes.some((a) => a.error) ? "Finish the values to try"
    : combos.length > SWEEP_MAX ? `${combos.length} combinations; keep it to ${SWEEP_MAX}`
    : comboProblem ? `A combination won't train: ${comboProblem[0]} ${comboProblem[1]}` : null;

  return (
    <section ref={ref} className="card stack train-panel" aria-label="Train a model">
      <div className="card-head"><h2>Train a model</h2></div>
      <div className="train-controls">
        <label className="field">Export
          <select value={e.id} onChange={(ev) => choose({ exportId: Number(ev.target.value) })}>
            {done.map((x) => <option key={x.id} value={x.id}>Export #{x.id} · {fmtInt(x.n_rows)} rows · {(x.summary?.targets ?? []).join(", ")}</option>)}
          </select>
        </label>
        <label className="field">Model
          <select value={model} onChange={(ev) => choose({ model: ev.target.value })}>
            {shown.map((m) => <option key={m.key} value={m.key} disabled={!m.compatible}>{m.label}{m.compatible ? "" : " — not compatible"}</option>)}
            {planned.length > 0 && (
              <optgroup label="Planned">
                {planned.map((m) => <option key={m.key} value={m.key} disabled>{m.label} (planned)</option>)}
              </optgroup>
            )}
          </select>
        </label>
        {gpuChoice && gpu && (
          <div className="field">
            <span id="train-device">Train on</span>
            <div className="segmented soft full" role="radiogroup" aria-labelledby="train-device">
              {([false, true] as const).map((on) => (
                <button key={String(on)} type="button" role="radio" aria-checked={gpuOn === on} className={gpuOn === on ? "on" : ""}
                  onClick={() => setGpuOn(on)}>{on ? "GPU" : "CPU"}</button>
              ))}
            </div>
            <span className="muted small-text num">{gpu.name}{gpu.memory_mb ? ` · ${fmtInt(Math.round(gpu.memory_mb / 1024))} GB` : ""}</span>
          </div>
        )}
        <label className="field">Cross-validation
          <select value={cvFolds} onChange={(ev) => setCvFolds(Number(ev.target.value))}>
            <option value={0}>Off</option>
            {Array.from({ length: 19 }, (_, i) => i + 2).map((k) => <option key={k} value={k}>{k} folds</option>)}
          </select>
          <span className="muted small-text">
            {cvFolds ? `Scores ${cvFolds} folds of the training rows too: about ${cvFolds + 1}× the time` : "Validation and test scores only"}
          </span>
        </label>
      </div>
      {models.data?.some((m) => !m.compatible) && (
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <label className="check small-text"><input type="checkbox" checked={showAll} onChange={(ev) => setShowAll(ev.target.checked)} /> show incompatible models</label>
        </div>
      )}
      {targets.length > 1 && <TargetOrder targets={targets} order={order} onChange={setOrder} />}
      {catalog && selected?.compatible ? (
        <>
          <ModelSettings settings={catalog} values={values} onChange={setValues} kinds={kinds} />
          <Sweep catalog={catalog} values={values} kinds={kinds} axes={axes} onChange={setAxes} />
        </>
      ) : selected?.compatible && (
        <div className="muted small-text">{selected.label}'s settings aren't in the app yet; it trains with its defaults.</div>
      )}
      {dropped.length > 0 && (
        <span className="muted small-text">Not carried over (no longer settings of this model): {dropped.join(", ")}</span>
      )}
      <ErrorAlert error={start.error} />
      <div className="row train-footer">
        {blocked && <span className="small-text" role={settingProblems ? "alert" : undefined} style={settingProblems ? { color: "var(--bad)" } : undefined}>{blocked}</span>}
        <button type="button" className="primary" disabled={Boolean(blocked) || start.isPending} onClick={() => start.mutate()}>
          {start.isPending ? "Starting…" : combos.length ? `Train ${combos.length} runs` : order.length > 1 ? "Train chain" : "Train"}
        </button>
      </div>
    </section>
  );
});
