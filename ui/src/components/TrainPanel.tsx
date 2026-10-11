import { forwardRef, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { fmtInt } from "../format";
import { changed, prefillValues, problems, restore, runFields } from "../settings";
import { pickExport } from "../runs";
import type { ExportJob, ModelRun, SettingValue, TrainPrefill } from "../types";
import { ModelSettings } from "./ModelSettings";
import { ErrorAlert } from "./ui";

const settingsKey = (exportId: number, model: string) => `rf.trainSettings.${exportId}.${model}`;
function remembered(exportId: number, model: string): unknown {
  try { return JSON.parse(localStorage.getItem(settingsKey(exportId, model)) ?? "null"); } catch { return null; }
}

/** Which export the panel trains on and, for Train again, the run to start from. `nonce` makes a repeat request new. */
export interface TrainRequest { exportId: number; prefill?: TrainPrefill; nonce: number }

/** Which of the export's targets a run trains, and in what chain order: the ticked ones, top to bottom. */
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
  exports: ExportJob[]; request: TrainRequest | null; onStarted: (r: ModelRun) => void;
}>(function TrainPanel({ exports, request, onStarted }, ref) {
  const qc = useQueryClient();
  const done = exports.filter((e) => e.status === "done");
  const [exportId, setExportId] = useState<number | null>(null);
  const e = pickExport(done, exportId);
  const targets: string[] = e?.summary?.targets ?? [];
  const models = useQuery({ queryKey: ["exportModels", e?.id], queryFn: () => api.modelsForExport(e!.id), staleTime: 60_000, enabled: !!e });
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  const [model, setModel] = useState("");
  const [gpuOn, setGpuOn] = useState(false);
  const [order, setOrder] = useState<string[]>(targets);
  const [valuesByModel, setValuesByModel] = useState<Record<string, Record<string, SettingValue>>>({});
  const [dropped, setDropped] = useState<string[]>([]);
  const [showAll, setShowAll] = useState(false);
  const [pending, setPending] = useState<TrainPrefill | null>(null);

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
  }, [exportId, e?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  // Each export starts with all its targets; Train again's order (applied below) comes after.
  useEffect(() => {
    if (e) setOrder(targets);
  }, [e?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!pending || !e || pending.exportId !== e.id || !models.data) return;
    const catalog = models.data.find((m) => m.key === pending.model)?.settings;
    setModel(pending.model);
    setOrder(pending.order.filter((t) => targets.includes(t)));
    if (catalog) {
      const { values: v, dropped: d } = prefillValues(pending.values, catalog);
      setValuesByModel((old) => ({ ...old, [`${e.id}:${pending.model}`]: v }));
      try { localStorage.setItem(settingsKey(e.id, pending.model), JSON.stringify(v)); } catch { /* not remembered */ }
      setDropped(d);
    }
    setPending(null);
  }, [pending, models.data, e?.id]); // eslint-disable-line react-hooks/exhaustive-deps
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
  const start = useMutation({
    mutationFn: () => api.startRun(e!.id, {
      model, order, device: useGpu ? "cuda" : "cpu",
      ...(catalog ? { params: changed(values, catalog, kinds), ...runFields(values, catalog) } : { cv: 5 }),
    }),
    onSuccess: (run) => {
      qc.setQueryData<ModelRun[]>(["runs", e!.dataset_id, e!.experiment_id], (old) => [run, ...(old ?? []).filter((r) => r.id !== run.id)]);
      qc.invalidateQueries({ queryKey: ["runs", e!.dataset_id] });
      qc.invalidateQueries({ queryKey: ["experiment", e!.experiment_id] });
      onStarted(run);
    },
  });
  if (!e) return null;
  const shown = models.data?.filter((m) => !m.planned && (showAll || m.compatible)) ?? [];
  const planned = models.data?.filter((m) => m.planned) ?? [];
  const choose = (next: { exportId?: number; model?: string }) => {
    start.reset();
    setDropped([]);
    if (next.exportId !== undefined) setExportId(next.exportId);
    if (next.model !== undefined) setModel(next.model);
  };
  const blocked = models.isLoading ? "Loading models…" : !model ? "Choose a model"
    : !selected?.compatible ? "This model can't train on this export"
    : order.length === 0 ? "Choose at least one target" : settingProblems > 0 ? "Fix the settings marked in red" : null;

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
      </div>
      {models.data?.some((m) => !m.compatible) && (
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <label className="check small-text"><input type="checkbox" checked={showAll} onChange={(ev) => setShowAll(ev.target.checked)} /> show incompatible models</label>
        </div>
      )}
      {targets.length > 1 && <TargetOrder targets={targets} order={order} onChange={setOrder} />}
      {catalog && selected?.compatible ? (
        <ModelSettings settings={catalog} values={values} onChange={setValues} kinds={kinds} />
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
          {start.isPending ? "Starting…" : order.length > 1 ? "Train chain" : "Train"}
        </button>
      </div>
    </section>
  );
});
