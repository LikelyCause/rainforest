import { useEffect, useMemo, useState } from "react";
import { useDebounced } from "../../hooks";
import { useChartTheme } from "../../theme";
import { Chart } from "../Chart";
import { mapOption } from "./options";
import { AnalysisState, SkippedList, useAnalysis, type ExploreState } from "./state";
import type { AnalysisParams, Dataset, MapResult } from "../../types";

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
const KNOBS = {
  tsne: { key: "perplexity", label: "Perplexity", def: 30, min: 5, max: 100, hint: "Roughly how many neighbours each point balances; higher shows broader structure." },
  umap: { key: "nn", label: "Neighbours", def: 15, min: 5, max: 100, hint: "Local neighbourhood size; higher keeps more global structure." },
} as const;

export function MapView({ ds, st }: { ds: Dataset; st: ExploreState }) {
  const method = st.get("method") === "umap" ? "umap" : "tsne";
  const knob = KNOBS[method];
  const knobValue = clamp(Number(st.get(knob.key)) || knob.def, knob.min, knob.max);
  const params: AnalysisParams = useMemo(
    () => ({ ...st.base, method, ...(method === "tsne" ? { perplexity: knobValue } : { n_neighbors: knobValue }) }),
    [st.base, method, knobValue],
  );
  const a = useAnalysis<MapResult>(ds.id, "map", params);

  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className="row explore-controls">
        <div className="segmented soft" role="radiogroup" aria-label="Embedding method">
          {(["tsne", "umap"] as const).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={method === m} className={method === m ? "on" : ""}
              onClick={() => st.set({ method: m === "tsne" ? null : "umap" })}>{m === "tsne" ? "t-SNE" : "UMAP"}</button>
          ))}
        </div>
        <KnobInput key={method} label={knob.label} hint={knob.hint} value={knobValue} min={knob.min} max={knob.max}
          onCommit={(v) => st.set({ [knob.key]: v === knob.def ? null : String(v) })} />
        <span className="muted small-text">Fixed seed{method === "tsne" ? "; reduced to 50 PCA components first" : ""}.</span>
      </div>
      <AnalysisState a={a} datasetId={ds.id}>
        {(r) => <MapChart r={r} />}
      </AnalysisState>
    </div>
  );
}

function KnobInput({ label, hint, value, min, max, onCommit }: {
  label: string; hint: string; value: number; min: number; max: number; onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const debounced = useDebounced(text, 600);
  useEffect(() => {
    const v = Number(debounced);
    if (debounced !== "" && Number.isFinite(v) && v >= min && v <= max && Math.round(v) !== value) onCommit(Math.round(v));
  }, [debounced]); // eslint-disable-line react-hooks/exhaustive-deps
  const n = Number(text);
  const bad = text !== "" && (!Number.isFinite(n) || n < min || n > max);
  return (
    <label className="row" style={{ gap: 6 }} title={hint}>
      <span className="secondary small-text">{label}</span>
      <input type="number" min={min} max={max} step={1} value={text} onChange={(e) => setText(e.target.value)}
        style={{ width: 76, ...(bad ? { borderColor: "var(--bad)" } : {}) }} aria-invalid={bad} />
      {bad && <span className="small-text" style={{ color: "var(--bad)" }}>{min}–{max}</span>}
    </label>
  );
}

function MapChart({ r }: { r: MapResult }) {
  const t = useChartTheme();
  const option = useMemo(() => mapOption(t, r), [t, r]);
  return (
    <>
      <div className="card chart-card" style={{ padding: 8 }}>
        <Chart option={option} height={580} ariaLabel={`${r.method === "umap" ? "UMAP" : "t-SNE"} map of ${r.n} rows`} />
        <span className="muted small-text" style={{ padding: "0 8px 4px" }}>
          {r.target ? <>Coloured by <code>{r.target}</code>{r.target_kind === "categorical" && " — click legend entries to hide classes"}. </> : "No target chosen, so points are uncoloured. "}
          Distances between far-apart clusters aren't meaningful; nearby points are similar rows.
        </span>
      </div>
      <SkippedList skipped={r.skipped} />
    </>
  );
}
