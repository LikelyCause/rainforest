import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import { viewKey } from "../../view";
import { useFields } from "../../hooks";
import { useChartTheme } from "../../theme";
import { fmtInt, METHOD_LABEL, TYPE_SHORT } from "../../format";
import { Chart } from "../Chart";
import { Empty, ErrorAlert } from "../ui";
import { corrHeatmapOption } from "./options";
import type { ExploreState } from "./state";
import type { CorrMatrix, Dataset } from "../../types";

export function CorrelationsView({ ds, st }: { ds: Dataset; st: ExploreState }) {
  if (!st.isRaw) {
    return (
      <div className="card">
        <Empty title="Correlations use the raw fields"
          action={<button type="button" onClick={() => st.set({ data: null })}>Switch to raw data</button>}>
          The heatmap reads the associations profiled for each original column, so it isn't available for an export's encoded features.
        </Empty>
      </div>
    );
  }
  return <Heatmap ds={ds} st={st} />;
}

function Heatmap({ ds, st }: { ds: Dataset; st: ExploreState }) {
  const t = useChartTheme();
  const add = useMemo(() => (st.get("add") ?? "").split(",").map(Number).filter((n) => n > 0), [st.params]);
  const threshold = Math.min(Math.max(Number(st.get("min") ?? 0) || 0, 0), 0.95);
  const q = useQuery({
    queryKey: ["corrMatrix", ds.id, st.kept, st.target, add, viewKey(st.view)],
    queryFn: () => api.corrMatrix(ds.id, { kept: st.kept, target: st.target, limit: 60, add, view: st.view }),
    placeholderData: (prev) => prev,
  });
  const m = q.data;
  const [sel, setSel] = useState<number | null>(null);
  useEffect(() => setSel(null), [m]);
  const option = useMemo(() => (m ? corrHeatmapOption(t, m, threshold) : null), [t, m, threshold]);
  const setAdd = (ids: number[]) => st.set({ add: ids.length ? ids.join(",") : null });

  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className="row explore-controls">
        <label className="row threshold" style={{ gap: 8 }}>
          <span className="secondary small-text">Hide below</span>
          <input type="range" min={0} max={0.95} step={0.05} value={threshold}
            onChange={(e) => st.set({ min: Number(e.target.value) ? e.target.value : null })} aria-label="Strength threshold" />
          <span className="num small-text" style={{ width: 32 }}>{threshold.toFixed(2)}</span>
        </label>
        <FieldAdder datasetId={ds.id} shown={m?.order ?? []} onAdd={(id) => setAdd([...new Set([...add, id])])} />
        {add.length > 0 && m && (
          <div className="chips">
            {add.map((id) => (
              <button key={id} type="button" className="chip on" title="Remove from the heatmap" onClick={() => setAdd(add.filter((x) => x !== id))}>
                {m.fields.find((f) => f.id === id)?.name ?? `#${id}`} ✕
              </button>
            ))}
          </div>
        )}
        <span className="spacer" />
        {m && (
          <span className="muted small-text">
            {m.order.length} fields, {st.target ? <>most associated with <code>{st.target}</code></> : "strongest associations"}{q.isFetching && " · updating…"}
          </span>
        )}
      </div>
      <ErrorAlert error={q.error} />
      {m && !m.order.length && <div className="card"><Empty title="No associations yet">Profiling found no field pairs to compare{st.kept ? " among the kept fields" : ""}.</Empty></div>}
      {m && m.order.length > 0 && option && (
        <div className={`explore-split${sel !== null ? " with-panel" : ""}`}>
          <div className="card chart-card" style={{ padding: 8 }}>
            <Chart option={option} height={Math.max(380, m.order.length * 14 + 190)} ariaLabel="Association heatmap"
              onItemClick={(p) => p.componentType === "series" && setSel(p.value[3])} />
            <div className="row muted small-text" style={{ padding: "0 8px 4px", justifyContent: "space-between" }}>
              <span>Colour = association strength 0–1 (|Spearman ρ|, Cramér's V or correlation ratio η, by field types). Clustered order.</span>
              <span className="legend-ramp" aria-hidden="true"><i />0 → 1</span>
            </div>
          </div>
          {sel !== null && m.cells[sel] && <CellPanel ds={ds} m={m} cell={m.cells[sel]} onClose={() => setSel(null)} />}
        </div>
      )}
      {!m && q.isLoading && <div className="card"><span className="skeleton" style={{ height: 320 }} /></div>}
    </div>
  );
}

function CellPanel({ ds, m, cell, onClose }: { ds: Dataset; m: CorrMatrix; cell: CorrMatrix["cells"][number]; onClose: () => void }) {
  const f = (id: number) => m.fields.find((x) => x.id === id);
  const signed = cell.method === "spearman";
  return (
    <aside className="card side-panel stack" aria-label="Pair details">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Pair</h2>
        <button type="button" className="ghost small" aria-label="Close" onClick={onClose}>✕</button>
      </div>
      {[cell.a, cell.b].map((id) => {
        const x = f(id);
        return (
          <div key={id} className="pair-field">
            <div className="row" style={{ justifyContent: "space-between", flexWrap: "nowrap" }}>
              <b className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{x?.name ?? `#${id}`}</b>
              {x && <span className="type-tag">{TYPE_SHORT[x.type] ?? x.type}</span>}
            </div>
            <div className="row small-text" style={{ justifyContent: "space-between" }}>
              <span className="muted">{x?.action === "target" ? "◎ target" : x?.action ?? ""}</span>
              <Link to={`/d/${ds.id}/review/${id}`}>Open in Review →</Link>
            </div>
          </div>
        );
      })}
      <div className="kv" style={{ gridTemplateColumns: "1fr" }}>
        <div><span>Method</span><b>{METHOD_LABEL[cell.method] ?? cell.method}</b></div>
        <div><span>Value</span><b>{signed && cell.value > 0 ? "+" : ""}{cell.value.toFixed(3)}</b></div>
        <div><span>Strength</span><b>{cell.strength.toFixed(3)}</b></div>
        <div><span>Rows compared</span><b>{fmtInt(cell.n)}</b></div>
      </div>
      {signed && <span className="muted small-text">{cell.value >= 0 ? "Rise together" : "One rises as the other falls"} (rank correlation).</span>}
    </aside>
  );
}

function FieldAdder({ datasetId, shown, onAdd }: { datasetId: number; shown: number[]; onAdd: (id: number) => void }) {
  const fields = useFields(datasetId);
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  const have = new Set(shown);
  const matches = q.trim()
    ? (fields.data ?? []).filter((f) => !have.has(f.id) && f.name.toLowerCase().includes(q.trim().toLowerCase())).slice(0, 30)
    : [];
  return (
    <div ref={ref} style={{ position: "relative" }} onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <input type="search" placeholder="Add a field…" value={q} aria-label="Add a field to the heatmap" style={{ width: 220 }}
        onFocus={() => setOpen(true)} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onKeyDown={(e) => { if (e.key === "Enter" && matches[0]) { onAdd(matches[0].id); setQ(""); } }} />
      {open && q.trim() && (
        <div className="popover combobox">
          <div className="combo-list">
            {matches.length === 0 && <span className="muted small-text" style={{ padding: 6 }}>No other fields match</span>}
            {matches.map((f) => (
              <button key={f.id} type="button" className="ghost menu-item" onClick={() => { onAdd(f.id); setQ(""); setOpen(false); }}>
                <span className="mono grow" style={{ overflow: "hidden", textOverflow: "ellipsis", textAlign: "left" }}>{f.name}</span>
                <span className="type-tag">{TYPE_SHORT[f.semantic_type] ?? f.semantic_type}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
