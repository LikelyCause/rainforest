import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useChartTheme } from "../../theme";
import { fmtPct } from "../../format";
import { Chart } from "../Chart";
import { loadingsOption, miOption, pcaOption } from "./options";
import { AnalysisState, SkippedList, useAnalysis, type ExploreState } from "./state";
import type { Dataset, ImportanceResult } from "../../types";

const TOP_MI = 30;
const PCS = 5;
const TOP_FEATURES = 20;
const MAX_PCA_BARS = 30;

export function ImportanceView({ ds, st }: { ds: Dataset; st: ExploreState }) {
  const a = useAnalysis<ImportanceResult>(ds.id, "importance", st.base);
  return (
    <AnalysisState a={a} datasetId={ds.id}>
      {(r) => (
        <div className="stack tools-gap-16">
          <MiCard ds={ds} r={r} />
          {r.pca && r.pca.explained.length > 0 && <PcaCards pca={r.pca} />}
          <SkippedList skipped={r.skipped} />
        </div>
      )}
    </AnalysisState>
  );
}

function MiCard({ ds, r }: { ds: Dataset; r: ImportanceResult }) {
  const t = useChartTheme();
  const nav = useNavigate();
  const [all, setAll] = useState(false);
  const rows = useMemo(() => {
    const sorted = [...r.mi].sort((a, b) => b.value - a.value);
    return all ? sorted : sorted.slice(0, TOP_MI);
  }, [r, all]);
  const option = useMemo(() => miOption(t, rows), [t, rows]);
  if (!r.mi.length) {
    return (
      <section className="card">
        <div className="card-head"><h2>Mutual information</h2></div>
        <span className="secondary">
          Mutual information needs a target. <Link to={`/?target=${ds.id}`}>Choose a target</Link> to rank fields by how much they tell you about it.
        </span>
      </section>
    );
  }
  return (
    <section className="card chart-card">
      <div className="card-head">
        <h2>Mutual information with {r.target}</h2>
        {r.mi.length > TOP_MI && (
          <label className="check small-text">
            <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> Show all {r.mi.length}
          </label>
        )}
      </div>
      <span className="muted small-text tools-chart-note">
        How much knowing each field reduces uncertainty about the target; catches non-linear links that correlation misses. 0 = independent.
      </span>
      <Chart option={option} height={rows.length * 22 + 16} ariaLabel="Mutual information by field"
        onBarClick={(i) => { const id = rows[i]?.field_id; if (id) nav(`/d/${ds.id}/review/${id}`); }} />
    </section>
  );
}

function PcaCards({ pca }: { pca: NonNullable<ImportanceResult["pca"]> }) {
  const t = useChartTheme();
  const explained = pca.explained.slice(0, MAX_PCA_BARS);
  const cumulative = pca.cumulative.slice(0, MAX_PCA_BARS);
  const pcaOpt = useMemo(() => pcaOption(t, explained, cumulative), [t, pca]);
  const comps = pca.components.slice(0, PCS);
  const features = useMemo(() => {
    const best = new Map<string, number>();
    for (const c of comps) for (const l of c.loadings) best.set(l.feature, Math.max(best.get(l.feature) ?? 0, Math.abs(l.value)));
    return [...best.entries()].sort((a, b) => b[1] - a[1]).slice(0, TOP_FEATURES).map(([f]) => f);
  }, [pca]);
  const loadOpt = useMemo(() => {
    const lookup = comps.map((c) => new Map(c.loadings.map((l) => [l.feature, l.value])));
    return loadingsOption(t, comps.map((_, i) => i), features, (i, f) => lookup[i].get(f));
  }, [t, pca, features]);
  const to80 = pca.cumulative.findIndex((c) => c >= 0.8);

  return (
    <div className="grid-2">
      <section className="card chart-card">
        <div className="card-head"><h2>PCA explained variance</h2><span className="muted small-text">{pca.n_features} features</span></div>
        <span className="muted small-text tools-chart-note">
          {to80 >= 0 ? <>{to80 + 1} component{to80 ? "s" : ""} explain {fmtPct(pca.cumulative[to80], 0)} of the variance.</> : <>The first {pca.cumulative.length} components explain {fmtPct(pca.cumulative[pca.cumulative.length - 1], 0)}.</>}
        </span>
        <Chart option={pcaOpt} height={300} ariaLabel="Explained variance per principal component" />
      </section>
      {comps.length > 0 && features.length > 0 && (
        <section className="card chart-card">
          <div className="card-head"><h2>Loadings</h2><span className="muted small-text">top {features.length} features × first {comps.length} components</span></div>
          <Chart option={loadOpt} height={features.length * 22 + 90} ariaLabel="PCA loadings heatmap" />
        </section>
      )}
    </div>
  );
}
