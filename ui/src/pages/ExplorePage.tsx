import { NavLink, Navigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useDatasetContext } from "../components/DatasetLayout";
import { useAccess } from "../hooks";
import { CorrelationsView } from "../components/explore/CorrelationsView";
import { MapView } from "../components/explore/MapView";
import { TestsView } from "../components/explore/TestsView";
import { ImportanceView } from "../components/explore/ImportanceView";
import { SAMPLES, useExploreState, type ExploreState } from "../components/explore/state";
import { familyShort, fmtCompact } from "../format";
import type { Dataset } from "../types";

const VIEWS = [
  { key: "correlations", label: "Correlations" },
  { key: "map", label: "Map" },
  { key: "tests", label: "Feature tests" },
  { key: "importance", label: "Importance" },
] as const;
type View = (typeof VIEWS)[number]["key"];

export function ExplorePage() {
  const ds = useDatasetContext();
  const view = useParams().view ?? "correlations";
  const st = useExploreState(ds);
  if (!VIEWS.some((v) => v.key === view)) return <Navigate to={`/d/${ds.id}/explore`} replace />;
  const search = st.params.toString();
  return (
    <div className="page explore">
      <h1>Explore</h1>
      <Toolbar ds={ds} st={st} sampled={view !== "correlations"} />
      <nav className="tabs-inline" aria-label="Analyses">
        {VIEWS.map((v) => (
          <NavLink key={v.key} to={{ pathname: `/d/${ds.id}/explore/${v.key}`, search }}
            className={() => (view === v.key ? "on" : "")} aria-current={view === v.key ? "page" : undefined}>
            {v.label}
          </NavLink>
        ))}
      </nav>
      <ViewBody view={view as View} ds={ds} st={st} />
    </div>
  );
}

// Only the visible view mounts, so only it requests an analysis.
function ViewBody({ view, ds, st }: { view: View; ds: Dataset; st: ExploreState }) {
  switch (view) {
    case "map": return <MapView ds={ds} st={st} />;
    case "tests": return <TestsView ds={ds} st={st} />;
    case "importance": return <ImportanceView ds={ds} st={st} />;
    default: return <CorrelationsView ds={ds} st={st} />;
  }
}

function Toolbar({ ds, st, sampled }: { ds: Dataset; st: ExploreState; sampled: boolean }) {
  const mine = useAccess(ds).source === "mine";
  const exports = useQuery({ queryKey: ["exports", ds.id], queryFn: () => api.exports(ds.id) });
  const done = (exports.data ?? []).filter((e) => e.status === "done");
  const unknown = !st.isRaw && !done.some((e) => `export:${e.id}` === st.data);
  return (
    <div className="explore-toolbar" role="toolbar" aria-label="Analysis data">
      <label className="row" style={{ gap: 6 }}>
        <span className="secondary small-text">Data</span>
        <select value={st.data} onChange={(e) => st.set({ data: e.target.value === "raw" ? null : e.target.value })}>
          <option value="raw">Raw fields</option>
          {done.map((e) => (
            <option key={e.id} value={`export:${e.id}`}>
              Export #{e.id}{e.experiment_name ? ` · ${e.experiment_name}` : ""} · {familyShort(e.model_family)} · train split · {new Date(e.finished_at ?? e.created_at).toLocaleDateString()}
            </option>
          ))}
          {unknown && <option value={st.data}>Export #{st.data.slice(7)}</option>}
        </select>
      </label>
      {st.isRaw && <DecisionsPicker ds={ds} st={st} />}
      {st.isRaw && (
        <label className="check small-text" title={mine && !st.view ? "Only the fields you kept in your assessment; otherwise every field except targets"
          : "Only fields marked keep; otherwise every field except targets"}>
          <input type="checkbox" checked={st.kept} onChange={(e) => st.set({ kept: e.target.checked ? "1" : null })} />
          {mine && !st.view ? "Kept only (your choices)" : "Kept only"}
        </label>
      )}
      {st.targets.length > 0 && (
        <label className="row" style={{ gap: 6 }}>
          <span className="secondary small-text">Target</span>
          {st.targets.length === 1 ? <code className="target-code">◎ {st.target}</code> : (
            <select value={st.target ?? ""} onChange={(e) => st.set({ target: e.target.value === st.targets[0] ? null : e.target.value })}>
              {st.targets.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          )}
        </label>
      )}
      {sampled && <div className="row" style={{ gap: 6 }}>
        <span className="secondary small-text" id="sample-label">Sample</span>
        <div className="segmented soft" role="radiogroup" aria-labelledby="sample-label">
          {SAMPLES.map((n) => (
            <button key={n} type="button" role="radio" aria-checked={st.sample === n} className={st.sample === n ? "on" : ""}
              onClick={() => st.set({ sample: n === 5000 ? null : String(n) })}>{fmtCompact(n)}</button>
          ))}
        </div>
        <span className="muted small-text">rows{ds.row_count ? ` of ${fmtCompact(ds.row_count)}` : ""}</span>
      </div>}
    </div>
  );
}

/** Whose keep / drop and types raw-data views use: the baseline in progress, a finalized baseline, or a team
 * experiment's adjusted copy. */
function DecisionsPicker({ ds, st }: { ds: Dataset; st: ExploreState }) {
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id) });
  const xs = useQuery({ queryKey: ["experiments", ds.id], queryFn: () => api.experiments(ds.id) });
  const finalized = (lines.data ?? []).flatMap((l) => l.versions.filter((v) => v.phase === "finalized"));
  const adjusted = (xs.data ?? []).filter((x) => x.assessor_id === null && !x.is_default && !x.archived_at);
  const cur = ds.baseline;
  return (
    <label className="row" style={{ gap: 6 }}>
      <span className="secondary small-text">Decisions</span>
      <select value={st.view ?? ""} onChange={(e) => st.set({ decisions: e.target.value || null })} aria-label="Whose decisions">
        <option value="">{cur ? `${cur.label} (${cur.phase === "finalized" ? "finalized" : "as it stands"})` : "Current"}</option>
        {finalized.filter((v) => v.id !== cur?.id || cur?.phase !== "finalized").map((v) => (
          <option key={v.id} value={`baseline:${v.id}`}>{v.label}</option>
        ))}
        {adjusted.map((x) => <option key={x.id} value={`experiment:${x.id}`}>Experiment: {x.name}</option>)}
        {st.view && lines.data && xs.data && !finalized.some((v) => `baseline:${v.id}` === st.view)
          && !adjusted.some((x) => `experiment:${x.id}` === st.view) && <option value={st.view}>Unavailable ({st.view})</option>}
      </select>
    </label>
  );
}
