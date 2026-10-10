import { Link } from "react-router-dom";
import { useExperiment } from "../hooks";
import { baselineLabel } from "../format";
import { parseView, type View } from "../view";
import type { Dataset } from "../types";

/** Review and Bulk edit on a team experiment's copy of its baseline: says so, and leads back. */
export function ExperimentModeBar({ ds, view }: { ds: Dataset; view: View }) {
  const v = parseView(view);
  const x = useExperiment(v?.kind === "experiment" ? v.id : null);
  if (!v) return null;
  if (v.kind === "baseline") {
    return (
      <div className="alert info mode-bar">
        <span className="grow">Browsing a finalized baseline, read-only.</span>
        <Link to={`/d/${ds.id}/baselines`}>Baselines →</Link>
      </div>
    );
  }
  const d = x.data;
  return (
    <div className="alert info mode-bar">
      <span className="grow">
        {d ? <>Adjusting experiment <b>{d.name}</b> (from {baselineLabel(d.recipe_family, d.recipe_version)}).
          {d.can_adjust ? " Changes apply to this experiment only; the baseline stays as the team agreed it." : " Read-only: only its owner adjusts it."}
          {d.adjustments.count > 0 && <> · <b className="num">{d.adjustments.count}</b> field{d.adjustments.count === 1 ? "" : "s"} adjusted</>}
          {d.adjustments.targets_changed && <span className="badge warn" style={{ marginLeft: 6 }}>targets changed</span>}</> : "Loading the experiment…"}
      </span>
      <Link to={`/d/${ds.id}/experiments/${v.id}`}>Back to the experiment →</Link>
    </div>
  );
}
