import { Link } from "react-router-dom";
import { useExperiment } from "../hooks";
import { baselineLabel } from "../format";
import { parseView, type View } from "../view";
import type { Dataset } from "../types";
import { Banner, Chip } from "./kit";

/** A page on a team experiment's copy of its baseline, or a finalized baseline: says so, and leads back. The header's
 * second row says the same on every dataset page, so pages show this only where that row isn't. */
export function ExperimentModeBar({ ds, view }: { ds: Dataset; view: View }) {
  const v = parseView(view);
  const x = useExperiment(v?.kind === "experiment" ? v.id : null);
  if (!v) return null;
  if (v.kind === "baseline") {
    return (
      <Banner tone="info" className="mode-bar" action={<Link to={`/d/${ds.id}/baselines`}>Versions and compare</Link>}>
        Browsing a finalized baseline, read-only.
      </Banner>
    );
  }
  const d = x.data;
  return (
    <Banner tone="info" className="mode-bar" action={<Link to={`/d/${ds.id}/experiments/${v.id}`}>Back to the experiment</Link>}>
      {d ? <>Adjusting experiment <b>{d.name}</b> (from {baselineLabel(d.recipe_family, d.recipe_version, false, d.recipe_name)}).
        {d.can_adjust ? " Changes apply to this experiment only; the baseline stays as the team agreed it." : " Read-only: only its owner adjusts it."}
        {d.adjustments.count > 0 && <> · <b className="num">{d.adjustments.count}</b> field{d.adjustments.count === 1 ? "" : "s"} adjusted</>}
        {d.adjustments.targets_changed && <> <Chip tone="warn">targets changed</Chip></>}</> : "Loading the experiment…"}
    </Banner>
  );
}
