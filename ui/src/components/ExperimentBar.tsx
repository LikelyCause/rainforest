import { Link } from "react-router-dom";
import { useExperiment } from "../hooks";
import { Tabs } from "./kit";
import type { Dataset } from "../types";

/** The second row on an experiment's page: where you are in it, and its places. Owned by the Experiments work. */
export function ExperimentBar({ ds, experimentId }: { ds: Dataset; experimentId: number }) {
  const x = useExperiment(experimentId);
  const base = `/d/${ds.id}/experiments/${experimentId}`;
  return (
    <>
      <Link to={`/d/${ds.id}/experiments`} className="secondary small-text">Experiments</Link>
      <span className="crumb-sep" aria-hidden="true">/</span>
      <b className="version-name">{x.data?.name ?? "…"}</b>
      <Tabs label="Experiment" className="sub" items={[
        { to: `${base}?tab=runs`, label: "Runs" },
        { to: `${base}?tab=train`, label: "Train" },
        { to: `${base}?tab=exports`, label: "Exports" },
        { to: `${base}?tab=adjustments`, label: "Adjustments" },
        { to: `${base}?tab=notes`, label: "Notes" },
      ]} />
      <span className="spacer" />
    </>
  );
}
