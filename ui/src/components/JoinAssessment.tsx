import { useMutation } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useMe, useTeam } from "../hooks";
import { useInvalidateTeam } from "../pages/DatasetTeamPage";
import type { Dataset } from "../types";
import { ErrorAlert } from "./ui";
import { Banner } from "./kit";

/** For a lead looking at a dataset being assessed without being one of its assessors: join with one click. While the
 * first baseline is still in set-up, joining would end set-up with default choices, so a lead is led to the set-up
 * form instead. */
export function JoinAssessment({ ds }: { ds: Dataset }) {
  const me = useMe().data;
  const team = useTeam(ds.id, ds.phase === "assessing" && !ds.baseline?.setup);
  const invalidate = useInvalidateTeam(ds.id);
  const join = useMutation({ mutationFn: () => api.assign(ds.id, me!.id), onSuccess: invalidate });
  if (ds.baseline?.setup) {
    // Assessing now would fix the first baseline's model type and targets before anyone chose them.
    const lead = me?.role === "lead";
    return (
      <Banner tone="info" className="join-banner"
        action={lead ? <Link className="btn primary small" to={`/d/${ds.id}/baseline`}>Set up the baseline</Link> : undefined}>
        {lead ? "Set up this dataset's first baseline before anyone assesses it: its model type, targets and who assesses it."
          : "A lead sets up this dataset's first baseline (model type, targets, assessors) before anyone assesses it."}
      </Banner>
    );
  }
  if (ds.phase !== "assessing" || me?.role !== "lead" || !team.data || team.data.me) return null;
  return (
    <Banner tone="info" className="join-banner"
      action={<button type="button" className="primary small" disabled={join.isPending} onClick={() => join.mutate()}>Assess it myself</button>}>
      You aren't assessing this dataset, so its fields are read-only for you. Assess it yourself, or assign others on
      the <Link to={`/d/${ds.id}/baseline`}>Baseline</Link> tab.
      <ErrorAlert error={join.error} />
    </Banner>
  );
}
