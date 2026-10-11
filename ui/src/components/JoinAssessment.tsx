import { useMutation } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useMe, useTeam } from "../hooks";
import { useInvalidateTeam } from "../pages/DatasetTeamPage";
import type { Dataset } from "../types";
import { ErrorAlert } from "./ui";

/** For a lead looking at a dataset being assessed without being one of its assessors: join with one click. */
export function JoinAssessment({ ds }: { ds: Dataset }) {
  const me = useMe().data;
  const team = useTeam(ds.id, ds.phase === "assessing");
  const invalidate = useInvalidateTeam(ds.id);
  const join = useMutation({ mutationFn: () => api.assign(ds.id, me!.id), onSuccess: invalidate });
  if (ds.baseline?.setup) {
    // Assessing now would fix the first baseline's model type and targets before anyone chose them.
    return (
      <div className="alert info row" style={{ justifyContent: "space-between" }}>
        <span>{me?.role === "lead" ? "Set up this dataset's first baseline: its model type, targets and who assesses it."
          : "A lead sets up this dataset's first baseline (model type, targets, assessors) before anyone assesses it."}</span>
        {me?.role === "lead" && <Link className="btn primary small" to={`/d/${ds.id}/team`}>Set it up</Link>}
      </div>
    );
  }
  if (ds.phase !== "assessing" || me?.role !== "lead" || !team.data || team.data.me) return null;
  return (
    <div className="alert info row" style={{ justifyContent: "space-between" }}>
      <span>You aren't assessing this dataset, so its fields are read-only for you. Assess it yourself, or assign
        others on the Team tab.</span>
      <button type="button" className="primary small" disabled={join.isPending} onClick={() => join.mutate()}>
        Assess it myself
      </button>
      <ErrorAlert error={join.error} />
    </div>
  );
}
