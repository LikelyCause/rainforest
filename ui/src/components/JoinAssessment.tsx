import { useMutation } from "@tanstack/react-query";
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
