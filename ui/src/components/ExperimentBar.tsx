import { useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useExperiment, useMe } from "../hooks";
import { addBlocked, experimentTab, privacyNote, sourceChip } from "../experiments";
import { Chip, Icon, LensChip, Tabs, useDismiss } from "./kit";
import type { Dataset } from "../types";

/** The name, with a menu of this dataset's other experiments to switch to (on the same tab). */
function ExperimentSwitcher({
  ds,
  experimentId,
  name,
  tab,
}: {
  ds: Dataset;
  experimentId: number;
  name: string;
  tab: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useDismiss(ref, open, () => setOpen(false));
  const list = useQuery({ queryKey: ["experiments", ds.id], queryFn: () => api.experiments(ds.id), enabled: open });
  const shown = (list.data ?? []).filter((x) => !x.archived_at || x.id === experimentId);
  return (
    <div ref={ref} className="exp-switcher">
      <button
        type="button"
        className="ghost exp-name"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen(!open)}
      >
        {name}
        <Icon name="chevron-down" />
      </button>
      {open && (
        <div className="popover exp-switcher-menu" role="menu" aria-label="Switch experiment">
          {list.isPending && <span className="muted small-text">Loading…</span>}
          {shown.map((x) => (
            <Link
              key={x.id}
              role="menuitem"
              to={`/d/${ds.id}/experiments/${x.id}?tab=${tab}`}
              onClick={() => setOpen(false)}
              aria-current={x.id === experimentId ? "true" : undefined}
              className={x.id === experimentId ? "on" : undefined}
            >
              <span className="exp-grow">{x.name}</span>
              <span className="muted small-text num">{x.run_count ?? 0} runs</span>
            </Link>
          ))}
          <hr />
          <Link role="menuitem" to={`/d/${ds.id}/experiments`} onClick={() => setOpen(false)}>
            All experiments
          </Link>
        </div>
      )}
    </div>
  );
}

/** The second row on an experiment's page: where you are in it, its places, what it is built from, and Train. */
export function ExperimentBar({ ds, experimentId }: { ds: Dataset; experimentId: number }) {
  const x = useExperiment(experimentId);
  const me = useMe().data;
  const [params] = useSearchParams();
  const tab = experimentTab(params.get("tab"));
  const base = `/d/${ds.id}/experiments/${experimentId}`;
  const d = x.data;
  const chip = d ? sourceChip(d, me) : null;
  const adjusted = d?.adjustments.count ?? 0;
  const canTrain = Boolean(d && me && !addBlocked(d, me, ds.status === "refreshing"));
  // The same queries the tabs poll, so the counts move as jobs start.
  const runList = useQuery({
    queryKey: ["runs", ds.id, experimentId],
    queryFn: () => api.runs(ds.id, experimentId),
    enabled: !x.error,
  });
  const exportList = useQuery({
    queryKey: ["exports", ds.id, experimentId],
    queryFn: () => api.exports(ds.id, experimentId),
    enabled: !x.error,
  });
  const runs = runList.data?.length ?? d?.runs.length;
  const exports = exportList.data?.length ?? d?.exports.length;
  // Missing, or not shared with this viewer: the page says so; here, only the way back.
  if (x.error)
    return (
      <Link to={`/d/${ds.id}/experiments`} className="secondary small-text">
        Experiments
      </Link>
    );
  return (
    <>
      <Link to={`/d/${ds.id}/experiments`} className="secondary small-text">
        Experiments
      </Link>
      <span className="crumb-sep" aria-hidden="true">
        /
      </span>
      {d ? (
        <ExperimentSwitcher ds={ds} experimentId={experimentId} name={d.name} tab={tab} />
      ) : (
        <b className="version-name">…</b>
      )}
      <Tabs
        label="Experiment"
        className="sub exp-tabs"
        items={[
          { to: `${base}?tab=runs`, label: "Runs", count: runs, active: tab === "runs" },
          // A Viewer can't train, so the tab would hold only a banner.
          ...(me?.role === "viewer" ? [] : [{ to: `${base}?tab=train`, label: "Train", active: tab === "train" }]),
          { to: `${base}?tab=exports`, label: "Exports", count: exports, active: tab === "exports" },
          {
            to: `${base}?tab=adjustments`,
            label: "Adjustments",
            count: d && d.assessor_id === null && !d.is_default ? adjusted : undefined,
            active: tab === "adjustments",
          },
          { to: `${base}?tab=notes`, label: "Notes", active: tab === "notes" },
        ]}
      />
      <span className="spacer" />
      {d?.private && (
        <Chip tone="private" icon="lock" title={privacyNote(d) ?? undefined}>
          Private
        </Chip>
      )}
      {d?.archived_at && <Chip>Archived</Chip>}
      {chip && (
        <LensChip lens={chip.lens} title={chip.title}>
          {chip.text}
        </LensChip>
      )}
      {adjusted > 0 && (
        <Link to={`${base}?tab=adjustments`} className="exp-chip-link">
          <Chip tone="experiment" icon="flask">
            {adjusted} field{adjusted === 1 ? "" : "s"} adjusted
          </Chip>
        </Link>
      )}
      {d?.chosen && (
        <Link
          to={`${base}?tab=runs&run=${d.chosen.run_id}`}
          className="exp-chip-link"
          title={`The chosen model: run #${d.chosen.run_id}${d.chosen.by ? `, chosen by ${d.chosen.by}` : ""}`}
        >
          <Chip tone="good" icon="check">
            Chosen: {d.chosen.model_label ?? d.chosen.model} · run #{d.chosen.run_id}
            {d.chosen.by && <span className="exp-chosen-by"> · by {d.chosen.by}</span>}
          </Chip>
        </Link>
      )}
      {d?.adjustments.targets_changed && (
        <Chip
          tone="warn"
          title="Its targets differ from the baseline's, so its results aren't comparable with experiments on the baseline's targets"
        >
          Targets changed
        </Chip>
      )}
      {canTrain && tab !== "train" && (
        <Link className="btn primary" to={`${base}?tab=train`}>
          <Icon name="play" />
          Train a model
        </Link>
      )}
    </>
  );
}
