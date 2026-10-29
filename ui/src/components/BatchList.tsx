import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { batchProgress } from "../batches";
import { fmtAgo, fmtDate, fmtInt } from "../format";
import type { Batch } from "../types";
import { Chip, ProgressBar, StatusBadge } from "./kit";
import { ErrorAlert } from "./ui";

const SHOWN = 5;
const KIND_LABEL: Record<Batch["kind"], string> = { one_vs_rest: "One per class", search: "Search" };

/** The experiment's batches, newest first, each linking to its results; polled while one runs. */
export function BatchList({ experimentId }: { experimentId: number }) {
  const [all, setAll] = useState(false);
  const batches = useQuery({
    queryKey: ["batches", experimentId],
    queryFn: () => api.batches(experimentId),
    refetchInterval: (q) => ((q.state.data as Batch[] | undefined)?.some((b) => b.status === "running") ? 3000 : false),
  });
  const rows = batches.data ?? [];
  if (!rows.length) return batches.error ? <ErrorAlert error={batches.error} /> : null;
  const shown = all ? rows : rows.slice(0, SHOWN);
  return (
    <section className="card table-wrap batches-card" aria-labelledby="batches-title">
      <div className="batches-head row exp-between">
        <h2 id="batches-title">
          Batches <span className="muted num">{fmtInt(rows.length)}</span>
        </h2>
        {rows.length > SHOWN && (
          <button type="button" className="small ghost" onClick={() => setAll(!all)}>
            {all ? `Show the latest ${SHOWN}` : `Show all ${fmtInt(rows.length)}`}
          </button>
        )}
      </div>
      <ErrorAlert error={batches.error} />
      <table className="data batches">
        <thead>
          <tr>
            <th scope="col">Batch</th>
            <th scope="col">Export</th>
            <th scope="col">Runs</th>
            <th scope="col">Started</th>
            <th scope="col">
              <span className="exp-sr">Status</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {shown.map((b) => {
            const p = batchProgress(b);
            return (
              <tr key={b.id}>
                <td className="batch-name">
                  <div className="row exp-nowrap exp-gap-2">
                    <Link to={`batches/${b.id}`}>{b.name}</Link>
                    <Chip>{KIND_LABEL[b.kind]}</Chip>
                  </div>
                  {b.skipped.length > 0 && (
                    <div className="muted small-text">
                      {fmtInt(b.skipped.length)} class{b.skipped.length === 1 ? "" : "es"} skipped
                    </div>
                  )}
                </td>
                <td className="num secondary">#{b.export_id}</td>
                <td>
                  <span className="batch-runs small-text num">
                    <ProgressBar value={p.ended} max={p.total} label={`${b.name}: ${p.text}`} width={96} />
                    <span>{p.text}</span>
                  </span>
                </td>
                <td className="secondary small-text" title={fmtDate(b.created_at)}>
                  {b.created_by_name ?? "—"} · {fmtAgo(b.created_at)}
                </td>
                <td>
                  <StatusBadge status={b.status} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
