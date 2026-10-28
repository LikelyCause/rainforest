import { Link } from "react-router-dom";
import { useLead } from "./DatasetLayout";
import { NameChips, ReopenedTable } from "./ReuseCard";
import { Banner } from "./kit";
import { ErrorAlert } from "./ui";
import { familyShort, fmtAgo, fmtInt } from "../format";
import { usePulls } from "../hooks";
import {
  flattenLines,
  pullHeadline,
  removedFields,
  reopenPath,
  reopenedNames,
  shownPulls,
  uncomparedNote,
  versionInProgress,
} from "../pulls";
import { reuseGroups } from "../recipes";
import type { Dataset, FamilyDrift, Pull } from "../types";

function FamilyBlock({ ds, family, drift }: { ds: Dataset; family: string; drift: FamilyDrift }) {
  const lead = useLead();
  const g = reuseGroups(drift);
  const names = reopenedNames({ [family]: drift }, family);
  return (
    <div className="stack bl-gap-8" role="group" aria-label={`${familyShort(family)} baseline against this pull`}>
      <h3>{familyShort(family)} baseline</h3>
      <div>
        <b className="num">{fmtInt(g.carried.length)}</b> fields unchanged:{" "}
        <span className="chips inline-chips">
          <NameChips names={g.carried} />
        </span>
      </div>
      {g.needsAssessing ? (
        <div className="stack bl-gap-4">
          <span>
            <b className="num">{fmtInt(names.length)}</b> new or drifted, for a new version:
          </span>
          <ReopenedTable groups={g.reopened} />
        </div>
      ) : (
        <Banner tone="info">Nothing new or drifted against this baseline.</Banner>
      )}
      <div>
        <b className="num">{fmtInt(g.missing.length)}</b> in the baseline but no longer in the data:{" "}
        <span className="chips inline-chips">
          <NameChips names={g.missing} />
        </span>
      </div>
      {lead && names.length > 0 && !versionInProgress(ds) && (
        <div>
          <Link className="btn" to={reopenPath(ds.id, family, names)}>
            Start a new version with these fields reopened
          </Link>
        </div>
      )}
    </div>
  );
}

/** A newer pull than the summary's: one that failed (the data in use is still the summary's) or is still running. */
function Notice({ pull }: { pull: Pull }) {
  if (pull.status === "failed")
    return (
      <ErrorAlert
        error={{ message: pull.error ?? "The pull failed." }}
        title={`${pullHeadline(pull)} ${fmtAgo(pull.finished_at ?? pull.started_at)}; the data is as it was.`}
      />
    );
  return (
    <Banner tone="info">
      {pullHeadline(pull)}, started {fmtAgo(pull.started_at)}
      {pull.created_by_name ? ` by ${pull.created_by_name}` : ""}.
    </Banner>
  );
}

function Summary({ ds, pull }: { ds: Dataset; pull: Pull }) {
  const lines = flattenLines(pull.flatten_notes);
  const removed = removedFields(pull);
  const families = Object.entries(pull.drift ?? {});
  const note = uncomparedNote(pull, ds.baselines);
  return (
    <>
      <div className="card-head">
        <h2>{pullHeadline(pull)}</h2>
        <span className="muted small-text">
          {pull.finished_at ? fmtAgo(pull.finished_at) : `started ${fmtAgo(pull.started_at)}`}
          {pull.created_by_name ? ` · by ${pull.created_by_name}` : ""}
        </span>
      </div>
      {pull.per_source && pull.per_source.length > 0 && (
        <ul>
          {pull.per_source.map((s) => (
            <li key={s.label}>
              <b>{s.label}</b>: <span className="num">{fmtInt(s.documents)}</span> documents
              {s.removed_duplicates ? (
                <>
                  , <span className="num">{fmtInt(s.removed_duplicates)}</span> duplicates removed
                </>
              ) : null}
            </li>
          ))}
        </ul>
      )}
      {lines.length > 0 && (
        <ul className="secondary small-text" aria-label="How the documents were flattened">
          {lines.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      )}
      {removed.length > 0 && (
        <details>
          <summary>
            <b className="num">{fmtInt(removed.length)}</b> field{removed.length === 1 ? "" : "s"} removed by this pull
          </summary>
          <span className="chips inline-chips">
            <NameChips names={removed} />
          </span>
        </details>
      )}
      {families.map(([family, d]) => (
        <FamilyBlock key={family} ds={ds} family={family} drift={d} />
      ))}
      {note && <span className="muted">{note}</span>}
    </>
  );
}

/** A connected dataset's pulls: the newest finished one (what came in, what flattening did, how it compares with
 * each finalized baseline), under a notice when a later pull failed or is running. Nothing for a dataset that isn't
 * connected or hasn't pulled. */
export function PullCard({ ds }: { ds: Dataset }) {
  const pulls = usePulls(ds.id, ds.source !== null, ds.status === "refreshing");
  const { summary, notice } = shownPulls(pulls.data);
  if (!ds.source || (!summary && !notice)) return <ErrorAlert error={pulls.error} />;
  return (
    <section className="card stack pull-card" aria-label="Latest pull">
      {notice && <Notice pull={notice} />}
      {summary && <Summary ds={ds} pull={summary} />}
    </section>
  );
}
