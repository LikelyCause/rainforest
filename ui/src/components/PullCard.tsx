import { Link } from "react-router-dom";
import { useLead } from "./DatasetLayout";
import { NameChips, ReopenedTable } from "./ReuseCard";
import { Banner } from "./kit";
import { ErrorAlert } from "./ui";
import { familyShort, fmtAgo, fmtInt } from "../format";
import { usePulls } from "../hooks";
import { flattenLines, pullHeadline, reopenPath, reopenedNames, uncomparedNote } from "../pulls";
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
      {lead && names.length > 0 && (
        <div>
          <Link className="btn" to={reopenPath(ds.id, family, names)}>
            Start a new version with these fields reopened
          </Link>
        </div>
      )}
    </div>
  );
}

function Summary({ ds, pull }: { ds: Dataset; pull: Pull }) {
  const lines = flattenLines(pull.flatten_notes);
  const families = Object.entries(pull.drift ?? {});
  const note = uncomparedNote(pull, ds.baselines);
  return (
    <section className="card stack pull-card" aria-label="Latest pull">
      <div className="card-head">
        <h2>{pullHeadline(pull)}</h2>
        <span className="muted small-text">
          {pull.finished_at ? fmtAgo(pull.finished_at) : `started ${fmtAgo(pull.started_at)}`}
          {pull.created_by_name ? ` · by ${pull.created_by_name}` : ""}
        </span>
      </div>
      {pull.status === "failed" && (
        <ErrorAlert
          error={{ message: pull.error ?? "The pull failed." }}
          title="This pull failed; the data is as it was."
        />
      )}
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
      {pull.status === "done" && (
        <>
          {families.map(([family, d]) => (
            <FamilyBlock key={family} ds={ds} family={family} drift={d} />
          ))}
          {note && <span className="muted">{note}</span>}
        </>
      )}
    </section>
  );
}

/** The latest pull of a connected dataset: what came in, what flattening did, and how it compares with each
 * finalized baseline. Nothing for a dataset that isn't connected or hasn't pulled. */
export function PullCard({ ds }: { ds: Dataset }) {
  const pulls = usePulls(ds.id, ds.source !== null, ds.status === "refreshing");
  const pull = pulls.data?.[0];
  if (!ds.source || !pull) return <ErrorAlert error={pulls.error} />;
  return <Summary ds={ds} pull={pull} />;
}
