import { baselineLabel } from "../format";
import { reuseGroups } from "../recipes";
import type { ReuseSummary } from "../types";
import { Banner, Chip } from "./kit";

export const NameChips = ({ names }: { names: string[] }) =>
  names.length ? (
    <>
      {names.map((n) => (
        <span key={n} className="name-chip">
          {n}
        </span>
      ))}
    </>
  ) : (
    <span className="muted">none</span>
  );

/** The reopened fields grouped by why, with the numbers behind each. */
export function ReopenedTable({ groups }: { groups: ReturnType<typeof reuseGroups>["reopened"] }) {
  return (
    <div className="table-wrap">
      <table className="data" aria-label="Reopened fields">
        <thead>
          <tr>
            <th>Reason</th>
            <th>Field</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {groups.flatMap((grp) =>
            grp.fields.map((f, i) => (
              <tr key={f.name}>
                <td>{i === 0 ? <Chip tone="warn">{grp.label}</Chip> : null}</td>
                <td>{f.name}</td>
                <td className="secondary">{f.detail}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}

/** How a reused recipe matched this dataset: carried, reopened (with why), missing, and the targets. */
export function ReuseCard({ reuse, label }: { reuse: ReuseSummary; label: string }) {
  const g = reuseGroups(reuse);
  const list = (names: string[]) => <NameChips names={names} />;
  return (
    <section className="card stack reuse-card" aria-label="Reuse summary">
      <div className="card-head">
        <h2>
          Started from {reuse.source.dataset_name ?? `dataset #${reuse.source.dataset_id}`} ·{" "}
          {reuse.source.family
            ? baselineLabel(reuse.source.family, reuse.source.version, false, reuse.source.name)
            : `v${reuse.source.version}`}
        </h2>
        <span className="muted small-text">{label} of this dataset</span>
      </div>
      <div className="stack bl-gap-8">
        <div>
          <b className="num">{g.carried.length}</b> carried over and locked:{" "}
          <span className="chips inline-chips">{list(g.carried)}</span>
        </div>
        {g.reopened.length === 0 ? (
          <Banner tone="info">Nothing needs assessing. A lead confirms by finalizing it in the team review.</Banner>
        ) : (
          <div className="stack bl-gap-4">
            <span>
              <b className="num">{reuse.reopened.length}</b> reopened for assessment:
            </span>
            <ReopenedTable groups={g.reopened} />
          </div>
        )}
        <div>
          <b className="num">{g.missing.length}</b> in the recipe but not in this data:{" "}
          <span className="chips inline-chips">{list(g.missing)}</span>
        </div>
        <div>
          Targets carried over: <span className="chips inline-chips">{list(g.targets)}</span>
        </div>
        {g.targetsNotCarried.map((t) => (
          <Banner key={t.name} tone="warn">
            Target <b>{t.name}</b> not carried over: {t.problem}
          </Banner>
        ))}
      </div>
    </section>
  );
}
