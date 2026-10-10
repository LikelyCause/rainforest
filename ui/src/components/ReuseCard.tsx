import { reuseGroups } from "../recipes";
import type { ReuseSummary } from "../types";

/** How a reused recipe matched this dataset: carried, reopened (with why), missing, and the targets. */
export function ReuseCard({ reuse, version }: { reuse: ReuseSummary; version: number }) {
  const g = reuseGroups(reuse);
  const list = (names: string[]) => names.length ? names.map((n) => <code key={n} className="target-code">{n}</code>) : <span className="muted">none</span>;
  return (
    <section className="card stack" aria-label="Reuse summary">
      <div className="card-head">
        <h2>Started from {reuse.source.dataset_name ?? `dataset #${reuse.source.dataset_id}`} · v{reuse.source.version}</h2>
        <span className="muted small-text">Recipe v{version} of this dataset</span>
      </div>
      <div className="stack" style={{ gap: 6 }}>
        <div><b className="num">{g.carried.length}</b> carried over and locked: <span className="chips" style={{ display: "inline-flex" }}>{list(g.carried)}</span></div>
        {g.reopened.length === 0 ? (
          <div className="alert info"><span className="grow">Nothing needs assessing. A lead confirms by finalizing on the Team page.</span></div>
        ) : (
          <div className="stack" style={{ gap: 4 }}>
            <span><b className="num">{reuse.reopened.length}</b> reopened for assessment:</span>
            <table className="data team" aria-label="Reopened fields">
              <thead><tr><th>Reason</th><th>Field</th><th>Details</th></tr></thead>
              <tbody>
                {g.reopened.flatMap((grp) => grp.fields.map((f, i) => (
                  <tr key={f.name}>
                    <td>{i === 0 ? <span className="badge warn">{grp.label}</span> : null}</td>
                    <td className="mono">{f.name}</td>
                    <td className="secondary">{f.detail}</td>
                  </tr>
                )))}
              </tbody>
            </table>
          </div>
        )}
        <div><b className="num">{g.missing.length}</b> in the recipe but not in this data: <span className="chips" style={{ display: "inline-flex" }}>{list(g.missing)}</span></div>
        <div>Targets carried over: <span className="chips" style={{ display: "inline-flex" }}>{list(g.targets)}</span></div>
        {g.targetsNotCarried.map((t) => (
          <div key={t.name} className="alert warn"><span className="grow">Target <code>{t.name}</code> not carried over: {t.problem}</span></div>
        ))}
      </div>
    </section>
  );
}
