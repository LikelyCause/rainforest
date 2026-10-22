import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useChartTheme } from "../../theme";
import { fmtInt, fmtNum, fmtP, TYPE_SHORT } from "../../format";
import { Chart } from "../Chart";
import { Empty } from "../ui";
import { Banner, Icon } from "../kit";
import { testDetailOption } from "./options";
import { AnalysisState, SkippedList, useAnalysis, type ExploreState } from "./state";
import type { Dataset, TestRow, TestsResult } from "../../types";

const ALPHA = 0.05;
type SortKey = "name" | "test" | "effect" | "p_adj" | "n";

const cmp = (a: number | string | null, b: number | string | null) => {
  if (a === null) return b === null ? 0 : 1;
  if (b === null) return -1;
  return typeof a === "string" ? a.localeCompare(String(b)) : a - (b as number);
};

export function TestsView({ ds, st }: { ds: Dataset; st: ExploreState }) {
  const a = useAnalysis<TestsResult>(ds.id, "tests", st.base);
  return (
    <AnalysisState a={a} datasetId={ds.id}>
      {(r) => <TestsTable ds={ds} r={r} />}
    </AnalysisState>
  );
}

function TestsTable({ ds, r }: { ds: Dataset; r: TestsResult }) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "p_adj", desc: false });
  const [sel, setSel] = useState<number | null>(null);
  const rows = useMemo(() => {
    const out = r.rows.map((row, i) => ({ row, i }));
    out.sort((x, y) => {
      const d = cmp(x.row[sort.key], y.row[sort.key]);
      // Nulls stay last either way; ties fall back to effect size.
      const s = x.row[sort.key] === null || y.row[sort.key] === null ? d : sort.desc ? -d : d;
      return s || cmp(y.row.effect, x.row.effect);
    });
    return out;
  }, [r, sort]);
  const significant = r.rows.filter((x) => x.p_adj !== null && x.p_adj < ALPHA).length;
  const th = (key: SortKey, text: string, num = false) => (
    <th
      className={`sortable${num ? " num" : ""}`}
      aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}
    >
      <button
        type="button"
        className="th-sort"
        onClick={() => setSort({ key, desc: sort.key === key ? !sort.desc : key === "effect" || key === "n" })}
      >
        {text}
        {sort.key === key && (
          <Icon name="chevron-down" size={12} className={sort.desc ? "tools-sort" : "tools-sort up"} />
        )}
      </button>
    </th>
  );
  const selected = sel !== null ? r.rows[sel] : null;

  return (
    <div className="stack tools-gap-12">
      <div className="row secondary small-text">
        <span>
          Each field tested against <b>{r.target}</b> ({r.target_kind}) · <b>{significant}</b> of {r.rows.length}{" "}
          significant at adjusted p &lt; {ALPHA} (Benjamini–Hochberg)
        </span>
      </div>
      {r.excluded_classes.length > 0 && (
        <Banner tone="info">
          Classes with fewer than 5 sampled rows are left out:{" "}
          {r.excluded_classes.map((c, i) => (
            <span key={c.name}>
              {i > 0 && ", "}
              <code>{c.name}</code> ({fmtInt(c.n)})
            </span>
          ))}
        </Banner>
      )}
      {r.rows.length === 0 ? (
        <div className="card">
          <Empty title="No fields could be tested" />
        </div>
      ) : (
        <div className={`explore-split${selected ? " with-panel wide" : ""}`}>
          <div className="table-wrap tools-tests-table">
            <table className="data compact">
              <thead>
                <tr>
                  {th("name", "Field")}
                  {th("test", "Test")}
                  {th("effect", "Effect", true)}
                  {th("p_adj", "Adj. p", true)}
                  {th("n", "n", true)}
                </tr>
              </thead>
              <tbody>
                {rows.map(({ row, i }) => {
                  const sig = row.p_adj !== null && row.p_adj < ALPHA;
                  return (
                    <tr
                      key={i}
                      className={`clickable${sel === i ? " selected" : ""}`}
                      onClick={() => setSel(sel === i ? null : i)}
                      tabIndex={0}
                      onKeyDown={(e) => e.key === "Enter" && setSel(sel === i ? null : i)}
                      aria-selected={sel === i}
                    >
                      <td>
                        <span className="row tools-nowrap tools-gap-6">
                          <span className="tools-ellipsis tools-test-name">{row.name}</span>
                          <span className="type-tag">{TYPE_SHORT[row.type] ?? row.type}</span>
                        </span>
                      </td>
                      <td className="secondary">{row.test}</td>
                      <td className="num">
                        {fmtNum(row.effect, 2)} <span className="muted small-text">{row.effect_name}</span>
                      </td>
                      <td className="num">
                        {fmtP(row.p_adj)}
                        <span
                          className={`sig${sig ? " on" : ""}`}
                          title={sig ? `Significant (adjusted p < ${ALPHA})` : undefined}
                          aria-label={sig ? "significant" : undefined}
                        >
                          {sig ? "●" : ""}
                        </span>
                      </td>
                      <td className="num">{fmtInt(row.n)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {selected && <TestDetailPanel ds={ds} r={r} row={selected} onClose={() => setSel(null)} />}
        </div>
      )}
      <SkippedList skipped={r.skipped} />
    </div>
  );
}

function TestDetailPanel({ ds, r, row, onClose }: { ds: Dataset; r: TestsResult; row: TestRow; onClose: () => void }) {
  const t = useChartTheme();
  const option = useMemo(
    () =>
      row.detail ? testDetailOption(t, row.detail, { field: row.name, target: r.target, classes: r.classes }) : null,
    [t, row, r],
  );
  const caption = !row.detail
    ? null
    : row.detail.kind === "box"
      ? `${r.target_kind === "numeric" ? r.target : row.name} by ${r.target_kind === "numeric" ? row.name : r.target}`
      : row.detail.kind === "levels"
        ? `Share of each ${row.name} level within each ${r.target} class`
        : `${r.target} against ${row.name}`;
  return (
    <aside className="card side-panel stack" aria-label={`Test details for ${row.name}`}>
      <div className="row tools-between tools-nowrap">
        <h2 className="tools-ellipsis">{row.name}</h2>
        <button type="button" className="ghost small tools-icon-btn" aria-label="Close" onClick={onClose}>
          <Icon name="x" />
        </button>
      </div>
      {row.field_id !== null && (
        <Link className="small-text" to={`/d/${ds.id}/review/${row.field_id}`}>
          Open field
        </Link>
      )}
      <div className="kv tools-kv-2">
        <div>
          <span>{row.test}</span>
          <b>{fmtNum(row.statistic)}</b>
        </div>
        <div>
          <span>p</span>
          <b>{fmtP(row.p)}</b>
        </div>
        <div>
          <span>{row.effect_name}</span>
          <b>{fmtNum(row.effect, 3)}</b>
        </div>
        <div>
          <span>Adj. p</span>
          <b>{fmtP(row.p_adj)}</b>
        </div>
        {row.secondary && (
          <>
            <div>
              <span>{row.secondary.test}</span>
              <b>{fmtNum(row.secondary.statistic)}</b>
            </div>
            <div>
              <span>p</span>
              <b>{fmtP(row.secondary.p)}</b>
            </div>
          </>
        )}
        <div>
          <span>Rows</span>
          <b>{fmtInt(row.n)}</b>
        </div>
      </div>
      {option ? (
        <>
          <span className="secondary small-text">{caption}</span>
          <Chart option={option} height={300} ariaLabel={caption ?? "Test detail"} />
        </>
      ) : (
        <span className="muted small-text">No chart for this test.</span>
      )}
    </aside>
  );
}
