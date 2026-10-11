import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useFields, useMe, useTeam } from "../hooks";
import { useDatasetContext } from "../components/DatasetLayout";
import { Empty, ErrorAlert } from "../components/ui";
import { ReuseCard } from "../components/ReuseCard";
import { StartBaseline } from "../components/StartBaseline";
import { baselineLabel, familyShort, fmtAgo, fmtDate, fmtInt, TYPE_SHORT } from "../format";
import { fmtChange, pickable, startLabel } from "../recipes";
import { toast } from "../toast";
import type { BaselineLine, Dataset, RecipeVersion } from "../types";
import { PhaseBadge, useInvalidateTeam } from "./DatasetTeamPage";

export function BaselinesPage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const team = useTeam(ds.id);
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id) });
  const [starting, setStarting] = useState(false);
  const lead = me?.role === "lead";
  const t = team.data;
  const cur = ds.baseline;
  const inProgress = Boolean(t && t.recipe.phase !== "finalized" && !cur?.setup);
  const canStart = Boolean(lead && ds.status === "ready" && t && (cur?.setup || t.recipe.phase === "finalized"));
  const all = (lines.data ?? []).flatMap((l) => l.versions);
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Baselines</h1>
        <Link to={`/d/${ds.id}/team`}>Team page →</Link>
      </div>
      <span className="secondary">
        A baseline is the team's agreed preparation of this dataset for one model type. Each is assessed blind, reviewed
        and finalized by a lead; a change is a new version. Experiments start from a finalized baseline.
      </span>
      <ErrorAlert error={team.error ?? lines.error} />
      {canStart && (starting || cur?.setup ? <StartBaseline ds={ds} onCancel={cur?.setup ? undefined : () => setStarting(false)} /> : (
        <div><button type="button" className="primary" onClick={() => setStarting(true)}>Start a baseline…</button></div>
      ))}
      {cur?.setup && !lead && <div className="alert info">A lead sets up this dataset's first baseline: its model type, targets and assessors.</div>}
      {t?.recipe.reuse && <ReuseCard reuse={t.recipe.reuse} label={t.recipe.label ?? ""} />}
      {t && inProgress && (
        <div className="alert info">
          <span className="grow">
            {t.recipe.label} is {t.recipe.phase === "assessing" ? "being assessed" : "in team review"}
            {t.recipe.reopened != null ? `: ${t.recipe.reopened} field${t.recipe.reopened === 1 ? "" : "s"} reopened, the rest carried over and locked.`
              : "."} Finalize it before starting another baseline or version. Exports keep using finalized versions.
          </span>
          <Link to={`/d/${ds.id}/team`}>Go to the team process →</Link>
        </div>
      )}
      {(lines.data ?? []).map((l) => (
        <BaselineCard key={l.family} ds={ds} line={l} canChange={Boolean(lead && t?.recipe.phase === "finalized" && l.latest_finalized_id)} />
      ))}
      {all.length > 1 && <Diff ds={ds} versions={all} />}
    </div>
  );
}

function BaselineCard({ ds, line, canChange }: { ds: Dataset; line: BaselineLine; canChange: boolean }) {
  const [changing, setChanging] = useState(false);
  const latest = line.versions.find((v) => v.id === line.latest_finalized_id);
  return (
    <section className="card stack" aria-label={`${line.label} baseline`}>
      <div className="card-head">
        <h2>{line.label} baseline</h2>
        <span className="muted small-text">{line.versions.length} version{line.versions.length === 1 ? "" : "s"}</span>
      </div>
      <History versions={line.versions} />
      {canChange && latest && (changing ? <NewVersion ds={ds} parent={latest} onCancel={() => setChanging(false)} /> : (
        <div className="row">
          <span className="secondary grow">{latest.label} is finalized and frozen. To change it, start v{line.versions[0].version + 1} and pick the fields to reopen.</span>
          <button type="button" onClick={() => setChanging(true)}>New version…</button>
        </div>
      ))}
    </section>
  );
}

function History({ versions }: { versions: RecipeVersion[] }) {
  return (
    <div className="table-wrap" aria-label="Version history">
      <table className="data team">
        <thead><tr><th>Version</th><th>Phase</th><th>Started</th><th>Finalized</th><th>Fields worked</th><th>Built from</th></tr></thead>
        <tbody>
          {versions.map((v) => (
            <tr key={v.id}>
              <td><b>v{v.version}</b></td>
              <td><PhaseBadge phase={v.phase} /></td>
              <td title={fmtDate(v.created_at)}>{v.created_by_name ?? "—"} · {fmtAgo(v.created_at)}</td>
              <td title={fmtDate(v.finalized_at)}>{v.finalized_at ? `${v.finalized_by_name ?? "—"} · ${fmtAgo(v.finalized_at)}` : <span className="muted">—</span>}</td>
              <td className="num">{v.reopened == null ? "all fields" : `${fmtInt(v.reopened)} reopened`}</td>
              <td className="secondary">{builtFrom(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function builtFrom(v: RecipeVersion): string {
  if (v.reuse) {
    const s = v.reuse.source;
    return `Reused ${s.family ? baselineLabel(s.family, s.version, false, s.name) : `v${s.version}`} of ${s.dataset_name ?? `dataset #${s.dataset_id}`}${v.parent_id === null ? " (since deleted)" : ""}`;
  }
  if (v.parent_version && v.parent_family) return baselineLabel(v.parent_family, v.parent_version, false, v.parent_name);
  return "the suggestions";
}

function Diff({ ds, versions }: { ds: Dataset; versions: RecipeVersion[] }) {
  const [a, setA] = useState(versions[1].id);
  const [b, setB] = useState(versions[0].id);
  const name = (id: number) => versions.find((v) => v.id === id)?.label ?? `#${id}`;
  const diff = useQuery({ queryKey: ["recipeDiff", ds.id, a, b], queryFn: () => api.recipeDiff(ds.id, a, b), enabled: a !== b });
  const pick = (value: number, set: (n: number) => void, label: string) => (
    <label className="field">{label}
      <select value={value} onChange={(e) => set(Number(e.target.value))}>
        {versions.map((v) => <option key={v.id} value={v.id}>{v.label}{v.phase === "finalized" ? "" : ` (${v.phase}, as it stands)`}</option>)}
      </select>
    </label>
  );
  const d = diff.data;
  return (
    <section className="card stack" aria-label="Compare versions">
      <div className="card-head"><h2>Compare versions</h2><span className="muted small-text">Team decisions and settings only (decisions only across model types); assessments are never part of a version.</span></div>
      <div className="row" style={{ alignItems: "flex-end" }}>
        {pick(a, setA, "From")}
        {pick(b, setB, "To")}
      </div>
      <ErrorAlert error={diff.error} />
      {a === b ? <span className="muted small-text">Pick two different versions.</span> : d && (
        d.fields.length === 0 && !d.targets ? <Empty title="No differences">{name(a)} and {name(b)} decide every field the same way.</Empty> : (
          <table className="data diff" aria-label="Differences">
            <thead><tr><th>Field</th><th>What changed ({name(a)} → {name(b)})</th></tr></thead>
            <tbody>
              {d.targets && (
                <tr><td><b>Targets</b></td><td>{d.targets.before.join(", ") || "none"} → {d.targets.after.join(", ") || "none"}</td></tr>
              )}
              {d.fields.map((f) => (
                <tr key={f.name}>
                  <td className="mono">{f.name}</td>
                  <td>
                    {f.change === "changed" ? (
                      <ul className="plain">{f.changes.map((c) => <li key={`${c.family}:${c.key}`}>{fmtChange(c, familyShort)}</li>)}</ul>
                    ) : <span className="badge">{f.change === "added" ? `only in ${name(b)}` : `only in ${name(a)}`}</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      )}
    </section>
  );
}

function NewVersion({ ds, parent, onCancel }: { ds: Dataset; parent: RecipeVersion; onCancel: () => void }) {
  const fields = useFields(ds.id);
  const invalidate = useInvalidateTeam(ds.id);
  const [search, setSearch] = useState("");
  const [types, setTypes] = useState<Set<string>>(new Set());
  const [attention, setAttention] = useState(false);
  const [chosen, setChosen] = useState<Set<number>>(new Set());
  const all = fields.data ?? [];
  const visible = useMemo(() => pickable(all, { search, types, attention }), [all, search, types, attention]);
  const typeCounts = useMemo(() => {
    const m = new Map<string, number>();
    all.forEach((f) => m.set(f.semantic_type, (m.get(f.semantic_type) ?? 0) + 1));
    return [...m.entries()].sort((x, y) => y[1] - x[1]);
  }, [all]);
  const next = (ds.baselines.find((b) => b.family === parent.family)?.version ?? parent.version) + 1;
  const start = useMutation({
    mutationFn: () => api.newVersion(ds.id, parent.family, [...chosen]),
    onSuccess: (t) => {
      invalidate();
      onCancel();
      toast({ message: t.recipe.phase === "assessing" ? `${t.recipe.label} started: assign assessors on the Team page`
        : `${t.recipe.label} started: edit the team decision in Review, then finalize` });
    },
  });
  const toggle = (id: number) => setChosen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  return (
    <section className="stack" aria-label="New version">
      <div className="card-head">
        <h2>Start {familyShort(parent.family)} v{next} from {parent.label}: fields to reopen</h2>
        <span className="muted small-text num">{fmtInt(chosen.size)} of {fmtInt(all.length)} chosen</span>
      </div>
      <div className="row">
        <input type="search" placeholder="Search fields" aria-label="Search fields" value={search} onChange={(e) => setSearch(e.target.value)} />
        <button type="button" className={`chip attention${attention ? " on" : ""}`} aria-pressed={attention} onClick={() => setAttention(!attention)}>⚑ Needs attention</button>
        <div className="chips" role="group" aria-label="Filter by type">
          {typeCounts.map(([t, n]) => (
            <button key={t} type="button" className={`chip${types.has(t) ? " on" : ""}`} aria-pressed={types.has(t)}
              onClick={() => setTypes((s) => { const x = new Set(s); if (x.has(t)) x.delete(t); else x.add(t); return x; })}>
              {t.replace("_", " ")} <b className="num">{n}</b>
            </button>
          ))}
        </div>
      </div>
      <div className="row">
        <button type="button" className="small" onClick={() => setChosen((s) => new Set([...s, ...visible.map((f) => f.id)]))}>Choose all {visible.length} shown</button>
        <button type="button" className="small ghost" disabled={chosen.size === 0} onClick={() => setChosen(new Set())}>Clear</button>
      </div>
      <div className="table-wrap" style={{ maxHeight: 320 }}>
        <table className="data team" aria-label="Fields">
          <thead><tr><th aria-label="Reopen" style={{ width: 36 }} /><th>Field</th><th>Type</th><th>Decision</th></tr></thead>
          <tbody>
            {visible.map((f) => (
              <tr key={f.id} onClick={() => toggle(f.id)} style={{ cursor: "pointer" }}>
                <td><input type="checkbox" aria-label={`Reopen ${f.name}`} checked={chosen.has(f.id)} onChange={() => toggle(f.id)} onClick={(e) => e.stopPropagation()} /></td>
                <td className="mono">{f.name}</td>
                <td>{TYPE_SHORT[f.semantic_type] ?? f.semantic_type}</td>
                <td>{f.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ justifyContent: "flex-end" }}>
        <button type="button" className="ghost" onClick={onCancel}>Cancel</button>
        <button type="button" className="primary" disabled={start.isPending} onClick={() => start.mutate()}>{startLabel(next, chosen.size)}</button>
      </div>
      <ErrorAlert error={start.error} />
    </section>
  );
}
