import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useFields, useMe, useTeam } from "../hooks";
import { useDatasetContext } from "../components/DatasetLayout";
import { Empty, ErrorAlert } from "../components/ui";
import { ReuseCard } from "../components/ReuseCard";
import { familyShort, fmtAgo, fmtDate, fmtInt, TYPE_SHORT } from "../format";
import { fmtChange, pickable, startLabel } from "../recipes";
import { toast } from "../toast";
import type { Dataset, RecipeVersion, TeamState } from "../types";
import { PhaseBadge, useInvalidateTeam } from "./DatasetTeamPage";

export function VersionsPage() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const team = useTeam(ds.id);
  const versions = useQuery({ queryKey: ["versions", ds.id], queryFn: () => api.versions(ds.id) });
  const lead = me?.role === "lead";
  const t = team.data;
  const canReuse = Boolean(lead && t && t.recipe.version === 1 && t.recipe.phase === "assessing" && !t.recipe.reuse
    && t.assessors.length === 0 && ds.status === "ready");
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Recipe versions {t && <PhaseBadge phase={t.recipe.phase} version={t.recipe.version} />}</h1>
        <Link to={`/d/${ds.id}/team`}>Team page →</Link>
      </div>
      <ErrorAlert error={team.error ?? versions.error} />
      {canReuse && <StartFromRecipe ds={ds} />}
      {t?.recipe.reuse && <ReuseCard reuse={t.recipe.reuse} version={t.recipe.version} />}
      {t && lead && t.recipe.phase === "finalized" && <NewVersion ds={ds} team={t} />}
      {t && t.recipe.phase !== "finalized" && t.recipe.version > 1 && (
        <div className="alert info">
          <span className="grow">
            Recipe v{t.recipe.version} is {t.recipe.phase === "assessing" ? "being assessed" : "in team review"}
            {t.recipe.reopened != null ? `: ${t.recipe.reopened} field${t.recipe.reopened === 1 ? "" : "s"} reopened, the rest carried over and locked.`
              : ": every field is open; a lead edits the team decision directly."} Exports keep using the latest finalized version.
          </span>
          <Link to={`/d/${ds.id}/team`}>Go to the team process →</Link>
        </div>
      )}
      {versions.data && <History versions={versions.data} />}
      {versions.data && versions.data.length > 1 && <Diff ds={ds} versions={versions.data} />}
    </div>
  );
}

function History({ versions }: { versions: RecipeVersion[] }) {
  return (
    <section className="card stack" aria-label="Version history">
      <div className="card-head"><h2>History</h2><span className="muted small-text">Finalized versions are frozen; each change is a new version.</span></div>
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
    </section>
  );
}

function builtFrom(v: RecipeVersion): string {
  if (v.reuse) {
    const s = v.reuse.source;
    return `Reused v${s.version} of ${s.dataset_name ?? `dataset #${s.dataset_id}`}${v.parent_id === null ? " (since deleted)" : ""}`;
  }
  return v.parent_version ? `v${v.parent_version}` : "the suggestions";
}

function Diff({ ds, versions }: { ds: Dataset; versions: RecipeVersion[] }) {
  const [a, setA] = useState(versions[1].version);
  const [b, setB] = useState(versions[0].version);
  const diff = useQuery({ queryKey: ["recipeDiff", ds.id, a, b], queryFn: () => api.recipeDiff(ds.id, a, b), enabled: a !== b });
  const pick = (value: number, set: (n: number) => void, label: string) => (
    <label className="field">{label}
      <select value={value} onChange={(e) => set(Number(e.target.value))}>
        {versions.map((v) => <option key={v.id} value={v.version}>v{v.version}{v.phase === "finalized" ? "" : ` (${v.phase}, as it stands)`}</option>)}
      </select>
    </label>
  );
  const d = diff.data;
  return (
    <section className="card stack" aria-label="Compare versions">
      <div className="card-head"><h2>Compare versions</h2><span className="muted small-text">Team decisions and settings only; assessments are never part of a version.</span></div>
      <div className="row" style={{ alignItems: "flex-end" }}>
        {pick(a, setA, "From")}
        {pick(b, setB, "To")}
      </div>
      <ErrorAlert error={diff.error} />
      {a === b ? <span className="muted small-text">Pick two different versions.</span> : d && (
        d.fields.length === 0 && !d.targets ? <Empty title="No differences">v{a} and v{b} decide every field the same way.</Empty> : (
          <table className="data diff" aria-label="Differences">
            <thead><tr><th>Field</th><th>What changed (v{a} → v{b})</th></tr></thead>
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
                    ) : <span className="badge">{f.change === "added" ? `only in v${b}` : `only in v${a}`}</span>}
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

function NewVersion({ ds, team }: { ds: Dataset; team: TeamState }) {
  const fields = useFields(ds.id);
  const invalidate = useInvalidateTeam(ds.id);
  const [open, setOpen] = useState(false);
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
  const next = team.recipe.version + 1;
  const start = useMutation({
    mutationFn: () => api.newVersion(ds.id, [...chosen]),
    onSuccess: (t) => {
      invalidate();
      setOpen(false);
      toast({ message: t.recipe.phase === "assessing" ? `Recipe v${t.recipe.version} started: assign assessors on the Team page`
        : `Recipe v${t.recipe.version} started: edit the team decision in Review, then finalize` });
    },
  });
  const toggle = (id: number) => setChosen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  if (!open) {
    return (
      <section className="card stack" aria-label="New version">
        <div className="card-head"><h2>Change the recipe</h2></div>
        <span className="secondary">Recipe v{team.recipe.version} is finalized and frozen. To change it, start v{next} from it and pick the fields to reopen: they go through blind assessment and review again, the rest are carried over and locked. Reopen nothing to edit the team decision directly.</span>
        <div><button type="button" className="primary" onClick={() => setOpen(true)}>Start v{next}…</button></div>
      </section>
    );
  }
  return (
    <section className="card stack" aria-label="New version">
      <div className="card-head">
        <h2>Start v{next}: fields to reopen</h2>
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
        <button type="button" className="ghost" onClick={() => setOpen(false)}>Cancel</button>
        <button type="button" className="primary" disabled={start.isPending} onClick={() => start.mutate()}>{startLabel(next, chosen.size)}</button>
      </div>
      <ErrorAlert error={start.error} />
    </section>
  );
}

function StartFromRecipe({ ds }: { ds: Dataset }) {
  const invalidate = useInvalidateTeam(ds.id);
  const recipes = useQuery({ queryKey: ["finalizedRecipes"], queryFn: api.finalizedRecipes });
  const [pick, setPick] = useState("");
  const options = (recipes.data ?? []).filter((r) => r.dataset_id !== ds.id);
  const apply = useMutation({
    mutationFn: () => api.applyRecipe(ds.id, Number(pick)),
    onSuccess: (t) => {
      invalidate();
      toast({ message: t.recipe.phase === "review" ? "Recipe applied: nothing needs assessing. Confirm by finalizing."
        : `Recipe applied: ${t.recipe.reopened} field${t.recipe.reopened === 1 ? "" : "s"} to assess` });
    },
  });
  return (
    <section className="card stack" aria-label="Start from recipe">
      <div className="card-head"><h2>Start from a recipe</h2><span className="muted small-text">Before anyone assesses this dataset</span></div>
      <span className="secondary">Reuse a finalized recipe of any dataset. Fields match by name; those with the same type and no drift are carried over and locked, the rest are assessed as usual.</span>
      {options.length === 0 && recipes.data ? <span className="muted">No finalized recipes yet.</span> : (
        <div className="row">
          <label className="field">Recipe
            <select value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Recipe to start from">
              <option value="">Choose a recipe…</option>
              {options.map((r) => (
                <option key={r.id} value={r.id}>{r.dataset_name} · v{r.version} · {r.field_count} fields · {r.finalized_by_name ?? "—"}, {fmtAgo(r.finalized_at)}</option>
              ))}
            </select>
          </label>
          <button type="button" className="primary" disabled={!pick || apply.isPending} onClick={() => apply.mutate()}>Start from this recipe</button>
        </div>
      )}
      <ErrorAlert error={recipes.error ?? apply.error} />
    </section>
  );
}
