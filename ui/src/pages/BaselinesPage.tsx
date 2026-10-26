import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useFields, useTeam } from "../hooks";
import { useCan, useDatasetContext, useLead } from "../components/DatasetLayout";
import { Empty, ErrorAlert } from "../components/ui";
import { Banner, Chip, Icon } from "../components/kit";
import { PullCard } from "../components/PullCard";
import { ReuseCard } from "../components/ReuseCard";
import { StartBaseline } from "../components/StartBaseline";
import { familyShort, fmtAgo, fmtDate, fmtInt, TYPE_SHORT } from "../format";
import { parseReopen } from "../pulls";
import { builtFrom, fmtChange, pickable, startLabel } from "../recipes";
import { toast } from "../toast";
import type { BaselineLine, Dataset, FamilyKey, RecipeVersion } from "../types";
import { FAMILIES } from "../baselines";
import { PhaseBadge, useInvalidateTeam } from "./DatasetTeamPage";
import { BaselineTabs } from "./BaselinePage";

export function BaselinesPage() {
  const ds = useDatasetContext();
  const team = useTeam(ds.id);
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id) });
  const [params, setParams] = useSearchParams();
  // The Baseline tab links here to start another model type (?start=linear) or a new version (?new=tree).
  const startFamily = FAMILIES.find((f) => f === params.get("start"));
  const newFamily = FAMILIES.find((f) => f === params.get("new"));
  // ?v=<recipe id>: the version switcher (or Browse's way back) brings that version into view.
  const focus = Number(params.get("v")) || null;
  const [starting, setStarting] = useState(Boolean(startFamily));
  const reopenValue = params.get("reopen");
  const reopen = useMemo(() => parseReopen(reopenValue), [reopenValue]);
  const clearParam = (...keys: string[]) => {
    if (keys.some((k) => params.has(k))) {
      const n = new URLSearchParams(params);
      keys.forEach((k) => n.delete(k));
      setParams(n, { replace: true });
    }
  };
  const lead = useLead();
  const canDo = useCan();
  const t = team.data;
  const cur = ds.baseline;
  const inProgress = Boolean(t && t.recipe.phase !== "finalized" && !cur?.setup);
  const canStart = Boolean(lead && ds.status === "ready" && t && (cur?.setup || t.recipe.phase === "finalized"));
  const all = (lines.data ?? []).flatMap((l) => l.versions);
  return (
    <div className="page bls-page">
      <BaselineTabs ds={ds} />
      <div className="bl-title">
        <h1>Versions and compare</h1>
        <p className="secondary">
          A baseline is the team's agreed preparation of this dataset for one model type. Each is assessed blind,
          reviewed and finalized by a lead; a change is a new version. Experiments start from a finalized baseline.
        </p>
      </div>
      <ErrorAlert error={team.error ?? lines.error} />
      {canStart &&
        (starting || cur?.setup ? (
          <StartBaseline
            ds={ds}
            family={startFamily as FamilyKey | undefined}
            onCancel={
              cur?.setup
                ? undefined
                : () => {
                    setStarting(false);
                    clearParam("start");
                  }
            }
          />
        ) : (
          <div>
            <button type="button" className="primary" onClick={() => setStarting(true)}>
              <Icon name="plus" size={14} />
              Start a baseline…
            </button>
          </div>
        ))}
      {cur?.setup && !lead && (
        <Banner tone="info">
          A lead sets up this dataset's first baseline: its model type, targets and assessors.
        </Banner>
      )}
      {t?.recipe.reuse && <ReuseCard reuse={t.recipe.reuse} label={t.recipe.label ?? ""} />}
      <PullCard ds={ds} />
      {t && inProgress && (
        <Banner
          tone="info"
          action={
            <Link className="btn small" to={`/d/${ds.id}/${t.recipe.phase === "review" ? "team" : "baseline"}`}>
              {t.recipe.phase === "review" ? "Open the team review" : "Go to the Baseline tab"}
            </Link>
          }
        >
          {t.recipe.label} is {t.recipe.phase === "assessing" ? "being assessed" : "in team review"}
          {t.recipe.reopened != null
            ? `: ${t.recipe.reopened} field${t.recipe.reopened === 1 ? "" : "s"} reopened, the rest carried over and locked.`
            : "."}{" "}
          Finalize it before starting another baseline or version. Exports keep using finalized versions.
        </Banner>
      )}
      {(lines.data ?? []).map((l) => (
        <BaselineCard
          key={l.family}
          ds={ds}
          line={l}
          focus={focus}
          canChange={Boolean(lead && t?.recipe.phase === "finalized" && l.latest_finalized_id)}
          canRename={canDo("renameBaseline") && !cur?.setup}
          startOpen={newFamily === l.family}
          reopen={newFamily === l.family ? reopen : undefined}
          onClose={() => clearParam("new", "reopen")}
        />
      ))}
      {all.length > 1 && <Diff ds={ds} versions={all} />}
    </div>
  );
}

function BaselineCard({
  ds,
  line,
  focus,
  canChange,
  canRename,
  startOpen,
  reopen,
  onClose,
}: {
  ds: Dataset;
  line: BaselineLine;
  focus: number | null;
  canChange: boolean;
  canRename: boolean;
  startOpen?: boolean;
  reopen?: string[];
  onClose?: () => void;
}) {
  const [changing, setChanging] = useState(Boolean(startOpen));
  useEffect(() => {
    if (startOpen) setChanging(true);
  }, [startOpen]);
  const [renaming, setRenaming] = useState(false);
  const latest = line.versions.find((v) => v.id === line.latest_finalized_id);
  return (
    <section className="card stack" aria-label={`${line.name ?? line.label} baseline`}>
      <div className="card-head">
        {renaming ? (
          <RenameBaseline ds={ds} line={line} onDone={() => setRenaming(false)} />
        ) : (
          <h2 className="row bl-gap-8">
            {line.name ?? `${line.label} baseline`}
            {line.name && <Chip>{line.label}</Chip>}
          </h2>
        )}
        <span className="row bl-gap-12">
          {canRename && !renaming && (
            <button type="button" className="ghost small" onClick={() => setRenaming(true)}>
              {line.name ? "Rename" : "Name it"}
            </button>
          )}
          <span className="muted small-text">
            {line.versions.length} version{line.versions.length === 1 ? "" : "s"}
          </span>
        </span>
      </div>
      <History ds={ds} versions={line.versions} focus={focus} />
      {canChange &&
        latest &&
        (changing ? (
          <NewVersion
            key={reopen?.join(",")}
            ds={ds}
            parent={latest}
            reopen={reopen}
            onCancel={() => {
              setChanging(false);
              onClose?.();
            }}
          />
        ) : (
          <div className="row">
            <span className="secondary grow">
              {latest.label} is finalized and frozen. To change it, start v{line.versions[0].version + 1} and pick the
              fields to reopen.
            </span>
            <button type="button" onClick={() => setChanging(true)}>
              New version…
            </button>
          </div>
        ))}
    </section>
  );
}

/** A lead names or renames a baseline in place; blank removes the name. Labels everywhere follow. */
function RenameBaseline({ ds, line, onDone }: { ds: Dataset; line: BaselineLine; onDone: () => void }) {
  const qc = useQueryClient();
  const [name, setName] = useState(line.name ?? "");
  const save = useMutation({
    mutationFn: () => api.renameBaseline(ds.id, line.family, name),
    onSuccess: () => {
      qc.invalidateQueries();
      toast({
        message: name.trim()
          ? `${line.label} baseline renamed to ${name.trim()}`
          : `${line.label} baseline's name removed`,
      });
      onDone();
    },
  });
  return (
    <form
      className="stack grow bl-gap-6"
      onSubmit={(e) => {
        e.preventDefault();
        save.mutate();
      }}
      onKeyDown={(e) => e.key === "Escape" && onDone()}
    >
      <div className="row bl-gap-8">
        <input
          autoFocus
          aria-label={`Name of the ${line.label} baseline`}
          value={name}
          maxLength={60}
          placeholder={`${line.label} baseline`}
          onChange={(e) => setName(e.target.value)}
          className="bls-name-input"
        />
        <button type="submit" className="primary" disabled={save.isPending}>
          Save
        </button>
        <button type="button" className="ghost" onClick={onDone}>
          Cancel
        </button>
      </div>
      <ErrorAlert error={save.error} />
    </form>
  );
}

/** "30 agreed · 6 settled in review · 1 kept as it was", from what the version's team review left. */
function reviewText(v: RecipeVersion): string | null {
  const r = v.review;
  if (!r) return null;
  const parts = [`${fmtInt(r.adopted)} agreed`];
  if (r.disagreements > 0)
    parts.push(r.settled == null ? `${fmtInt(r.disagreements)} disagreed` : `${fmtInt(r.settled)} settled in review`);
  if (r.open_at_finalize)
    parts.push(`${fmtInt(r.open_at_finalize)} kept as ${r.open_at_finalize === 1 ? "it was" : "they were"}`);
  return parts.join(" · ");
}

function History({ ds, versions, focus }: { ds: Dataset; versions: RecipeVersion[]; focus: number | null }) {
  const focused = useRef<HTMLTableRowElement>(null);
  const has = versions.some((v) => v.id === focus);
  useEffect(() => {
    if (has) focused.current?.scrollIntoView({ block: "center" });
  }, [has]);
  return (
    <div className="table-wrap" aria-label="Version history">
      <table className="data">
        <thead>
          <tr>
            <th>Version</th>
            <th>Phase</th>
            <th>Started</th>
            <th>Finalized</th>
            <th>Fields worked</th>
            <th>Team review</th>
            <th>Built from</th>
            <th aria-label="Browse" />
          </tr>
        </thead>
        <tbody>
          {versions.map((v) => (
            <tr
              key={v.id}
              ref={v.id === focus ? focused : undefined}
              className={v.id === focus ? "bls-focus" : undefined}
              aria-current={v.id === focus ? "true" : undefined}
            >
              <td>
                <b>v{v.version}</b>
              </td>
              <td>
                <PhaseBadge phase={v.phase} />
              </td>
              <td title={fmtDate(v.created_at)}>
                {v.created_by_name ?? "—"} · {fmtAgo(v.created_at)}
              </td>
              <td title={fmtDate(v.finalized_at)}>
                {v.finalized_at ? (
                  `${v.finalized_by_name ?? "—"} · ${fmtAgo(v.finalized_at)}`
                ) : (
                  <span className="muted">—</span>
                )}
              </td>
              <td className="num">
                {v.reopened == null ? `${fmtInt(ds.column_count)} fields` : `${fmtInt(v.reopened)} reopened`}
              </td>
              <td
                className="num secondary"
                title={
                  v.review && v.review.settled == null
                    ? "Finalized before the review was recorded: what the lead settled isn't known"
                    : undefined
                }
              >
                {reviewText(v) ?? <span className="muted">—</span>}
              </td>
              <td className="secondary">{builtFrom(v)}</td>
              <td>
                {v.phase === "finalized" && (
                  <Link
                    className="btn ghost small"
                    to={`/d/${ds.id}/review?view=baseline:${v.id}`}
                    title="Its fields and decisions, read-only"
                  >
                    Browse
                  </Link>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Diff({ ds, versions }: { ds: Dataset; versions: RecipeVersion[] }) {
  const [a, setA] = useState(versions[1].id);
  const [b, setB] = useState(versions[0].id);
  const name = (id: number) => versions.find((v) => v.id === id)?.label ?? `#${id}`;
  const diff = useQuery({
    queryKey: ["recipeDiff", ds.id, a, b],
    queryFn: () => api.recipeDiff(ds.id, a, b),
    enabled: a !== b,
  });
  const pick = (value: number, set: (n: number) => void, label: string) => (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => set(Number(e.target.value))}>
        {versions.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label}
            {v.phase === "finalized" ? "" : ` (${v.phase}, as it stands)`}
          </option>
        ))}
      </select>
    </label>
  );
  const d = diff.data;
  return (
    <section className="card stack" aria-label="Compare versions">
      <div className="card-head">
        <h2>Compare versions</h2>
        <span className="muted small-text">
          Team decisions and settings only (decisions only across model types); assessments are never part of a version.
        </span>
      </div>
      <div className="row bls-pick">
        {pick(a, setA, "From")}
        {pick(b, setB, "To")}
      </div>
      <ErrorAlert error={diff.error} />
      {a === b ? (
        <span className="muted small-text">Pick two different versions.</span>
      ) : (
        d &&
        (d.fields.length === 0 && !d.targets ? (
          <Empty title="No differences">
            {name(a)} and {name(b)} decide every field the same way.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table className="data diff" aria-label="Differences">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>
                    What changed ({name(a)} → {name(b)})
                  </th>
                </tr>
              </thead>
              <tbody>
                {d.targets && (
                  <tr>
                    <td>
                      <b>Targets</b>
                    </td>
                    <td>
                      {d.targets.before.join(", ") || "none"} → {d.targets.after.join(", ") || "none"}
                    </td>
                  </tr>
                )}
                {d.fields.map((f) => (
                  <tr key={f.name}>
                    <td>{f.name}</td>
                    <td>
                      {f.change === "changed" ? (
                        <ul className="plain">
                          {f.changes.map((c) => (
                            <li key={`${c.family}:${c.key}`}>{fmtChange(c, familyShort)}</li>
                          ))}
                        </ul>
                      ) : (
                        <Chip>{f.change === "added" ? `only in ${name(b)}` : `only in ${name(a)}`}</Chip>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))
      )}
    </section>
  );
}

/** `reopen`: field names to start with chosen (from a pull's summary); the rest of the fields are one click away. */
function NewVersion({
  ds,
  parent,
  reopen,
  onCancel,
}: {
  ds: Dataset;
  parent: RecipeVersion;
  reopen?: string[];
  onCancel: () => void;
}) {
  const fields = useFields(ds.id);
  const invalidate = useInvalidateTeam(ds.id);
  const [search, setSearch] = useState("");
  const [types, setTypes] = useState<Set<string>>(new Set());
  const [attention, setAttention] = useState(false);
  const [chosen, setChosen] = useState<Set<number>>(new Set());
  const seeded = useRef(false);
  useEffect(() => {
    if (seeded.current || !reopen?.length || !fields.data) return;
    seeded.current = true;
    const want = new Set(reopen);
    setChosen(new Set(fields.data.filter((f) => want.has(f.name)).map((f) => f.id)));
  }, [fields.data, reopen]);
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
      toast({
        message:
          t.recipe.phase === "assessing"
            ? `${t.recipe.label} started: assign assessors on the Baseline tab`
            : `${t.recipe.label} started: edit the team decision in Fields, then finalize`,
      });
    },
  });
  const toggle = (id: number) =>
    setChosen((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  return (
    <section className="stack" aria-label="New version">
      <div className="card-head">
        <h2>
          Start {familyShort(parent.family)} v{next} from {parent.label}: fields to reopen
        </h2>
        <span className="muted small-text num">
          {fmtInt(chosen.size)} of {fmtInt(all.length)} chosen
        </span>
      </div>
      <div className="row">
        <input
          type="search"
          placeholder="Search fields"
          aria-label="Search fields"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <button
          type="button"
          className={`chip attention${attention ? " on" : ""}`}
          aria-pressed={attention}
          onClick={() => setAttention(!attention)}
        >
          <Icon name="alert" size={13} />
          Needs a look
        </button>
        <div className="chips" role="group" aria-label="Filter by type">
          {typeCounts.map(([t, n]) => (
            <button
              key={t}
              type="button"
              className={`chip${types.has(t) ? " on" : ""}`}
              aria-pressed={types.has(t)}
              onClick={() =>
                setTypes((s) => {
                  const x = new Set(s);
                  if (x.has(t)) x.delete(t);
                  else x.add(t);
                  return x;
                })
              }
            >
              {t.replace("_", " ")} <b className="num">{n}</b>
            </button>
          ))}
        </div>
      </div>
      <div className="row">
        <button
          type="button"
          className="small"
          onClick={() => setChosen((s) => new Set([...s, ...visible.map((f) => f.id)]))}
        >
          Choose all {visible.length} shown
        </button>
        <button type="button" className="small ghost" disabled={chosen.size === 0} onClick={() => setChosen(new Set())}>
          Clear
        </button>
      </div>
      <div className="table-wrap bls-reopen">
        <table className="data" aria-label="Fields">
          <thead>
            <tr>
              <th aria-label="Reopen" className="bls-check" />
              <th>Field</th>
              <th>Type</th>
              <th>Decision</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((f) => (
              <tr key={f.id} onClick={() => toggle(f.id)} className="clickable">
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Reopen ${f.name}`}
                    checked={chosen.has(f.id)}
                    onChange={() => toggle(f.id)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </td>
                <td>{f.name}</td>
                <td>{TYPE_SHORT[f.semantic_type] ?? f.semantic_type}</td>
                <td>{f.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row bls-end">
        <button type="button" className="ghost" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" className="primary" disabled={start.isPending} onClick={() => start.mutate()}>
          {startLabel(next, chosen.size)}
        </button>
      </div>
      <ErrorAlert error={start.error} />
    </section>
  );
}
