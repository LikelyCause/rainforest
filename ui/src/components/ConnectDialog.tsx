import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { fmtInt, fmtPct } from "../format";
import { budgetMessage, flattenKindLabel, parseJsonDocument } from "../sources";
import type { CollectionSpec, Dataset, FlattenKind, PreviewResult, SourceInfo } from "../types";
import { Banner, Dialog } from "./kit";
import { ErrorAlert } from "./ui";

const RESERVED_LABEL = "__source_rank";
/** Kinds the preview flags: the pull reads them differently from what the source holds. */
const FLAGGED = new Set<FlattenKind>(["mixed", "json"]);

interface Pick {
  label: string;
  filter: string;
  projection: string;
}
const BLANK: Pick = { label: "", filter: "", projection: "" };

/** What the preview and the pull send for a collection: blank filters and projections are left out. */
function specOf(name: string, pick: Pick, withLabel: boolean): CollectionSpec {
  const spec: CollectionSpec = { name };
  if (withLabel && pick.label.trim()) spec.label = pick.label.trim();
  if (pick.filter.trim()) spec.filter = pick.filter.trim();
  if (pick.projection.trim()) spec.projection = pick.projection.trim();
  return spec;
}

const labelOf = (name: string, pick: Pick | undefined) => pick?.label.trim() || name;

export function ConnectDialog({
  sources,
  onClose,
  onCreated,
}: {
  sources: SourceInfo[];
  onClose: () => void;
  onCreated: (ds: Dataset) => void;
}) {
  const qc = useQueryClient();
  const [connection, setConnection] = useState(sources[0]?.name ?? "");
  const [database, setDatabase] = useState("");
  const [typedDatabase, setTypedDatabase] = useState("");
  // The order collections are ticked in is the order they are pulled and the order ties are broken in.
  const [chosen, setChosen] = useState<string[]>([]);
  const [picks, setPicks] = useState<Record<string, Pick>>({});
  const [nameEdit, setNameEdit] = useState<string | null>(null);
  const [addSourceEdit, setAddSourceEdit] = useState(false);
  const [dedupeEdit, setDedupeEdit] = useState<boolean | null>(null);
  const [matchOn, setMatchOn] = useState("");
  const [preview, setPreview] = useState<PreviewResult | null>(null);

  const databases = useQuery({
    queryKey: ["source-databases", connection],
    queryFn: () => api.sourceDatabases(connection),
    enabled: !!connection,
    staleTime: 60_000,
  });
  const listable = databases.data?.listable ?? true;
  const collections = useQuery({
    queryKey: ["source-collections", connection, database],
    queryFn: () => api.sourceCollections(connection, database),
    enabled: !!connection && !!database,
    staleTime: 60_000,
  });

  const several = chosen.length > 1;
  const addSource = several || addSourceEdit;
  const dedupe = dedupeEdit ?? several;
  const labels = chosen.map((n) => labelOf(n, picks[n]));
  const name = nameEdit ?? labels.join(" + ");

  const problems = (() => {
    const out: Record<string, string> = {};
    for (const n of chosen) {
      const p = picks[n] ?? BLANK;
      const f = parseJsonDocument(p.filter);
      const pr = parseJsonDocument(p.projection);
      if (f) out[`${n}:filter`] = f;
      if (pr) out[`${n}:projection`] = pr;
      const label = labelOf(n, p);
      if (label === RESERVED_LABEL) out[`${n}:label`] = `${RESERVED_LABEL} is reserved`;
      else if (labels.indexOf(label) !== labels.lastIndexOf(label)) out[`${n}:label`] = "Labels must be unique";
    }
    return out;
  })();
  const invalid = Object.keys(problems).length > 0;

  const stale = () => setPreview(null);
  const pick = (n: string, patch: Partial<Pick>) => {
    setPicks({ ...picks, [n]: { ...(picks[n] ?? BLANK), ...patch } });
    stale();
  };
  const pickConnection = (c: string) => {
    setConnection(c);
    setDatabase("");
    setTypedDatabase("");
    setChosen([]);
    stale();
  };
  const pickDatabase = (d: string) => {
    setDatabase(d);
    setChosen([]);
    stale();
  };
  const toggle = (n: string) => {
    setChosen(chosen.includes(n) ? chosen.filter((c) => c !== n) : [...chosen, n]);
    stale();
  };
  const moveUp = (i: number) => {
    const next = [...chosen];
    [next[i - 1], next[i]] = [next[i], next[i - 1]];
    setChosen(next);
    stale();
  };

  const run = useMutation({
    mutationFn: () =>
      api.sourcePreview(connection, { database, collections: chosen.map((n) => specOf(n, picks[n] ?? BLANK, false)) }),
    onSuccess: setPreview,
  });
  const matchColumns = matchOn
    .split(",")
    .map((c) => c.trim())
    .filter(Boolean);
  const unknownMatch = preview ? matchColumns.filter((c) => !preview.columns.some((col) => col.name === c)) : [];
  const connect = useMutation({
    mutationFn: () =>
      api.connectDataset({
        name: name.trim(),
        source: {
          connection,
          database,
          collections: chosen.map((n) => specOf(n, picks[n] ?? BLANK, true)),
          add_source: addSource,
          dedupe,
          ...(dedupe && matchColumns.length > 0 ? { match_on: matchColumns } : {}),
        },
      }),
    onSuccess: (ds) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      onCreated(ds);
      onClose();
    },
  });

  const flagged = preview?.columns.filter((c) => FLAGGED.has(c.kind)) ?? [];
  const fits = preview?.budget.fits ?? false;
  const canPreview = chosen.length > 0 && !invalid && !run.isPending;
  const canConnect = !!preview && fits && !invalid && name.trim() !== "" && !connect.isPending;

  return (
    <Dialog
      title="Connect a MongoDB collection"
      wide
      onClose={onClose}
      labelledBy="connect-title"
      actions={
        <>
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="primary" disabled={!canConnect} onClick={() => connect.mutate()}>
            Connect
          </button>
        </>
      }
    >
      <div className="connect-body stack">
        <label className="field">
          Connection
          <select value={connection} onChange={(e) => pickConnection(e.target.value)}>
            {sources.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </label>

        {databases.isLoading ? (
          <span className="muted small-text">Loading databases…</span>
        ) : listable ? (
          <label className="field">
            Database
            <select value={database} onChange={(e) => pickDatabase(e.target.value)}>
              <option value="">Choose a database</option>
              {(databases.data?.databases ?? []).map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <div className="field">
            <label htmlFor="connect-db">Database</label>
            <div className="row connect-row">
              <input
                id="connect-db"
                className="home-grow"
                value={typedDatabase}
                onChange={(e) => setTypedDatabase(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && typedDatabase.trim()) pickDatabase(typedDatabase.trim());
                }}
                placeholder="This account can't list databases; type the name"
              />
              <button type="button" disabled={!typedDatabase.trim()} onClick={() => pickDatabase(typedDatabase.trim())}>
                Load collections
              </button>
            </div>
          </div>
        )}
        <ErrorAlert error={databases.error} />

        {database && (
          <fieldset className="connect-collections">
            <legend className="field-label">Collections</legend>
            {collections.isLoading && <span className="muted small-text">Loading collections…</span>}
            {collections.data?.length === 0 && <span className="muted small-text">No collections in {database}.</span>}
            <div className="connect-list">
              {(collections.data ?? []).map((c) => (
                <label key={c.name} className="check connect-item">
                  <input type="checkbox" checked={chosen.includes(c.name)} onChange={() => toggle(c.name)} />
                  <span className="mono connect-name">{c.name}</span>
                  <span className="muted small-text num">{fmtInt(c.documents)} documents</span>
                </label>
              ))}
            </div>
            <ErrorAlert error={collections.error} />
          </fieldset>
        )}

        {chosen.length > 0 && (
          <div className="stack connect-picks">
            {several && (
              <span className="secondary small-text">
                Collections are pulled in this order. List the most trusted first: when duplicates are left out, the row
                from the collection listed first is kept.
              </span>
            )}
            {chosen.map((n, i) => {
              const p = picks[n] ?? BLANK;
              return (
                <div key={n} className="card connect-pick">
                  <div className="row connect-pick-head">
                    <b className="mono">{n}</b>
                    {several && (
                      <button
                        type="button"
                        className="ghost small"
                        aria-label={`Move ${n} up`}
                        disabled={i === 0}
                        onClick={() => moveUp(i)}
                      >
                        Move up
                      </button>
                    )}
                  </div>
                  <div className="connect-pick-fields">
                    <label className="field">
                      Label
                      <input value={p.label} placeholder={n} onChange={(e) => pick(n, { label: e.target.value })} />
                      {problems[`${n}:label`] && <span className="connect-bad">{problems[`${n}:label`]}</span>}
                    </label>
                    <label className="field">
                      Filter (JSON)
                      <input
                        className="mono"
                        value={p.filter}
                        placeholder='{"year": {"$gte": 2020}}'
                        onChange={(e) => pick(n, { filter: e.target.value })}
                      />
                      {problems[`${n}:filter`] && <span className="connect-bad">{problems[`${n}:filter`]}</span>}
                    </label>
                    <label className="field">
                      Projection (JSON)
                      <input
                        className="mono"
                        value={p.projection}
                        placeholder='{"notes": 0}'
                        onChange={(e) => pick(n, { projection: e.target.value })}
                      />
                      {problems[`${n}:projection`] && (
                        <span className="connect-bad">{problems[`${n}:projection`]}</span>
                      )}
                    </label>
                  </div>
                </div>
              );
            })}
            <div className="row connect-row">
              <button type="button" disabled={!canPreview} onClick={() => run.mutate()}>
                {run.isPending ? "Sampling…" : preview ? "Preview again" : "Preview"}
              </button>
              <span className="muted small-text">Samples 200 documents per collection.</span>
            </div>
            <ErrorAlert error={run.error} />
          </div>
        )}

        {preview && (
          <div className="stack connect-preview">
            <span className="num">
              {fmtInt(preview.columns.length)} columns · about {fmtInt(preview.estimated_rows)} rows ·{" "}
              {budgetMessage(preview.estimated_cells, preview.budget.max_cells)}
            </span>
            {!fits && (
              <Banner tone="bad" title="Too large to pull">
                Narrow the filter or add a projection to take fewer documents or fields.
              </Banner>
            )}
            {flagged.length > 0 && (
              <Banner tone="warn">
                {flagged.length} column{flagged.length === 1 ? "" : "s"} read differently from the source:{" "}
                {flagged
                  .slice(0, 6)
                  .map((c) => c.name)
                  .join(", ")}
                {flagged.length > 6 ? ` and ${flagged.length - 6} more` : ""}.
              </Banner>
            )}
            <div className="connect-columns" tabIndex={0} role="region" aria-label="Columns the pull will create">
              <table className="data">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Read as</th>
                    <th className="num">Present</th>
                  </tr>
                </thead>
                <tbody>
                  {preview.columns.map((c) => (
                    <tr key={c.name}>
                      <td className="mono">{c.name}</td>
                      <td className={FLAGGED.has(c.kind) ? "connect-bad" : "secondary"}>{flattenKindLabel(c.kind)}</td>
                      <td className="num">{fmtPct(c.present_share, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {chosen.length > 0 && (
          <div className="stack connect-options">
            <label className="field">
              Dataset name
              <input value={name} onChange={(e) => setNameEdit(e.target.value)} />
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={addSource}
                disabled={several}
                onChange={(e) => setAddSourceEdit(e.target.checked)}
              />
              Add a <span className="mono">source</span> column naming each row's collection
            </label>
            <label className="check">
              <input type="checkbox" checked={dedupe} onChange={(e) => setDedupeEdit(e.target.checked)} />
              Leave out exact duplicate rows{several ? " across collections" : ""}
            </label>
            {dedupe && (
              <div className="stack connect-match">
                {preview?.compared_on ? (
                  <span className="secondary small-text">
                    Compared on: <span className="mono">{preview.compared_on.join(", ") || "no shared columns"}</span>
                  </span>
                ) : (
                  <span className="secondary small-text">
                    Compared on the columns every collection has, less <span className="mono">_id</span>, the{" "}
                    <span className="mono">source</span> column and identifiers.
                  </span>
                )}
                <label className="field">
                  Match on (optional)
                  <input
                    className="mono"
                    value={matchOn}
                    placeholder="Comma-separated columns, e.g. vin, model_year"
                    onChange={(e) => setMatchOn(e.target.value)}
                  />
                  {unknownMatch.length > 0 && (
                    <span className="connect-bad">Not in the preview: {unknownMatch.join(", ")}</span>
                  )}
                </label>
              </div>
            )}
          </div>
        )}
        <ErrorAlert error={connect.error} />
      </div>
    </Dialog>
  );
}
