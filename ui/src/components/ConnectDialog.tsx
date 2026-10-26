import { useId, useReducer, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { fmtInt, fmtPct } from "../format";
import {
  BLANK_PICK,
  budgetMessage,
  comparedOnText,
  connectBlocker,
  editDraft,
  flattenKindLabel,
  labelOf,
  matchColumns,
  newDraft,
  parseJsonDocument,
  previewKey,
  previewRequest,
  specOf,
  unknownColumns,
  type ConnectEdit,
} from "../sources";
import type { Dataset, FlattenKind, PreviewResult, SourceInfo } from "../types";
import { Banner, Dialog } from "./kit";
import { ErrorAlert } from "./ui";

const RESERVED_LABEL = "__source_rank";
/** Kinds the preview flags: the pull reads them differently from what the source holds. */
const FLAGGED = new Set<FlattenKind>(["mixed", "json"]);

/** An input's validation message, tied to it for assistive tech. */
const invalidProps = (message: string | undefined, id: string) =>
  message ? { "aria-invalid": true, "aria-describedby": id } : {};

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
  const uid = useId();
  const [d, dispatch] = useReducer(editDraft, sources[0]?.name ?? "", newDraft);
  // Kept with the request it answered: an edit that changes the request hides it, one that doesn't (a label) keeps it.
  const [sampled, setSampled] = useState<{ key: string; result: PreviewResult } | null>(null);
  const { connection, database, chosen, picks } = d;

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

  const key = previewKey(d);
  const preview = sampled?.key === key ? sampled.result : null;
  const several = chosen.length > 1;
  const addSource = several || d.addSource;
  const dedupe = d.dedupe ?? several;
  const labels = chosen.map((n) => labelOf(n, picks[n]));
  const name = d.name ?? labels.join(" + ");

  const problems = (() => {
    const out: Record<string, string> = {};
    for (const n of chosen) {
      const p = picks[n] ?? BLANK_PICK;
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

  const run = useMutation({
    mutationFn: (req: { key: string; body: ReturnType<typeof previewRequest> }) =>
      api.sourcePreview(connection, req.body).then((result) => ({ key: req.key, result })),
    onSuccess: setSampled,
  });
  const matchOn = matchColumns(d.matchOn);
  const unknownMatch = unknownColumns(matchOn, preview);
  const connect = useMutation({
    mutationFn: () =>
      api.connectDataset({
        name: name.trim(),
        source: {
          connection,
          database,
          collections: chosen.map((n) => specOf(n, picks[n] ?? BLANK_PICK, true)),
          add_source: addSource,
          dedupe,
          ...(dedupe && matchOn.length > 0 ? { match_on: matchOn } : {}),
        },
      }),
    onSuccess: (ds) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      onCreated(ds);
      onClose();
    },
  });
  const edit = (e: ConnectEdit) => {
    // An error about the collections of another database would read as one about these.
    if (e.type === "connection" || e.type === "database") {
      run.reset();
      connect.reset();
    }
    dispatch(e);
  };

  const flagged = preview?.columns.filter((c) => FLAGGED.has(c.kind)) ?? [];
  const fits = preview?.budget.fits ?? false;
  const canPreview = chosen.length > 0 && !invalid && !run.isPending;
  const blocker = connectBlocker({ chosen: chosen.length, invalid, preview, name, dedupe, unknownMatch });
  const comparedOn = comparedOnText(preview);
  const ids = {
    why: `${uid}-why`,
    name: `${uid}-name`,
    match: `${uid}-match`,
    pick: (i: number, part: string) => `${uid}-${i}-${part}`,
  };
  const nameProblem = chosen.length > 0 && !name.trim() ? "Name the dataset" : undefined;
  const matchProblem = unknownMatch.length > 0 ? `Not in the preview: ${unknownMatch.join(", ")}` : undefined;

  return (
    <Dialog
      title="Connect a MongoDB collection"
      wide
      onClose={onClose}
      labelledBy="connect-title"
      actions={
        <>
          {blocker && (
            <span id={ids.why} className="muted small-text connect-why">
              {blocker}
            </span>
          )}
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            disabled={blocker !== null || connect.isPending}
            aria-describedby={blocker ? ids.why : undefined}
            onClick={() => connect.mutate()}
          >
            Connect
          </button>
        </>
      }
    >
      <div className="connect-body stack">
        <label className="field">
          Connection
          <select value={connection} onChange={(e) => edit({ type: "connection", value: e.target.value })}>
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
            <select value={database} onChange={(e) => edit({ type: "database", value: e.target.value })}>
              <option value="">Choose a database</option>
              {(databases.data?.databases ?? []).map((db) => (
                <option key={db} value={db}>
                  {db}
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
                value={d.typedDatabase}
                onChange={(e) => edit({ type: "typedDatabase", value: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && d.typedDatabase.trim())
                    edit({ type: "database", value: d.typedDatabase.trim() });
                }}
                placeholder="This account can't list databases; type the name"
              />
              <button
                type="button"
                disabled={!d.typedDatabase.trim()}
                onClick={() => edit({ type: "database", value: d.typedDatabase.trim() })}
              >
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
                  <input
                    type="checkbox"
                    checked={chosen.includes(c.name)}
                    onChange={() => edit({ type: "toggle", name: c.name })}
                  />
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
              const p = picks[n] ?? BLANK_PICK;
              const bad = (part: string) => problems[`${n}:${part}`];
              const message = (part: string) =>
                bad(part) && (
                  <span id={ids.pick(i, part)} className="connect-bad">
                    {bad(part)}
                  </span>
                );
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
                        onClick={() => edit({ type: "moveUp", index: i })}
                      >
                        Move up
                      </button>
                    )}
                  </div>
                  <div className="connect-pick-fields">
                    <label className="field">
                      Label
                      <input
                        value={p.label}
                        placeholder={n}
                        onChange={(e) => edit({ type: "pick", name: n, patch: { label: e.target.value } })}
                        {...invalidProps(bad("label"), ids.pick(i, "label"))}
                      />
                      {message("label")}
                    </label>
                    <label className="field">
                      Filter (JSON)
                      <input
                        className="mono"
                        value={p.filter}
                        placeholder='{"year": {"$gte": 2020}}'
                        onChange={(e) => edit({ type: "pick", name: n, patch: { filter: e.target.value } })}
                        {...invalidProps(bad("filter"), ids.pick(i, "filter"))}
                      />
                      {message("filter")}
                    </label>
                    <label className="field">
                      Projection (JSON)
                      <input
                        className="mono"
                        value={p.projection}
                        placeholder='{"notes": 0}'
                        onChange={(e) => edit({ type: "pick", name: n, patch: { projection: e.target.value } })}
                        {...invalidProps(bad("projection"), ids.pick(i, "projection"))}
                      />
                      {message("projection")}
                    </label>
                  </div>
                </div>
              );
            })}
            <div className="row connect-row">
              <button type="button" disabled={!canPreview} onClick={() => run.mutate({ key, body: previewRequest(d) })}>
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
              <input
                value={name}
                onChange={(e) => edit({ type: "name", value: e.target.value })}
                {...invalidProps(nameProblem, ids.name)}
              />
              {nameProblem && (
                <span id={ids.name} className="connect-bad">
                  {nameProblem}
                </span>
              )}
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={addSource}
                disabled={several}
                onChange={(e) => edit({ type: "addSource", value: e.target.checked })}
              />
              Add a <span className="mono">source</span> column naming each row's collection
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={dedupe}
                onChange={(e) => edit({ type: "dedupe", value: e.target.checked })}
              />
              Leave out exact duplicate rows{several ? " across collections" : ""}
            </label>
            {dedupe && (
              <div className="stack connect-match">
                {comparedOn !== null ? (
                  <span className="secondary small-text">
                    Compared on: <span className="mono">{comparedOn}</span>
                  </span>
                ) : (
                  <span className="secondary small-text">
                    Compared on the columns every collection has, less <span className="mono">_id</span>, the{" "}
                    <span className="mono">source</span> column and identifiers.
                  </span>
                )}
                <label className="field">
                  Match on (optional, replaces those columns)
                  <input
                    className="mono"
                    value={d.matchOn}
                    placeholder="Comma-separated columns, e.g. vin, model_year"
                    onChange={(e) => edit({ type: "matchOn", value: e.target.value })}
                    {...invalidProps(matchProblem, ids.match)}
                  />
                  {matchProblem && (
                    <span id={ids.match} className="connect-bad">
                      {matchProblem}
                    </span>
                  )}
                </label>
              </div>
            )}
          </div>
        )}
        <ErrorAlert error={connect.error} title="Couldn't connect." />
      </div>
    </Dialog>
  );
}
