import { useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, deploymentTestDownload, uploadDeploymentTest } from "../api";
import { useDatasetContext } from "../components/DatasetLayout";
import { Banner, Chip, confirmDialog, Icon, ProgressBar, StatusBadge } from "../components/kit";
import { ConfusionTable } from "../components/RunDetail";
import { CopyCommand, ErrorAlert } from "../components/ui";
import { curlExample, fieldWarnings, modelThreshold, scoreColumns } from "../deployments";
import { addBlocked } from "../experiments";
import { fmtAgo, fmtDate, fmtInt, fmtMetric, fmtPct } from "../format";
import { useMe } from "../hooks";
import { isActive } from "../runs";
import { toast } from "../toast";
import type {
  DeploymentDetail,
  DeploymentKey,
  DeploymentTest,
  DeploymentTestDetail,
  NewDeploymentKey,
  TestMetrics,
} from "../types";
import { DeploymentStatus } from "./DeploymentsPage";

const ACCEPT = ".csv,.tsv,.txt,.parquet,.pq";

function Models({ d, runHref }: { d: DeploymentDetail; runHref: (runId: number) => string }) {
  const binary = d.models.some((m) => m.kind === "binary");
  return (
    <section className="card table-wrap batches-card" aria-labelledby="models-title">
      <div className="batches-head row exp-between">
        <h2 id="models-title">
          Models <span className="muted num">{fmtInt(d.models.length)}</span>
        </h2>
        <span className="muted small-text">Export #{d.export_id}</span>
      </div>
      <div className="exp-scroll-x deploy-models">
        <table className="data batches">
          <thead>
            <tr>
              <th scope="col">{binary ? "Class" : "Target"}</th>
              <th scope="col">Run</th>
              <th scope="col" className={binary ? "num" : undefined}>
                {binary ? "Threshold" : "Predicts"}
              </th>
              {binary && <th scope="col">Chosen by</th>}
            </tr>
          </thead>
          <tbody>
            {d.models.map((m) => {
              const t = modelThreshold(m);
              return (
                <tr key={m.run_id}>
                  <td className="batch-class-name">
                    <b>{m.label}</b>
                  </td>
                  <td className="batch-nowrap">
                    <Link to={runHref(m.run_id)} title={m.model_label ?? undefined}>
                      Run {m.run_id}
                    </Link>
                  </td>
                  <td className={binary ? "num" : undefined}>{t.value}</td>
                  {binary && (
                    <td className="small-text secondary batch-note-text" title={m.threshold_note}>
                      {t.fallback && <Icon name="alert" size={13} />} {t.rule}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="batch-foot muted small-text">
        <p>
          {binary
            ? "Each model scores its class against the rest. Scores are calibrated probabilities; a record matches a class when its score is at or above that model's threshold, and its top class is the highest score."
            : d.models[0]?.kind === "regression"
              ? "The run predicts a number for each record."
              : "The run gives each class's probability; a record's top class is the most likely one."}
        </p>
      </div>
    </section>
  );
}

function Keys({ d, locked }: { d: DeploymentDetail; locked: boolean }) {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [fresh, setFresh] = useState<NewDeploymentKey | null>(null);
  const keys = useQuery({ queryKey: ["deploymentKeys", d.id], queryFn: () => api.deploymentKeys(d.id) });
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["deploymentKeys", d.id] });
    qc.invalidateQueries({ queryKey: ["deployment", d.id] });
    qc.invalidateQueries({ queryKey: ["deployments", d.dataset_id] });
  };
  const create = useMutation({
    mutationFn: () => api.createDeploymentKey(d.id, name.trim()),
    // The key is shown once: it mustn't outlive the banner in the mutation cache.
    gcTime: 0,
    onSuccess: (k) => {
      setFresh(k);
      setName("");
      refresh();
    },
  });
  const revoke = useMutation({
    mutationFn: (k: DeploymentKey) => api.revokeDeploymentKey(d.id, k.id),
    onSuccess: (k) => {
      toast({ message: `Revoked ${k.name}`, detail: "Calls with it are refused from now on." });
      refresh();
    },
  });
  const onRevoke = async (k: DeploymentKey) => {
    const ok = await confirmDialog({
      title: `Revoke ${k.name}?`,
      body: `Calls with ${k.prefix}… are refused from now on. This can't be undone; create another key to call it again.`,
      confirm: "Revoke the key",
      danger: true,
    });
    if (ok) revoke.mutate(k);
  };
  const list = keys.data ?? [];
  return (
    <section className="card stack" aria-labelledby="keys-title">
      <div className="row exp-between">
        <h2 id="keys-title">
          Keys <span className="muted num">{fmtInt(d.keys)}</span>
        </h2>
        <span className="muted small-text">A key calls this deployment&apos;s predict and nothing else.</span>
      </div>
      {fresh && (
        <Banner
          tone="good"
          title={`${fresh.name} created.`}
          action={
            <button
              type="button"
              className="small"
              onClick={() => {
                setFresh(null);
                create.reset();
              }}
            >
              Done
            </button>
          }
        >
          <span className="deploy-key-once">Copy it now: you won&apos;t see this key again.</span>
          <CopyCommand command={fresh.key} label="Copy the key" />
        </Banner>
      )}
      <ErrorAlert error={keys.error} />
      {list.length > 0 ? (
        <div className="exp-scroll-x">
          <table className="data batches deploy-keys">
            <thead>
              <tr>
                <th scope="col">Name</th>
                <th scope="col">Key</th>
                <th scope="col">Created</th>
                <th scope="col">Last used</th>
                <th scope="col">
                  <span className="exp-sr">Status</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {list.map((k) => (
                <tr key={k.id} className={k.revoked_at ? "deploy-revoked" : undefined}>
                  <td>{k.name}</td>
                  <td className="mono">{k.prefix}…</td>
                  <td className="secondary small-text" title={fmtDate(k.created_at)}>
                    {k.created_by_name ?? "Someone"}, {fmtAgo(k.created_at)}
                  </td>
                  <td className="secondary small-text" title={k.last_used_at ? fmtDate(k.last_used_at) : undefined}>
                    {k.last_used_at ? fmtAgo(k.last_used_at) : "Never"}
                  </td>
                  <td className="batch-nowrap">
                    {k.revoked_at ? (
                      <Chip title={fmtDate(k.revoked_at)}>Revoked {fmtAgo(k.revoked_at)}</Chip>
                    ) : (
                      d.can_manage && (
                        <button
                          type="button"
                          className="small ghost"
                          disabled={revoke.isPending}
                          onClick={() => onRevoke(k)}
                        >
                          Revoke
                        </button>
                      )
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        keys.data && <span className="muted small-text">No keys yet.</span>
      )}
      <ErrorAlert error={revoke.error} title="Not revoked." />
      {d.can_manage ? (
        <form
          className="row deploy-key-form"
          onSubmit={(ev) => {
            ev.preventDefault();
            if (name.trim()) create.mutate();
          }}
        >
          <input
            value={name}
            maxLength={200}
            aria-label="New key's name"
            placeholder="Name, e.g. the script that calls it"
            onChange={(ev) => {
              if (create.isError) create.reset();
              setName(ev.target.value);
            }}
          />
          <button
            type="submit"
            disabled={!name.trim() || create.isPending || locked}
            title={locked ? "A refresh is running on this dataset; try again once it finishes" : undefined}
          >
            <Icon name="plus" />
            Create key
          </button>
        </form>
      ) : (
        <span className="muted small-text">Its creator or a lead creates and revokes keys.</span>
      )}
      <ErrorAlert error={create.error} title="No key created." />
    </section>
  );
}

function CallIt({ d }: { d: DeploymentDetail }) {
  if (!d.fields) return null;
  const text = curlExample(window.location.origin, d.id, d.fields, d.example);
  return (
    <section className="card stack" aria-labelledby="call-title">
      <h2 id="call-title">Call it</h2>
      <span className="secondary small-text">
        Replace <code>rf_…</code> with a key. Each record carries every one of the {fmtInt(d.fields.length)} fields
        below (send <code>null</code> for a missing value; other fields are ignored). A call takes up to
        PREDICT_MAX_RECORDS records, 1,000 unless the server sets another number.
      </span>
      <CopyCommand command={text} />
      <details className="small-text">
        <summary>The fields a record carries</summary>
        <ul className="deploy-fields num">
          {d.fields.map((f) => (
            <li key={f.name}>
              <span className="mono">{f.name}</span> <span className="muted">{f.dtype}</span>
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}

function TestsCard({
  d,
  tests,
  blocked,
  selected,
  onSelect,
}: {
  d: DeploymentDetail;
  tests: DeploymentTest[];
  blocked: string | null;
  selected: DeploymentTest | null;
  onSelect: (id: number) => void;
}) {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<{ name: string; frac: number } | null>(null);
  const [failed, setFailed] = useState<unknown>(null);
  const upload = async (file: File | undefined) => {
    if (!file) return;
    setFailed(null);
    setProgress({ name: file.name, frac: 0 });
    try {
      const t = await uploadDeploymentTest(d.id, file, (frac) => setProgress({ name: file.name, frac }));
      await qc.invalidateQueries({ queryKey: ["deploymentTests", d.id] });
      onSelect(t.id);
    } catch (e) {
      setFailed(e);
    } finally {
      setProgress(null);
    }
  };
  return (
    <section className="card stack" aria-labelledby="tests-title">
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(ev) => {
          void upload(ev.target.files?.[0]);
          ev.target.value = "";
        }}
      />
      <div className="row exp-between">
        <h2 id="tests-title">Test with a file</h2>
        <button
          type="button"
          className="primary"
          disabled={Boolean(blocked) || progress !== null}
          title={blocked ?? undefined}
          onClick={() => input.current?.click()}
        >
          <Icon name="upload" />
          Upload CSV or Parquet
        </button>
      </div>
      <span className="secondary small-text">
        The file needs the fields predict takes; other columns are ignored. With a {d.models[0]?.target} column, the
        test also measures the models against it. The newest 20 tests are kept.
      </span>
      {blocked && <span className="muted small-text">{blocked}</span>}
      {progress && (
        <div className="row export-progress">
          <ProgressBar value={progress.frac} max={1} label={`Uploading ${progress.name}`} />
          <span className="muted small-text num">
            Uploading {progress.name} · {fmtPct(progress.frac, 0)}
          </span>
        </div>
      )}
      <ErrorAlert error={failed} title="Upload refused." />
      {tests.length > 0 && (
        <div className="exp-scroll-x">
          <table className="data batches deploy-tests">
            <thead>
              <tr>
                <th scope="col">File</th>
                <th scope="col">Status</th>
                <th scope="col" className="num">
                  Rows
                </th>
                <th scope="col">Labels</th>
                <th scope="col">Started</th>
              </tr>
            </thead>
            <tbody>
              {tests.map((t) => (
                <tr
                  key={t.id}
                  className={`run-row clickable${selected?.id === t.id ? " selected" : ""}`}
                  aria-selected={selected?.id === t.id}
                  onClick={() => onSelect(t.id)}
                >
                  <td>
                    <button
                      type="button"
                      className="link deploy-test-open"
                      onClick={(ev) => {
                        ev.stopPropagation();
                        onSelect(t.id);
                      }}
                    >
                      {t.filename}
                    </button>
                  </td>
                  <td>
                    {isActive(t.status) ? (
                      <span className="row export-progress">
                        <ProgressBar
                          value={Math.max(t.progress, 0.02)}
                          max={1}
                          label={`Test of ${t.filename}: ${t.status_message ?? t.status}`}
                        />
                        <span className="muted small-text">{t.status_message}</span>
                      </span>
                    ) : (
                      <StatusBadge status={t.status} />
                    )}
                  </td>
                  <td className="num">{fmtInt(t.rows)}</td>
                  <td className="secondary small-text">
                    {t.has_labels === null ? "—" : t.has_labels ? "Labelled" : "No labels"}
                  </td>
                  <td className="muted small-text" title={fmtDate(t.created_at)}>
                    {t.created_by_name ?? "Someone"}, {fmtAgo(t.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Tile({ label, value, hint, warn }: { label: string; value: string; hint?: string; warn?: boolean }) {
  return (
    <div className={`stat${warn ? " warn" : ""}`} title={hint}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function Metrics({ m, binary, target }: { m: TestMetrics; binary: boolean; target: string }) {
  if (m.kind === "number") {
    return (
      <div className="stats deploy-tiles">
        <Tile label="MAE" value={fmtMetric(m.mae)} hint="Mean absolute error" />
        <Tile label="RMSE" value={fmtMetric(m.rmse)} hint="Root mean squared error" />
        <Tile label="R²" value={fmtMetric(m.r2)} />
        <Tile label="Labelled rows" value={fmtInt(m.labelled_rows)} />
        <Tile
          label="Label not a number"
          value={fmtInt(m.rows_label_not_a_number)}
          warn={m.rows_label_not_a_number > 0}
          hint={`Rows whose ${target} isn't a number; they aren't measured`}
        />
      </div>
    );
  }
  return (
    <div className="stack exp-gap-3">
      <div className="stats deploy-tiles">
        <Tile
          label="Top-1 accuracy"
          value={fmtMetric(m.top1_accuracy)}
          hint="The share of labelled rows whose top class is their label"
        />
        <Tile label="Labelled rows" value={fmtInt(m.labelled_rows)} />
        <Tile
          label="Label has no model"
          value={fmtInt(m.rows_label_without_model)}
          warn={m.rows_label_without_model > 0}
          hint="Labelled rows whose class no model scores (skipped, left out or unseen); they count as wrong"
        />
        {m.nothing_matched_share !== null && (
          <Tile
            label="Nothing matched"
            value={fmtPct(m.nothing_matched_share)}
            hint="The share of all rows where no model's score reached its threshold"
          />
        )}
        {m.rows_without_label > 0 && <Tile label="No label" value={fmtInt(m.rows_without_label)} />}
      </div>
      <div className="exp-scroll-x">
        <table className="data batches deploy-classes">
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col" className="num" title="Labelled rows of the class">
                Rows
              </th>
              {binary && (
                <th scope="col" className="num">
                  Threshold
                </th>
              )}
              <th scope="col" className="num" title={binary ? "At its threshold" : "As the top class"}>
                Precision
              </th>
              <th scope="col" className="num" title={binary ? "At its threshold" : "As the top class"}>
                Recall
              </th>
              <th scope="col" className="num" title={binary ? "At its threshold" : "As the top class"}>
                F1
              </th>
              <th scope="col" className="num" title="Average precision of its score on the file">
                PR AUC
              </th>
            </tr>
          </thead>
          <tbody>
            {m.classes.map((c) => (
              <tr key={c.label}>
                <td className="batch-class-name">{c.label}</td>
                <td className="num secondary">{fmtInt(c.support)}</td>
                {binary && <td className="num">{c.threshold === null ? "—" : c.threshold.toFixed(2)}</td>}
                <td className="num">{fmtMetric(c.precision)}</td>
                <td className="num">{fmtMetric(c.recall)}</td>
                <td className="num">{fmtMetric(c.f1)}</td>
                <td className="num">{fmtMetric(c.pr_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {m.confusion.labels.length > 0 && (
        <ConfusionTable
          labels={m.confusion.labels}
          counts={m.confusion.counts}
          extra={["Other"]}
          caption={(total) =>
            `Top class against ${target} · the ${m.confusion.labels.length === 20 ? "20 most frequent labels" : `${m.confusion.labels.length} labels`}, ${fmtInt(total)} rows; Other is a top class outside them, or none`
          }
        />
      )}
    </div>
  );
}

function TestDetail({ d, test }: { d: DeploymentDetail; test: DeploymentTest }) {
  const [showAll, setShowAll] = useState(false);
  const detail = useQuery({
    queryKey: ["deploymentTest", test.id, test.status],
    queryFn: () => api.deploymentTest(test.id),
    enabled: test.status === "done",
  });
  const binary = d.models.some((m) => m.kind === "binary");
  const target = d.models[0]?.target ?? "the target";
  if (test.status === "failed")
    return (
      <section className="card stack" aria-labelledby="result-title">
        <h2 id="result-title">{test.filename}</h2>
        <Banner tone="bad" title="The test failed.">
          {test.error}
        </Banner>
      </section>
    );
  if (isActive(test.status))
    return (
      <section className="card stack" aria-labelledby="result-title">
        <h2 id="result-title">{test.filename}</h2>
        <span className="muted" role="status">
          {test.status_message ?? "Queued"}…
        </span>
      </section>
    );
  const t: DeploymentTestDetail | undefined = detail.data;
  const warnings = fieldWarnings(test.warnings);
  const cols = scoreColumns(test.labels ?? [], showAll);
  const results = t?.results ?? [];
  return (
    <section className="card stack deploy-result" aria-labelledby="result-title">
      <div className="row exp-between">
        <div className="stack exp-gap-1">
          <h2 id="result-title">{test.filename}</h2>
          <span className="secondary small-text num">
            {fmtInt(test.rows)} rows · {test.has_labels ? `labelled by ${target}` : `no ${target} column`} · scored{" "}
            {fmtAgo(test.finished_at)}
          </span>
        </div>
        <div className="row exp-gap-2">
          <a className="btn small" href={deploymentTestDownload(test.id, "csv")}>
            <Icon name="download" size={14} />
            CSV
          </a>
          <a className="btn small" href={deploymentTestDownload(test.id, "parquet")}>
            <Icon name="download" size={14} />
            Parquet
          </a>
        </div>
      </div>
      {test.notes.length > 0 && (
        <ul className="muted small-text deploy-notes">
          {test.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      {warnings.length > 0 && (
        <Banner
          tone={warnings.some((w) => w.flagged) ? "bad" : "warn"}
          title="Values that didn't convert to their field's type were scored as missing."
          role="note"
        >
          <table className="data deploy-warnings">
            <thead>
              <tr>
                <th scope="col">Field</th>
                <th scope="col" className="num">
                  Didn&apos;t convert
                </th>
                <th scope="col" className="num">
                  Share
                </th>
                <th scope="col">
                  <span className="exp-sr">Flag</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {warnings.map((w) => (
                <tr key={w.field}>
                  <td className="mono">{w.field}</td>
                  <td className="num">
                    {fmtInt(w.failed)} of {fmtInt(w.values)}
                  </td>
                  <td className="num">{fmtPct(w.share)}</td>
                  <td>{w.flagged && <Chip tone="bad">More than half failed</Chip>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Banner>
      )}
      <ErrorAlert error={detail.error} />
      {test.metrics ? (
        <>
          <Metrics m={test.metrics} binary={binary} target={target} />
          {binary && (
            <span className="muted small-text" role="note">
              Each model was calibrated on its validation split and its threshold chosen there too, so its validation
              numbers are mildly optimistic. This file, which the models haven&apos;t seen, gives the honest number.
            </span>
          )}
        </>
      ) : (
        <span className="muted small-text">
          The file has no {target} column, so there is nothing to measure the predictions against.
        </span>
      )}
      <div className="row exp-between">
        <h3 className="deploy-h3">
          Results{" "}
          <span className="muted small-text num">
            {results.length < (test.rows ?? 0)
              ? `the first ${fmtInt(results.length)} of ${fmtInt(test.rows)}; download them all`
              : fmtInt(results.length)}
          </span>
        </h3>
        {cols.toggle && (
          <button type="button" className="small" aria-pressed={showAll} onClick={() => setShowAll(!showAll)}>
            {showAll ? "Hide the scores" : `Show all scores (${fmtInt((test.labels ?? []).length)})`}
          </button>
        )}
      </div>
      {detail.isLoading && <span className="skeleton exp-skel-line" />}
      {results.length > 0 && (
        <div className="exp-scroll-x deploy-results">
          <table className="data batches">
            <thead>
              <tr>
                <th scope="col" className={test.id_field ? undefined : "num"}>
                  {test.id_field ?? "Row"}
                </th>
                {test.has_labels && <th scope="col">{target}</th>}
                {results[0].top !== null && <th scope="col">Top</th>}
                {binary && <th scope="col">Matches</th>}
                {cols.shown.map((l) => (
                  <th key={l} scope="col" className="num">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.row}>
                  <td className={test.id_field ? undefined : "num secondary"}>{test.id_field ? r.id : r.row}</td>
                  {test.has_labels && <td>{r.label ?? <span className="muted">—</span>}</td>}
                  {results[0].top !== null && (
                    <td className="batch-nowrap">
                      {r.top ? (
                        <>
                          <b>{r.top.label}</b> <span className="muted num">{fmtMetric(r.top.score)}</span>
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                  )}
                  {binary && <td>{r.matches?.length ? r.matches.join(", ") : <span className="muted">none</span>}</td>}
                  {cols.shown.map((l) => (
                    <td key={l} className="num">
                      {fmtMetric(r.scores[l])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

/** One deployment: its models and thresholds, its keys, how to call it, and tests with uploaded files. Keyed by the
 * deployment, so moving to another starts with its own newest test and no key on show. */
export function DeploymentPage() {
  return <DeploymentView key={useParams().deploymentId} />;
}

function DeploymentView() {
  const ds = useDatasetContext();
  const me = useMe().data;
  const qc = useQueryClient();
  const id = Number(useParams().deploymentId);
  const dep = useQuery({ queryKey: ["deployment", id], queryFn: () => api.deployment(id), enabled: !!id });
  const d = dep.data;
  const x = useQuery({
    queryKey: ["experiment", d?.experiment_id],
    queryFn: () => api.experiment(d!.experiment_id),
    enabled: !!d,
  });
  const tests = useQuery({
    queryKey: ["deploymentTests", id],
    queryFn: () => api.deploymentTests(id),
    enabled: !!d,
    refetchInterval: (q) =>
      (q.state.data as DeploymentTest[] | undefined)?.some((t) => isActive(t.status)) ? 1500 : false,
  });
  const [chosen, setChosen] = useState<number | null>(null);
  const status = useMutation({
    mutationFn: (to: "retire" | "activate") =>
      to === "retire" ? api.retireDeployment(id) : api.activateDeployment(id),
    onSuccess: (next) => {
      qc.setQueryData(["deployment", id], next);
      qc.invalidateQueries({ queryKey: ["deployments", ds.id] });
      toast({ message: next.status === "active" ? `${next.name} is active again` : `Retired ${next.name}` });
    },
  });

  const back = `/d/${ds.id}/deployments`;
  if (dep.error && !d) {
    const missing = (dep.error as { status?: number }).status === 404;
    return (
      <div className="state-card card">
        {missing ? (
          <h1>Deployment #{id} doesn&apos;t exist, or isn&apos;t shared with you</h1>
        ) : (
          <ErrorAlert error={dep.error} />
        )}
        <div>
          <Link className="btn" to={back}>
            Back to the deployments
          </Link>
        </div>
      </div>
    );
  }
  if (!d)
    return (
      <div className="page" aria-busy="true">
        <span className="skeleton exp-skel-line short" />
      </div>
    );

  const locked = ds.status === "refreshing";
  const xid = d.experiment_id;
  const runHref = (runId: number) => `/d/${ds.id}/experiments/${xid}?tab=runs&export=${d.export_id}&run=${runId}`;
  const writable = x.data && me ? addBlocked(x.data, me, locked) : "";
  const blocked = writable || (d.status === "retired" ? "Retired: reactivate it to test it." : null);
  const list = tests.data ?? [];
  const selected = list.find((t) => t.id === chosen) ?? list[0] ?? null;

  const onRetire = async () => {
    const ok = await confirmDialog({
      title: `Retire ${d.name}?`,
      body: "Predict calls are refused, with every key, until it is reactivated. Its keys and tests stay.",
      confirm: "Retire it",
      danger: true,
    });
    if (ok) status.mutate("retire");
  };

  return (
    <div className="page batch-page deployment-page">
      <div className="stack exp-gap-1">
        <Link to={back} className="secondary small-text batch-back">
          <Icon name="arrow-left" size={14} />
          Deployments
        </Link>
        <div className="row batch-title">
          <h1>{d.name}</h1>
          <Chip>Deployment #{d.id}</Chip>
          <DeploymentStatus d={d} />
          <span className="spacer" />
          {d.can_manage &&
            (d.status === "active" ? (
              <button type="button" className="small" disabled={status.isPending} onClick={onRetire}>
                Retire…
              </button>
            ) : (
              <button
                type="button"
                className="small primary"
                disabled={status.isPending || locked}
                title={locked ? "A refresh is running on this dataset; try again once it finishes" : undefined}
                onClick={() => status.mutate("activate")}
              >
                Reactivate
              </button>
            ))}
        </div>
        <span className="secondary small-text" title={fmtDate(d.created_at)}>
          Experiment <Link to={`/d/${ds.id}/experiments/${xid}?tab=runs`}>{d.experiment_name}</Link>
          {d.experiment_archived_at && " (archived)"}
          {d.batch_id !== null && (
            <>
              , batch{" "}
              <Link to={`/d/${ds.id}/experiments/${xid}/batches/${d.batch_id}`}>
                {d.batch_name ?? `Batch #${d.batch_id}`}
              </Link>
            </>
          )}{" "}
          · Export #{d.export_id} · deployed by {d.created_by_name ?? "someone"} {fmtAgo(d.created_at)}
          {d.retired_at ? ` · retired ${fmtAgo(d.retired_at)}` : ""}
        </span>
      </div>
      <ErrorAlert error={status.error} title={d.status === "active" ? "Not retired." : "Not reactivated."} />
      {d.problem && (
        <Banner tone="bad" title="This deployment can't score.">
          {d.problem}
        </Banner>
      )}
      {d.status === "retired" && (
        <Banner tone="warn">Retired: predict calls are refused until it is reactivated.</Banner>
      )}

      <div className="deploy-grid">
        <Models d={d} runHref={runHref} />
        <Keys d={d} locked={locked} />
      </div>
      <CallIt d={d} />
      <ErrorAlert error={tests.error} />
      <TestsCard d={d} tests={list} blocked={blocked} selected={selected} onSelect={setChosen} />
      {selected && <TestDetail key={selected.id} d={d} test={selected} />}
    </div>
  );
}
