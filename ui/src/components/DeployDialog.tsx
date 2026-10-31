import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { deployRequest, F1_RULE, type DeployModel, type RuleDraft } from "../deployments";
import { fmtInt } from "../format";
import { toast } from "../toast";
import type { ThresholdRule } from "../types";
import { Dialog, Icon, Segmented } from "./kit";
import { ErrorAlert } from "./ui";

const RULES: { value: ThresholdRule["kind"]; label: string }[] = [
  { value: "f1", label: "Best F1" },
  { value: "precision", label: "Precision at least" },
  { value: "fixed", label: "Fixed" },
];

function RuleValue({ draft, onChange, label }: { draft: RuleDraft; onChange: (v: string) => void; label: string }) {
  if (draft.kind === "f1") return null;
  return (
    <input
      type="number"
      inputMode="decimal"
      className="deploy-rule-value"
      min={0}
      max={1}
      step={0.01}
      value={draft.value}
      aria-label={label}
      placeholder={draft.kind === "precision" ? "0.9" : "0.5"}
      onChange={(ev) => onChange(ev.target.value)}
    />
  );
}

function modelsText(models: DeployModel[]): string {
  if (models.length === 1 && !models[0].binary) return models[0].label;
  const names = models.map((m) => m.label);
  const list =
    names.length > 8 ? `${names.slice(0, 8).join(", ")} and ${fmtInt(names.length - 8)} more` : names.join(", ");
  return `${fmtInt(models.length)} one-vs-rest model${models.length === 1 ? "" : "s"}: ${list}`;
}

/** Deploy runs of one experiment: a one-vs-rest batch (`runIds` null), some of a batch's rows, or one run. The API
 * checks what may share a deployment and that scoring reproduces training; its refusals stay in the dialog. */
export function DeployDialog({
  experimentId,
  datasetId,
  batchId = null,
  runIds,
  models,
  defaultName,
  origin,
  onClose,
}: {
  experimentId: number;
  datasetId: number;
  batchId?: number | null;
  runIds: number[] | null;
  models: DeployModel[];
  /** The API's name when none is given: the batch's name or the model's label. */
  defaultName: string;
  /** Where the runs come from, as a line of the dialog. */
  origin: string;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [rule, setRule] = useState<RuleDraft>(F1_RULE);
  const [overrides, setOverrides] = useState<Record<number, RuleDraft>>({});
  const binary = models.filter((m) => m.binary);
  const built = deployRequest({ name, batchId, runIds, models, rule, overrides });
  const create = useMutation({
    mutationFn: () => api.createDeployment(experimentId, built.request!),
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ["deployments", datasetId] });
      toast({ message: `Deployed ${d.name}` });
      nav(`/d/${datasetId}/models/${d.id}`);
    },
  });
  // A refusal is about the request as it was sent; once the form changes it no longer applies.
  const edit = () => create.isError && create.reset();
  // Once sent, the deployment may be created whatever the dialog does, and success opens it; so the dialog stays
  // open (Cancel, Escape and the overlay alike) until the check answers.
  const close = () => {
    if (!create.isPending) onClose();
  };
  const setOverride = (runId: number, d: RuleDraft | null) => {
    edit();
    setOverrides((o) => {
      const next = { ...o };
      if (d) next[runId] = d;
      else delete next[runId];
      return next;
    });
  };
  const overridden = Object.keys(overrides).filter((id) => binary.some((m) => m.runId === Number(id))).length;

  return (
    <Dialog
      title="Deploy"
      wide
      onClose={close}
      labelledBy="deploy-title"
      actions={
        <>
          {create.error != null && (
            <div className="batch-error">
              <ErrorAlert error={create.error} title="Not deployed." />
            </div>
          )}
          {create.isPending ? (
            <span className="small-text batch-why deploy-progress" role="status">
              <span className="meter indeterminate" aria-hidden="true">
                <span />
              </span>
              Checking that scoring through the deployment reproduces training…
            </span>
          ) : (
            built.error && (
              <span id="deploy-why" className="small-text batch-why exp-bad">
                {built.error}
              </span>
            )
          )}
          <button type="button" onClick={close} disabled={create.isPending}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            disabled={built.error !== null || create.isPending}
            aria-describedby={built.error ? "deploy-why" : undefined}
            onClick={() => create.mutate()}
          >
            {create.isPending ? "Deploying…" : "Deploy"}
          </button>
        </>
      }
    >
      <div className="batch-body stack">
        <p className="muted small-text exp-flush">
          A deployment scores new records with these models, through their export&apos;s own preprocessing, for testing
          them on data they haven&apos;t seen. It is frozen: retraining or exporting again doesn&apos;t change it.
        </p>
        <dl className="batch-from num">
          <dt>From</dt>
          <dd>{origin}</dd>
          <dt>Models</dt>
          <dd>{modelsText(models)}</dd>
        </dl>
        <label className="field">
          Name
          <input
            data-autofocus
            value={name}
            maxLength={200}
            placeholder={defaultName}
            onChange={(ev) => {
              edit();
              setName(ev.target.value);
            }}
          />
        </label>
        {binary.length > 0 && (
          <>
            <div className="field">
              <span>{binary.length === 1 ? "Threshold" : "Threshold for every model"}</span>
              <div className="row deploy-rule">
                <Segmented
                  label="Threshold rule"
                  value={rule.kind}
                  options={RULES}
                  onChange={(kind) => {
                    edit();
                    setRule({ kind, value: kind === rule.kind ? rule.value : "" });
                  }}
                />
                <RuleValue
                  draft={rule}
                  label={rule.kind === "precision" ? "Minimum precision" : "Threshold"}
                  onChange={(value) => {
                    edit();
                    setRule({ ...rule, value });
                  }}
                />
              </div>
              <span className="muted small-text">
                {rule.kind === "f1"
                  ? "The threshold with the best F1 on each model's validation split."
                  : rule.kind === "precision"
                    ? "The lowest threshold whose validation precision reaches it. A model that can't reach it takes the best-F1 threshold, and says so."
                    : "The same threshold, from 0 to 1, for every model."}{" "}
                A record scoring at or above a model&apos;s threshold matches its class.
              </span>
            </div>
            {binary.length > 1 && (
              <details className="deploy-overrides" open={overridden > 0 || undefined}>
                <summary className="small-text">
                  A model&apos;s own threshold{overridden ? ` (${overridden} set)` : ""}
                </summary>
                <div className="table-wrap batch-classes">
                  <table className="data batch-class-table">
                    <thead>
                      <tr>
                        <th scope="col">Class</th>
                        <th scope="col">Threshold</th>
                      </tr>
                    </thead>
                    <tbody>
                      {binary.map((m) => {
                        const own = overrides[m.runId];
                        return (
                          <tr key={m.runId}>
                            <td className="batch-class-name">{m.label}</td>
                            <td>
                              <div className="row deploy-rule">
                                <select
                                  aria-label={`Threshold rule for ${m.label}`}
                                  value={own?.kind ?? ""}
                                  onChange={(ev) =>
                                    setOverride(
                                      m.runId,
                                      ev.target.value
                                        ? { kind: ev.target.value as RuleDraft["kind"], value: own?.value ?? "" }
                                        : null,
                                    )
                                  }
                                >
                                  <option value="">As every model</option>
                                  {RULES.map((r) => (
                                    <option key={r.value} value={r.value}>
                                      {r.label}
                                    </option>
                                  ))}
                                </select>
                                {own && (
                                  <RuleValue
                                    draft={own}
                                    label={`${own.kind === "precision" ? "Minimum precision" : "Threshold"} for ${m.label}`}
                                    onChange={(value) => setOverride(m.runId, { ...own, value })}
                                  />
                                )}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </>
        )}
        <span className="muted small-text">
          <Icon name="info" size={13} /> Before it is stored, the export&apos;s test rows are scored from the
          dataset&apos;s raw values and must reproduce each model&apos;s scores; this can take a few seconds.
        </span>
      </div>
    </Dialog>
  );
}
