import { useCallback } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { api } from "../api";
import { can } from "../can";
import { useFields, useInvalidateTeam, useMe } from "../hooks";
import { baselineLabel, fmtInt } from "../format";
import { showError, toast } from "../toast";
import { useView } from "../view";
import { confirmDialog, type Stage } from "./kit";
import type { Dataset, TeamState } from "../types";
import type { Access } from "../team";

/** The next step: always this slot, always the primary style; only the words change.
 * `card` repeats it as the Review complete card's action: focused when it appears, so Enter takes it,
 * and left out while it would only say what's still to do. */
export function NextStep({ ds, stage, access, team, card }: {
  ds: Dataset; stage: Stage; access: Access; team: TeamState | undefined; card?: boolean;
}) {
  const me = useMe().data;
  const nav = useNavigate();
  const loc = useLocation();
  const view = useView();
  const fields = useFields(ds.id, true, view);
  const invalidate = useInvalidateTeam(ds.id);
  // Focus once, when the card's button first appears (a callback ref that stays the same).
  const focusOnce = useCallback((el: HTMLElement | null) => { if (card) el?.focus(); }, [card]);
  const submit = useMutation({
    mutationFn: () => api.submitAssessment(ds.id),
    onSuccess: (t) => {
      invalidate();
      toast({ tone: "good", message: t.recipe.phase === "review" ? "Submitted. Everyone is in: team review is open." : "Assessment submitted" });
      nav(`/d/${ds.id}/baseline`);
    },
    onError: (e) => showError("Couldn't submit", e),
  });
  const finalize = useMutation({
    mutationFn: (acceptOpen: boolean) => api.finalize(ds.id, acceptOpen),
    onSuccess: (t) => {
      invalidate();
      toast({ tone: "good", message: `${t.recipe.label} finalized` });
      nav(`/d/${ds.id}/experiments?finalized=${t.recipe.id}`);
    },
    onError: (e) => showError("Couldn't finalize", e),
  });
  const here = (p: string) => loc.pathname === p || loc.pathname.startsWith(`${p}/`);
  const base = `/d/${ds.id}`;
  const lead = me?.role === "lead";
  const label = team?.recipe.label ?? ds.baseline?.label ?? "this baseline";
  const shortLabel = ds.baseline ? baselineLabel(ds.baseline.family, ds.baseline.version, true, ds.baseline.name) : label;
  // On the card: focused when it appears, and marked so Enter on the card can find it.
  const mark = card ? { ref: focusOnce, "data-next-step": "" } : {};
  const enter = card ? <> <kbd>↵</kbd></> : null;

  if (stage === "setup") {
    return lead && !here(`${base}/baseline`) ? <Link {...mark} className="btn primary" to={`${base}/baseline`}>Set up the baseline{enter}</Link> : null;
  }
  if (stage === "assessing" && access.source === "mine" && team?.me && !team.me.submitted) {
    const all = fields.data ?? [];
    const todo = all.filter((f) => !f.reviewed);
    if (todo.length > 0) {
      if (here(`${base}/review`)) {
        return card ? null : <button type="button" className="primary" disabled title={`Review every field first (${todo.length} left)`}>Submit · {todo.length} left</button>;
      }
      return <Link {...mark} className="btn primary" to={`${base}/review/${todo[0].id}`}>Next field · {fmtInt(todo.length)} left <kbd className="hide-phone">U</kbd></Link>;
    }
    if (all.length === 0) return null;
    const ask = async () => {
      const changed = team.me ? team.me.total : all.length;
      const ok = await confirmDialog({
        title: "Submit your assessment?",
        body: <p>{fmtInt(changed)} fields. Your choices stay private until everyone submits. You can unsubmit until then.</p>,
        confirm: "Submit", cancel: "Keep reviewing",
      });
      if (ok) submit.mutate();
    };
    return (
      <button {...mark} type="button" className="primary" disabled={submit.isPending} onClick={ask}>
        {card ? "Submit your assessment" : "Submit assessment"}{enter}
      </button>
    );
  }
  if (stage === "review" && team?.review) {
    const open = team.review.open;
    const askFinalize = async () => {
      const r = team.review!;
      const next = `A later change is a new version, v${team.recipe.version + 1}.`;
      const ok = await confirmDialog({
        title: `Finalize ${shortLabel}?`,
        body: <>
          {r.adopted === 0 && r.disagreements === 0
            // Nothing went through assessment (say, only new fields confirmed directly): no counts to report.
            ? <p>Its decisions freeze as a snapshot that experiments build on. {next}</p>
            : <p className="num">{fmtInt(r.adopted)} agreed and {fmtInt(r.disagreements - open)} settled, {fmtInt(open)} open. It freezes as a snapshot that experiments build on. {next}</p>}
          {open > 0 && <p><b>{open} disagreement{open === 1 ? " is" : "s are"} still open</b> and will keep {open === 1 ? "its" : "their"} current team decision{open === 1 ? "" : "s"}.</p>}
        </>,
        confirm: `Finalize ${shortLabel}`, cancel: "Not yet",
      });
      if (ok) finalize.mutate(open > 0);
    };
    if (!lead) return here(`${base}/team`) ? null : <Link {...mark} className="btn primary" to={`${base}/team`}>See the review queue{enter}</Link>;
    if (open > 0 && !here(`${base}/team`)) return <Link {...mark} className="btn primary" to={`${base}/team`}>Settle {fmtInt(open)} open field{open === 1 ? "" : "s"}{enter}</Link>;
    return <button {...mark} type="button" className={open > 0 ? "" : "primary"} disabled={finalize.isPending} onClick={askFinalize}>Finalize {shortLabel}{enter}</button>;
  }
  if (stage === "finalized" && !here(`${base}/experiments`)) {
    return (
      <Link {...mark} className="btn primary" to={`${base}/experiments`}>
        {card ? "Continue to Experiments" : can(me, "createExperiment") ? "Start an experiment" : "See experiments"}{enter}
      </Link>
    );
  }
  return null;
}
