import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { eligibleTarget, FAMILIES, startBaselineDefaults, type StartBaselineForm } from "../baselines";
import { defaultTask, taskLabel } from "../decision";
import { baselineLabel, familyShort, fmtAgo } from "../format";
import { useFields, useMe, useOptions } from "../hooks";
import { useInvalidateTeam } from "../pages/DatasetTeamPage";
import { toast } from "../toast";
import type { Dataset, FamilyKey } from "../types";
import { ErrorAlert } from "./ui";

/** A lead starts a baseline: its model type (fixed from then on), what it starts from, its targets and who
 * assesses it. On a new dataset this sets up the first baseline. */
export function StartBaseline({ ds, onCancel }: { ds: Dataset; onCancel?: () => void }) {
  const nav = useNavigate();
  const me = useMe().data;
  const { data: catalog } = useOptions();
  const fields = useFields(ds.id);
  const users = useQuery({ queryKey: ["users"], queryFn: api.users });
  const lines = useQuery({ queryKey: ["baselines", ds.id], queryFn: () => api.baselines(ds.id) });
  const setup = Boolean(ds.baseline?.setup);
  const others = useQuery({ queryKey: ["finalizedRecipes"], queryFn: api.finalizedRecipes, enabled: setup });
  const invalidate = useInvalidateTeam(ds.id);
  const [form, setForm] = useState<StartBaselineForm | null>(null);
  useEffect(() => {
    if (!form && me && fields.data) setForm(startBaselineDefaults(ds, me, fields.data));
  }, [form, me, fields.data, ds]);
  const own = useMemo(() => (lines.data ?? []).flatMap((l) => l.versions.filter((v) => v.phase === "finalized")), [lines.data]);
  const fromOther = (others.data ?? []).filter((r) => r.dataset_id !== ds.id);
  const start = useMutation({
    mutationFn: (f: StartBaselineForm) => api.startBaseline(ds.id, {
      ...f, source_recipe_id: f.start === "scratch" ? null : f.source_recipe_id, targets: f.start === "recipe" ? [] : f.targets,
    }),
    onSuccess: (t) => {
      invalidate();
      toast({ message: t.recipe.phase === "review" ? `${t.recipe.label} started: nothing needs assessing; a lead confirms by finalizing.`
        : `${t.recipe.label} started: ${t.assessors.length} assessor${t.assessors.length === 1 ? "" : "s"}` });
      nav(t.recipe.phase === "review" ? `/d/${ds.id}/team` : `/d/${ds.id}/review`);
    },
  });
  if (!form || !me) return <span className="skeleton" style={{ width: 240, height: 22 }} />;
  const set = (patch: Partial<StartBaselineForm>) => setForm({ ...form, ...patch });
  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const nextVersion = (f: FamilyKey) => (setup ? 1 : (ds.baselines.find((b) => b.family === f)?.version ?? 0) + 1);
  const people = (users.data ?? []).filter((u) => u.active && u.role !== "viewer").sort((a, b) => Number(b.id === me.id) - Number(a.id === me.id) || a.name.localeCompare(b.name));
  const missingSource = form.start !== "scratch" && !form.source_recipe_id;
  const blocked = missingSource ? "Choose what to start from" : form.assessors.length === 0 ? "Choose at least one assessor"
    : form.start !== "recipe" && form.targets.length === 0 ? "Choose at least one target" : null;

  return (
    <form className="card stack start-baseline" aria-label="Start a baseline"
      onSubmit={(e) => { e.preventDefault(); if (!blocked) start.mutate(form); }}>
      <div className="card-head">
        <h2>{setup ? "Set up the first baseline" : "Start a baseline"}</h2>
        <span className="muted small-text">The model type is fixed once it starts.</span>
      </div>

      <fieldset className="bare stack" style={{ gap: 6 }}>
        <legend className="field-label">Model type</legend>
        <div className="segmented" role="radiogroup" aria-label="Model type">
          {FAMILIES.map((f) => (
            <button key={f} type="button" role="radio" aria-checked={form.family === f} className={form.family === f ? "on" : ""}
              disabled={form.start === "recipe"} onClick={() => set({ family: f })}
              title={catalog?.families.find((x) => x.value === f)?.examples}>
              {familyShort(f)}
            </button>
          ))}
        </div>
        <span className="muted small-text">
          {form.start === "recipe" ? "Takes the model type of the baseline it reuses."
            : `${baselineLabel(form.family, nextVersion(form.family))} · for ${catalog?.families.find((x) => x.value === form.family)?.examples ?? familyShort(form.family)}`}
        </span>
      </fieldset>

      <fieldset className="bare stack" style={{ gap: 6 }}>
        <legend className="field-label">Start from</legend>
        <label className="check"><input type="radio" name="start" checked={form.start === "scratch"} onChange={() => set({ start: "scratch" })} />
          The suggestions for {familyShort(form.family).toLowerCase()} models</label>
        <label className="check"><input type="radio" name="start" checked={form.start === "baseline"} disabled={own.length === 0}
          onChange={() => set({ start: "baseline", source_recipe_id: own[0]?.id ?? null })} />
          Another baseline of this dataset{own.length === 0 ? " (none finalized yet)" : ""}</label>
        {form.start === "baseline" && (
          <select aria-label="Baseline to start from" value={form.source_recipe_id ?? ""} onChange={(e) => set({ source_recipe_id: Number(e.target.value) })}>
            {own.map((v) => <option key={v.id} value={v.id}>{baselineLabel(v.family, v.version)} · finalized {fmtAgo(v.finalized_at)}</option>)}
          </select>
        )}
        {setup && (
          <label className="check"><input type="radio" name="start" checked={form.start === "recipe"} disabled={fromOther.length === 0}
            onChange={() => set({ start: "recipe", source_recipe_id: fromOther[0]?.id ?? null })} />
            Another dataset's baseline{fromOther.length === 0 && others.data ? " (none finalized yet)" : ""}</label>
        )}
        {form.start === "recipe" && (
          <select aria-label="Dataset baseline to start from" value={form.source_recipe_id ?? ""} onChange={(e) => set({ source_recipe_id: Number(e.target.value) })}>
            {fromOther.map((r) => <option key={r.id} value={r.id}>{r.dataset_name} · {r.label} · {r.field_count} fields</option>)}
          </select>
        )}
        <span className="muted small-text">
          {form.start === "baseline" ? "Its decisions and the settings that suit this model type are filled in; every field is still assessed."
            : form.start === "recipe" ? "Fields that match by name, type and statistics are carried over and locked; the rest are assessed."
            : "Every field starts from the profile-based suggestion."}
        </span>
      </fieldset>

      {form.start !== "recipe" && (
        <fieldset className="bare stack" style={{ gap: 4 }}>
          <legend className="field-label">Targets <span className="muted small-text">· fixed for this baseline; experiments can change them. Each is predicted as shown; change it to a number or bands per field in Review</span></legend>
          <div className="check-grid">
            {(fields.data ?? []).map((f) => {
              const why = eligibleTarget(f);
              const task = defaultTask(f.semantic_type, f.n_unique);
              return (
                <label key={f.id} className={`check${why ? " muted" : ""}`} title={why ?? undefined}>
                  <input type="checkbox" disabled={Boolean(why) && !form.targets.includes(f.name)} checked={form.targets.includes(f.name)}
                    onChange={() => set({ targets: toggle(form.targets, f.name) })} />
                  <span className="mono">{f.name}</span>
                  {!why && task && <span className="muted small-text"> · {taskLabel(task).toLowerCase()}</span>}
                </label>
              );
            })}
          </div>
        </fieldset>
      )}

      <fieldset className="bare stack" style={{ gap: 4 }}>
        <legend className="field-label">Assessors <span className="muted small-text">· each assesses every field blind; review opens when all have submitted</span></legend>
        <div className="check-grid">
          {people.map((u) => (
            <label key={u.id} className="check">
              <input type="checkbox" checked={form.assessors.includes(u.id)} onChange={() => set({ assessors: toggle(form.assessors, u.id) })} />
              {u.name}{u.id === me.id ? " (you)" : ""} <span className="muted small-text">{u.role}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <ErrorAlert error={users.error ?? lines.error ?? start.error} />
      <div className="row" style={{ justifyContent: "flex-end" }}>
        {onCancel && <button type="button" className="ghost" onClick={onCancel}>Cancel</button>}
        <button type="submit" className="primary" disabled={Boolean(blocked) || start.isPending} title={blocked ?? undefined}>
          {form.start === "recipe" ? "Start from this baseline" : `Start ${baselineLabel(form.family, nextVersion(form.family))}`}
        </button>
      </div>
    </form>
  );
}
