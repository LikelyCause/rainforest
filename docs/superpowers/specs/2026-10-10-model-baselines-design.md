# Model baselines and experiment adjustments

Goal: the team agrees on how to prepare a dataset for one kind of model at a time. Each agreed preparation is a
**baseline** for one model type (Trees, Linear or Neural). Experiments start from a baseline and may adjust it; the
baseline itself only changes through the team process.

Decisions made with the owner (2026-10-10):
- One baseline per model type: Trees (LightGBM, XGBoost, random forest, histogram GBM), Linear (logistic regression),
  Neural (MLP). A dataset can have one of each. Adding models to a type is a registry change and needs no new baseline
  type. Planned models (linear SVM, k-NN, more neural network kinds) appear in the model list as "planned", not
  selectable.
- Focus: the Trees baseline with LightGBM is the path that must work end to end and is tested on dev-01; the other
  types keep working as they do but aren't the subject of this work.
- The model type is chosen when a baseline is started and can't change afterwards. The top-bar Model switch goes away;
  the top bar shows a baseline chip linking to the Baselines page instead.
- A new baseline starts either from scratch (the suggestions for its model type) or populated from another baseline of
  the same dataset.
- The lead is an assessor like anyone else (included by ticking themself) and finalizes once every assigned assessor
  has submitted.
- Experiments use a finalized baseline as their starting point and can adjust it, including a field's type and the
  targets; adjustments belong to the experiment, are shown as differences from the baseline, and are logged to MLflow.
- Unchanged from the team design: blind assessment, review opening by itself, lead-only finalize, versioned and
  frozen finalized versions, reuse on new data, personal experiments private until review.

## 1. Baselines

A baseline is the existing recipe, given a model type. `meta.recipes` gains `family` (`tree | linear | neural`,
required). A dataset has up to three baseline lines, one per family; each line is versioned as recipes are now
(v1, v2, …, each `assessing → review → finalized`, finalized snapshots immutable). A finalized snapshot holds the
decisions (keep / drop / target, type, notes) and the settings for its own family only.

Labels: "Trees baseline v2", "Linear baseline v1". Every place that shows a recipe version shows this label.

**One baseline in progress at a time per dataset.** The working tables (`field_decisions`, `field_settings`,
`assessments`, `assessment_settings`) serve the baseline being worked, as they serve the version being worked today;
finalized baselines live in their snapshots. Starting a baseline (any family) or a new version needs the dataset's
baseline in progress, if any, to be finalized first. (Alternative: key the working tables by baseline so several can
be assessed at once. It touches about 50 queries in 13 modules and doubles the blind-data surface to test; not worth it
for one team. Revisit if teams run baselines in parallel.)

`meta.datasets.model_family` stays, maintained by the server as the current baseline's family (set whenever a
baseline or version starts and loads the working tables); a `PATCH` with a different family is refused with 422. It
keeps the SQL views, the suggestion columns and older queries working without a join; clients can't set
it.

## 2. Starting a baseline

A lead starts one from the dataset's **Baselines** page (today's Versions page, renamed; `/versions` redirects), or,
for a new dataset, from the Team page. A new dataset already has a first baseline row (Trees v1, assessing, nobody
assigned); while nobody has started on it, it is **set up in place**: the start dialog changes its family, starting
point and targets on that row and assigns the assessors, rather than adding a v2 or a second baseline. API:
`POST /api/datasets/{id}/baselines` (`family`, `start` = `scratch | baseline | recipe`, `source_recipe_id`, `targets`,
`assessors`); `GET …/baselines` lists them per family. One dialog:
1. **Model type**: Trees, Linear or Neural. Types that already have a baseline are offered as "new version of …".
2. **Start from**: *Scratch* (the suggestions for this type) or *Another baseline* (a finalized baseline of this
   dataset, any type; shows its type, version and finalized date). Populating copies keep / drop / target, types and
   notes, and each field's settings where they are valid for the new type; other settings take the new type's
   suggestion. For a new dataset, *A recipe from another dataset* (today's reuse on new data) is the third option.
3. **Targets**: prefilled from the source (or the dataset's suggested targets), editable here; fixed once assessment
   starts.
4. **Assessors**: the team's members, the starting lead ticked by default.

Every field is open for assessment, prefilled from the source. (Today's "reopen only some fields" stays for new
versions of the same baseline.)

## 3. Assessment and review

As today, scoped to the baseline: assessors see one model type's settings; the decision panel shows the baseline's
type as a fixed badge. Review, adoption rules and finalizing are unchanged. Finalizing records the snapshot for the
baseline's family.

Starting any baseline or version loads the working tables from its starting point (`recipes.load_working`): the source
snapshot's decisions, its settings where valid for the new family, else the suggestions. The working tables therefore
always hold exactly the version in progress; finalized baselines are only ever read from their snapshots.

## 4. Experiments

An experiment's source is a finalized baseline version (team experiment) or one assessor's assessment of a baseline in
progress (personal, as today). The export's model type is the baseline's.

**Adjustments.** The experiment owner can change, per field: the role (keep / drop / target), the type, and the prep
settings (imputation, missing indicator, transform, scaling, encoding and encoder parameters), within the baseline's
model type. A type change whose settings don't fit the new type takes the new type's suggested settings.

Storage (decided in implementation): each team experiment gets a **full copy** of its baseline when it is created,
shaped like the working tables (`meta.experiment_decisions`, `meta.experiment_settings`, `meta.experiment_batches`
for bulk undo; `experiments.target_order`; migration 013). The adjustments are computed, not stored: the diff between
the baseline snapshot and the copy. A sparse override table was the first idea; the copy won because every existing
read and write path (Review, Bulk edit, the finder, bulk undo, Explore) already works on relations of that shape, so an
experiment is just another decision source. Revert deletes a field's rows (or all) and reseeds them from the snapshot.

The copy is read and written through `view=experiment:<id>` on the field routes (`GET …/fields`, `GET
/api/fields/{id}`, `PUT …/decision`, bulk and its undo, `…/fields/query`) and Explore. `view=baseline:<recipe_id>`
reads a finalized baseline's snapshot the same way, read-only. Without a view, routes work on the baseline in progress
as before.

**Who adjusts.** Only the experiment's owner, while it isn't archived. The default per-baseline Team experiments
(`is_default`, named like "Team · Trees v1", made on the first export without an experiment) are not adjustable, so the
baseline's own results stay as agreed; neither are Imported or personal experiments. Anyone else wanting changes
starts their own experiment from the baseline.

**Changed targets.** Targets are the baseline's by default; an experiment may add or remove targets (a run then picks
among the experiment's targets, as today). Because the baseline's choices were made against its targets, an experiment
with different targets:
- is checked for leakage against its own target set at preview time (`leakage.leak_names` over the stored
  associations), so any kept field that looks like a leak of its targets is warned about before export; a field newly
  made a target gets the leak screen in the background, and a target with no associations at all is named ("No leak
  check has run for target(s) …"). The dataset's stored leakage flags and target state, which belong to the working
  copy, are left alone;
- has its export split stratified on its own targets;
- is badged "targets changed" on the experiment page, in run comparisons and in MLflow (`rainforest.targets_changed`
  tag), and its export preview warns "Targets differ from …", so its results aren't read as comparable with
  experiments on the baseline's targets.

Editing reuses Review and Bulk edit in an **experiment mode** (*Adjust fields* on the experiment page opens
`?view=experiment:<id>`, with a mode bar): the panel says which settings changed from the baseline, with *Revert to
baseline*. The experiment page has an Adjustments card (per-field revert, revert everything) and "N adjusted" /
"targets changed" badges.

Exports of an experiment are built from the baseline snapshot plus its adjustments, frozen at export time. The
export's model family is always the baseline's (another `family` is a 422). The manifest config records `baseline`
(recipe id, family, version, label) and `adjustments` (count, fields, targets changed, targets, per-field detail); the
MLflow run records them as tags (`rainforest.baseline`, `rainforest.baseline_family`, `rainforest.adjusted`,
`rainforest.targets_changed`), the param `prep.adjusted_fields` and the artifact `adjustments.json`, so a result can
always be traced to the exact preparation.

Team experiments stay visible to everyone; only the owner adjusts them. Personal experiments behave as today.

## 5. Elsewhere

- **Explore** has a "Decisions" picker (`?decisions=`) for raw data: the baseline in progress (the viewer's own
  assessment for "Kept only" while assessing), any finalized baseline, or an adjustable team experiment's copy. The
  cache key includes it (`team | assessment:r:u | experiment:x | baseline:r`).
- **Reuse on new data** reuses one finalized baseline into a new dataset's baseline of the same family.
- **Datasets list** shows each dataset's baselines as chips (type, version, phase).
- **SQL views**: `v_field_profiles` / `v_fields` gain `baseline_family`, `baseline_version`; `v_model_metrics` gains
  `experiment_id`, `baseline_family`, `baseline_version`.
- **Planned models** (`linear_svm`, `knn`, `cnn`, `rnn`) are listed by `/api/models` and per export with
  `planned: true`; training one is a 422.

## 6. Migration

Existing recipe versions become baselines of the dataset's current `model_family`, keeping their snapshots (settings
for the other families are dropped from the snapshot's use but left in the stored JSON); versions become unique per
(dataset, family) (migration 012). Default Team experiments are flagged `is_default` and renamed "Team · Trees vN".
Existing team experiments get their copy at startup (`experiments.backfill_copies`) and so start with no
adjustments. dev-01 currently has no datasets.

## Build order

Each its own PR, CI green and checked on dev-01:
1. Baselines: `recipes.family`, labels, Baselines page, start dialog (type, start from, targets, assessors), Model
   switch removed, export family from the baseline, planned models in the model list.
2. Experiment adjustments: table, experiment mode in Review and Bulk edit, exports from baseline + adjustments,
   manifest and MLflow records.
3. Explore baseline picker, reuse by family, Datasets chips, views, docs.

## Testing

Unit and DB-backed API tests per step: a baseline's family is fixed (422 on change); starting needs the one in progress
finalized (409); populate copies decisions and only family-valid settings; blind no-leak tests extended to the new
endpoints; adjustments are owner-only, never alter the baseline, and an export reproduces baseline + adjustments; MLflow
records the adjustments. Playwright on dev-01: lead starts a Trees baseline from scratch with two members and
themself, finalizes; starts a Linear baseline populated from it; a member runs a LightGBM experiment with two
adjustments and the run shows them in MLflow; an experiment that changes a field's type and drops a target trains with
LightGBM and is badged "targets changed".
