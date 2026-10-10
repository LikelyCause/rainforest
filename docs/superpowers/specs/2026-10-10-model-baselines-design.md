# Model baselines and experiment adjustments

Goal: the team agrees on how to prepare a dataset for one kind of model at a time. Each agreed preparation is a
**baseline** for one model type (Trees, Linear or Neural). Experiments start from a baseline and may adjust it; the
baseline itself only changes through the team process.

Decisions made with the owner (2026-10-10):
- One baseline per model type: Trees (LightGBM, XGBoost, random forest, histogram GBM), Linear (logistic regression),
  Neural (MLP). A dataset can have one of each. Adding models to a type (e.g. a linear SVM) is a registry change and
  needs no new baseline type.
- The model type is chosen when a baseline is started and can't change afterwards. The top-bar Model switch goes away.
- A new baseline starts either from scratch (the suggestions for its model type) or populated from another baseline of
  the same dataset.
- The lead is an assessor like anyone else and finalizes once every assigned assessor has submitted.
- Experiments use a finalized baseline as their starting point and can adjust it; adjustments belong to the
  experiment, are shown as differences from the baseline, and are logged to MLflow.
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

`meta.datasets.model_family` is removed; the family comes from the baseline in view.

## 2. Starting a baseline

A lead starts one from the dataset's **Baselines** page (today's Versions page, renamed), or, for a new dataset, from
the prompt that replaces "nobody is assigned yet". One dialog:
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

## 4. Experiments

An experiment's source is a finalized baseline version (team experiment) or one assessor's assessment of a baseline in
progress (personal, as today). The export's model type is the baseline's.

**Adjustments.** The experiment owner can change, per field: keep / drop, and the prep settings (imputation, missing
indicator, transform, scaling, encoding and encoder parameters). Not the type, not the targets (a run already chooses
which targets it trains). Stored in `meta.experiment_adjustments` (experiment_id, field_id, action, settings columns,
updated_by, updated_at); a field without a row follows the baseline. Editing reuses Review and Bulk edit in an
**experiment mode** (entered from the experiment page): the panel shows the baseline value next to each changed setting
and *Revert to baseline*. The experiment page lists the adjustments ("4 fields adjusted from Trees baseline v2").

Exports of an experiment are built from the baseline snapshot plus its adjustments, frozen at export time. The
manifest and the MLflow run record the baseline (id, family, version) and the adjustments (as a JSON artifact, a count
param and a `rainforest.adjusted` tag), so a result can always be traced to the exact preparation.

Team experiments stay visible to everyone; only the owner adjusts them. Personal experiments behave as today.

## 5. Elsewhere

- **Explore** has a baseline picker where it uses decisions ("Kept only"): any finalized baseline, or the one in
  progress (the viewer's own assessment while assessing). The cache key includes it.
- **Reuse on new data** reuses one finalized baseline into a new dataset's baseline of the same family.
- **Datasets list** shows each dataset's baselines as chips (type, version, phase).
- **SQL views** (`v_fields`, `v_model_metrics`) gain the baseline family and version.

## 6. Migration

Existing recipe versions become baselines of the dataset's current `model_family`, keeping their snapshots (settings
for the other families are dropped from the snapshot's use but left in the stored JSON). Existing experiments point at
those baselines with no adjustments. dev-01 currently has no datasets.

## Build order

Each its own PR, CI green and checked on dev-01:
1. Baselines: `recipes.family`, labels, Baselines page, start dialog (type, start from, targets, assessors), Model
   switch removed, export family from the baseline.
2. Experiment adjustments: table, experiment mode in Review and Bulk edit, exports from baseline + adjustments,
   manifest and MLflow records.
3. Explore baseline picker, reuse by family, Datasets chips, views, docs.

## Testing

Unit and DB-backed API tests per step: a baseline's family is fixed (422 on change); starting needs the one in progress
finalized (409); populate copies decisions and only family-valid settings; blind no-leak tests extended to the new
endpoints; adjustments are owner-only, never alter the baseline, and an export reproduces baseline + adjustments; MLflow
records the adjustments. Playwright on dev-01: lead starts a Trees baseline from scratch with two members and
themself, finalizes; starts a Linear baseline populated from it; a member runs an experiment with two adjustments and
the run shows them in MLflow.
