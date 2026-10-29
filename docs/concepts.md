# Concepts

The terms Rainforest uses, in the order they build on each other. Every other page uses these names. Where the
code or the database uses a different name, the table at the end maps it.

## Data

**Dataset.** A file you upload (CSV or Parquet) or a sample you add. Rainforest copies it to Parquet, loads it into
Postgres as `raw.ds_<id>`, and profiles every column.

**Field.** One column of a dataset. Each field gets an inferred type (`numeric_continuous`, `numeric_discrete`,
`categorical`, `binary`, `text`, `datetime`, `identifier` or `constant`), statistics and suggestions. See
[Data preparation](data-preparation.md#type-inference).

**Target.** A field you want to predict. A dataset can have several targets, or none (unsupervised). A target must be
binary, categorical or numeric.

**Predict as.** How a model predicts a target. Each target has one of these values:

| Predict as | Meaning |
|---|---|
| *Classes* | One of the field's values (classification). |
| *Number* | The value itself (regression). |
| *Bands* | The number cut into ranges, each treated as a class. On a class target it is *One vs rest*: chosen classes against the rest. |

See [Predicting numbers and bands](data-preparation.md#predicting-numbers-and-bands).

**Derived field.** A column computed from the dataset's own fields, such as a ratio or a grouping of codes. It is
reviewed and exported like any other field.

## Decisions

**Decision.** What happens to a field. Each decision has two parts:

- The shared part: keep, drop or target, *Predict as*, type, notes, and whether it is reviewed.
- The settings: imputation, missing indicator, transform, scaling, encoding and encoder parameters. These belong to
  one model family.

**Model family.** The kind of model the data is prepared for. There are three:

| Family (UI) | Code value | Models |
|---|---|---|
| *Trees* | `tree` | LightGBM, XGBoost, histogram GBM, random forest |
| *Linear* | `linear` | logistic regression, ridge regression |
| *Neural* | `neural` | MLP |

The UI calls this the *model type*. Each family has its own suggested settings.

**Suggestion.** The decision and settings Rainforest proposes for a field, with the evidence behind each one. A
field stays at its suggestion until you change it. Confirming or annotating a suggestion leaves it a suggestion. A
field counts as *edited by you* once you change its keep or drop, or its type. Its settings count as *customised*
once they differ from the suggestion.

## The team process

**Baseline.** The team's agreed decisions for one dataset and one model family. A dataset has at most one baseline
per family, so at most three. A baseline can have a name, such as "Dropout risk".

**Version.** A numbered edition of a baseline, such as *Trees v2*. You change a finalized baseline by starting a new
version.

**Phase.** Where a version is in the team process:

| Phase | What happens |
|---|---|
| *Assessing* | Assessors review the fields, each on their own. |
| *Team review* | The team settles the fields where assessors disagreed. |
| *Finalized* | The version is frozen and can be exported. |

Only one version per dataset is in progress (assessing or in review) at a time.

**Assessor.** A person a lead assigns to review the fields of a version.

**Assessment.** One assessor's own decisions for a version. Nobody else sees an assessment while the version is
assessing, the lead included.

**Team review.** The phase after every assessor submits. Fields where every assessor made the same core choice are
*adopted*. The rest wait in the review queue.

**Review queue.** The fields assessors disagreed on, shown side by side with comments in *Fields* during team review. A
lead sets the team decision for each.

**Team decision.** The decision that counts for the version: adopted, or set by a lead in team review.

**Snapshot.** The frozen record of a finalized version: every field's decision, its settings for the baseline's
family, its inferred type and key statistics, and the target order.

**Carried-over field (locked).** In a new version, a field the lead didn't reopen. It keeps the previous version's
decision, and nobody can change it in that version. The *Fields* list marks it with 🔒.

**Reopened field.** In a new version, a field the lead chose for assessment again.

## Experiments and training

**Experiment.** A workspace for exports and model runs, built from one source. There are two kinds:

| Kind | Built from | Who sees it | Who adds exports and runs |
|---|---|---|---|
| Team experiment | A finalized baseline version | Everyone | Anyone (except Viewers) |
| Personal experiment | Your own assessment of the version being assessed | Only you while the version is assessing, then everyone | Only you |

A personal experiment is **private** (🔒) while its version is assessing.

Each finalized version also has a default team experiment, named like *Team · Trees v2*. It is read-only for
adjustments.

**Adjustment.** A change a team experiment makes to its own copy of the baseline, such as dropping a field or
changing a setting. Adjustments never change the baseline.

**Export.** A model-ready dataset built from an experiment's source: split into train, validation and test,
preprocessed, and written as Parquet with its fitted preprocessor and a manifest. An export freezes its source, so it
always rebuilds the same way.

**Model run.** One training job on one export. It records its settings, metrics and model files.

**Chain.** The models a run trains, one per target, in an order. Each model also sees the predictions for the
targets before it.

**One-vs-rest run.** A run that trains one binary model: one class of a class target against every other class,
calibrated so its scores are probabilities. See
[One class against the rest](experiments-and-training.md#one-class-against-the-rest).

**Batch.** Runs started together on one export: one-vs-rest runs, one per class of a target (*Per class*), or one
run per combination of settings, ranked on validation (*Search*). The runs train one at a time. See
[Batches](experiments-and-training.md#batches).

**Deployment.** Finished runs of one export, frozen with the export's preprocessing, that score new records sent
through an API call with a key or in an uploaded file. It is for testing, not production serving. See
[Deployments](deployments.md).

### A Rainforest experiment is not an MLflow experiment

Rainforest logs model runs to [MLflow](mlflow.md). The word "experiment" means different things in each:

| | Rainforest experiment | MLflow experiment |
|---|---|---|
| Scope | One source (a baseline version or an assessment) | One whole dataset |
| How many per dataset | As many as people create | One, named `rainforest/<dataset id>-<dataset name>` |
| Link between them | — | Each MLflow run carries the tags `rainforest.experiment_id` and `rainforest.experiment_name` |

An experiment's *Open its runs in MLflow ↗* link, on its *Notes* tab, opens the dataset's MLflow experiment filtered to
that experiment's runs.

## Roles

| Role (UI) | Code value | In short |
|---|---|---|
| *Team Lead* | `lead` | Runs the team process and manages people. |
| *Team Member* | `member` | Assesses when assigned, uploads, experiments and trains. |
| *Viewer* | `viewer` | Looks, changes nothing. |

See [Roles and permissions](team-workflow.md#roles-and-permissions).

## UI names and code names

| In the UI | In the code and database |
|---|---|
| Baseline version | `recipe` (`meta.recipes`, `recipe_id`, `recipe_version`) |
| Model type | `family` (`tree`, `linear`, `neural`; `model_family`) |
| *Assessing* / *Team review* / *Finalized* | phase `assessing` / `review` / `finalized` |
| Reopened version | `scope = 'reopened'` (a full version is `all`) |
| *Classes* / *Number* / *Bands* or *One vs rest* | `target_task` `classes` / `regression` / `bins` |
| Bands definition | `target_bins` |
| Settings | `meta.field_settings` (per family) |
| Team decision | `meta.field_decisions` |
| Assessment | `meta.assessments`, `meta.assessment_settings` |
| Default team experiment | `meta.experiments.is_default` |
| Experiment source | `recipe:<id>` or `assessment:<recipe id>:<user id>` |
| Model run | `meta.model_runs` |
| Chain kind | `classifier_chain` or `target_chain`; `one_vs_rest` for a one-vs-rest run |
| Batch (*Per class* / *Search*) | `meta.run_batches`, kind `one_vs_rest` / `search` |
| Deployment, its keys and its tests | `meta.deployments`, `meta.deployment_keys`, `meta.deployment_tests` |
