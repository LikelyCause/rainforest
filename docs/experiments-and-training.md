# Experiments and training

How you turn a finalized baseline into a model-ready export and train models on it. Terms are defined in
[Concepts](concepts.md).

## Experiments

Exports and the model runs trained on them belong to an experiment. The dataset's *Experiments* tab lists them.

To create one, choose *New experiment*, give it a name, and choose what it is built from. Right after a lead
finalizes a version, *New experiment from vN* does the same with that version chosen.

| Built from | Kind | Who sees it | Who adds exports and runs |
|---|---|---|---|
| A finalized baseline, such as *Trees baseline v2 · finalized* | Team experiment | Everyone | Anyone who can experiment |
| *My assessment (Trees v2)*, offered while you assess the version being worked | Personal experiment | You only, until team review opens | You only |

If you export a finalized baseline from the API without naming an experiment, the export goes into that version's
default team experiment, *Team · Trees v2*. Rainforest makes it on first use.

A personal experiment's exports use your own choices as they stand when you export. Fields a later version carried
over (locked) take the version's values. The export freezes that snapshot, so it rebuilds the same way after you
change your assessment.

### Privacy

A personal experiment is **private** (🔒) while its baseline version is assessing. Nobody else sees it, the lead
included. That covers its exports, runs, files, *Explore* results on its exports, and anything about it in the
*Datasets* list. Opening one by URL answers as if it didn't exist.

From team review onwards, a personal experiment is visible to the whole team, like the assessment it was built from.

The SQL console never shows a private experiment or its exports, runs or metrics. This uses row-level security. The
console doesn't know who is asking, so this includes the owner. See [SQL](sql.md#privacy).

### The experiment page

On an experiment's page, the second row of the top bar names the experiment, with a menu to switch to another one of
the dataset's experiments. It shows what the experiment is built from, such as *From Trees baseline v2 · as
finalized*, whether it is private, and *Train a model*. Its tabs are:

| Tab | What it holds |
|---|---|
| *Runs* | The [runs table](#the-runs-table), with a run's detail beside it. |
| *Train* | The [Train tab](#the-train-tab): what to train, the model's settings, and *Train*. |
| *Exports* | What goes into an export, the split and rows, and the [exports](#exports) made. |
| *Adjustments* | What the experiment changes from its baseline. See [Adjusting an experiment](#adjusting-an-experiment). |
| *Notes* | Its name, notes, best run, chosen model, *Open its runs in MLflow*, and *Archive*. |

A new experiment opens on *Exports*.

The owner (for a personal experiment, its assessor) or a lead can rename it, edit its notes, and archive or unarchive
it, all on *Notes*. Archiving asks to confirm. An archived experiment takes no new exports or runs.

## Adjusting an experiment

A team experiment holds its own copy of its baseline. Its owner can change that copy without touching the baseline.

Per field, you can change:

- the role: keep, drop or target
- a target's *Predict as*
- the type
- the settings: missing values, missing indicator, transform, scaling, encoding and encoder parameters

Changes stay within the baseline's model type. A type change brings the new type's suggested settings where the old
ones don't fit.

1. Choose *Adjust fields* on the experiment's *Adjustments* tab. It opens *Fields · Focus* and *Fields · Table* on the
   experiment's copy (`?view=experiment:<id>`). The top bar's second row names the experiment, with the *Experiment
   copy* chip and *Back to the experiment*.
2. Change fields as in *Focus*. The decision panel says what differs from the baseline, with *Revert*.
3. Check the experiment's *Adjustments* tab. It lists every changed field, and the targets when they differ, with a
   per-field *Revert* and *Revert everything*. *Revert everything* asks to confirm.

The experiments list badges an adjusted experiment *N adjusted* and *Targets changed*. Its page shows *N fields
adjusted* and *Targets changed*.

### Details

- Only the owner adjusts, and only while the experiment isn't archived.
- The default *Team ·* experiments, and other people's experiments, are read-only here. Start your own experiment to
  try changes.
- A personal experiment follows its assessment instead.
- You can change the targets. The export then uses the experiment's targets. Its preview warns that the results
  aren't comparable with experiments on the baseline's targets. A field newly made a target gets the
  [leak screen](data-preparation.md#associations-and-the-leak-screen).
- Each export freezes the baseline and the adjustments as they stand, and records them in its manifest and in
  [MLflow](mlflow.md).

## Exports

An export is built from its experiment's source, frozen at export time:

- for a team experiment, the finalized version's snapshot, or the experiment's adjusted copy of it
- for a personal experiment, its assessor's assessment

A team export is never built from decisions being worked on. So a finalized version always exports the same way
while another version is in progress. An assessor can export their own assessment, privately, while they assess.

The model family is always the baseline's.

### What an export records

Each export records its baseline (`recipe_id`) and its experiment. The list shows the baseline as *Trees v2*. On *Runs*,
the export is noted with *N fields adjusted* and *targets changed* when they apply.

The manifest's `config` records:

| Key | Contents |
|---|---|
| `source` | `recipe` or `assessment` |
| `recipe_id`, `recipe_version` | the baseline version |
| `experiment_id` | the experiment |
| `baseline` | `recipe_id`, `family`, `version`, `name`, `label` |
| `adjustments` | `count`, `fields`, `targets_changed`, `targets`, and `detail`: each adjusted field's changes |
| `assessor_id` | for a personal export only |
| `pull_id` | for a connected dataset only: the pull its data came from |

A connected dataset's data changes when a lead refreshes it, so an export of one also records the pull it was built
from: the newest finished pull when the export starts. The export row carries it as `pull_id`, the manifest's `config` repeats it, and each run trained on the export logs it to MLflow as the parameter
`rainforest.pull_id`. A refresh never touches an export's files, so a run on an older export still trains on the pull
that export names.

### Before you export

*Exports*, under *What will be exported*, summarises what goes in: kept, dropped, target and customised fields, and
kept fields by type.

It warns about:

- unreviewed fields
- kept fields that look like a leak of the export's own targets
- targets no leak check has run for
- targets that differ from the baseline's
- a baseline finalized before the connected dataset's latest pull (*Baseline finalized before pull #3; consider a new
  version*)

These block the export:

- no target
- no kept features
- a field in the baseline that a refresh took out of the data
- a target class with a single row
- a numeric target with values that don't read as numbers, or with a single value
- a band with fewer than 2 rows

Once an export exists, it shows how many decisions changed since, with *Re-export with same split*.

You can adjust the validation and test shares, and, under *Advanced*, the seed and *Max rows to fit on*. *Export for
trees models* (or linear, or neural) builds it.

### Rows

Under *Rows* in the export panel, you can leave rows out before anything else happens:

| Option | What it does | Example |
|---|---|---|
| *Leave out exact duplicate rows* | drops exact duplicates | — |
| *Leave out rows where …* | drops rows whose field holds given values | `discharge_disposition_id` 11, 13, 14, 19–21, for patients who died or went to hospice |
| *One row per* … *first by* … | keeps only the first row per key | one row per patient number, first by encounter number |

Under the split bar, the panel counts what the rules leave, with the export's own row-rule and split code, so the
preview and the export agree: *69,990 rows after the row rules (1,246 left out)*, then the exact train, validation and
test rows the export will make with these shares and seed, a grouped split included. Kept rows with no target value
are named too, since the split leaves them out. The count updates a moment after you stop changing the rules or
shares.

The export row says how many rows each rule removed (`split_info.rows_removed`).

### Grouped split

*Keep rows together by* names a field, such as a patient number. Every row sharing its value lands in the same split
and, for cross-validation, the same fold. So a model is never scored on someone it trained on.

The split then uses scikit-learn's `GroupShuffleSplit`. Its shares are of groups, so row shares come out close but
not exact; the panel shows the split's actual counts and how many groups there are. Folds use `StratifiedGroupKFold`
(`GroupKFold` when every target is a number). The manifest records `split_info.grouped_on`, and the export writes
`groups.npy`.

### Leakage-safe split

Rainforest assigns the train, validation and test split first. Then it fits every statistic on training rows only,
up to *Max rows to fit on* (every row from the UI, 500k from the API) and the [Memory](#memory) budget. Then it
transforms the full file in chunks.

The split is stratified on the combination of targets. Details:

- Combinations with a single row are pooled.
- If the combinations still can't be stratified, it stratifies on one target, else not at all.
- A numeric target stratifies on bands: its cut points, else 5 equal-count bands. So each split covers its range.
- A target with a class that has a single row blocks the export, naming the class.

### Memory

Both the fit and the transform are sized by output width:

| Budget | Setting | Default |
|---|---|---|
| Dense output per transform chunk (rows × features × 8 bytes) | `EXPORT_CHUNK_MB` | 256 MB |
| Fit sample | `EXPORT_FIT_MB` | 512 MB |

So a wide export fits on fewer rows. The manifest records `fit_sample_rows` and `chunk_rows`. Peak memory runs at a
few times these budgets. A 300k-row export peaks at about 1.3–2 GB.

### Output

An export writes:

- float32 features
- `__split`
- `__row_id`, which joins back to `raw.ds_<id>`
- the targets
- `preprocessor.joblib`, the fitted preprocessor
- `manifest.json`: feature → source field, class mappings, model family, a NaN report, and `inf_values` (±inf per
  field, written as missing)

Feature names never take a target's name, `__row_id` or `__split`. A clashing feature gets a numbered suffix.

### Targets

Targets follow their *Predict as*:

| Predict as | Written as |
|---|---|
| *Classes* | label-encoded |
| *One vs rest* | binary: the chosen classes against `(rest)` |
| *Number* | float64 |
| *Bands* | class codes labelled like `< 10`, `[10, 25)`, `≥ 80`. Equal-sized bands are cut on the training split only. |

Numeric targets are read as profiling reads them, with thousands separators and decimal commas.

The manifest records:

- `target_tasks`
- `target_bins`: `edges`, `labels`, `k`
- `target_stats`: the training `mean`, `std`, `min` and `max` of each number
- `target_kinds`: `binary`, `multiclass` or `regression`
- `target_classes`, with no entry for a number

### Encodings worth knowing

- A binary field becomes one 0/1 column named after its second level (`smoker_yes`).
- A binary field's missing values take the most frequent level, or the level you name as the fill value. Any other
  fill label adds a `<field>__isna` column instead of a third code.
- Numeric fill values are applied before log1p or winsorizing, so they are in the field's own units.
- A field retyped to a date has its date format detected once over the whole column when you save. So every chunk of
  the export reads it the same way.

## Training

### Compatibility

- Tree models train on any export.
- Linear and neural models need an export prepared for their family. They are refused if the export contains NaN.
- The *Train* tab hides incompatible models. *Show incompatible models* lists them, disabled and marked *not
  compatible*.
- Planned models (linear SVM, k-nearest neighbours, convolutional and recurrent networks) are listed as *planned*.
  You can't train them yet.

### The Train tab

*Train a model*, in the top bar, opens the experiment's *Train* tab. It picks:

- the export: the newest finished one, unless you choose another
- the model
- CPU or GPU, when a GPU is available (see [GPU training](#gpu-training))
- the cross-validation folds
- which of the export's targets to train, and in which chain order. Tick one target for a single model, or several
  and order them.
- the model's settings

A summary beside it, *Ready to train*, sums up the model, the export, the changed settings, the scoring and the number
of runs, above *Train*. Starting a run takes you to *Runs*, where it appears at the top as it trains.

Each export, on *Exports*, has:

- *Train on it*, which opens *Train* with that export
- *Files ▾*: *Dataset (.parquet)*, *Pipeline (.joblib)*, *Manifest (.json)*
- *Use from code*: the CLI command and class mappings

### Chains

A run trains one model per target, in the export's target order unless you choose another. Chains use `cv=5` by
default (*Chain folds*: *Off* or 2–10; *Off* passes the true values down).

During training, each chained prediction is **out-of-fold**. So a later target's model learns from realistic
predictions of the earlier targets, not the ground truth.

| Targets | Chain | Passed down the chain |
|---|---|---|
| Class targets only (bands included) | scikit-learn's `ClassifierChain(model, order=<targets>, cv=5)` | all binary: P(class 1); any multiclass: the predicted class code |
| Any number | Rainforest's `TargetChain`, a classifier or a regressor per target | binary: P(class 1); multiclass: the predicted class; number: the predicted value |

### Details

- Every model has a regressor: LightGBM, XGBoost on CPU or GPU, histogram GBM, random forest, ridge for the linear
  family, and the MLP.
- Classifier-only parameters (`class_weight`, `scale_pos_weight`, `C`, …) are left out for a regressor, and the
  number-target loss (`objective`) for a classifier.
- The report records `chain_kind` (`classifier_chain` or `target_chain`) and `target_tasks`.
- The linear model (`logistic_regression`) trains ridge regression for a number, so a run whose targets are all
  numbers is named *Ridge regression*, and one with both kinds *Logistic and ridge regression*. *Train* names it the
  same way for the targets you pick, and shows ridge's settings for a number. The model key stays
  `logistic_regression`.
- *Model* in the run detail downloads the chain as `run<id>_<model>_classifier_chain.joblib`, or
  `…_target_chain.joblib` when any target is a number.
- NaN-capable models are wrapped, so scikit-learn's finiteness check doesn't reject the NaN they handle natively.
- A class that is missing from training (a rare class that only landed in validation or test) keeps its own
  probability column, set to zero. So metrics stay aligned with the class names. The report lists these classes under
  `missing_from_train`.
- A target with a single class in the training split stops the run before fitting.

### Model settings

*Train* lays out the chosen model's parameters. Every trainable model has them: LightGBM, XGBoost, histogram GBM,
random forest, logistic or ridge regression, and the MLP. The MLP's layers are a choice of presets, such as *128 × 64*.

Each parameter has its own control: a dropdown for a choice (boosting type, loss, class weights), a number, or an
on/off checkbox. The controls change with the model. They are grouped:

1. run
2. network (the MLP's)
3. trees
4. learning
5. sampling
6. regularization
7. *For class targets*, shown only when you train one
8. *For number targets*, shown only when you train one

Every other parameter that changes the model sits under *Advanced*. Groups start folded unless they hold a change,
and *Find a setting* searches them all.

Each setting shows a one-line explanation. A changed setting is marked, shows its default and has *Reset*. *Reset
all* puts every setting back. A folded group sums up its settings in a line.

### Details

- Settings that depend on another appear only when it calls for them. Examples: DART's drop settings with DART, or
  a loss's own setting, such as the quantile, with that loss. They then train with the value shown, even if you leave
  it unchanged.
- A value outside its range blocks *Train*.
- A combination the library refuses is caught when you ask for the run, by fitting the settings on 64 synthetic
  rows. An example is LightGBM's random forest without bagging or a feature sample.
- A loss that needs non-negative numbers (Poisson, Tweedie) is refused for a target whose training minimum breaks it.
  Gamma needs positive numbers. Squared log error needs numbers above -1.
- *Train* remembers each model's last settings per export, in your browser.
- A run stores the settings that differ from the defaults. Its detail lists them, with *Train again*, which opens
  *Train* with the run's export, model, chain order and settings.

### Trying several values

Under the settings, *Try several values* takes up to two settings and the values to try for each. For example,
`15, 31, 63` leaves, or `0.03, 0.1` learning rate.

*Train* becomes *Train N runs*. It starts one run per combination, at most 50. Each is a normal run of the
experiment, so the runs table and its ★ compare them. Every combination's values are checked before any starts.
The runs then start one by one, and one the API refuses stops the rest.

### Cross-validation

A run can also fit its chain on K folds of the training rows. It uses the same model, settings, chain folds and
device. It scores each held-out fold with the same metrics. So each metric gets a mean and a spread (the standard
deviation over folds), beside validation and test.

K is the *Train* tab's *Cross-validation*: *Off*, or 2–20 folds. The default is 5. A run takes about K + 1 times as
long.

### Details

- Validation and test rows never take part.
- The saved model is still the one fit on every training row.
- Folds keep the class shares of the chain's first class target when its rarest class has at least K training rows.
  Otherwise they are drawn at random, and the run says so.
- If a fold's training rows would lose a target's second class (or a number target's spread), or a fold fails to fit,
  cross-validation is skipped with a note. The run still completes with its model and its validation and test scores.
- The run detail's cross-validation view shows how long the folds took. Under *Per fold*, it shows every fold's
  scores.
- Cross-validation is separate from *Chain folds*, which only decides what chained targets pass down.

## Metrics

Every run is scored per target on the validation and test splits.

**Class targets** (bands included) get:

- accuracy, balanced accuracy, log-loss
- ROC-AUC (one-vs-rest macro for multiclass)
- precision, recall and F1, macro and weighted. Binary targets also get the positive class's values and
  specificity.
- confusion matrices and a per-class table
- ROC curves: one-vs-rest per class for multiclass targets, up to 200 points each
- PR-AUC (`pr_auc`; `pr_auc_macro` for multiclass). Read this score when the positive class is rare.
- for a binary target, a threshold table: precision, recall, F1 and the share flagged at each cut from 0.05 to 0.95
- the top importances per chain link

**Number targets** get:

- `rmse`, `mae`, `r2`, and `mape` (only when no actual value is 0)
- up to 1,000 sampled (actual, predicted) pairs
- a 30-bin histogram of the errors
- range scores: actual and predicted values put into up to 5 ranges of roughly equal count, cut on the training rows.
  Tied values can make fewer ranges. These give `band_accuracy`, `band_f1_macro`, a confusion matrix, and per-range
  precision, recall and F1.

## The runs table

*Runs* lists the runs of one export at a time; the picker above the table chooses the export. The export's targets are
the columns: AUC, PR AUC and F1 (macro) for each, or R² and RMSE for a number.

The *Validation / Test / Cross-validation* switch picks the split. *Cross-validation* shows each mean ± spread, and —
for runs without cross-validation.

In each column, ★ marks the run that scored best on the validation split for that target and metric (lowest for RMSE).
So choosing it doesn't use the test split. On *Test* or *Cross-validation*, the star can sit on a lower number than
another run's. Its tooltip gives the validation score it was picked on.

A lead can mark one run per experiment as the chosen model (see [Choosing a model](#choosing-a-model)).

### Run detail

Select a run to open its detail beside the table (below it on a phone). The detail has *Train again*, *MLflow ↗*,
*Model* (download), the run's note and, for a lead, *Mark as chosen*. Per target, it shows:

- its headline metrics. For a binary target, precision, recall and F1 are the positive class's. Otherwise they are
  macro averages.
- for a class target, tabs for the confusion matrix (correct predictions shaded blue, errors red), the ROC curves, and
  the per-class table ending in the macro and weighted averages, plus *Threshold* for a binary target
- for a number, R², RMSE, MAE and MAPE tiles over tabs: the predicted-vs-actual scatter, the error histogram,
  and *Ranges* (the range confusion matrix and per-range precision, recall and F1)
- a *Top features* tab, when the run recorded them

A *Test / Validation* switch applies to every target. With cross-validation, the switch adds *Cross-validation*. It
shows each target's headline metrics as mean ± spread, with every fold's value on hover, and a note when the folds
weren't stratified or cross-validation was skipped.

A failed run's detail shows its settings, why it failed in one sentence (the full message under *Details*), and *Train
again*.

### Choosing a model

Each run has a note that its author or a lead can write: why this run, what it showed. A lead can *Mark as chosen*, one
run per experiment, and *Unmark* it. The runs table shows *Chosen* beside it, with who chose it on hover. The
experiments list shows *Chosen: LightGBM*, and the *Datasets* page shows the chosen model as the dataset's *Best
result*.

## GPU training

XGBoost can train on an NVIDIA GPU when the API runs the GPU image. Setting that up is in
[Deployment](deployment.md#gpu).

Training on the GPU is a per-run choice. When the API sees a usable GPU, the *Train* tab shows a CPU / GPU
switch with the GPU's name, for models that support it. It defaults to GPU. On the GPU, XGBoost uses
`device="cuda"` and `tree_method="hist"`.

The runs table has a *Device* column, and each run's `report.json` records `device`. Other models always train on
the CPU. *Explore*'s analyses run on the CPU.

## Training from the command line

```bash
make train EXPORT=1                          # LightGBM
make train EXPORT=1 MODEL=random_forest
docker compose exec api python -m app.modeling.train --export-id 1 --model xgboost \
    --order column_b,column_a --param n_estimators=1000 --param learning_rate=0.03
make train EXPORT=1 MODEL=xgboost DEVICE=cuda  # GPU image only
make models                                  # what's registered / installed, and on which devices
```

`--order` takes the export's own target names. A generated sample's are `column_a` and `column_b`, plus `column_c`
when generated with a numeric target.

CLI runs write `classifier_chain.joblib` (on disk, for either kind of chain; the download and MLflow name a target
chain `target_chain.joblib`) and `report.json` under
`/data/models/export<id>_<model>_<time>/`. They aren't recorded in `meta.model_runs`. So they don't appear in the UI's
runs table or `meta.v_model_metrics`, and they aren't logged to MLflow.

## Adding a model

Add one `ModelSpec` to `api/app/modeling/registry.py`. It needs:

- a key and a label
- its family
- whether it supports NaN
- a builder and default parameters

Optionally, it takes how to read its importances, a description, the package it requires, `cuda_params` for GPU
training, and `build_regressor` for number targets. The model then appears in the CLI, the API and the UI. *Train*
shows its settings once they are listed in `CATALOG` in `api/app/modeling/params.py`.

To bake an extra package into the API image, set `API_EXTRA_PIP` in `.env` (for example, `catboost`). See
[Configuration](configuration.md).
