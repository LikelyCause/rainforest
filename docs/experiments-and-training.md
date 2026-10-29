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
| *Runs* | The experiment's [batches](#batches) and the [runs table](#the-runs-table), with a run's detail beside it. |
| *Train* | The [Train tab](#the-train-tab): what to train, the model's settings, *Train*, and the batches. |
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
| `adjustments` | `count`, `fields`, `targets_changed`, `targets`, `detail`: each adjusted field's changes, and `removed_by_refresh`: the baseline's fields a refresh took out of the data (not counted as adjustments) |
| `assessor_id` | for a personal export only |
| `pull_id` | for a connected dataset only: the pull its data came from |
| `derived` | the derived fields the export reads, with the derived fields they read: each one's `name`, `kind` and `spec` as defined when the export started, in the order they were added. One that reads a field no longer in the dataset also has `unavailable`, saying which: the export still reads its column, but it can't be computed from new records |

The export row's `config` also carries `input_schema`, written when the export finishes. It has one entry per source
field the export reads: its features, targets and row-rule fields, and the fields its derived fields read. Each entry
holds:

- `dtype`: the column's polars dtype in the dataset's Parquet copy
- `datetime_format`: for a date feature, the format the export read it with; for a date field read only by a row rule
  or a derived field, the format detected over its whole column
- `integer_codes`: true for a category stored as integers, so `7.0` and `07` read as `7`

Scoring reads new records with these and the frozen `derived` formulas, so removing a derived field later doesn't
change an export. Exports made before these were recorded have neither key.

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
- target-encoded fields that follow a first target other than the baseline's

These block the export:

- no target
- no kept features
- a field the export keeps, or one of its targets, that a refresh took out of the data (a field it drops may go).
  For a team experiment that is what its own copy kept or targeted, adjustments included, and for a personal one what
  its assessment did, as the refresh recorded them before deleting the field; for an experiment made after that
  refresh, what its baseline kept or targeted
- a target class with a single row
- a numeric target with values that don't read as numbers, or with a single value
- a band with fewer than 2 rows
- a target-encoded field whose target the experiment doesn't predict

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

The fit sample is also where target encodings are learned. A target encoding needs every class of its target in it,
or the export stops and says which class is missing.

### Output

An export writes:

- float32 features
- `__split`
- `__row_id`, which joins back to `raw.ds_<id>`
- the targets
- `preprocessor.joblib`, the fitted preprocessor
- `manifest.json`: feature → source field, class mappings, model family, a NaN report, `inf_values` (±inf per
  field, written as missing), `out_of_fold_features`, and `target_aware` (per target-encoded field: the option, the
  target, its kind, and which class each feature averages)

On the fit rows (the `fit_sample_rows` training rows drawn with the export's seed), the `out_of_fold_features` hold
out-of-fold values, so they differ from `preprocessor.transform` there by design. Every other row matches it.

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
- A field retyped to a date has its date format detected once over the whole column when you save. A date field with
  no recorded format has it detected over the whole column when the export starts. So every chunk of the export, and
  scoring, read it the same way.
- A target-encoded field's columns average its target. Refitting `preprocessor.joblib` needs `y` as a DataFrame with
  that target's column, encoded as the export writes it. `transform` needs no target.

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

Under *Train*, two buttons start a [batch](#batches) with the same export, model, settings, device and
cross-validation: *Train one model per class…*, shown when the export has a class target, and *Search settings…*.

Each export, on *Exports*, has:

- *Train on it*, which opens *Train* with that export
- *Files ▾*: *Dataset (.parquet)*, *Pipeline (.joblib)*, *Manifest (.json)*
- *Use from code*: *Python script (.py)* (see [Rebuild an export in Python](#rebuild-an-export-in-python)), the CLI
  command and class mappings

### Rebuild an export in Python

*Use from code → Python script (.py)* downloads one file, `export<id>_<dataset>.py`, that rebuilds the export outside
the app. Hand it to someone with the original data and they get the same model-ready rows, split, features and target
codes the app's `model_ready.parquet` holds, without access to Rainforest.

```bash
uv run export12_customers.py customers.csv            # rebuild it from the uploaded file and check it
uv run export12_customers.py customers.csv --out model_ready.parquet
uv run export12_customers.py customers.csv --clean clean.parquet
uv run export12_customers.py customers.csv --compare export12_model_ready.parquet
uv run export9_concrete_strength.py                   # a sample dataset: no file needed
```

`uv run` installs the library versions the script names in its header (the app's own); any environment with them
works too. From Python, next to the file:

```python
from export12_customers import load_model_ready, load_clean, xy

data = load_model_ready("customers.csv")     # like model_ready.parquet: __row_id, __split, features, targets
X_train, y_train = xy(data, "train")
clean = load_clean("customers.csv")          # the decisions applied, nothing fitted
```

The script holds:

| Part | What it is |
|---|---|
| The data | How the data is read: the uploaded file with the app's reading rules (separator, missing-value markers such as `?`, column names), and the derived fields the export uses, with their definitions as they were at export time |
| The decisions | `kept_fields()`: one line per kept field, such as `keep.numeric('age', imputation='median', add_missing_indicator=True)`, with the team's notes and the experiment's adjustments as comments; the fields left out; `targets()` in chain order, such as `predict.bands(5)`; the row rules, split shares and seed, and the export's *Max rows to fit on* and memory budgets |
| Rebuilding | Rainforest's own export code, copied from the app, so the script runs the same steps rather than a rewrite of them |
| The checks | What the app read (rows, and each original column's type, missing values and a hash of its values) and what it exported (each column's hash, and sums for the float columns) |

Before anything else, the script checks that the data reads as the app read it, and stops (exit code 3) if it
doesn't, naming the columns that differ. Derived fields aren't checked here: they are computed from the columns that
are. `--no-check` goes on anyway; the result is still checked. After rebuilding, the script checks the result against
the export: *Identical to the app's export*, or columns that differ only by floating-point rounding (another
machine's maths libraries can round the last digits of a log or a text feature differently), or what differs (exit
code 1). `--compare` checks value by value against the export's own `model_ready.parquet`: row ids, the split and
class codes must be equal, and a feature may differ by rounding (up to 1e-4 of the column's largest value).

| Exit code | Meaning |
|---|---|
| 0 | The result matches the export, exactly or but for floating-point rounding |
| 1 | The result differs from the export (or from the file given to `--compare`) |
| 2 | The command line is wrong, or the data file isn't the kind this export reads |
| 3 | The data doesn't read as the app read it |
| 4 | The rebuild failed; the error is printed |

Edit the decisions to try other choices: the script then builds what the app would have exported with them, and its
check says the result no longer matches the export.

Where the data comes from:

| The dataset | The script |
|---|---|
| An uploaded file | Takes the file as its first argument |
| A sample library dataset | Downloads it from its UCI source and builds it as the app's copy was built, or takes the app's Parquet copy from `api/app/sample_library/data/`; the *Concrete strength* build also installs `fastexcel` |
| A generated sample | Makes it again with the app's generator and the settings it was made with, and takes no file. A sample generated before Rainforest recorded those settings is recognised by its default name when making it again gives the same data; otherwise the script can't be written, and generating the sample again fixes that |

The script is written when you ask for it, by the app as it is then. An export made by an earlier version whose
library versions differ says so in the script's header; its check then shows whether the result still matches. An
export that target-encodes a field can't be written as a script, and *Python script (.py)* says so.

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

This is for comparing a few values by eye. For more settings, ranges, more runs or a ranked leaderboard, use
*Search settings…*; [When to use which](#when-to-use-which) sets the two side by side.

### One class against the rest

A one-vs-rest run trains one binary model: one class of a class target against every other class. A
[one-model-per-class batch](#one-model-per-class) starts one per class; through the API, a run with `positive_class`
(and `target`, when the export has several targets) is one. The export's other targets take no part, and nothing is
chained. The run is named for the class, as in "LightGBM · vehicle = V07 vs rest".

- The model is calibrated on the validation split, so its scores are probabilities that compare across classes:
  isotonic regression when the validation split has at least 200 rows of the class and 200 of the rest, else a
  sigmoid. The report's `calibration` gives the method and the Brier score before and after, on validation and test.
- Every score the run reports, and the saved model's predictions, are calibrated.
- The validation split also chooses thresholds, so its scores are slightly optimistic; the test split gives the
  honest number.
- `threshold_curve` holds precision, recall, F1 and the share flagged at every threshold from 0 to 1 in steps of
  0.01, for validation and test. `f1_threshold` keeps the threshold with the best F1 on validation and its F1 on
  each split.
- `balance` weights the classes by their training counts: LightGBM and XGBoost get a positive-class weight of
  negatives / positives, and random forest, histogram GBM and logistic regression get balanced class weights. MLP has
  no class weights; the report notes that balance was ignored. Balance can't be combined with a class-weight setting
  changed from its default.
- Cross-validation folds are calibrated on the validation split too.
- The training split needs rows of the class and of the rest, and so does the validation split.
- Its scores are one class's against the rest, not the target's, so the runs table's ★ and the experiment's best run
  leave it out.

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
- Target encodings' out-of-fold values come from the export's own folds, not cross-validation's. So cross-validation
  scores on target-encoded features are slightly optimistic. Validation and test scores aren't affected.

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
  (and, for a one-vs-rest run, a threshold curve from 0 to 1 in steps of 0.01)
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
another run's. Its tooltip gives the validation score it was picked on. One-vs-rest runs never get the star, and the
experiment's best run (on *Notes*, in the experiments list and as the dataset's *Best result*) leaves them out too:
their scores are one class's against the rest. Compare them on their [batch page](#one-vs-rest-results).

A run started by a batch carries the batch's chip, such as *Per class #3* or *Search #4*, with the batch's name on
hover. The chip opens the batch's page.

A lead can mark one run per experiment as the chosen model (see [Choosing a model](#choosing-a-model)).

### Batches on Runs

Above the runs table, *Batches* lists the experiment's batches, newest first: the latest 5, with *Show all N*. Each
row has the batch's name, which opens its page, its kind (*Per class* or *Search*), how many classes it skipped, its
export, its runs as a bar and a count (*12 of 40 done · 1 failed · 27 waiting*), who started it and when, and its
status. The list isn't shown until the experiment has a batch.

### Run detail

Select a run to open its detail beside the table (below it on a phone). The detail has *Train again*, *Deploy…* (for
a finished run with one target, to anyone who can add runs; see [Deployments](deployments.md#the-deploy-dialog)),
*MLflow ↗*, *Model* (download), the run's note and, for a lead, *Mark as chosen*. Per target, it shows:

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

For a [one-vs-rest run](#one-class-against-the-rest), the headline precision, recall and F1 are at 0.5, but the run's
own threshold is the one with the best F1 on validation. A note beside the switch says so and gives that threshold
and its F1 on validation and test, as in *This run's threshold is 0.34, the highest F1 on validation: F1 0.712 on
validation, 0.688 on test.* When no threshold has a positive F1, 0.5 stands in, marked *(fallback)*, and the note
says why. *Threshold* then reads the run's fine curve (every 0.01 from 0 to 1) and its slider starts at that
threshold.

A failed run's detail shows its settings, why it failed in one sentence (the full message under *Details*), and *Train
again*.

### Choosing a model

Each run has a note that its author or a lead can write: why this run, what it showed. A lead can *Mark as chosen*, one
run per experiment, and *Unmark* it. The runs table shows *Chosen* beside it, with who chose it on hover. The
experiments list shows *Chosen: LightGBM*, and the *Datasets* page shows the chosen model as the dataset's *Best
result*.

## Batches

A batch is many runs started together on one export. There are two kinds:

| Kind | Chip | Runs |
|---|---|---|
| [One model per class](#one-model-per-class) | *Per class* | a [one-vs-rest run](#one-class-against-the-rest) for each class of a class target |
| [Settings search](#settings-search) | *Search* | one run per combination of settings, ranked on validation |

Both start from the [Train tab](#the-train-tab), with its export, model, settings, device and cross-validation. The
dialog sums these up under *Model*, *Settings*, *Scoring* and *Device*; change them on *Train* before opening it.
Anyone who can add runs to the experiment can start a batch.

Before anything starts, every run the batch plans gets a single run's checks, the 64-row dry fit included. A refusal
stays in the dialog and starts nothing. A batch is held to `BATCH_MAX_RUNS` runs (500 by default; see
[Configuration](configuration.md)): the dialog counts the runs it would start, as *40 runs · at most 500 in one
batch*, and won't start a larger one.

*Start N runs* takes you to *Runs* on the batch's export, where its runs appear at the top. A message names the batch
and its run count, with the classes it skipped and, for a random search, how many draws the checks refused.

### One model per class

*Train one model per class…* opens *Train one model per class*. It trains a binary model for each class of the
target, each calibrated and thresholded on validation as [One class against the rest](#one-class-against-the-rest)
describes.

- *Target* picks the class target, when the export has several.
- The class table lists each class with its rows in *Train*, *Valid* and *Test*, ticked to train. *Tick all* and
  *Untick all* change every row. The last column says *Trains*, or why the class won't.
- A class is skipped when it isn't ticked (*Not selected*), has fewer than `OVR_MIN_POSITIVES` rows in the training
  split (10 by default), or when the validation split has no rows or none of the class. Such a class could train but
  couldn't be calibrated or given a threshold. *Not trained: N classes* lists the skipped ones, grouped by reason.
- *Balance the classes*, offered for every model with class weights (all but MLP) and on by default, weights each
  run's classes by their training counts (`balance`; see [One class against the rest](#one-class-against-the-rest)).
  It is off, and can't be turned on, while a class-weight setting is changed from its default on *Train*.
- *Name* defaults to the model and the target, as in *LightGBM · one model per class of vehicle*.
- Values under *Try several values* aren't used: each run takes the setting's own value, and the dialog says so.

### Settings search

*Search settings…* opens *Search settings*: runs with different values of the model's settings, ranked on the
validation split. Each run takes the settings on *Train*, with the searched ones replaced. It needs the model's
settings to be listed in the app (see [Adding a model](#adding-a-model)).

- *Train* chooses what each combination trains, when the export has a class target. *The whole target* (*The ticked
  targets* when several are ticked) trains one run per combination on the targets ticked on *Train*, in their chain
  order, ranked on the first. *One model per class* trains each combination as one-vs-rest runs, once per
  class under *Classes to tune on*, and ranks it on the mean of their scores. Those classes start as the most
  frequent, the median and the rarest class that train; a class that can't train is greyed out. *Target* picks the
  class target, when there are several.
- *Settings to search*: *Choose a setting to search*, then *Add a setting* for more. It starts with the settings under
  *Try several values*. Each one takes *Values*, typed as in *Try several values*, or, for a number setting, a
  *Range* from a low to a high. A range can be on a *Log scale* and in *Whole numbers*; a whole-number setting is
  always drawn as whole numbers. Values and ends are checked against the setting's own limits.
- *Combinations*: *Grid* tries every combination of the values, and takes no ranges. *Random* draws the number under
  *Draw* of distinct combinations from the values and ranges; the same *Seed* draws the same ones. A combination the
  checks refuse is drawn again, up to 10 times the number of draws; past that, the batch is refused. A grid
  combination the checks refuse refuses the batch, naming it.
- *Rank on* is the validation metric that orders the leaderboard. The default is PR AUC for a binary target or a
  one-vs-rest search, F1 (macro) for several classes and R² for a number. The others are the metrics of that kind of
  target.
- *Balance the classes*, for a one-vs-rest search, as in
  [One model per class](#one-model-per-class). It is also off while a class-weight setting is searched.
- *Name* defaults to the model and the searched settings, as in *LightGBM · search over num_leaves, learning_rate*.

The preview counts the runs: for a one-vs-rest search, combinations × classes.

### When to use which

| | *Try several values* | *Search settings…* |
|---|---|---|
| Settings | one or two | as many as the model has |
| Values | lists | lists, or ranges drawn at random |
| Runs | up to 50 | up to `BATCH_MAX_RUNS` |
| Targets | the targets and order ticked on *Train* | the targets and order ticked on *Train*, or one model per class |
| How they run | ordinary runs, started one after another; one refused stops the rest | a batch: checked first, then queued one at a time |
| Compared on | the runs table and its ★ | the batch page's leaderboard and chart |

Use *Try several values* to see how one or two settings move the scores. Use *Search settings…* to tune: more
settings, a range, a random search, or settings that must hold up across classes before a one-model-per-class batch.

### How a batch runs

A batch's runs wait as *Waiting in its batch* and train one at a time: the next starts when the previous one ends,
whether it finished or failed. Other exports, runs and Explore jobs queue between them, so a long batch doesn't hold
up everyone else's work.

- After an API restart, a batch trains the run the restart interrupted once more (*retried* beside its class on the
  batch page; interrupted again, it fails) and carries on with its waiting runs.
- *Cancel the batch*, on its page, asks to confirm, then cancels the runs that haven't started. A run already training
  finishes, and finished runs stay. With no run waiting, there is nothing to cancel.
- While a batch has runs to finish, its experiment can't be archived and its dataset can't be refreshed or deleted.
  The refusal names the batch to cancel.

### The batch page

The page opens from the batch's name under [*Batches*](#batches-on-runs) or from a run's batch chip. *Runs*, at the
top, goes back. The title row has the batch's name, its chip (*Per class #3*), its status and
[*Deploy…*](#deploying-from-a-batch). Under it: the export, who started it and when, and when it ended.

- *Progress*: a bar and a count of its runs, as *12 of 40 done · 1 failed · 27 waiting*, with *Cancel the batch*
  while it runs. *Not trained* lists the classes it skipped, by reason.
- *What it trains*: the model and device, and *classes balanced* when they are; the target and its classes; the
  settings every run shares; for a search, what was *Searched*, the *Combinations* (*Grid · 12 combinations* or
  *Random · 20 draws · seed 0*) and what it is *Ranked on*; and the scoring.

The page follows the batch as it runs.

### One-vs-rest results

A one-model-per-class batch has a table with a row per class: *One model per class*. Select a row to open its run on
*Runs*. Every column sorts.

| Column | What it shows |
|---|---|
| *Class* | the class, with *retried* or a threshold note when there is one |
| *Positives*: *Train*, *Valid*, *Test* | the class's rows in each split |
| *Validation* and *Test*: *PR AUC*, *ROC AUC*, *F1* | the run's scores; F1 is at the run's threshold |
| *Threshold* | the threshold with the best F1 on validation |
| *Calibration* | *Isotonic* or *Sigmoid*, and the Brier score before → after calibration on test |
| *Weak* | *Weak* when the model is unlikely to be useful, with the reason on hover |

A row is *Weak* when its validation PR AUC is below `OVR_WEAK_PR_AUC` (0.5 by default) or its class has fewer than
30 rows in the test split, where test scores say little. A row still waiting or training shows its status instead
of scores.

*Curve*, at the end of a finished row, opens the run's precision, recall and F1 by threshold on validation, with the
chosen threshold marked. Beside it are the threshold and the calibration: the method, the validation rows it was fit
on, and the Brier score before and after, on test and on validation. The validation pair is in-sample: the
calibrator was fit there. When the export has no test rows, the table's Brier pair is the validation one, marked
`*`.

Each model is calibrated on the validation split and its threshold is chosen there too, so the validation F1 is
mildly optimistic. The test columns give the honest number, and a note under the table says so.

### Search results

A search has a *Leaderboard*: one row per combination, best first on the ranking metric on validation. Its columns
are the *Rank*, each searched setting's value, the metric, for a one-vs-rest search each class's value of it, *CV ±*
(the spread over the cross-validation folds) and *Fit* (the time it took to train). A combination is ranked once all
of its runs have the metric, so a one-vs-rest mean is never a partial one. Until then its row shows *Waiting* or
*Training*, and a failed or cancelled combination says so. For a search on the whole target, select a row to open its
run; for a one-vs-rest search, each class's value opens its run.

*Use the top settings for a one-vs-rest batch…*, above the leaderboard, and *Per class…* on a finished row open
*Train one model per class* with that combination's settings over the batch's own, on the batch's model and export.
They are offered when the export has a class target, to anyone who can add runs.

Under the leaderboard, a chart plots the metric by a searched setting chosen under *Across*, with the best
combination ringed and each combination's settings on hover. *Against* picks a second setting: each combination is
then a point coloured by its metric, darker for better. Only settings whose values differ between combinations are
offered. The chart fills in as combinations finish.

### Deploying from a batch

*Deploy…*, in the title row, opens the [deploy dialog](deployments.md#the-deploy-dialog):

- For a one-model-per-class batch with nothing selected, it deploys the batch's finished runs, once the batch has
  ended.
- With rows selected, it deploys those runs. A row's checkbox is open once its run (for a combination, all its runs)
  has finished. The bar at the bottom counts them, with *Clear* and *Deploy them…* (*Deploy it…* for one).

A search is deployed by selecting rows: one row of a whole-target search, or one row of a one-vs-rest search, which
selects its run for each class. [What can be deployed](deployments.md#what-can-be-deployed) says what may share a
deployment.

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
