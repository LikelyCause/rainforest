# Regression targets, with binning as an option

Goal: a numeric target (a price, an amount, a duration) can be predicted as a number, or, when bands are what's
wanted, cut into classes. Both are decisions about the target, made and reviewed like any other preparation choice,
versioned with the baseline and adjustable per experiment. Ingest stays faithful to the source file.

Decisions made with the owner (2026-10-10): regression support, with binning as an option.

## 1. How a target is predicted

A target field gets a **target task**, part of its decision alongside keep / drop / target and type:

| Task | For | The model predicts |
|---|---|---|
| `classes` | binary, categorical, discrete numeric with ≤ 100 values (today's targets) | one of the field's values |
| `regression` | continuous or discrete numeric | a number |
| `bins` | continuous or discrete numeric | a band, e.g. `[10, 25)` |

- The default comes from the field: binary and categorical → `classes`; continuous numeric → `regression`; discrete
  numeric → `classes` with ≤ 100 distinct values, else `regression`. Types that can't be targets (text, identifier,
  datetime, constant) still can't.
- `bins` takes either **quantile bins** (`k`, 2–20, default 5: equal-count bands) or **edges** (an ascending list;
  values below the first or above the last go to open-ended end bands). Labels read like `[10, 25)`, `< 10`, `≥ 80`.
  Quantile edges are computed on the training split of each export, so the bands never see validation or test rows;
  the edges used are recorded in the manifest.
- Stored with the decision (`target_task`, `target_bins` JSON) in the team decision, assessments, experiment copies
  and baseline snapshots. A disagreement on the task of a target counts as a disagreement on the core decision.
- Experiments can change a target's task like any other adjustment (it shows in the adjustments and MLflow).

## 2. Export

- Regression targets are written as float columns; `bins` targets as their band index with the band labels as its
  classes (from then on a classification target).
- The split stratifies on classification targets as today; a regression target is stratified on its training-free
  quantile bands (5) so each split covers the range; rows with a missing target are dropped as today.
- The manifest records each target's task (`target_tasks`), the bins and edges, and for regression targets the
  training mean and standard deviation.

## 3. Training

- A target chain mixes classifiers and regressors. Each model in the chain sees the features plus the earlier
  targets' predictions (class for a classification target, value for a regression one), out-of-fold with `cv`
  folds as today. Runs whose targets are all classification keep using scikit-learn's ClassifierChain unchanged;
  any regression target uses Rainforest's own `TargetChain`, saved the same way (`classifier_chain.joblib` keeps its
  name; the report says which kind it is).
- Every model gets a regression counterpart: LightGBM → `LGBMRegressor`, XGBoost → `XGBRegressor` (GPU too),
  histogram GBM and random forest → their regressors, logistic regression → ridge regression, MLP → `MLPRegressor`.
  Models without one (none today) refuse a regression target with a clear message. Estimator parameters that only
  apply to classifiers (e.g. `class_weight`) are left out for regressors.
- **Metrics** per regression target and split: RMSE, MAE, R², and MAPE when no actual value is 0; plus the
  evaluation details the run page draws: up to 1,000 sampled (actual, predicted) pairs and a 30-bin residual
  histogram. Binned targets get the classification metrics, ROC curves and confusion matrix as today.
- Comparing runs: the headline for a regression target is R² (validation split picks the ★, as today), with RMSE
  next to it; classification keeps AUC and macro-F1.

## 4. In the app

- **Review / decision panel:** for a target, a *Predict as* control (*Classes* / *Number* / *Bands*), with the band
  settings (quantile count, or edges) and a preview of the band counts on the field's data.
- **Start dialog and target pickers:** numeric fields become eligible targets; the picker says how each will be
  predicted by default.
- **Run detail:** for a regression target, tiles RMSE, MAE, R², MAPE; a predicted-vs-actual scatter with the
  diagonal; a residual histogram. Runs table columns per target show R² and RMSE for regression targets.
- **Explore:** already treats numeric targets as numeric; its target picker offers them.
- **MLflow:** metrics `<split>.<target>.rmse|mae|r2|mape`, the task per target as params, and the bins in the
  manifest artifact.
- **Samples:** *Generate* gains a *with a numeric target* option (a continuous `column_c` driven by the same latent
  factors), and the library gains one regression dataset (UCI [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), CC BY 4.0, 1,030 rows; strength in MPa) next to Student grades, whose
  `G3` can be predicted as a number.

## Build order

Each its own PR, CI green, checked on dev-01:
1. Target tasks and bins in decisions (all four sources, snapshots, adoption, adjustments), eligibility, export
   (float targets, bins on the training split, stratification, manifest).
2. Training: regressors in the registry, `TargetChain`, regression metrics and evaluation details, MLflow, run
   comparison.
3. UI: *Predict as* in the decision panel, start dialog and pickers, regression run detail and runs table; sample
   generator option and the Concrete dataset; docs.

## Testing

Unit: default task per type; quantile and edge binning (edges from training rows only, labels, open ends); mixed
chain fit/predict with out-of-fold features; regression metrics on known values; registry regressors build. API:
setting and reviewing a task (blind until review, disagreement on task), an experiment changing a task, export of
regression and binned targets (manifest, stratification), a LightGBM regression run end to end with metrics and
MLflow. dev-01: Student grades with `G3` as a number and as 4 bands; Concrete strength as a number; a chain mixing
`G1` as classes and `G3` as a number.
