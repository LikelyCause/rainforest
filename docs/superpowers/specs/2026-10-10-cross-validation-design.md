# Cross-validation scores

Goal: a run reports how stable its scores are. Besides training the model as now, it fits the same model on K folds
of the training rows and reports each metric as a mean ± spread across folds, next to validation and test.

Decisions made with the owner (2026-10-10): CV adds scores and changes nothing else (the model, validation and test
stay as they are); K is set per run, default 5; it is separate from *Chain folds*; the ★ stays on validation.

## 1. The setting

- *Cross-validation* in the Train panel's top row (beside Export, Model and Train on), for every model: *Off*, or 2–20
  folds; default 5. It sits outside the model's settings because it applies to any model, catalogued or not.
- API: `POST /api/exports/{id}/runs` takes `cv_folds` (0 for off, or 2–20; default 5; `422` otherwise). Stored in
  the run's `config`; *Train again* brings it back.
- Separate from `cv` (*Chain folds*), which keeps its meaning: how chained targets pass predictions down.
- Cost: a run fits the whole chain K + 1 times, so it takes about that much longer; the panel says so under the
  setting ("about 6× the time of a run without it" for K = 5).

## 2. What it computes

- Folds are cut from the **training split** only; validation and test rows never take part. Folds are stratified on
  the chain's first class target when it has one, else shuffled, with the run's seed.
- Each fold fits the run's full chain (model, settings, chain folds, device) on the other K − 1 folds and predicts
  the held-out fold. The held-out predictions are scored per target with the same metrics as validation and test.
- The run's model is still the one trained on all training rows; CV models are discarded (not saved, not logged as
  models).
- Per target and metric the report keeps `mean`, `std` and the per-fold values (`report["cv"]` and the run's
  `metrics["cv"]` = means, so the runs table can read it like the other splits).
- Too few rows: K is lowered to the smallest training class count when that is below K (at least 2), and the report
  says so ("3 folds: the rarest class has 3 training rows"); below 2, CV is skipped with that note. A fold that fails
  fails the run, naming the fold.
- Progress: "Cross-validating: fold 2 of 5", then the final fit as today.
- MLflow: `cv_<target>_<metric>` (mean) and `cv_std_<target>_<metric>` per target and metric, and the `cv_folds`
  param.

## 3. Where it shows

- **Run detail:** each target's headline metrics gain a *CV (5-fold)* column beside Validation and Test, as
  "0.871 ± 0.012"; hovering shows the per-fold values. A note appears when K was lowered or CV skipped.
- **Runs table:** the *Test / Validation* switch gains *CV*, showing each metric's mean ± spread (— for runs without
  CV). The ★ is picked on validation whatever the switch shows, as today.
- Runs trained before this, or with CV off, show — in CV places.

## Testing

Unit: folds use only training rows; stratification on the first class target; K lowered and skipped cases; mean and
std match the per-fold values; a numbers target gets r2/rmse/mae CV. API: `cv_folds` accepted (0, 2–20), refused
(1, 21), stored and returned; default 5; a run reports `metrics.cv`; CV off reports none. UI: the switch's CV column,
the ± format, Train again restores the setting. dev-01: train with CV 5 and Off on a class and a number export,
compare timing, check run detail, runs table and MLflow.
