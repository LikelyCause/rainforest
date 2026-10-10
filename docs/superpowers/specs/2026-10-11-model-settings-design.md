# Model settings in the Train menu

Goal: every model parameter that matters is settable from the Train menu with a proper control (a dropdown for a
choice, a number input with its range, a switch for on/off), starting with LightGBM. The other models get the same
treatment one at a time, after the owner has tried LightGBM's.

Decisions made with the owner (2026-10-11): curated and complete (the important parameters up front, every other
model parameter in a collapsed Advanced list); LightGBM first; Claude designs the interaction.

## 1. The parameter catalog

Each model publishes its parameters as data, in `api/app/modeling/params.py`, served with the model list
(`GET /api/models`, `GET /api/exports/{id}/models` gain `settings`). One entry per parameter:

| Key | Meaning |
|---|---|
| `name` | The estimator's parameter name (`num_leaves`). |
| `label`, `help` | What the form shows ("Leaves per tree"; one sentence on what raising it does). |
| `group` | `run`, `trees`, `learning`, `sampling`, `regularization`, `classes`, `numbers`, `advanced`. |
| `kind` | `choice` (with `options`: value + label + optional help), `int`, `float`, `bool`, or `int_or_none` (a number, or a labelled "no limit" choice such as `max_depth = -1`). |
| `min`, `max`, `step`, `log` | Range for numbers; `log` makes the input step by factors (learning rate). |
| `default` | Rainforest's default (the registry's `default_params`, else the library's). |
| `applies` | `both`, `classes` (classifier links only) or `numbers` (regressor links only). |
| `when` | Shown and sent only when another setting has a value (dart's drop settings when `boosting_type = dart`). |

The catalog replaces the hand-kept `CLASSIFIER_ONLY` list for any model that has one: `applies` routes each value to
the classifier, the regressor or both. A model without a catalog keeps today's behaviour (no settings in the menu,
API `params` checked as now).

## 2. LightGBM's settings

**Run** (every model): `seed` (default 42) and `cv` (chain folds: off, or 2–10; default 5). These are run fields, not
estimator parameters; the form shows them first.

| Group | Parameters |
|---|---|
| Trees | `boosting_type` (gbdt, dart, rf), `n_estimators` (10–10,000), `num_leaves` (2–4,096), `max_depth` (no limit, or 1–64), `min_child_samples` (1–10,000) |
| Learning | `learning_rate` (0.001–1, log steps) |
| Sampling | `subsample` (0.1–1), `subsample_freq` (0 = off, 1–100), `colsample_bytree` (0.1–1) |
| Regularization | `reg_alpha` (0–100), `reg_lambda` (0–100), `min_split_gain` (0–100) |
| For class targets | `class_weight` (none, balanced) |
| For number targets | `objective` (squared error, absolute error, Huber, Fair, quantile, MAPE, Poisson, gamma, Tweedie), with `alpha` when Huber or quantile |
| Advanced | `max_bin`, `min_child_weight`, `subsample_for_bin`, `extra_trees`, `path_smooth`, `max_delta_step`, `feature_fraction_bynode`, `linear_tree`, `data_sample_strategy` (bagging, goss) with `top_rate`/`other_rate` when goss, dart's `drop_rate`, `max_drop`, `skip_drop`, `uniform_drop`, `xgboost_dart_mode` when dart, and `is_unbalance`/`scale_pos_weight` for class targets |

Left out on purpose: `n_jobs`, `verbose`, `random_state` and `importance_type`, which the app sets itself or which
don't change the model.

## 3. Checks

- Each value is checked against its entry (kind, range, options, `when`) before a run starts; `422` names the setting
  and the rule.
- LightGBM's own cross-setting rules (e.g. `rf` needs bagging: `subsample < 1` and `subsample_freq > 0`) are caught
  by a dry fit on 64 synthetic rows when the run is requested, so the error appears in the menu, not minutes later.
- An objective that needs non-negative or positive numbers (Poisson, gamma, Tweedie) is refused for a target whose
  training minimum (manifest `target_stats`) breaks it.

## 4. In the Train menu

- Under the model picker: **Model settings**, reading *Defaults* or *3 changed*. Clicking it opens the settings in
  the menu, which widens and scrolls.
- Groups in the order above, each a short list of rows: label, control, the default beside it ("default 63"), a ↺
  on a changed row (with the decision panel's edited marker) and the one-line help under the label on hover or
  focus. *For class targets* and *For number targets* show only when the export has such a target. *Advanced* is
  collapsed with a search box and a count of changed settings inside it.
- *Reset all* at the top. A value that breaks its rule is flagged as you type and Train stays disabled until it's
  fixed; the dry-fit error, when any, shows above Train.
- The menu remembers the last settings per model for each export in the viewer's browser.

## 5. After a run

- The run stores only the settings that differ from the defaults (`config.params`, as today) plus seed and cv.
- Run detail shows them ("Settings: num_leaves 127, learning_rate 0.02") and a **Train again with these settings**
  button that opens the Train menu filled with them.
- MLflow already logs every estimator parameter; nothing changes there.

## Build order

One PR: catalog module and LightGBM's entries, API (catalog in the model lists, validation, dry fit, routing by
`applies`), Train menu settings, run detail's settings and *Train again*, docs. Other models follow, one PR each,
once the owner has tried LightGBM's.

## Testing

Unit: catalog entries match the estimator (every name is a real parameter of LGBMClassifier/LGBMRegressor or a
LightGBM native one; defaults pass their own rules); validation per kind; `when`; routing (`class_weight` reaches
classifiers only, `objective` regressors only). API: a run with changed settings stores and applies them (the fitted
booster's params show them); a bad value and a bad combination are refused with the setting named; Poisson on a
negative target is refused. UI: form state (changed count, reset, `when`, validation, remembered settings,
Train-again prefill). dev-01: change LightGBM settings in the browser, train, check run detail, MLflow and a retrain.
