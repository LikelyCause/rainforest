# Batch training, hyperparameter search, and test deployments

Three linked features:

- **One-vs-rest batches.** Train one binary model per class of a target (this vehicle against all others) from a
  single export, in one action.
- **Search batches.** Explore a hyperparameter space wider than today's two-setting grid, then apply the chosen settings
  to a one-vs-rest batch.
- **Test deployments.** Deploy a batch (or any run) inside Rainforest and score new data through an API endpoint or by
  uploading a file in the UI.

Decisions made with the owner (2026-10-28):

- This replaces the vehicle–engine verification project as the next piece of work.
- One-vs-rest is the primary pattern: identify one vehicle against all others in a large dataset, for every distinct
  value of the target, without queueing 50 runs by hand.
- Searching a hyperparameter space as a batch is wanted too.
- Once the models look right, deploy them **inside Rainforest, for testing**: an API endpoint to call, and a UI page to
  upload a dataset and see predictions. This reverses the earlier choice to keep Rainforest out of the serving path,
  for testing only. It is not production serving.
- The owner approved the playback and asked for design, plan and build to proceed without stopping. Choices the
  owner didn't make are marked *(my call)*.

## 1. Intent

A data scientist has a dataset whose target has 50 vehicles. They tune LightGBM on a few of them, then press one
button to train a model per vehicle and get a table of 50 results, so the weak vehicles stand out. They deploy the
batch and send it records from a script, or upload a held-out file and see, for each row, every vehicle's score, the
most likely vehicle, and which vehicles' models said yes. If the file has the true labels, they also see how each
model did.

### Goals

- A **batch** groups runs started together. Kinds: `one_vs_rest` and `search`.
- **One-vs-rest batch:** from one export with a class target, one binary run per chosen class, with the target turned
  binary at training time (this class vs the rest). No new export per class.
- **Search batch:** a grid over any number of settings, or random sampling from ranges, ranked on the validation split.
  It can be scoped to a few classes of a one-vs-rest target, so settings are tuned on representative vehicles before
  the full batch.
- **Batch results:** one table, one row per run, with the metrics that matter for the batch kind. Sortable, with weak
  runs flagged, and a path from a search's best row to a one-vs-rest batch with those settings.
- **Deployments:** freeze a set of runs (a batch, or chosen runs) with a threshold per binary model. Score raw records
  through the same preprocessing the export used.
- **Prediction API:** `POST /api/deployments/{id}/predict`, authenticated by the signed-in session or by a deployment
  API key.
- **Test page:** upload a CSV or Parquet file, score every row in a background job, see and download the results, and
  see per-class metrics when the file carries the target.

### Non-goals

- Production serving: SLAs, autoscaling, a separate serving process, GPU inference, canary or A/B routing.
- A model registry with stages and promotion.
- Deploying outside Rainforest. The standalone bundle and script remain the out-of-Rainforest path, through the open
  export-to-code work (PR #42).
- Chains in one-vs-rest batches. Each one-vs-rest run has a single target.
- Search strategies beyond grid and random (Bayesian or successive halving), and early stopping of whole search arms.

### Anti-goals (a failure even if it works)

- Deployment scores that differ from what the same model produced at training time for the same rows. The
  preprocessing path must be the export's own.
- An API key stored or logged in clear, or returned after creation.
- A batch that silently skips classes. Every class not trained is listed with the reason (too few positives, for
  example).
- A deployment that changes behaviour because its dataset was later refreshed or re-exported. A deployment is frozen
  files.

## 2. Constraints

- Runs today: `POST /api/exports/{id}/runs` validates the model, parameters, order and device, then inserts
  `meta.model_runs` and `jobs.submit(_run, ...)`. Training goes through `modeling/train.py` (ClassifierChain or
  TargetChain, with cross-validation via `crossval.py`), and logs to MLflow. Jobs queue on `JOB_SLOTS` (one by
  default), in ticket order.
- The UI's *Try several values* (`ui/src/sweep.ts`) builds up to `SWEEP_MAX = 50` combinations of at most two settings,
  client side, and starts one run each. Runs carry the settings that differ from the defaults.
- One-vs-rest today is a **target decision** (`target_task` with positive classes, migration 018), baked into the
  export. A batch must not need one export per class.
- The export writes `preprocessor.joblib` (a fitted `ColumnTransformer`) and a manifest. Its rows are made from raw
  fields by `transform.to_pandas_frame(df, specs, coercers)` with field specs from the export's frozen snapshot, and
  derived fields come from `derived.compute`. Scoring new data must reuse these.
- Refresh lock (`routers/_lock.py`): starting runs on a refreshing dataset is refused. Batch start follows the same
  rule. Deployments are unaffected because they are frozen files.
- Roles: Viewers read only. Members and Leads write within experiments they may write (`experiments.require_writable`).
- Single API replica, in-process jobs, `/data` on a PVC.

## 3. Batches

### Data model *(my call)*

```sql
CREATE TABLE meta.run_batches (
    id            BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES meta.experiments ON DELETE CASCADE,
    export_id     BIGINT NOT NULL REFERENCES meta.exports ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('one_vs_rest', 'search')),
    name          TEXT NOT NULL,
    spec          JSONB NOT NULL,   -- model, device, base params, target, classes or space, sampler, ranking metric
    skipped       JSONB NOT NULL DEFAULT '[]',   -- [{class, reason}]
    status        TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'done', 'cancelled')),
    created_by    BIGINT REFERENCES meta.users,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
ALTER TABLE meta.model_runs ADD COLUMN batch_id BIGINT REFERENCES meta.run_batches ON DELETE SET NULL;
```

A batch's status is derived when its runs end: `done` when none is pending or running. The batch's progress is its
runs' statuses.

### One-vs-rest

- **Start:** the Train panel gains *Train one model per class…* when the export has a class target (classes or bands;
  not numbers). The dialog picks the target (when the export has several), the classes, the model and its settings
  (the panel's current ones), and the device.
- **Classes:** default all of them. A class is skipped, with the reason listed, when its training split has fewer than
  `OVR_MIN_POSITIVES` (default 10) rows or no validation rows. The user may untick classes. The cap is
  `BATCH_MAX_RUNS` (default 500) runs per batch.
- **Training *(my call)*:** each run's config carries `target` and `positive_class`. `train.py` trains a single binary
  model on `y == positive_class` (other targets of the export are ignored; no chain). Every metric is the binary set
  that exists today (PR-AUC, ROC-AUC, the threshold table, precision, recall and F1 of the positive class,
  cross-validation). The run's label names the class: "LightGBM · vehicle = V07 vs rest".
- **Class weights:** each run uses the model's class-weight setting as given. For LightGBM and XGBoost the dialog
  offers *balance the classes* (on by default): `class_weight="balanced"` or `scale_pos_weight` = negatives / positives
  on that run's training rows.

### Search

- **Space:** any number of the model's settings (from its `modeling/params.py` catalogue). Each is either a list of
  values, as today, or, for numbers, a range with a scale (linear or log) and an integer flag.
- **Sampler:** *grid*, every combination of the lists (ranges are not allowed in grid), or *random*, N samples, seeded,
  drawn from lists and ranges. Each combination is checked by the existing setting checks and the 64-row dry fit before
  any run starts, and invalid samples are redrawn up to a limit. The cap is `BATCH_MAX_RUNS`.
- **Scope:** any export. On a class target, a search may also be one-vs-rest for a few chosen classes (default: up to 3,
  the most frequent, the median and the rarest above the minimum). Then each combination is trained once per chosen
  class and ranked on the mean of their scores.
- **Ranking metric (validation split) *(my call)*:** PR-AUC for a binary or one-vs-rest target, macro-F1 for a
  multiclass target, R² for a number. The user can switch to another metric present in the reports.
- **Results:** a leaderboard (rank, settings, metric, the spread over cross-validation folds when present, time), with
  a 1-D or 2-D view of metric by setting for the chosen axes. *Use these settings for a one-vs-rest batch* opens the
  one-vs-rest dialog prefilled.

### Batch results page

`/d/<dataset>/experiments/<experiment>/batches/<id>`:

- the spec (model, settings, target, classes or space), progress, *Cancel* (the pending runs are cancelled; running ones
  finish), and the skipped classes with their reasons;
- for one-vs-rest, one row per class: positives in train, valid and test; PR-AUC, ROC-AUC and F1 at the run's
  threshold on validation and test; the chosen threshold.
  - A row is flagged *weak* when its validation PR-AUC is below `OVR_WEAK_PR_AUC` (default 0.5) or the class has fewer
    than 30 test positives.
  - Clicking a row opens the run's existing detail.
- *Deploy…* opens the deployment dialog with the batch's runs.

Batches are listed on the experiment page above its runs. The runs also stay in the runs table, tagged with their
batch, so nothing about single runs changes.

### API

| Method | Path | |
|---|---|---|
| POST | `/api/exports/{id}/batches` | body `{kind, name?, model, device, params, target?, classes?, balance?, space?, sampler?: {kind: grid\|random, n?, seed?}, ranking?, search_classes?}` → `202` with the batch and its run ids; the same `422`s as a run; `409` while refreshing |
| GET | `/api/experiments/{id}/batches` | the batches with counts by status |
| GET | `/api/batches/{id}` | the spec, skipped classes, and rows (the run summaries above) |
| POST | `/api/batches/{id}/cancel` | cancels its pending runs |

MLflow: each run is tagged `rainforest.batch_id`, `rainforest.batch_kind` and `rainforest.positive_class` when set.

## 4. Thresholds

Every binary model in a deployment has a threshold. *(my call)* The default is the validation-split threshold that
maximises F1, taken from a finer threshold sweep than today's table (0.01 steps, stored in the run's report as
`threshold_curve`). In the deployment dialog the user can switch all models at once, or one model, to *precision at
least p* (the lowest threshold whose validation precision reaches p, falling back to the F1 choice with a warning) or
to a fixed value. Multiclass models have no threshold; their prediction is the top class.

## 5. Deployments

### Data model *(my call)*

```sql
CREATE TABLE meta.deployments (
    id            BIGSERIAL PRIMARY KEY,
    dataset_id    BIGINT NOT NULL REFERENCES meta.datasets ON DELETE CASCADE,
    experiment_id BIGINT NOT NULL REFERENCES meta.experiments ON DELETE CASCADE,
    export_id     BIGINT NOT NULL REFERENCES meta.exports ON DELETE RESTRICT,
    batch_id      BIGINT REFERENCES meta.run_batches ON DELETE SET NULL,
    name          TEXT NOT NULL,
    models        JSONB NOT NULL,  -- [{run_id, label, kind: binary|multiclass|regression, positive_class?, threshold?, threshold_rule}]
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    created_by    BIGINT REFERENCES meta.users,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    retired_at    TIMESTAMPTZ
);
CREATE TABLE meta.deployment_keys (
    id            BIGSERIAL PRIMARY KEY,
    deployment_id BIGINT NOT NULL REFERENCES meta.deployments ON DELETE CASCADE,
    name          TEXT NOT NULL,
    prefix        TEXT NOT NULL,            -- the first 8 characters, shown in the list
    hash          TEXT NOT NULL UNIQUE,     -- SHA-256 of the key
    created_by    BIGINT REFERENCES meta.users,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);
CREATE TABLE meta.deployment_tests (
    id            BIGSERIAL PRIMARY KEY,
    deployment_id BIGINT NOT NULL REFERENCES meta.deployments ON DELETE CASCADE,
    filename      TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    rows          BIGINT,
    has_labels    BOOLEAN,
    metrics       JSONB,
    error         TEXT,
    owner         INTEGER,
    created_by    BIGINT REFERENCES meta.users,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
```

- A deployment is made from runs of **one export**, so all of them share one preprocessor and one input schema. The
  runs must be done. Creating it copies nothing: the export's files and the runs' model files are already immutable.
  `export_id ... ON DELETE RESTRICT` (and the same check when a run is deleted) keeps them while a deployment uses
  them. Deleting a dataset retires and removes its deployments first.
- *Retire* stops it answering (`410`) and keeps its record and tests. A retired deployment can be reactivated.
- Who may: Members and Leads create deployments from experiments they may write. The creator and Leads retire them and
  manage keys. Viewers see deployments and their tests, and may not upload tests or call predict except with a key.

### Scoring *(my call)*

`api/app/scoring.py` is the one place that turns raw records into predictions:

- **Load** (cached per deployment in the API process, LRU of `DEPLOYMENT_CACHE` deployments, default 4): the export's
  field specs and coercers from its frozen snapshot, `preprocessor.joblib`, derived-field formulas, and each run's
  model.
- **Score a polars frame of raw fields:** check the required source fields are present (a missing one is an error
  naming it; extra columns are ignored); compute derived fields with `derived.compute`; then
  `to_pandas_frame` → `transform_into` → each model's `predict_proba` (or `predict` for numbers). One-vs-rest models
  give P(positive); a multiclass model gives its class probabilities.
- **Output per record:**
  - `scores` (label → probability, or the number for a regression model);
  - `top` (the label with the highest score among binary and multiclass outputs, and that score);
  - `matches` (the labels whose binary model scores at or above its threshold, best first);
  - `none_matched` (true when no binary model says yes).
- **Equality test:** scoring the export's own test rows through a deployment must reproduce each run's stored test
  predictions exactly. This is the anti-goal's guard.

### Prediction API

- `POST /api/deployments/{id}/predict`, body `{"records": [{"<field>": value, ...}, ...]}` with at most
  `PREDICT_MAX_RECORDS` (default 1,000) records. The response is `{"deployment": id, "models": [...labels], "results":
  [{scores, top, matches, none_matched}, ...]}` in record order. `422` names a missing field. `410` when retired.
- **Authentication:** the signed-in session (for the UI), or `Authorization: Bearer rf_<32 random chars>` from a key of
  that deployment.
  - Keys are created on the deployment page and shown once. Only the SHA-256 and an 8-character prefix are stored.
  - Keys can be revoked.
  - `last_used_at` is updated at most once a minute.
  - A key authorises only its deployment's predict call. It isn't a user session, has no other access, and is exempt
    from the same-origin check for that one route.
- **Request limits:** a body over 5 MB gets `413`. The per-request record cap above applies.
- **Docs:** the deployment page shows a `curl` example with the endpoint, a key placeholder and two records built from
  the dataset's fields.

### Test page

`/d/<dataset>/deployments/<id>`:

- the models with their thresholds, the keys, and the `curl` example;
- **Test with a file:**
  - Upload CSV or Parquet, using the same reader as dataset upload: separators, null tokens, 20 GB limit, streamed to
    `DATA_DIR/deployments/<id>/tests/<test id>/`.
  - A background job scores it in chunks of 50,000 rows. Status and progress are shown like an export's.
- **Results:**
  - The first 500 rows: the record's identifying field (the dataset's identifier field when one exists, else the row
    number), `top`, `matches` and every score. A per-model score column is hidden behind *Show all scores* when there
    are more than 10 models.
  - *Download* gives the full results as CSV or Parquet.
- **Labels:** when the file has the dataset's target column, the test also reports:
  - top-1 accuracy;
  - per class: precision, recall and F1 of its model at its threshold, and PR-AUC on the file;
  - the share of rows where nothing matched;
  - the confusion of `top` against the label for the 20 most frequent classes.
- The last 20 tests per deployment are kept and older ones are deleted with their files.
- Deployments are listed on the dataset's *Deployments* step (a new top-bar step after *Experiments*).

## 6. Code layout

- Migration `025_batches_and_deployments.sql`.
- `api/app/batches.py` (planning: classes and skips, grid and random samplers, validation of the space) and
  `api/app/routers/batches.py`.
- `api/app/modeling/train.py`: `positive_class` support (binarise, label, metrics, class balance) and the
  `threshold_curve`.
- `api/app/scoring.py`, `api/app/deployments.py` (create, retire, keys, tests job) and
  `api/app/routers/deployments.py`; the key path in `api/app/auth.py`.
- UI:
  - `ui/src/batches.ts` (pure helpers: class planning preview, sampler preview, leaderboard sorting, weak flags);
  - `components/BatchDialog.tsx` (one-vs-rest and search, from the Train panel), `pages/BatchPage.tsx`;
  - `ui/src/deployments.ts`, `components/DeployDialog.tsx`, `pages/DeploymentsPage.tsx`, `pages/DeploymentPage.tsx`.
- Docs:
  - `docs/experiments-and-training.md`: batches and search;
  - a new `docs/deployments.md`: deploying, keys, predict, tests;
  - `docs/api.md` rows;
  - `docs/configuration.md` for the new settings: `OVR_MIN_POSITIVES`, `OVR_WEAK_PR_AUC`, `BATCH_MAX_RUNS`,
    `PREDICT_MAX_RECORDS`, `DEPLOYMENT_CACHE`.

## 7. Testing

- **Unit:** the samplers (grid count, random reproducible by seed, ranges with log scale and integers, invalid samples
  redrawn); class planning and skips; the threshold rules; scoring output shape (top, matches, none_matched); key
  hashing and verification.
- **API, with the test database:**
  - A one-vs-rest batch on the 50k sample's class target trains one run per class with the right labels and binary
    metrics, and lists skipped classes.
  - A random search on one export runs N valid combinations and ranks them.
  - A search scoped to 2 classes trains combinations × 2.
  - Cancel works.
  - Batch start is `409` while refreshing.
  - Deployment: create from a batch, predict through the session and through a key, `401` without one, `410` once
    retired, `422` for a missing field, `413` over the limit.
  - **The equality test:** deployment scores of the export's test rows equal each run's stored predictions.
  - A labelled test file gives the right top-1 accuracy on a constructed case.
  - A key never appears in any response after creation, or in the logs.
- **UI:** helper tests, and a headless check of the batch dialog, batch page, deploy dialog and test page against the
  live API.
- **Smoke:** `deploy/smoke.sh` gains a step: after its run, it starts a 3-class one-vs-rest batch on the demo, deploys
  it, and predicts two records through a key.

## 8. Decisions on the builder check *(my calls, 2026-10-28)*

1. **Calibration.** Class balancing shifts each one-vs-rest model's raw P(positive) by a different amount, so raw
   scores aren't comparable across vehicles.
   - Every one-vs-rest run is calibrated on its validation split after training: isotonic when the validation split
     has at least 200 positives and 200 negatives, else sigmoid (Platt). The calibrator is saved with the run, and
     the report keeps `calibration` (method, Brier score before and after).
   - `scores`, `top`, the thresholds (chosen on calibrated validation scores) and every test-page metric use
     calibrated probabilities.
   - Choosing a threshold on the split the model was calibrated on is mildly optimistic. The batch page and test page
     say so, and the test split gives the honest number.
2. **Queueing and restarts.** A batch is not 500 submitted jobs.
   - Its runs are inserted `pending` with the batch id. A batch feeder submits the next one only when the batch's
     previous run finishes, so other people's exports, runs and tests interleave with a batch rather than wait behind
     it.
   - On startup, `jobs.sweep()` fails the interrupted run as today, and running batches resume: the feeder retries
     the interrupted run once, then continues with the pending ones.
   - *Cancel* marks the pending runs cancelled.
3. **Frozen inputs.**
   - Exports freeze what scoring needs into `meta.exports.config`:
     - `derived`: each derived field's kind and spec at export time, using the same key PR #42 uses, so the two meet
       when it merges;
     - `input_schema`: each source field's raw dtype as read from the dataset's Parquet, and the `datetime_format`
       of every date field, inferred once over the whole column and pinned.
   - Exports made before this change have neither, so creating a deployment from one is refused with "Re-export to
     deploy".
   - **Equality check at creation.** Creating a deployment scores the export's test rows through `scoring.py` and
     compares them with each model's predictions on the export's own `model_ready.parquet` (tolerance 1e-9). A
     mismatch refuses the deployment and names the first differing model and row.
4. **What may share a deployment.** A deployment is one of two things:
   - a set of one-vs-rest runs of one export on one target, at most one run per class (a search batch is deployed
     by choosing its rows), or
   - a single run with one target (multiclass or number).

   Chains, mixed sets and two runs for one class are refused. The response keys `scores` by class label (or by the
   target's name for a number); `top` and `matches` apply to one-vs-rest sets, `top` alone to a multiclass run.
5. **Test-file labels.**
   - Every row is scored; the export's row rules don't apply.
   - Rows whose label has no model (skipped, unticked or unseen classes) count as wrong in top-1 accuracy and are
     also reported as "rows whose label has no model: N". They have no per-class row.
   - The "nothing matched" share is over all rows.
6. **Types of incoming values.**
   - Scoring converts each source field to its frozen `input_schema` dtype before preprocessing:
     - numbers through the existing numeric coercers;
     - integer-coded categories normalised (`7.0` and `07` become `7` when training values were integers);
     - dates with the pinned format.
   - A value that doesn't convert becomes missing and is counted. The predict response carries `warnings: {field:
     count}`, and a test reports the counts per field prominently. A field where more than half the values fail is
     flagged.

## 9. Left to the builder

- Exact layout of the dialogs and pages, reusing the Train panel's settings controls for the search space.
- How runs report `positive_class` in the existing runs table (a column, or the label).
- The LRU cache's eviction details and whether models load lazily per model or per deployment.
- The 1-D or 2-D search chart's form (follow the dataviz guidance already used by Explore).
