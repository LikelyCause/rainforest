# Batch Training and Deployments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-vs-rest and search batches from one export, and test deployments that score raw data through an API endpoint (session or key) and a UI upload page.

**Architecture:** Batches group runs; a feeder submits one batch run at a time and resumes after restarts. `train.py` gains `positive_class` training with balancing, calibration and a fine threshold curve. Exports freeze derived formulas and an input schema; `scoring.py` reuses the export's own preprocessing; deployments are verified by an equality check at creation and served by a predict route and a test-file job.

**Tech Stack:** FastAPI, psycopg 3, polars, scikit-learn (CalibratedClassifierCV / IsotonicRegression), LightGBM/XGBoost, React + TanStack Query + vitest.

**Spec:** `docs/superpowers/specs/2026-10-28-batch-training-and-deployments-design.md` (§8 holds the binding decisions on the builder check).

## Global Constraints

- New settings, passed through docker-compose `api` env and `.env.example` and documented in `docs/configuration.md` (test_config_wiring enforces): `OVR_MIN_POSITIVES` (10), `OVR_WEAK_PR_AUC` (0.5), `BATCH_MAX_RUNS` (500), `PREDICT_MAX_RECORDS` (1000), `DEPLOYMENT_CACHE` (4). The chart's ConfigMap carries them too.
- Migration `025_batches_and_deployments.sql` holds every schema change in this plan.
- Every batch and deployment write route refuses with the lock's 409 while the dataset is refreshing (`routers/_lock.py`) and requires `experiments.require_writable`; Viewers read only.
- API keys: `rf_` + 32 random URL-safe characters; only SHA-256 and the first 8 characters stored; never returned after creation, never logged.
- Scoring uses the export's own path (`to_pandas_frame` → `transform_into`) — no second implementation of preprocessing.
- Docs are current-state only; comments explain why; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A test file or predict request with an integer-coded category sent as `"7.0"` must convert to `7`, not become an unknown category. Pinned in Task 5.
2. A class with positives in train but none in validation must be skipped with a reason, not trained into an uncalibratable model. Pinned in Task 1.
3. An API restart in the middle of a batch resumes the batch rather than failing its pending runs. Pinned in Task 3.
4. A key from deployment A must not authorise deployment B, and a revoked key must stop working at once. Pinned in Task 6.
5. A random search whose sampled values fail the dry fit must redraw, and report how many were redrawn, rather than start invalid runs. Pinned in Task 1.

---

### Task 1: Migration 025, settings, and batch planning

**Files:** Create `api/app/migrations/025_batches_and_deployments.sql`, `api/app/batches.py`; modify `api/app/config.py`, `docker-compose.yml`, `.env.example`, `docs/configuration.md`, `deploy/chart/rainforest/templates/configmap.yaml` (+ values defaults); test `api/tests/test_batches_planning.py`.

**Interfaces — produces:**
- Tables `meta.run_batches`, `meta.deployments`, `meta.deployment_keys`, `meta.deployment_tests` exactly as spec §3/§5; `meta.model_runs.batch_id`; `meta.model_runs.status` gains `cancelled` if not present.
- `batches.plan_classes(class_counts: dict[str, dict[str,int]], wanted: list[str] | None, min_pos: int) -> tuple[list[str], list[dict]]` (trained classes, `[{class, reason}]` skips: "fewer than N positives in training", "no validation rows", "no validation positives").
- `batches.grid(space: dict[str, list]) -> list[dict]`; `batches.random_samples(space: dict[str, dict|list], n: int, seed: int, valid: Callable[[dict], str|None], max_redraws: int = 10*n) -> tuple[list[dict], int]` (samples, redraw count); range spec `{"low", "high", "log": bool, "integer": bool}`; ranges refused in grid.
- `batches.search_classes_default(class_counts, min_pos) -> list[str]` (most frequent, median, rarest above the minimum; ≤ 3, distinct).

- [ ] Step 1: failing tests — grid count = product; random reproducible by seed; log-range samples within bounds and log-uniform-ish (median near geometric mean over 2,000 draws); integer flag yields ints; a `valid` that rejects half the values → all returned samples valid and redraws counted; `max_redraws` exhausted → ValueError naming the setting; plan_classes skips the three reasons; default search classes for a skewed count dict.
- [ ] Step 2: run → fail; implement; run → pass; migration applies on the test DB (`pytest tests/test_migrations.py` or the bootstrap path the suite uses); test_config_wiring passes with the five settings.
- [ ] Step 3: commit.

### Task 2: One-vs-rest training, calibration and the threshold curve

**Files:** Modify `api/app/modeling/train.py`, `api/app/schemas.py` (`RunRequest` gains `positive_class: str | None`, `target: str | None`, `balance: bool = False`), `api/app/routers/runs.py` (validation: positive_class must be a class of a class target of the export; implies a single target), `api/app/modeling/mlflow_log.py` (params/tags `rainforest.positive_class`, `rainforest.batch_id`, `rainforest.batch_kind`); test `api/tests/test_ovr_training.py`.

**Interfaces — produces:**
- A run with `positive_class` trains one binary model on `y == class` for `target`; label "<Model> · <target> = <class> vs rest"; report keeps every binary metric that exists today plus `threshold_curve` (validation and test: precision, recall, F1, share flagged at thresholds 0.00–1.00 in 0.01 steps) and `calibration` `{method: isotonic|sigmoid, brier_before, brier_after}`; the saved model is the calibrated estimator (predict_proba returns calibrated P(positive)); all metrics computed on calibrated scores.
- `balance=True`: LightGBM/XGBoost/random forest/hist GBM get class weighting (class_weight="balanced" or scale_pos_weight = neg/pos on that run's training rows); models without a weighting option ignore it with a note in the report.
- `train.choose_threshold(curve, rule: {"kind": "f1"} | {"kind": "precision", "min": p} | {"kind": "fixed", "value": t}) -> (threshold, note|None)`.
- `train.predict_scores(model_dir: Path, X: np.ndarray) -> np.ndarray` — the one function deployments and the equality check use to score a run's model.

- [ ] Step 1: failing tests on the 50k sample export (fixture from tests/test_crossval.py or similar): a `positive_class` run → binary report, label, threshold_curve has 101 points per split, calibration recorded, predict_scores in [0,1]; isotonic vs sigmoid chosen by the 200/200 rule; balance on changes LightGBM's weighting (assert the fitted params); choose_threshold: F1 maximiser, precision rule with fallback note when unreachable, fixed; a positive_class not in the classes → 422; with a numeric target → 422.
- [ ] Step 2: implement; run the run/crossval/train test files + new tests; commit.

### Task 3: Batches API, the feeder, resume and cancel

**Files:** Create `api/app/routers/batches.py`; modify `api/app/batches.py` (feeder), `api/app/jobs.py` (`sweep` resumes running batches; a `cancelled` status for runs), `api/app/routers/runs.py` (the shared run validation extracted so batches reuse it), `api/app/main.py`, `docs/api.md`; test `api/tests/test_batches_api.py`.

**Interfaces — produces:**
- `POST /api/exports/{id}/batches` body per spec §3 (`kind`, `name?`, `model`, `device`, `params`, `target?`, `classes?`, `balance?`, `space?`, `sampler?`, `ranking?`, `search_classes?`, `cv_folds?`) → `202` `{batch, run_ids, skipped, redraws}`; validates every planned run with the single-run checks before inserting anything; `BATCH_MAX_RUNS` → 422 with the count.
- `GET /api/experiments/{id}/batches`, `GET /api/batches/{id}` (spec, skipped, rows: run summaries incl. validation/test PR-AUC, ROC-AUC, F1 at the F1 threshold, positives per split, weak flag by `OVR_WEAK_PR_AUC` or <30 test positives; for search: settings, ranking metric value, CV spread), `POST /api/batches/{id}/cancel`.
- Feeder: `batches.feed(batch_id)` submits the next pending run when none of the batch's runs is pending-submitted or running; called when a batch starts and when one of its runs ends (hook in runs' job completion); batch `done` when nothing is left; `jobs.sweep()` at startup calls `batches.resume_all()` which re-queues the interrupted run once (`report.retried=true`) then feeds.
- Search ranking: the batch's `ranking` metric on validation (defaults per spec), averaged over `search_classes` when one-vs-rest.

- [ ] Step 1: failing tests (DB): a 3-class one-vs-rest batch on the sample's class target runs 3 runs, labels right, skips listed; at most one batch run running at a time while another export job interleaves (submit an export after the batch and assert it starts before the batch's last run); random search n=4 → 4 valid runs ranked; one-vs-rest search on 2 classes × grid of 2 → 4 runs, ranking by mean; cancel → pending runs cancelled, batch cancelled; restart simulation (mark the owner dead, call sweep) → the interrupted run re-queued once, the rest continue; 409 while refreshing; Viewer 403; too many runs → 422.
- [ ] Step 2: implement; run new + runs/jobs/refresh_api tests; commit.

### Task 4: Exports freeze derived formulas and the input schema

**Files:** Modify `api/app/transform.py` / `api/app/routers/exports.py` (export config gains `derived` and `input_schema`), `docs/experiments-and-training.md`; test `api/tests/test_export_freeze.py`.

**Interfaces — produces:** `exports.config["derived"] = {name: {"kind", "spec"}}` for every derived field the export reads (same key as PR #42; check `git show origin/feature/export-to-code:api/app/export_code/gather.py` for its shape and match it); `exports.config["input_schema"] = {field: {"dtype": str(polars dtype), "datetime_format": str|None, "integer_codes": bool}}` for every source field the export reads (derived fields' inputs included); date formats inferred once over the full column and pinned.

- [ ] Step 1: failing tests: a new export carries both keys; a derived field removed after the export doesn't change the export's frozen spec; a date field's pinned format is the full-column one.
- [ ] Step 2: implement; run export tests; commit.

### Task 5: `scoring.py`

**Files:** Create `api/app/scoring.py`; test `api/tests/test_scoring.py`.

**Interfaces — produces:**
- `scoring.load(conn, deployment_or_spec) -> Scorer` (LRU of `DEPLOYMENT_CACHE` keyed by deployment id); `Scorer.score(df: pl.DataFrame) -> ScoreResult` where `ScoreResult` has `scores: dict[label, np.ndarray]`, `top: list[(label, score)]|None`, `matches: list[list[label]]|None`, `none_matched`, `warnings: dict[field, int]`; missing required field → `ValueError("Missing field: …")`.
- Conversion per spec §8.6 (numbers via existing coercers, integer codes normalised, pinned date formats; failures → null + counted).
- `scoring.equality_check(conn, export_id, models) -> str | None` (None when every model's scores on the export's test rows equal `train.predict_scores` on `model_ready.parquet` within 1e-9; else a message naming the model and row).

- [ ] Step 1: failing tests: equality check passes for a one-vs-rest set and a multiclass run on the sample export; `"7.0"`/`"07"` for an integer-coded category score identically to `7`; a non-convertible number → null + warning count; missing field → error; derived field computed from frozen spec; top/matches/none_matched on a constructed case.
- [ ] Step 2: implement; commit.

### Task 6: Deployments API, keys and predict

**Files:** Create `api/app/deployments.py`, `api/app/routers/deployments.py`; modify `api/app/auth.py` (key path for the predict route only; exempt from the same-origin check there), `api/app/main.py`, dataset/run deletion paths (retire/remove deployments; refuse deleting a run in an active deployment), `docs/api.md`, new `docs/deployments.md`; test `api/tests/test_deployments_api.py`.

**Interfaces — produces:**
- `POST /api/experiments/{id}/deployments` body `{name, run_ids?, batch_id?, threshold_rule?: {...}, thresholds?: {run_id: rule}}` → 201; validates spec §8.4 composition, frozen export inputs ("Re-export to deploy"), equality check; stores models with resolved thresholds.
- `GET /api/datasets/{id}/deployments`, `GET /api/deployments/{id}`, `POST /api/deployments/{id}/retire`, `POST /api/deployments/{id}/activate`.
- Keys: `POST /api/deployments/{id}/keys` `{name}` → `{id, name, prefix, key}` (key shown once); `GET …/keys` (no key); `DELETE /api/deployments/{id}/keys/{key_id}` revokes.
- `POST /api/deployments/{id}/predict` per spec §5 + `warnings`; session or Bearer key of that deployment; 401 / 410 / 413 / 422.

- [ ] Step 1: failing tests: create from a batch (thresholds resolved), refuse a chain / mixed set / duplicate class / pre-freeze export / equality mismatch (monkeypatch a model file); predict via session and via key; key of A on B → 401; revoked key → 401 immediately; retired → 410; >PREDICT_MAX_RECORDS → 413 or 422 per spec; a key appears in no later response and no log record (caplog); Viewer can't create, can read.
- [ ] Step 2: implement; commit.

### Task 7: Deployment tests job

**Files:** Modify `api/app/deployments.py`, `api/app/routers/deployments.py`, `api/app/jobs.py` (sweep fails interrupted tests), `docs/deployments.md`; test `api/tests/test_deployment_tests.py`.

**Interfaces — produces:** `POST /api/deployments/{id}/tests` (multipart file, CSV/TSV/TXT/Parquet via the upload reader) → 202 test row; job scores in 50,000-row chunks into `DATA_DIR/deployments/<id>/tests/<test id>/results.parquet`; metrics when the target column is present (spec §5 + §8.5); `GET /api/deployments/{id}/tests`, `GET /api/deployment-tests/{id}` (metrics, first 500 result rows), `GET /api/deployment-tests/{id}/download?format=csv|parquet`; keep last 20 per deployment.

- [ ] Step 1: failing tests: labelled file → top-1 accuracy, per-class P/R/F1/PR-AUC, no-model rows counted, nothing-matched share on a constructed case; unlabelled file → results only; retention deletes the 21st oldest with its files; warnings per field surfaced.
- [ ] Step 2: implement; commit.

### Task 8: UI — batch dialog and batches list

**Files:** Create `ui/src/batches.ts` (+ test), `ui/src/components/BatchDialog.tsx`; modify `ui/src/components/TrainPanel.tsx` (*Train one model per class…* and *Search settings…*), `ui/src/api.ts`, `ui/src/types.ts`, the experiment page (batches list above runs).

**Interfaces:** dialog for one-vs-rest (target, classes with positives counts from the export manifest/report, skips preview via the same rule, balance toggle, model settings from the panel) and search (axes: list or range+log+integer per setting; sampler grid/random n/seed; scope: classes for one-vs-rest search; ranking metric; run count preview with the cap); helpers tested: run count, skip preview, range parsing.

- [ ] Steps: helper tests RED → implement → typecheck/vitest/prettier/eslint → headless check against the live API (start api + vite as in previous projects) → commit.

### Task 9: UI — batch page

**Files:** Create `ui/src/pages/BatchPage.tsx`; modify routes, `ui/src/batches.ts` (sorting, weak flag) + tests.

**Interfaces:** spec §3 batch results page: progress, cancel, skipped, one-vs-rest table (sortable; weak flagged; row → run detail; calibration and threshold-optimism note), search leaderboard + 1-D/2-D chart (follow existing Explore chart conventions), *Use these settings for a one-vs-rest batch*, *Deploy…*.

- [ ] Steps: helper tests RED → implement → checks → headless check → commit.

### Task 10: UI — deploy dialog, deployments list, deployment page

**Files:** Create `ui/src/deployments.ts` (+ test), `components/DeployDialog.tsx`, `pages/DeploymentsPage.tsx`, `pages/DeploymentPage.tsx`; modify top bar steps (*Deployments* after *Experiments*), routes, `api.ts`, `types.ts`.

**Interfaces:** deploy dialog (runs preselected from a batch or chosen runs, threshold rule for all / per model, name; shows refusals); deployments list; deployment page (models + thresholds, keys create/revoke with the key shown once and copy, `curl` example built from the dataset's fields, retire/activate, test upload with progress, results table first 500 with *Show all scores* beyond 10 models, download, metrics when labelled, warnings per field).

- [ ] Steps: helper tests RED → implement → checks → headless check → commit.

### Task 11: Docs and smoke step

**Files:** `docs/experiments-and-training.md` (batches, search, calibration, thresholds), `docs/deployments.md` (complete), `docs/api.md`, `docs/configuration.md`, `README.md` (one line), `deploy/smoke.sh` (3-class one-vs-rest batch → deploy → predict two records with a key), `deploy/README.md` (the predict route behind the Ingress; body size).

- [ ] Steps: write; run `bash deploy/smoke.sh` on kind once; grep docs for history wording; commit.

### Task 12: Finish

- [ ] Full API suite with the test DB, ruff, UI checks, chart tests and lint; whole-branch review; push the branch; **poll GitHub CI on the branch's PR or on main after merge** and report the result.
