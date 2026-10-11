# Cross-validation Scores Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A run can also fit its chain on K folds of the training rows and report each metric's mean ± spread
across folds, beside validation and test.

**Architecture:** `train()` gains `cv_folds`. The chain construction and the per-split scoring become two local
functions (`make_chain`, `score`) so the final fit and every fold use the same code. A new `modeling/crossval.py`
holds the pure parts (fold assignment, aggregation). The run stores means as `metrics["cv"]` (so the runs table,
SQL view and MLflow treat CV like another split) and the full summary as `report["cv"]`.

**Tech Stack:** scikit-learn KFold/StratifiedKFold, FastAPI, React, vitest, Playwright on dev-01.

**Spec:** `docs/superpowers/specs/2026-10-10-cross-validation-design.md`

## Global Constraints

- `cv_folds`: 0 (off) or 2–20; API default 5; `422` for 1 or 21+. Separate from `cv` (chain folds).
- Folds come from the training split only; validation/test never take part. Seeded with the run's seed.
- The saved model is the one fit on all training rows; fold models are discarded.
- The ★ stays on validation.
- Docs current-state only; comments sparse; layouts checked 768/1024/1440.

## Review Focus

1. A class target whose rarest class has fewer training rows than K → the run still completes (folds unstratified,
   with a note), never a crash (Task 1 test).
2. A fold whose training rows lose a binary target's minority class → CV is skipped with a note; the run and its
   validation/test scores still complete (Task 1 test).
3. A metric that is undefined in some folds (ROC AUC with one class in the held-out fold) → mean over the folds that
   have it, never NaN in JSON (Task 1 test).
4. A number target (regression) → r2/rmse/mae CV like the others (Task 1 test).
5. Old runs and runs with CV off → "—" in the CV view, the switch still works (Task 3 test).

---

### Task 1: Folds, aggregation and CV inside `train()`

**Files:** Create `api/app/modeling/crossval.py`; modify `api/app/modeling/train.py`; test
`api/tests/test_crossval.py`.

**Interfaces — Produces:**
- `crossval.folds(y_strat: np.ndarray | None, n: int, k: int, seed: int, strat_name: str | None) ->
  tuple[list[tuple[np.ndarray, np.ndarray]], str | None]` — (train_idx, test_idx) pairs and a note.
- `crossval.summarize(per_fold: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, dict]]` —
  `{target: {metric: {"mean", "std", "folds": [float | None]}}}`, finite values only.
- `train(..., cv_folds: int = 0)`; `report["cv"] = {"folds", "stratified_on", "note", "seconds", "targets"}`
  or `{"folds": 0, "note": …}` when skipped; `metrics["cv"] = {target: {metric: mean}}`.

- [ ] **Step 1: Failing tests** (`test_crossval.py`): folds partition the given rows exactly once each; stratified
  folds keep class shares (binary 30/70 → each held-out fold within ±5 points); a rarest class below K → shuffled
  folds and a note naming the target; n < k → k lowered to n; summarize: mean/std of finite values, `None` for a
  missing fold value, metric absent everywhere omitted. Train-level: `train(..., cv_folds=3)` on the regression
  test export reports `metrics["cv"][t]["r2"]` and `report["cv"]["folds"] == 3`, per-fold list of length 3; a
  class export → `roc_auc`/`f1_macro` CV; `cv_folds=0` → no `"cv"` key; a binary target with 2 minority rows in
  training → run completes, `report["cv"]["folds"] == 0` with a note.
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3: Implement** `crossval.py` (`StratifiedKFold(k, shuffle=True, random_state=seed)` when the rarest
  class count ≥ k, else `KFold` with the note "Folds aren't stratified: the rarest class of {name} has {n} training
  rows"); in `train()`, extract `make_chain()` and `score(chain, X, Y)` from the existing code without changing
  behaviour, then after the final evaluation:

```python
if cv_folds:
    report_cv = _cross_validate(...)  # loops folds: say(f"Cross-validating: fold {i} of {k}"), make_chain().fit,
                                      # score held-out rows; a single-class fold → skip with a note
```

- [ ] **Step 4:** New tests PASS; full API suite green.
- [ ] **Step 5: Commit** "Cross-validation inside a run: folds of the training rows, mean ± spread per metric".

### Task 2: API, CLI and MLflow

**Files:** `api/app/schemas.py` (`RunRequest.cv_folds`), `api/app/routers/runs.py` (pass it), `api/app/modeling/
train.py` (`--cv-folds`), `api/app/modeling/mlflow_log.py` (`cv_std.<t>.<m>` metrics, `cv_folds` param);
tests in `api/tests/test_crossval.py`.

- [ ] **Step 1: Failing tests:** `RunRequest(cv_folds=1)` and `21` invalid, default 5, `0` allowed; `run_metrics`
  of a report with `cv` includes `cv.y.roc_auc` and `cv_std.y.roc_auc`; `run_params` includes `cv_folds`.
- [ ] **Step 2–4:** implement, PASS, full suite.
- [ ] **Step 5: Commit** "Runs take cv_folds (default 5); MLflow logs CV means and spreads".

### Task 3: UI

**Files:** `ui/src/types.ts` (config `cv_folds`, report `cv`), `ui/src/runs.ts` (`Split` adds `"cv"`,
`spreadFor`, `starTitle` for cv), `ui/src/components/TrainPanel.tsx` (Cross-validation select, hint, sent as
`cv_folds`, prefill), `ui/src/components/RunDetail.tsx` (switch gains *cross-validation* when the run has it;
`CvCard`: tiles "mean ± std", per-fold values in the tooltip, the note), `ui/src/pages/ExperimentWorkspace.tsx`
(switch gains *CV*; cells show ± spread), `ui/src/settings.ts`/`TrainPrefill` (cvFolds); tests `ui/src/runs.test.ts`.

- [ ] **Step 1: Failing tests:** `metricFor(r, "cv", …)` reads `metrics.cv`; `spreadFor` returns the std for the
  same metric key metricFor uses (AUC/F1, R²/RMSE) and undefined without CV; `starTitle(r, "cv", …)` says it's
  picked on validation without the test-split sentence.
- [ ] **Step 2–4:** implement; typecheck, vitest, build.
- [ ] **Step 5: Commit** "UI: cross-validation setting, CV in the runs table and run detail".

### Task 4: Docs, deploy, browser check

- [ ] README (Train panel, runs table, run detail, report), docs/api.md (`cv_folds`, `metrics.cv`,
  `report.cv`), docs/architecture.md (`crossval.py`, train flow).
- [ ] `DOCKER_CONTEXT=dev-01 make up` from this worktree (it carries the baseline-names branch too).
- [ ] Browser: throwaway Concrete strength (number) and Student dropout (classes) datasets: train LightGBM with
  CV 5 and Off; check runs table CV switch (± shown, — for Off), run detail CV tiles and tooltip, Train again keeps
  K, MLflow has `cv.*` metrics; time with/without; delete the datasets.
- [ ] Final review (fresh reviewer), fix pass, PR.
