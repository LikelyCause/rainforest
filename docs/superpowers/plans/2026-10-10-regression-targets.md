# Regression Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A numeric target can be predicted as a number (regression) or cut into bands (classes), chosen per target as part of the field's decision, and trained, scored and compared end to end.

**Architecture:** Two columns on every decision table (`target_task`, `target_bins`), kept coherent by one database trigger (cleared off targets, defaulted from the field's type, reset when the type no longer allows the task), copied by every path that copies a decision, and frozen into snapshots. Exports turn bands into class indices using edges computed on the training split and write regression targets as floats. Training uses a new `TargetChain` (classifier or regressor per target) whenever a run has a regression target; all-class runs keep scikit-learn's `ClassifierChain` unchanged.

**Tech Stack:** FastAPI, psycopg 3, Postgres 16, polars/pandas/numpy, scikit-learn, LightGBM, XGBoost, MLflow, React 18 + react-query + ECharts, vitest.

**Spec:** `docs/superpowers/specs/2026-10-10-regression-targets-design.md` (approved 2026-10-10: a task disagreement is a core disagreement; validation R² picks the ★).

## Global Constraints

- Tasks: `classes`, `regression`, `bins`. Default: binary and categorical → `classes`; numeric_continuous → `regression`; numeric_discrete → `classes` with ≤ 100 distinct values, else `regression`. Text, identifier, datetime and constant fields can't be targets.
- `bins`: `{"k": n}` (quantile, 2–20, default 5) or `{"edges": [...]}` (1–19 finite, strictly ascending numbers). Labels `< a`, `[a, b)`, `≥ b`. Quantile edges come from the training split only and are recorded in the manifest.
- A task (or bins) disagreement between assessors is a disagreement on the core decision.
- Regression metrics per target and split: `rmse`, `mae`, `r2`, and `mape` only when no actual value is 0. Details: ≤ 1,000 sampled (actual, predicted) pairs and a 30-bin residual histogram.
- Regression headline is R² (validation picks the ★); RMSE shown next to it (lower is better).
- Concrete Compressive Strength: UCI dataset 165, CC BY 4.0, DOI 10.24432/C5PK67, 1,030 rows.
- Three PRs, each CI green and checked on dev-01: PR A = Tasks 1–4, PR B = Tasks 5–7, PR C = Tasks 8–11.
- Docs describe the current state only (no "now", no PR numbers). Comments sparse; explain why, not what.
- Test commands: API `"$PT" <args>` where `PT=/private/tmp/claude-501/-Users-sentry-Development/195de4a6-33ad-44b5-b513-b33406064d16/scratchpad/pt` (runs pytest in `api/` against the dev-01 test DB on 127.0.0.1:55432; needs the ssh tunnel up); UI `cd ui && npm run typecheck && npm test`.

## Review Focus

1. **A copy path that drops the task.** Every path that copies a decision (assessor seeding, adoption at review, finalize → snapshot, new version via `load_working`, experiment copy, revert, bulk/undo, reuse onto another dataset) must carry `target_task`/`target_bins`; forgetting one silently turns a regression target back into classes. Pinned by `test_task_survives_every_copy_path` (Task 3).
2. **A save that omits the task.** A PUT without `target_task` (old clients, the save queue, `team_helpers.save`) must keep the stored task, not reset it to the default. Pinned by `test_put_without_task_keeps_it` (Task 3).
3. **Retyping a target.** A regression target retyped to categorical must fall back to that type's default, never keep an impossible task. Pinned by `test_trigger_resets_task_the_type_no_longer_allows` (Task 2).
4. **Explore on an export with a regression target.** `prepare_export` casts targets to int32 today, which would silently truncate prices. Pinned by `test_prepare_export_keeps_regression_target_numeric` (Task 4).
5. **Quantile bands collapsing.** A skewed column (most values 0) gives duplicate quantiles; bands must collapse to the distinct edges, and a single band must fail the export with a clear message. Pinned by `test_quantile_edges_collapse_duplicates` and `test_export_refuses_a_single_band` (Task 4).

---

## PR A — target tasks in decisions, and export

### Task 1: Target-task rules and ingest

**Files:**
- Modify: `api/app/semantics.py` (TARGET_TYPES, `target_problem`, new `TARGET_TASKS`, `default_task`, `bins_problem`, `validate_decision`, `option_catalog`)
- Modify: `api/app/schemas.py` (`DecisionIn`)
- Modify: `api/app/profiling.py:615-620` (targets keep numeric types)
- Test: `api/tests/test_data_correctness.py`, `api/tests/test_profiling.py`, `api/tests/test_semantics.py`

**Interfaces:**
- Produces: `semantics.TARGET_TASKS = ("classes", "regression", "bins")`; `semantics.default_task(t: str, n_unique: int | None) -> str | None`; `semantics.target_problem(t: str, n_unique: int | None, task: str | None = None) -> str | None` (task None = the default); `semantics.bins_problem(bins: dict | None) -> str | None`; `DecisionIn.target_task: Literal["classes","regression","bins"] | None = None`, `DecisionIn.target_bins: dict | None = None`; catalog `target_rule = {"types": [...], "numeric_types": [...], "max_levels": 100, "tasks": [...], "default_k": 5}`.

- [ ] **Step 1: Write the failing tests**

In `test_data_correctness.py`, replace `test_only_class_like_fields_can_be_targets` and `test_upload_target_with_too_many_levels_is_not_made_a_target`:

```python
@pytest.mark.parametrize("t,levels,task,ok", [
    ("binary", 2, None, True), ("categorical", 40, None, True), ("numeric_discrete", 7, None, True),
    ("categorical", MAX_TARGET_LEVELS + 1, None, False), ("numeric_continuous", 30, None, True),
    ("numeric_continuous", 5000, "regression", True), ("numeric_continuous", 30, "classes", False),
    ("numeric_discrete", 500, "classes", False), ("numeric_discrete", 500, "bins", True),
    ("categorical", 10, "regression", False), ("binary", 2, "bins", False),
    ("text", 10, None, False), ("datetime", 10, None, False), ("identifier", 10, None, False),
])
def test_target_eligibility_depends_on_type_and_task(t, levels, task, ok):
    errors = validate_decision({"action": "target", "semantic_type": t, "target_task": task,
                                "target_bins": {"k": 5} if task == "bins" else None}, n_unique=levels)
    assert (errors == []) is ok


@pytest.mark.parametrize("t,levels,task", [
    ("binary", 2, "classes"), ("categorical", 40, "classes"), ("numeric_discrete", 100, "classes"),
    ("numeric_discrete", 101, "regression"), ("numeric_continuous", 3, "regression"), ("text", 5, None),
])
def test_default_task(t, levels, task):
    assert default_task(t, levels) == task


@pytest.mark.parametrize("bins,ok", [
    ({"k": 5}, True), ({"k": 2}, True), ({"k": 20}, True), ({"k": 1}, False), ({"k": 21}, False),
    ({"k": 2.5}, False), ({"edges": [10]}, True), ({"edges": [10, 25.5, 80]}, True), ({"edges": []}, False),
    ({"edges": [3, 3]}, False), ({"edges": [5, 1]}, False), ({"edges": [1, float("inf")]}, False),
    ({"edges": list(range(20))}, False), ({"k": 5, "edges": [1]}, False), ({}, False), (None, False),
])
def test_bins_problem(bins, ok):
    assert (bins_problem(bins) is None) is ok


def test_upload_targets_keep_their_numeric_type():
    price = _profile([i / 3 for i in range(5000)], "price", is_target=True)
    assert price.is_target and price.inferred_type == "numeric_continuous"
    grade = _profile([i % 4 for i in range(5000)], "grade", is_target=True)
    assert grade.is_target and grade.inferred_type in ("numeric_discrete", "categorical")
    ids = _profile([f"id{i}" for i in range(5000)], "code", is_target=True)
    assert not ids.is_target and any("can't be a target" in r for r in ids.reasons)
```

Import `default_task, bins_problem` from `app.semantics` at the top of the file next to `MAX_TARGET_LEVELS`.

In `test_profiling.py`, replace `test_target_is_forced_to_class_labels`:

```python
def test_numeric_target_keeps_its_type_and_text_target_becomes_labels():
    p = profile_column(pl.Series("column_b", [0, 1, 2] * 300), 0, True, None, None, 50)
    assert p.is_target and p.inferred_type == "numeric_discrete"
    s = profile_column(pl.Series("tier", ["gold", "silver", "bronze free text"] * 300), 0, True, None, None, 50)
    assert s.is_target and s.inferred_type in ("categorical", "binary")
```

In `test_semantics.py` add:

```python
def test_catalog_publishes_the_target_rule():
    rule = option_catalog()["target_rule"]
    assert rule["tasks"] == ["classes", "regression", "bins"]
    assert set(rule["numeric_types"]) == {"numeric_continuous", "numeric_discrete"}
    assert rule["max_levels"] == 100 and rule["default_k"] == 5
```

- [ ] **Step 2: Run them to see them fail**

Run: `"$PT" tests/test_data_correctness.py tests/test_profiling.py tests/test_semantics.py -k "target or bins or catalog"`
Expected: FAIL (ImportError `default_task`, then assertion failures).

- [ ] **Step 3: Implement the rules in `semantics.py`**

Replace the block from `# Targets are class labels…` through `target_problem`:

```python
# A target is predicted as classes (binary, categorical, or a numeric field with few values) or, when numeric, as a
# number or as bands of numbers.
TARGET_TYPES = ("binary", "categorical", "numeric_discrete", "numeric_continuous")
CLASS_TYPES = ("binary", "categorical", "numeric_discrete")
MAX_TARGET_LEVELS = 100
TARGET_TASKS = ("classes", "regression", "bins")
DEFAULT_BIN_COUNT = 5
MAX_BIN_COUNT = 20


def default_task(t: str, n_unique: int | None) -> str | None:
    """How a target of type `t` is predicted unless someone chooses otherwise; None when it can't be a target.
    Mirrors meta.default_target_task (migration 014)."""
    if t in ("binary", "categorical"):
        return "classes"
    if t == "numeric_continuous":
        return "regression"
    if t == "numeric_discrete":
        return "classes" if (n_unique or 0) <= MAX_TARGET_LEVELS else "regression"
    return None


def target_problem(t: str, n_unique: int | None, task: str | None = None) -> str | None:
    """Why a field typed `t` with `n_unique` distinct values can't be a target predicted as `task` (its default when
    None), or None if it can."""
    if t not in TARGET_TYPES:
        return (f"a {TYPE_LABELS.get(t, t).lower()} field can't be a target; targets must be binary, categorical or "
                f"numeric (change the field's type first if its values are class labels)")
    task = task or default_task(t, n_unique)
    if task not in TARGET_TASKS:
        return f"unknown target task {task!r} (expected one of {', '.join(TARGET_TASKS)})"
    if task == "classes":
        if t not in CLASS_TYPES:
            return "a continuous field is predicted as a number or as bands, not as classes"
        if n_unique is not None and n_unique > MAX_TARGET_LEVELS:
            return (f"{n_unique:,} distinct values; predicting classes allows at most {MAX_TARGET_LEVELS}"
                    + ("" if t in ("binary", "categorical") else " (predict it as a number or as bands instead)"))
        return None
    if t not in NUMERIC:
        return f"only a numeric field can be predicted as {'a number' if task == 'regression' else 'bands'}"
    return None


def bins_problem(bins: dict | None) -> str | None:
    """What's wrong with a bands setting: exactly one of `k` (2-20 equal-count bands) or `edges` (1-19 ascending cut
    points)."""
    if not isinstance(bins, dict) or len(bins) != 1 or not ({"k", "edges"} & set(bins)):
        return "bands need either a count (k) or cut points (edges)"
    if "k" in bins:
        k = bins["k"]
        if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= MAX_BIN_COUNT:
            return f"the band count must be a whole number from 2 to {MAX_BIN_COUNT}"
        return None
    edges = bins["edges"]
    if (not isinstance(edges, list) or not 1 <= len(edges) <= MAX_BIN_COUNT - 1
            or not all(isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e) for e in edges)):
        return f"cut points must be 1 to {MAX_BIN_COUNT - 1} finite numbers"
    if any(b <= a for a, b in zip(edges, edges[1:])):
        return "cut points must be in ascending order, each different"
    return None
```

In `validate_decision`, replace the target line:

```python
    if d.get("action") == "target":
        if problem := target_problem(t, n_unique, d.get("target_task")):
            errors.append(problem)
        elif (d.get("target_task") or default_task(t, n_unique)) == "bins" and d.get("target_bins") is not None:
            if problem := bins_problem(d.get("target_bins")):
                errors.append(problem)
```

In `option_catalog` replace `target_rule`:

```python
        "target_rule": {"types": list(TARGET_TYPES), "numeric_types": sorted(NUMERIC), "max_levels": MAX_TARGET_LEVELS,
                        "tasks": list(TARGET_TASKS), "default_k": DEFAULT_BIN_COUNT},
```

- [ ] **Step 4: `DecisionIn` in `schemas.py`**

After `params`:

```python
    target_task: Literal["classes", "regression", "bins"] | None = Field(
        None, description="How a target is predicted; omitted keeps the stored task (or the type's default)")
    target_bins: dict[str, Any] | None = Field(None, description='Bands: {"k": 2-20} or {"edges": [ascending numbers]}')
```

- [ ] **Step 5: Ingest keeps numeric targets numeric (`profiling.py`)**

Replace the two `if is_target …` blocks:

```python
    if is_target and t not in NUMERIC and target_problem("categorical", n_unique):
        reasons = [f"named as a target but can't be a target: {target_problem('categorical', n_unique)}", *reasons]
        is_target = False
    if is_target and t not in NUMERIC and t not in ("binary", "categorical"):
        reasons = [f"target column; values look {t}, treated as class labels", *reasons]
        t = "binary" if n_unique == 2 else "categorical"
```

  Order matters for numeric columns that inference calls `identifier` (whole numbers, almost all unique, e.g. whole-dollar prices): before these two blocks, when `is_target and t == "identifier" and s.dtype.is_numeric()`, set `t = "numeric_continuous"` with the reason "target column; whole numbers that are almost all distinct, treated as a number". Test: `_profile(list(range(1000, 6000)), "price", is_target=True)` is a target typed `numeric_continuous`.

  `processing.py:104-106`'s warning becomes "Not used as target(s): {names} (text, identifiers and dates can't be targets, nor labels with more than 100 values)".

Import `NUMERIC` from `.semantics` if `profiling.py` doesn't already.

- [ ] **Step 6: Run the tests**

Run: `"$PT" tests/test_data_correctness.py tests/test_profiling.py tests/test_semantics.py`
Expected: PASS. Then `"$PT" -x -q 2>&1 | tail -5` to find other tests that asserted the old rule (`numeric_continuous` refused as a target in `test_targets_and_lists.py`, `test_team.py`, `test_api_cleanup.py`, `reuse` tests); update each one to the new rule (a continuous target is accepted with task `regression`; use a `text` field where a test needs a refused target). Expected after updates: all pass.

- [ ] **Step 7: Commit**

```bash
git add api/app/semantics.py api/app/schemas.py api/app/profiling.py api/tests
git commit -m "Target tasks: classes, number or bands, with defaults per type"
```

### Task 2: Migration 014 — columns, default function, trigger, backfill

**Files:**
- Create: `api/app/migrations/014_target_tasks.sql`
- Create: `api/tests/test_target_tasks.py`

**Interfaces:**
- Consumes: `semantics.default_task` (Task 1).
- Produces: columns `target_task TEXT NULL`, `target_bins JSONB NULL` on `meta.field_decisions`, `meta.assessments`, `meta.experiment_decisions`; SQL function `meta.default_target_task(semantic_type text, n_unique bigint) RETURNS text`; trigger `target_task_coherent` (BEFORE INSERT OR UPDATE) on all three tables; every finalized snapshot's decisions carry `target_task`/`target_bins`.

- [ ] **Step 1: Write the failing tests** (`api/tests/test_target_tasks.py`)

```python
"""Target tasks in the decision tables: the trigger that keeps them coherent, its agreement with semantics, and the
paths that carry them."""
import pytest

from app.semantics import SEMANTIC_TYPES, default_task

from .team_helpers import DB_URL, app_, fields, people, save, upload  # noqa: F401

pytestmark = pytest.mark.skipif(not DB_URL, reason="set RAINFOREST_TEST_DATABASE_URL to run the API tests")


def _sql_default(conn, t, n):
    return conn.execute("SELECT meta.default_target_task(%s, %s) AS t", (t, n)).fetchone()["t"]


def test_sql_default_matches_python(app_):
    from app.db import connection
    with connection() as conn:
        for t in SEMANTIC_TYPES:
            for n in (None, 2, 100, 101, 5000):
                assert _sql_default(conn, t, n) == default_task(t, n), (t, n)


def test_trigger_defaults_clears_and_resets(app_, people):
    from app.db import connection
    lead = people["lead"]
    ds = upload(lead, "x,price,c\n" + "".join(f"{i},{i * 1.5 + 0.25},{'pq'[i % 2]}\n" for i in range(300)))
    f = fields(lead, ds)
    with connection() as conn:
        def row(name):
            return conn.execute("SELECT action, target_task, target_bins FROM meta.field_decisions WHERE field_id = %s",
                                (f[name]["id"],)).fetchone()
        conn.execute("UPDATE meta.field_decisions SET action = 'target' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_task"] == "regression"
        conn.execute("UPDATE meta.field_decisions SET target_task = 'bins' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_bins"] == {"k": 5}
        conn.execute("UPDATE meta.field_decisions SET target_task = 'regression' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_bins"] is None
        conn.execute("UPDATE meta.field_decisions SET action = 'keep' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_task"] is None and row("price")["target_bins"] is None
        conn.rollback()


def test_trigger_resets_task_the_type_no_longer_allows(app_, people):
    from app.db import connection
    lead = people["lead"]
    ds = upload(lead, "x,price\n" + "".join(f"{i},{i * 1.5 + 0.25}\n" for i in range(300)))
    fid = fields(lead, ds)["price"]["id"]
    with connection() as conn:
        conn.execute("UPDATE meta.field_decisions SET action = 'target', target_task = 'regression' WHERE field_id = %s",
                     (fid,))
        conn.execute("UPDATE meta.field_decisions SET semantic_type = 'categorical' WHERE field_id = %s", (fid,))
        got = conn.execute("SELECT target_task FROM meta.field_decisions WHERE field_id = %s", (fid,)).fetchone()
        # 300 distinct values: too many for classes, and categorical can't be a number, so its default (classes)
        # is what the trigger can offer; validation reports the level limit when anyone saves it.
        assert got["target_task"] == "classes"
        conn.rollback()


def test_existing_targets_are_classes(app_, people):
    from app.db import connection
    with connection() as conn:
        bad = conn.execute("""SELECT count(*) AS n FROM meta.field_decisions
                               WHERE (action = 'target') <> (target_task IS NOT NULL)""").fetchone()["n"]
        assert bad == 0
        snaps = conn.execute("""SELECT count(*) AS n FROM meta.recipes r, jsonb_each(r.snapshot -> 'fields') e
                                 WHERE r.snapshot IS NOT NULL AND NOT (e.value -> 'decision' ? 'target_task')""").fetchone()
        assert snaps["n"] == 0
```

- [ ] **Step 2: Run them to see them fail**

Run: `"$PT" tests/test_target_tasks.py`
Expected: FAIL (`function meta.default_target_task does not exist`).

- [ ] **Step 3: Write the migration**

```sql
-- Migration 014: how a target is predicted (classes, a number, or bands of a number), on every decision table, kept
-- coherent by one trigger so every writer (including the ones that only move the target role) gets the same rule.
ALTER TABLE meta.field_decisions ADD COLUMN IF NOT EXISTS target_task TEXT, ADD COLUMN IF NOT EXISTS target_bins JSONB;
ALTER TABLE meta.assessments ADD COLUMN IF NOT EXISTS target_task TEXT, ADD COLUMN IF NOT EXISTS target_bins JSONB;
ALTER TABLE meta.experiment_decisions ADD COLUMN IF NOT EXISTS target_task TEXT,
    ADD COLUMN IF NOT EXISTS target_bins JSONB;

ALTER TABLE meta.field_decisions DROP CONSTRAINT IF EXISTS field_decisions_target_task_check,
    ADD CONSTRAINT field_decisions_target_task_check CHECK (target_task IN ('classes', 'regression', 'bins'));
ALTER TABLE meta.assessments DROP CONSTRAINT IF EXISTS assessments_target_task_check,
    ADD CONSTRAINT assessments_target_task_check CHECK (target_task IN ('classes', 'regression', 'bins'));
ALTER TABLE meta.experiment_decisions DROP CONSTRAINT IF EXISTS experiment_decisions_target_task_check,
    ADD CONSTRAINT experiment_decisions_target_task_check CHECK (target_task IN ('classes', 'regression', 'bins'));

-- Mirrors semantics.default_task.
CREATE OR REPLACE FUNCTION meta.default_target_task(semantic_type text, n_unique bigint) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN semantic_type IN ('binary', 'categorical') THEN 'classes'
        WHEN semantic_type = 'numeric_continuous' THEN 'regression'
        WHEN semantic_type = 'numeric_discrete' THEN CASE WHEN coalesce(n_unique, 0) <= 100 THEN 'classes'
                                                          ELSE 'regression' END
    END
$$;

CREATE OR REPLACE FUNCTION meta.target_task_coherent() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    levels bigint;
BEGIN
    IF NEW.action <> 'target' THEN
        NEW.target_task := NULL;
        NEW.target_bins := NULL;
        RETURN NEW;
    END IF;
    SELECT n_unique INTO levels FROM meta.fields WHERE id = NEW.field_id;
    IF NEW.target_task IS NULL
       OR (NEW.target_task IN ('regression', 'bins')
           AND NEW.semantic_type NOT IN ('numeric_continuous', 'numeric_discrete'))
       OR (NEW.target_task = 'classes' AND NEW.semantic_type = 'numeric_continuous') THEN
        NEW.target_task := meta.default_target_task(NEW.semantic_type, levels);
    END IF;
    IF NEW.target_task IS DISTINCT FROM 'bins' THEN
        NEW.target_bins := NULL;
    ELSIF NEW.target_bins IS NULL THEN
        NEW.target_bins := '{"k": 5}'::jsonb;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS target_task_coherent ON meta.field_decisions;
CREATE TRIGGER target_task_coherent BEFORE INSERT OR UPDATE ON meta.field_decisions
    FOR EACH ROW EXECUTE FUNCTION meta.target_task_coherent();
DROP TRIGGER IF EXISTS target_task_coherent ON meta.assessments;
CREATE TRIGGER target_task_coherent BEFORE INSERT OR UPDATE ON meta.assessments
    FOR EACH ROW EXECUTE FUNCTION meta.target_task_coherent();
DROP TRIGGER IF EXISTS target_task_coherent ON meta.experiment_decisions;
CREATE TRIGGER target_task_coherent BEFORE INSERT OR UPDATE ON meta.experiment_decisions
    FOR EACH ROW EXECUTE FUNCTION meta.target_task_coherent();

-- Every target so far was predicted as classes.
UPDATE meta.field_decisions SET target_task = 'classes' WHERE action = 'target' AND target_task IS NULL;
UPDATE meta.assessments SET target_task = 'classes' WHERE action = 'target' AND target_task IS NULL;
UPDATE meta.experiment_decisions SET target_task = 'classes' WHERE action = 'target' AND target_task IS NULL;

ALTER TABLE meta.recipes DISABLE TRIGGER recipes_frozen;
UPDATE meta.recipes r
   SET snapshot = jsonb_set(r.snapshot, '{fields}', coalesce((
         SELECT jsonb_object_agg(e.key, jsonb_set(e.value, '{decision}', (e.value -> 'decision') || jsonb_build_object(
                    'target_task', CASE WHEN e.value -> 'decision' ->> 'action' = 'target' THEN 'classes' END,
                    'target_bins', NULL)))
           FROM jsonb_each(r.snapshot -> 'fields') e), '{}'::jsonb))
 WHERE r.snapshot IS NOT NULL
   AND EXISTS (SELECT 1 FROM jsonb_each(r.snapshot -> 'fields') e WHERE NOT (e.value -> 'decision' ? 'target_task'));
ALTER TABLE meta.recipes ENABLE TRIGGER recipes_frozen;
```

Also check `meta.exports.snapshot` (frozen experiment snapshots): apply the same `UPDATE` to `meta.exports e SET snapshot = …` with `e.snapshot` in place of `r.snapshot` (exports have no frozen trigger).

- [ ] **Step 4: Run the tests**

Run: `"$PT" tests/test_target_tasks.py` then the whole suite `"$PT" > "$WS/t2.log" 2>&1; tail -3 "$WS/t2.log"`
Expected: PASS; the full suite still green (the trigger only fills columns no code reads yet).

- [ ] **Step 5: Commit**

```bash
git add api/app/migrations/014_target_tasks.sql api/tests/test_target_tasks.py
git commit -m "Store each target's task, kept coherent by one trigger"
```

### Task 3: Every decision path reads, writes and copies the task

**Files:**
- Modify: `api/app/assessment.py` (SHARED_COLS, `Source.decisions`, `write_decision`, `restore`, `seed`, `core_key`, `adopt`, `disagreements`)
- Modify: `api/app/routers/fields.py` (SHARED_COLS, `_merged_sql`, `save_decision`, undo SET list, `set_targets(tasks=…)`)
- Modify: `api/app/routers/team.py` (adoption `decided`/write, review queue and field review payloads)
- Modify: `api/app/recipes.py` (`snapshot`, `load_working`, `spec_rows`, `diff`, new `snapshot_target_tasks`)
- Modify: `api/app/experiments.py` (`seed_copy`)
- Modify: `api/app/reuse.py` (task-aware eligibility, pass tasks to `set_targets`)
- Test: `api/tests/test_target_tasks.py`

**Interfaces:**
- Consumes: Task 1 rules, Task 2 columns/trigger.
- Produces: `assessment.TARGET_COLS = ("target_task", "target_bins")`; `assessment.core_key(d: dict) -> tuple` = `(action, semantic_type, target_task, json.dumps(target_bins, sort_keys=True))`; `fields.set_targets(conn, dataset_id, names, user_id, tasks: dict[str, dict] | None = None)` where `tasks[name] = {"target_task": str, "target_bins": dict | None}`; `recipes.snapshot_target_tasks(snap) -> dict[str, dict]` (`{name: {"task": str, "bins": dict | None}}` for each target in chain order, `"classes"` when the snapshot lacks the key); field and snapshot decision dicts carry `target_task`, `target_bins`.

- [ ] **Step 1: Write the failing tests** (append to `test_target_tasks.py`)

```python
from .team_helpers import adjust, experiment, export, review_all, start, submit_and_finalize, wait  # noqa: E402

PRICES = "x,z,price,c\n" + "".join(f"{i % 17},{(i * 7) % 11},{i * 1.5 + (i % 5)},{'pq'[i % 2]}\n" for i in range(400))


def _task(client, ds, name, view=None):
    d = fields(client, ds, view)[name]
    got = client.get(f"/api/fields/{d['id']}" + (f"?view={view}" if view else "")).json()["decision"]
    return got["target_task"], got["target_bins"]


def test_put_sets_validates_and_keeps_the_task(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    fid = fields(lead, ds)["price"]["id"]
    assert _task(lead, ds, "price") == ("regression", None)
    assert save(lead, fid, target_task="bins", target_bins={"edges": [100, 300]}).status_code == 200
    assert _task(lead, ds, "price") == ("bins", {"edges": [100, 300]})
    r = save(lead, fid, target_task="classes")
    assert r.status_code == 422 and "continuous" in r.text
    r = save(lead, fid, target_task="bins", target_bins={"edges": [5, 1]})
    assert r.status_code == 422 and "ascending" in r.text


def test_put_without_task_keeps_it(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    fid = fields(lead, ds)["price"]["id"]
    save(lead, fid, target_task="bins", target_bins={"k": 4})
    cur = lead.get(f"/api/fields/{fid}").json()["decision"]
    body = {k: cur[k] for k in ("action", "semantic_type", "imputation", "impute_value", "add_missing_indicator",
                                "transform", "scaling", "encoding", "params", "notes", "reviewed")}
    assert lead.put(f"/api/fields/{fid}/decision", json={**body, "notes": "n"}).status_code == 200
    assert _task(lead, ds, "price") == ("bins", {"k": 4})


def test_a_task_disagreement_goes_to_review(app_, people):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [a, b], targets=("price",))
    fid = fields(a, ds)["price"]["id"]
    save(a, fid, target_task="regression")
    save(b, fid, target_task="bins", target_bins={"k": 3})
    for who in (a, b):
        review_all(who, ds)
        assert who.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
    queue = lead.get(f"/api/datasets/{ds['id']}/review").json()["queue"]
    row = next(q for q in queue if q["name"] == "price")
    assert {(x["target_task"], (x["target_bins"] or {}).get("k")) for x in row["assessments"]} == {
        ("regression", None), ("bins", 3)}


def test_agreed_task_is_adopted_finalized_and_carried(app_, people):
    """Review Focus 1: the task survives adoption, finalize, a new version, an experiment copy and a revert."""
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [a, b], targets=("price",))
    for who in (a, b):
        save(who, fields(who, ds)["price"]["id"], target_task="bins", target_bins={"k": 4})
    st = submit_and_finalize(lead, ds, a, b)
    rid = st["recipe"]["id"]
    from app.db import fetch_one
    snap = fetch_one("SELECT snapshot FROM meta.recipes WHERE id = %s", (rid,))["snapshot"]
    assert snap["fields"]["price"]["decision"]["target_task"] == "bins"
    assert snap["fields"]["price"]["decision"]["target_bins"] == {"k": 4}
    assert _task(lead, ds, "price", f"baseline:{rid}") == ("bins", {"k": 4})
    x = experiment(lead, ds, rid)
    view = f"experiment:{x['id']}"
    assert _task(lead, ds, "price", view) == ("bins", {"k": 4})
    adjust(lead, ds, x, "price", target_task="regression")
    adj = lead.get(f"/api/experiments/{x['id']}").json()["adjustments"]
    assert any(c["key"] == "target_task" for f in adj["fields"] for c in f["changes"])
    fid = fields(lead, ds, view)["price"]["id"]
    r = lead.post(f"/api/experiments/{x['id']}/adjustments/revert", json={"field_ids": [fid]})
    assert r.status_code == 200, r.text
    assert _task(lead, ds, "price", view) == ("bins", {"k": 4})
    start(lead, ds, "tree", [lead], start="baseline", source=rid, targets=("price",))
    assert _task(lead, ds, "price") == ("bins", {"k": 4})


def test_bulk_review_keeps_the_task(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    save(lead, fields(lead, ds)["price"]["id"], target_task="bins", target_bins={"k": 3}, reviewed=False)
    review_all(lead, ds)
    assert _task(lead, ds, "price") == ("bins", {"k": 3})


def test_reuse_carries_the_task(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    save(lead, fields(lead, ds)["price"]["id"], target_task="bins", target_bins={"edges": [200, 400]})
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    other = upload(lead, PRICES, targets="")
    start(lead, other, "tree", [lead], start="recipe", source=rid, targets=())
    assert _task(lead, other, "price") == ("bins", {"edges": [200, 400]})
```

Routes used: `GET /api/experiments/{id}` returns `adjustments`; revert is `POST /api/experiments/{id}/adjustments/revert` with `{"field_ids": [...]}`; a finalized baseline's decision reads through `GET /api/fields/{fid}?view=baseline:{rid}`. Also update `test_team.py:332`, which compares a snapshot decision dict exactly: add `"target_task": None, "target_bins": None`.

- [ ] **Step 2: Run them to see them fail**

Run: `"$PT" tests/test_target_tasks.py`
Expected: FAIL (`KeyError: 'target_task'` in the field response).

- [ ] **Step 3: `assessment.py`**

```python
SHARED_COLS = ("action", "semantic_type", "notes", "reviewed")
TARGET_COLS = ("target_task", "target_bins")


def core_key(d: dict) -> tuple:
    """What assessors must agree on for a field: its role, its type and, for a target, how it is predicted."""
    return (d["action"], d["semantic_type"], d.get("target_task"),
            json.dumps(d.get("target_bins"), sort_keys=True))
```

`Source.decisions()`: add `a.target_task, a.target_bins` / `e.target_task, e.target_bins` to the assessment and experiment relations, and to the baseline relation:

```sql
e.value -> 'decision' ->> 'target_task' AS target_task,
nullif(e.value -> 'decision' -> 'target_bins', 'null'::jsonb) AS target_bins,
```

`write_decision`: an explicit task is written; an omitted one stays (the trigger re-checks it against the new type):

```python
def write_decision(conn, src: Source, field_id: int, d: dict, decided: bool) -> None:
    """Shared columns of one field's decision; `decided` marks it a real change (source becomes 'user'). A target
    task left out of `d` keeps the stored one."""
    table, where = _decision_target(src)
    explicit = d.get("target_task") is not None
    conn.execute(
        sql.SQL("""UPDATE {} SET action = %(action)s, semantic_type = %(semantic_type)s, notes = %(notes)s,
                          reviewed = %(reviewed)s, source = CASE WHEN %(decided)s THEN 'user' ELSE source END,
                          target_task = CASE WHEN %(explicit)s THEN %(target_task)s ELSE target_task END,
                          target_bins = CASE WHEN %(explicit)s THEN %(target_bins)s ELSE target_bins END,
                          updated_at = now()
                    WHERE field_id = %(field_id)s AND {}""").format(table, where),
        {**{c: d[c] for c in SHARED_COLS}, "decided": decided, "field_id": field_id, "explicit": explicit,
         "target_task": d.get("target_task"), "target_bins": jsonb(d["target_bins"]) if d.get("target_bins") else None},
    )
```

`restore`: add `target_task = %s, target_bins = %s` with `dec.get("target_task")`, `jsonb(dec["target_bins"]) if dec.get("target_bins") else None`.

`seed`: insert `target_task, target_bins` from `d.target_task, d.target_bins`.

`adopt`: use `core_key` for agreement and return `"target_task": first.get("target_task"), "target_bins": first.get("target_bins")`. `disagreements`: `len({core_key(r) for r in rows}) > 1`. Update both docstrings ("same action, semantic type and target task").

Add `import json` at the top.

- [ ] **Step 4: `routers/fields.py`**

- `SHARED_COLS` gains nothing (it lists what `write_decision` always writes); `_merged_sql` selects `d.target_task, d.target_bins` after `d.semantic_type`.
- `save_decision`: before validation, keep a stored task when none was sent:

```python
        if d["action"] == "target" and d.get("target_task") is None:
            was = _merged(conn, field_id, fam, src)
            if was is not None and was["action"] == "target" and was["semantic_type"] == d["semantic_type"]:
                d = {**d, "target_task": was["target_task"], "target_bins": was["target_bins"]}
        elif d.get("target_task") == "bins" and d.get("target_bins") is None:
            was = _merged(conn, field_id, fam, src)
            if was is not None and was["target_task"] == "bins":
                d = {**d, "target_bins": was["target_bins"]}
        if d["action"] != "target":
            d = {**d, "target_task": None, "target_bins": None}
```

  Add to `test_put_without_task_keeps_it`: a PUT with `target_task="bins"` and no `target_bins` keeps `{"k": 4}`.

  (`errors = validate_decision(...)` follows unchanged.) Provenance compares what is stored after the write, since the trigger may adjust the task: replace the `decided = …` line and the `write_decision` call with

```python
        write_decision(conn, src, field_id, d, False)
        if core_key(prev) != core_key(_merged(conn, field_id, fam, src)):
            mark_decided(conn, src, field_id)
```

  where `assessment.mark_decided(conn, src, field_id)` runs `UPDATE {table} SET source = 'user' WHERE field_id = %s AND {where}` with `_decision_target(src)`. Keep `decided` as a local (`core_key(prev) != core_key(row)`) where later code in `save_decision` reads it.
- Bulk undo team path: add `target_task = r.target_task, target_bins = r.target_bins` to the `SET` list (`jsonb_populate_record` fills them; old batch snapshots give NULL, which the trigger turns into the default, i.e. `classes` for every old target).
- `set_targets(conn, dataset_id, names, user_id, tasks=None)`: eligibility uses each name's task (`(tasks or {}).get(n, {}).get("target_task")`) in `target_problem`; after the `UPDATE … action = 'target'` add

```python
    for n, t in (tasks or {}).items():
        if n in names and t.get("target_task"):
            conn.execute("UPDATE meta.field_decisions SET target_task = %s, target_bins = %s WHERE field_id = %s",
                         (t["target_task"], jsonb(t["target_bins"]) if t.get("target_bins") else None,
                          by_name[n]["id"]))
```

- [ ] **Step 5: `routers/team.py`**

Adoption: `decided = core_key(prev) != core_key(plan)`; pass `"target_task": plan["target_task"], "target_bins": plan["target_bins"]` to `write_decision`. Review queue: select `d.target_task, d.target_bins`; `"team"` gets both; each assessment dict adds `target_task`, `target_bins` (in both `review_queue` and `field_review`).

- [ ] **Step 6: `recipes.py`**

- `snapshot`: select `d.target_task, d.target_bins`; `"decision": {k: f[k] for k in ("action", "semantic_type", "notes", "target_task", "target_bins")}`.
- `snapshot_target_tasks`:

```python
def snapshot_target_tasks(snap: dict) -> dict[str, dict]:
    """How each of the snapshot's targets is predicted, in chain order."""
    return {t: {"task": snap["fields"][t]["decision"].get("target_task") or "classes",
                "bins": snap["fields"][t]["decision"].get("target_bins")} for t in snapshot_targets(snap)}
```

- `load_working`: pass `tasks={t: {"target_task": v["task"], "target_bins": v["bins"]} for t, v in snapshot_target_tasks(snap).items()} if snap else None` to `set_targets`.
- `spec_rows`: add `"target_task"`, `"target_bins"` from the snapshot decision.
- `diff`: compare `("action", "semantic_type", "target_task", "target_bins", "notes")`.

- [ ] **Step 7: `experiments.py` `seed_copy` and `reuse.py`**

`seed_copy`: the decision tuple adds `d.get("target_task"), jsonb(d["target_bins"]) if d.get("target_bins") else None`, and the INSERT lists `target_task, target_bins`.

`reuse.apply`: carried-target eligibility uses `target_problem(type, n_unique, entry["decision"].get("target_task"))`; the second eligibility loop likewise; `set_targets(conn, dataset_id, targets, user_id, tasks={t: {"target_task": …, "target_bins": …} for t in targets})` built from `recipes.snapshot_target_tasks(snap)`.

- [ ] **Step 8: Run the tests**

Run: `"$PT" tests/test_target_tasks.py`, then the full suite to the workspace log.
Expected: all pass. A failure in the full suite on a `before`/`after` dict comparison (history, `_decision_key`) is expected only if a test compared exact key sets; update such a test to include the two keys.

- [ ] **Step 9: Commit**

```bash
git add api/app api/tests
git commit -m "Carry each target's task through assessment, review, baselines, experiments and reuse"
```

### Task 4: Export bands and numbers; band preview; Explore on numeric exports

**Files:**
- Create: `api/app/bands.py`
- Modify: `api/app/transform.py` (`assign_splits` strata, `run_export(target_tasks=…)`, manifest)
- Modify: `api/app/routers/exports.py` (`export_spec`, `_run` summary, `_preview_of` tasks and checks)
- Modify: `api/app/routers/fields.py` (new `GET /api/fields/{id}/bands`)
- Modify: `api/app/analysis/prep.py` (`prepare_export` numeric target; `target_values` honours the task)
- Modify: `api/app/analysis/runner.py` (`_prepare` reads the target task; `data_version` hashes it)
- Test: `api/tests/test_bands.py` (new), `api/tests/test_export_transforms.py`, `api/tests/test_target_tasks.py`, `api/tests/test_explore.py`

**Interfaces:**
- Consumes: `recipes.snapshot_target_tasks` (Task 3).
- Produces:
  - `bands.quantile_edges(values: np.ndarray, k: int) -> list[float]` (NaN ignored, duplicates collapsed);
  - `bands.band_labels(edges: list[float]) -> list[str]`;
  - `bands.assign(values: np.ndarray, edges: list[float]) -> np.ndarray` (float array, band index, NaN stays NaN);
  - `bands.edges_for(bins: dict, train_values: np.ndarray) -> list[float]`;
  - `bands.thin_bands(values: np.ndarray, edges: list[float]) -> list[str]` (labels of bands with < 2 rows);
  - `transform.run_export(..., target_tasks: dict[str, dict] | None = None)` (`{name: {"task", "bins"}}`; None = all classes);
  - manifest keys `target_tasks: {t: "classes"|"regression"|"bins"}`, `target_bins: {t: {"edges", "labels", "k"}}`, `target_stats: {t: {"mean","std","min","max"}}` (regression, training rows), `target_kinds[t] = "regression"` for regression targets; `target_classes` has band labels for bins targets and no entry for regression targets;
  - export summary gains `target_tasks`;
  - `GET /api/fields/{field_id}/bands?k=…` or `?edges=1,2,3` → `{"edges", "labels", "counts", "missing"}` on the full column.

- [ ] **Step 1: Unit tests** (`api/tests/test_bands.py`)

```python
import numpy as np
import pytest

from app.bands import assign, band_labels, edges_for, quantile_edges


def test_quantile_edges_split_evenly():
    assert quantile_edges(np.arange(100, dtype=float), 4) == [24.75, 49.5, 74.25]


def test_quantile_edges_collapse_duplicates():
    v = np.array([0.0] * 90 + list(range(1, 11)), dtype=float)
    edges = quantile_edges(v, 5)
    assert edges == [1.0]
    assert np.unique(assign(v, edges)).size == 2


def test_quantile_edges_of_a_constant_column_are_empty():
    assert quantile_edges(np.zeros(20), 4) == []


def test_labels_stay_distinct_for_close_edges():
    labels = band_labels([1000000.1, 1000000.2])
    assert len(set(labels)) == 3


def test_quantile_edges_ignore_nan():
    assert quantile_edges(np.array([np.nan, 1, 2, 3, 4]), 2) == [2.5]


def test_labels_and_assignment():
    edges = [10.0, 25.0]
    assert band_labels(edges) == ["< 10", "[10, 25)", "≥ 25"]
    got = assign(np.array([3, 10, 24.9, 25, 99, np.nan]), edges)
    assert got[:5].tolist() == [0, 1, 1, 2, 2] and np.isnan(got[5])


def test_edges_for_uses_given_edges_or_training_quantiles():
    assert edges_for({"edges": [1, 2]}, np.arange(10.0)) == [1.0, 2.0]
    assert edges_for({"k": 2}, np.arange(10.0)) == [4.5]
    with pytest.raises(ValueError, match="one band"):
        edges_for({"k": 4}, np.zeros(50))
```

And in `test_export_transforms.py` (follow that file's existing helper that writes a parquet and calls `run_export` with `FieldSpec`s):

```python
def test_export_writes_regression_and_band_targets(tmp_path):
    n = 600
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.normal(size=n), "price": rng.gamma(2.0, 50.0, size=n), "kind": ["a", "b"] * (n // 2)})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    specs = [_spec("x", "numeric_continuous")]
    m = run_export(src, specs, ["price", "kind"], tmp_path / "out", {"seed": 1}, lambda *_: None,
                   target_tasks={"price": {"task": "bins", "bins": {"k": 4}}, "kind": {"task": "classes", "bins": None}})
    assert m["target_tasks"] == {"price": "bins", "kind": "classes"}
    assert len(m["target_bins"]["price"]["edges"]) == 3 and m["target_classes"]["price"][0].startswith("< ")
    out = pl.read_parquet(tmp_path / "out" / "model_ready.parquet")
    train = out.filter(pl.col("__split") == "train")["price"].to_numpy()
    assert set(np.unique(train)) == {0, 1, 2, 3}
    # Edges come from training rows only: the training split's bands are near-equal.
    counts = np.bincount(train)
    assert counts.max() - counts.min() <= 0.05 * len(train) + 2

    m2 = run_export(src, specs, ["price"], tmp_path / "out2", {"seed": 1}, lambda *_: None,
                    target_tasks={"price": {"task": "regression", "bins": None}})
    assert m2["target_kinds"]["price"] == "regression" and "price" not in m2["target_classes"]
    assert m2["target_stats"]["price"]["std"] > 0
    out2 = pl.read_parquet(tmp_path / "out2" / "model_ready.parquet")
    assert out2.schema["price"] == pl.Float64
    assert m2["split_info"]["stratified"] is True


def test_export_refuses_a_single_band(tmp_path):
    df = pl.DataFrame({"x": np.arange(300.0), "y": [0.0] * 295 + [1.0] * 5})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    with pytest.raises(ValueError, match="one band"):
        run_export(src, [_spec("x", "numeric_continuous")], ["y"], tmp_path / "o",
                   {"seed": 1}, lambda *_: None, target_tasks={"y": {"task": "bins", "bins": {"k": 3}}})


def test_prepare_export_keeps_regression_target_numeric(tmp_path):
    from app.analysis.prep import prepare_export
    n = 400
    df = pl.DataFrame({"x": np.arange(n, dtype=float), "price": np.arange(n) * 2.5 + 0.3})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    m = run_export(src, [_spec("x", "numeric_continuous")], ["price"], tmp_path / "o",
                   {"seed": 1}, lambda *_: None, target_tasks={"price": {"task": "regression", "bins": None}})
    p = prepare_export(tmp_path / "o", m, "price", 200)
    assert p.target_kind == "numeric" and np.any(p.y % 1 != 0)
```

`_spec(name, type)` is `test_export_transforms.py`'s helper (line 12) that fills `FieldSpec`'s nine required arguments from the type's suggestion.

- [ ] **Step 2: Run them to see them fail**

Run: `"$PT" tests/test_bands.py tests/test_export_transforms.py -k "band or regression or quantile or labels or edges"`
Expected: FAIL (`ModuleNotFoundError: app.bands`; `run_export() got an unexpected keyword argument 'target_tasks'`).

- [ ] **Step 3: `api/app/bands.py`**

```python
"""Cutting a numeric target into bands: equal-count edges from training values, or the cut points someone chose."""
from __future__ import annotations

import numpy as np


def quantile_edges(values: np.ndarray, k: int) -> list[float]:
    """Equal-count cut points. Repeated quantiles collapse; a cut at the minimum would leave the lowest band empty, so
    it moves up to the next value (90 zeros and 1..10 cut into `< 1` and `≥ 1`)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return []
    lo = v.min()
    above = v[v > lo]
    edges = {float(e) if e > lo else (float(above.min()) if above.size else None)
             for e in np.quantile(v, np.arange(1, k) / k)}
    return sorted(e for e in edges if e is not None)


def edges_for(bins: dict, train_values: np.ndarray) -> list[float]:
    edges = [float(e) for e in bins["edges"]] if "edges" in bins else quantile_edges(train_values, int(bins["k"]))
    if not edges:
        raise ValueError("these bands leave one band only: the training values are all the same")
    v = np.asarray(train_values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size and np.unique(np.searchsorted(edges, v, side="right")).size < 2:
        raise ValueError("these bands leave one band only in the training rows; choose other cut points or fewer bands")
    return edges


def _num(v: float) -> str:
    return f"{v:g}"


def band_labels(edges: list[float]) -> list[str]:
    return ([f"< {_num(edges[0])}"] + [f"[{_num(a)}, {_num(b)})" for a, b in zip(edges, edges[1:])]
            + [f"≥ {_num(edges[-1])}"])


def assign(values: np.ndarray, edges: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.searchsorted(np.asarray(edges, dtype=float), v, side="right").astype(float)
    out[np.isnan(v)] = np.nan
    return out
```

`_num(v)` uses `f"{v:.12g}"` (shortest form up to 12 significant digits) so neighbouring edges never share a label.

Also add `bands.thin_bands(values, edges) -> list[str]`: the labels of bands with fewer than 2 rows among `values` (an export needs every band in the data at least twice, and XGBoost refuses class labels with gaps):

```python
def thin_bands(values: np.ndarray, edges: list[float]) -> list[str]:
    idx = assign(values, edges)
    counts = np.bincount(idx[~np.isnan(idx)].astype(int), minlength=len(edges) + 1)
    return [label for label, n in zip(band_labels(edges), counts) if n < 2]
```

Test: `assert thin_bands(np.array([1.0, 2, 3, 50, 60]), [10, 20]) == ["[10, 20)"]` (the `[10, 20)` band is empty; `< 10` has 3 rows and `≥ 20` has 2).

- [ ] **Step 4: `transform.py`**

- `assign_splits(y, …, strata_edges: dict[str, list[float]] | None = None)`: a numeric target stratifies on bands, so each split covers its range: its own cut points for a `bins` target with explicit edges, else 5 equal-count bands of the labelled rows. Before `labels = y.loc[labeled].astype(str)`, add

```python
    strata = y.copy()
    for c, edges in (strata_edges or {}).items():
        vals = strata[c].to_numpy(dtype=float)
        strata[c] = bands.assign(vals, edges if edges else bands.quantile_edges(vals, 5))
    labels = strata.loc[labeled].astype(str)
```

- `run_export(..., target_tasks=None)`; each entry is `{"task", "bins", "decimal_comma"}`. Numeric targets are read the way profiling read them (`profiling.coerce_numeric`, thousands separators and decimal commas included), and a target whose values don't read as numbers fails the export instead of losing rows:

```python
    tasks = {t: (target_tasks or {}).get(t, {}).get("task") or "classes" for t in targets}
    numeric_targets = [t for t in targets if tasks[t] in ("regression", "bins")]
    raw = pl.read_parquet(parquet_path, columns=targets)
    y_raw = raw.to_pandas()
    for t in numeric_targets:
        num = coerce_numeric(raw[t], (target_tasks or {})[t].get("decimal_comma"))
        given_values = raw[t].fill_nan(None) if raw[t].dtype.is_float() else raw[t]
        lost = int(given_values.is_not_null().sum() - num.is_not_null().sum())
        if lost:
            raise ValueError(f"Target {t!r} has {lost:,} value(s) that don't read as numbers (or are infinite); "
                             "fix them or predict it as classes.")
        y_raw[t] = num.to_numpy()
    check_target_classes(y_raw[[t for t in targets if tasks[t] == "classes"]])
    given = {t: [float(e) for e in ((target_tasks or {})[t].get("bins") or {}).get("edges", [])] for t in numeric_targets}
    split, split_info = assign_splits(y_raw, float(config.get("valid_size", 0.15)),
                                      float(config.get("test_size", 0.15)), seed, strata_edges=given)
    train_rows = split == 0
    y_enc, classes = encode_targets(y_raw[[t for t in targets if tasks[t] == "classes"]])
    target_bins, target_stats = {}, {}
    for t in targets:
        vals = y_raw[t].to_numpy(dtype=float) if t in numeric_targets else None
        if tasks[t] == "bins":
            bins = (target_tasks or {})[t].get("bins") or {"k": DEFAULT_BIN_COUNT}
            edges = bands.edges_for(bins, vals[train_rows])
            labels = bands.band_labels(edges)
            thin = bands.thin_bands(vals[split != 255], edges)
            if thin:
                raise ValueError(f"Target {t!r}: band(s) {', '.join(thin)} have fewer than 2 rows; move or remove "
                                 "cut points, or use fewer bands.")
            y_enc[t] = bands.assign(vals, edges)
            classes[t] = labels
            target_bins[t] = {"edges": edges, "labels": labels, "k": bins.get("k")}
        elif tasks[t] == "regression":
            y_enc[t] = vals
            tr = vals[train_rows & ~np.isnan(vals)]
            if np.unique(tr).size < 2:
                raise ValueError(f"Target {t!r} has a single value in the training rows; there is nothing to predict.")
            target_stats[t] = {"mean": float(tr.mean()), "std": float(tr.std()), "min": float(tr.min()),
                               "max": float(tr.max())}
    y_enc = y_enc[targets]
```

  The rule is "every band has at least 2 rows among the exported rows"; a band missing from the training rows alone is reported by training as today (`missing_from_train`), except that it must not reach XGBoost with a gap, so `train()` refuses a run whose class target lacks a class in training **for XGBoost only** with "XGBoost needs every class in the training rows: {t} lacks …" (add that check next to `missing_from_train` in Task 6). Import `coerce_numeric` from `.profiling` (check for an import cycle; if `profiling` imports `transform`, move the import inside `run_export`).

  Add a test to `test_export_transforms.py`:

```python
def test_numeric_target_stored_as_text_is_read_like_profiling(tmp_path):
    df = pl.DataFrame({"x": np.arange(300.0), "price": [f"{1000 + i:,}.5" for i in range(300)]})
    df.write_parquet(tmp_path / "d.parquet")
    m = run_export(tmp_path / "d.parquet", [_spec("x", "numeric_continuous")], ["price"], tmp_path / "o",
                   {"seed": 1}, lambda *_: None, target_tasks={"price": {"task": "regression", "bins": None}})
    assert m["rows_written"] == 300 and m["target_stats"]["price"]["min"] >= 1000.5


def test_export_refuses_an_empty_band(tmp_path):
    df = pl.DataFrame({"x": np.arange(300.0), "y": np.r_[np.zeros(150), np.full(150, 100.0)]})
    df.write_parquet(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match=r"\[10, 20\)"):
        run_export(tmp_path / "d.parquet", [_spec("x", "numeric_continuous")], ["y"], tmp_path / "o",
                   {"seed": 1}, lambda *_: None, target_tasks={"y": {"task": "bins", "bins": {"edges": [10, 20]}}})
```

  Arrow schema: `pa.field(t, pa.float64() if tasks[t] == "regression" else pa.int32())`; the column write uses `astype(np.float64)` for regression, `astype(np.int32)` otherwise. Manifest:

```python
        "target_classes": {t: classes[t] for t in targets if tasks[t] != "regression"},
        "target_kinds": {t: "regression" if tasks[t] == "regression"
                         else ("binary" if len(classes[t]) == 2 else "multiclass") for t in targets},
        "target_tasks": tasks,
        "target_bins": target_bins,
        "target_stats": target_stats,
```

  Import `from . import bands` and `DEFAULT_BIN_COUNT` from `.semantics`.

- [ ] **Step 5: exports router**

`export_spec` returns `(specs, targets, tasks)` where `tasks = {t: {**v, "decimal_comma": <the field's stats decimal_comma>} for t, v in recipes.snapshot_target_tasks(snap).items()}` (read `coalesce((stats ->> 'decimal_comma')::boolean, false)` for the target names, as `spec_rows` does for features); `_run` passes `target_tasks=tasks` and adds `"target_tasks": manifest["target_tasks"]` to the summary. Every other caller of `export_spec` (grep) unpacks three values. `_preview_of`: `tasks = recipes.snapshot_target_tasks(snap)`; return `"target_tasks": {t: v["task"] for t, v in tasks.items()}`; `_lone_class_problem` gets only the `classes` targets; for a `bins` target with explicit edges, add a blocking line naming the bands with fewer than 2 rows in the column (`bands.thin_bands` on the column read with `coerce_numeric`); a regression target with fewer than 2 distinct values blocks with "has a single value".

- [ ] **Step 6: band preview endpoint (`routers/fields.py`)**

```python
@router.get("/fields/{field_id}/bands")
def band_preview(field_id: int, k: int | None = Query(None, ge=2, le=20), edges: str | None = None) -> dict:
    """How a numeric field would be cut into bands, on the whole column (an export cuts on its training rows)."""
    f = _field(field_id)
    if (k is None) == (edges is None):
        raise HTTPException(422, "Give either k or edges")
    ds = fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (f["dataset_id"],))
    vals = pl.read_parquet(ds["parquet_path"], columns=[f["name"]]).to_series().cast(pl.Float64, strict=False).to_numpy()
    try:
        cut = [float(e) for e in edges.split(",")] if edges else bands.quantile_edges(vals, k)
    except ValueError:
        raise HTTPException(422, "edges must be numbers separated by commas") from None
    if not cut:
        raise HTTPException(422, "Every value is the same, so there is nothing to cut into bands")
    if problem := bins_problem({"edges": cut}):
        raise HTTPException(422, problem)
    idx = bands.assign(vals, cut)
    present = idx[~np.isnan(idx)].astype(int)
    return {"edges": cut, "labels": bands.band_labels(cut),
            "counts": np.bincount(present, minlength=len(cut) + 1).tolist(), "missing": int(np.isnan(idx).sum())}
```

  Read the column with `coerce_numeric(series, f["stats"].get("decimal_comma"))` rather than a plain cast. `routers/fields.py` imports `numpy as np`, `from .. import bands`, `bins_problem` from `..semantics` and `coerce_numeric` from `..profiling`.

  Test (in `test_target_tasks.py`):

```python
def test_band_preview(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    fid = fields(lead, ds)["price"]["id"]
    r = lead.get(f"/api/fields/{fid}/bands?k=4").json()
    assert len(r["labels"]) == len(r["counts"]) and sum(r["counts"]) == 400
    r = lead.get(f"/api/fields/{fid}/bands?edges=100,300").json()
    assert r["labels"] == ["< 100", "[100, 300)", "≥ 300"]
    assert lead.get(f"/api/fields/{fid}/bands?edges=3,1").status_code == 422
```

- [ ] **Step 7: Explore**

`prepare_export`: `numeric = (manifest.get("target_kinds") or {}).get(target) == "regression"`; read the target as float (`.cast(pl.Float64).to_numpy()`, NaN kept) when numeric, sample with kind `"numeric"`, and return `Prepared(..., "numeric" if numeric else "categorical", y, [] if numeric else classes)`.

`target_values(s, semantic_type, task=None)`: when `task == "classes"` skip the numeric branch. The raw path's fields come from `analysis/runner.py` `_prepare` (lines 130-139): add `d.target_task` to its SELECT, and have `prepare_raw` pass the target field's `target_task` to `target_values`. `runner.data_version` (lines 32-43) hashes `field_id|action|semantic_type`; add `target_task` and `target_bins` to the SELECT and the hashed text so switching a target between Classes and Number invalidates cached results. Test in `test_explore.py` following its existing raw-mode tests: the same params give a different `data_version` after the target's task changes.

- [ ] **Step 8: API export test** (append to `test_target_tasks.py`)

```python
def test_regression_export_end_to_end(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid)
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert pv["target_tasks"] == {"price": "regression"} and not pv["blocking"]
    e = export(lead, ds, x)
    assert e["summary"]["target_tasks"] == {"price": "regression"}
    assert e["summary"]["target_kinds"] == {"price": "regression"}
```


- [ ] **Step 9: Run everything**

Run: `"$PT" tests/test_bands.py tests/test_export_transforms.py tests/test_target_tasks.py tests/test_explore.py`, then the full suite.
Expected: PASS.

- [ ] **Step 10: Commit, push, PR A**

```bash
git add api/app api/tests
git commit -m "Export numeric targets as numbers or as bands cut on the training rows"
```

Open PR A (`regression-a` branch from main with the spec and plan commits), wait for CI, read it, merge, deploy to dev-01 (`DOCKER_CONTEXT=dev-01 make up`), check migration 014 applied and an existing dataset's targets read `classes`.

## PR B — training, metrics, MLflow, comparison

### Task 5: Regressors in the registry

**Files:**
- Modify: `api/app/modeling/registry.py`
- Test: `api/tests/test_registry.py` (create if absent; else the file that tests `param_problems`)

**Interfaces:**
- Produces: `ModelSpec.build_regressor: Callable[[dict, int], Any] | None`, `ModelSpec.regressor_defaults: dict` (default `{}`); `registry.CLASSIFIER_ONLY` (parameter names that only make sense for a classifier); `registry.regressor_params(spec, params: dict) -> dict` (keeps only parameters the regressor accepts, LightGBM's native aliases included, minus `CLASSIFIER_ONLY`); `registry.regression_problem(spec, kinds: list[str]) -> str | None`; `importances` handles 1-D `coef_`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest

from app.modeling.registry import MODELS, ModelSpec, importances, regression_problem, regressor_params


@pytest.mark.parametrize("key", [k for k, m in MODELS.items() if m.available])
def test_every_model_builds_a_regressor_that_fits(key):
    spec = MODELS[key]
    est = spec.build_regressor({**spec.regressor_defaults, **regressor_params(spec, spec.default_params)}, 0)
    X = np.random.default_rng(0).normal(size=(80, 3))
    est.fit(X, X[:, 0] * 2 + 1)
    assert est.predict(X).shape == (80,)


def test_classifier_only_parameters_are_left_out():
    assert "C" not in regressor_params(MODELS["logistic_regression"], {"C": 2.0, "max_iter": 50})
    assert regressor_params(MODELS["lightgbm"], {"class_weight": "balanced", "is_unbalance": True,
                                                 "objective": "binary", "min_data_in_leaf": 5}) == {"min_data_in_leaf": 5}
    assert regressor_params(MODELS["xgboost"], {"scale_pos_weight": 3, "max_depth": 4}) == {"max_depth": 4}


def test_a_model_without_a_regressor_refuses_a_number():
    spec = ModelSpec("x", "X", "tree", True, lambda p, s: None)
    assert "number" in regression_problem(spec, ["binary", "regression"])
    assert regression_problem(spec, ["binary"]) is None


def test_ridge_importances_are_per_feature():
    est = MODELS["logistic_regression"].build_regressor({}, 0).fit(np.eye(4), [1.0, 2.0, 3.0, 4.0])
    assert importances(MODELS["logistic_regression"], est, list("abcd")).shape == (4,)
```

- [ ] **Step 2: Run** `"$PT" tests/test_registry.py` → FAIL (ImportError).

- [ ] **Step 3: Implement**

Builders `_lightgbm_reg` (`LGBMRegressor`), `_xgboost_reg` (`XGBRegressor`), `_hgb_reg` (`HistGradientBoostingRegressor`), `_random_forest_reg` (`RandomForestRegressor`), `_ridge` (`Ridge(**params)`; Ridge takes `random_state` only for some solvers, so pass `random_state=seed`), `_mlp_reg` (`MLPRegressor`). Each `ModelSpec` in `MODELS` gets `build_regressor=…`; logistic regression gets `regressor_defaults={"alpha": 1.0}`; `description` stays the classifier's. Update the module docstring's first line to "every model that can sit in a target chain".

```python
# Accepted by some regressors too (LGBMRegressor takes class_weight), but they describe classes.
CLASSIFIER_ONLY = {"class_weight", "is_unbalance", "unbalance", "unbalanced_sets", "scale_pos_weight", "objective",
                   "num_class", "num_classes", "multi_class", "C", "penalty", "l1_ratio"}


def regressor_params(spec: ModelSpec, params: dict[str, Any]) -> dict[str, Any]:
    """The run's parameters the regressor accepts, minus the ones that describe classes."""
    if spec.build_regressor is None:
        return {}
    known = set(spec.build_regressor({}, 0).get_params())
    if spec.key == "lightgbm":
        known |= _lightgbm_native_params()
    return {k: v for k, v in params.items() if k in known and k not in CLASSIFIER_ONLY}


def regression_problem(spec: ModelSpec, kinds: list[str]) -> str | None:
    if "regression" in kinds and spec.build_regressor is None:
        return f"{spec.label} can't predict a number yet; choose another model or predict the target as bands"
    return None
```

`importances` for `coef_`: `np.abs(np.atleast_2d(np.asarray(est.coef_, dtype=float))).mean(axis=0)`.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `"Regressor counterpart for every model"`.

### Task 6: TargetChain, regression metrics, train()

**Files:**
- Modify: `api/app/modeling/train.py`
- Modify: `api/app/routers/runs.py` (refuse a model without a regressor)
- Test: `api/tests/test_train_regression.py` (new)

**Interfaces:**
- Consumes: Task 4 manifest keys, Task 5 registry.
- Produces: `train.TargetChain(classifier, regressor, kinds: list[str], order: list[int], cv: int | None, handles_nan: bool)` with `.fit(X, Y)`, `.estimators_`, `.order_`, `.predict_all(X, n_classes: list[int]) -> list[np.ndarray]` (per target, target order: class-probability matrix or predicted values); `train.evaluate_regression(y, pred, seed=42) -> tuple[dict[str, float], dict]` (details `{"kind": "regression", "n", "scatter": {"actual", "predicted"}, "residuals": {"edges", "counts"}}`); report keys `target_tasks`, `target_kinds`, `chain_kind` (`"classifier_chain"` | `"target_chain"`), `chain_method` (`"per_target"` for a target chain).

- [ ] **Step 1: Failing tests**

```python
import json

import numpy as np
import polars as pl
import pytest

from app.modeling.train import TargetChain, evaluate_regression, train
from app.transform import run_export

from .test_export_transforms import _spec


def test_regression_metrics_on_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m, d = evaluate_regression(y, np.array([1.0, 2.0, 3.0, 5.0]))
    assert m["mae"] == pytest.approx(0.25) and m["rmse"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 1 / 5) and m["mape"] == pytest.approx(0.0625)
    assert d["kind"] == "regression" and len(d["residuals"]["counts"]) == 30 and len(d["scatter"]["actual"]) == 4


def test_mape_is_left_out_when_an_actual_is_zero():
    m, _ = evaluate_regression(np.array([0.0, 1.0, 2.0]), np.array([0.1, 1.0, 2.0]))
    assert "mape" not in m


def test_scatter_is_capped():
    y = np.arange(5000, dtype=float)
    _, d = evaluate_regression(y, y + 1)
    assert len(d["scatter"]["actual"]) == 1000


def test_target_chain_mixes_and_chains_out_of_fold():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    a = (X[:, 0] > 0).astype(float)
    b = 3 * a + X[:, 1] + rng.normal(scale=0.1, size=300)
    chain = TargetChain(RandomForestClassifier(n_estimators=20, random_state=0),
                        RandomForestRegressor(n_estimators=20, random_state=0),
                        ["binary", "regression"], [0, 1], cv=3, handles_nan=True).fit(X, np.column_stack([a, b]))
    assert chain.estimators_[1].n_features_in_ == 4
    proba, values = chain.predict_all(X, [2, 0])
    assert proba.shape == (300, 2) and values.shape == (300,)
    assert np.corrcoef(values, b)[0, 1] > 0.9


def _export(tmp_path, tasks):
    n = 600
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    df = pl.DataFrame({"x": x, "z": rng.normal(size=n), "g1": (x > 0).astype(int),
                       "g3": 10 + 4 * x + rng.normal(scale=0.5, size=n)})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    specs = [_spec("x", "numeric_continuous"), _spec("z", "numeric_continuous")]
    out = tmp_path / "export"
    run_export(src, specs, list(tasks), out, {"seed": 3}, lambda *_: None,
               target_tasks={t: {"task": k, "bins": {"k": 4} if k == "bins" else None} for t, k in tasks.items()})
    return out


def test_lightgbm_regression_run(tmp_path):
    report = train(_export(tmp_path, {"g3": "regression"}), "lightgbm", params={"n_estimators": 60}, cv=3,
                   out_dir=tmp_path / "m")
    m = report["metrics"]["valid"]["g3"]
    assert m["r2"] > 0.8 and {"rmse", "mae", "r2", "mape"} <= set(m)
    assert report["chain_kind"] == "target_chain" and report["target_tasks"] == {"g3": "regression"}
    assert report["evaluation"]["test"]["g3"]["kind"] == "regression"
    json.dumps(report)


def test_mixed_chain_classes_then_number(tmp_path):
    report = train(_export(tmp_path, {"g1": "classes", "g3": "regression"}), "lightgbm",
                   ["g1", "g3"], params={"n_estimators": 40}, cv=3, out_dir=tmp_path / "m")
    assert "roc_auc" in report["metrics"]["valid"]["g1"] and "r2" in report["metrics"]["valid"]["g3"]


def test_all_class_runs_keep_classifier_chain(tmp_path):
    report = train(_export(tmp_path, {"g1": "classes", "g3": "bins"}), "lightgbm", params={"n_estimators": 20},
                   cv=None, out_dir=tmp_path / "m")
    assert report["chain_kind"] == "classifier_chain" and len(report["target_classes"]["g3"]) == 4


def test_single_value_regression_target_is_refused(tmp_path):
    with pytest.raises(ValueError, match="single value"):
        n = 100
        df = pl.DataFrame({"x": np.arange(n, dtype=float), "y": np.ones(n)})
        df.write_parquet(tmp_path / "d.parquet")
        run_export(tmp_path / "d.parquet", [_spec("x", "numeric_continuous")], ["y"],
                   tmp_path / "e", {"seed": 1}, lambda *_: None, target_tasks={"y": {"task": "regression", "bins": None}})
```


- [ ] **Step 2: Run** `"$PT" tests/test_train_regression.py` → FAIL (ImportError `TargetChain`).

- [ ] **Step 3: Implement in `train.py`**

```python
class TargetChain:
    """A chain whose targets may be classes or numbers: one classifier or regressor per target, in `order`. Each
    later model sees the features plus the earlier targets' outputs, out-of-fold with `cv` folds (the ground truth
    when cv is None): P(class 1) for a binary target, the predicted class for a multiclass one, the predicted value
    for a number. Runs with only class targets use ClassifierChain instead, so their behaviour is unchanged."""

    chain_method_ = "per_target"

    def __init__(self, classifier, regressor, kinds: list[str], order: list[int], cv: int | None = 5,
                 handles_nan: bool = False):
        self.classifier, self.regressor, self.kinds = classifier, regressor, kinds
        self.order, self.cv, self.handles_nan = order, cv, handles_nan

    def _output(self, est, X: np.ndarray, kind: str) -> np.ndarray:
        return est.predict_proba(X)[:, 1] if kind == "binary" else est.predict(X)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TargetChain":
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_predict

        feats = np.zeros((X.shape[0], len(self.order)))
        self.estimators_ = []
        with config_context(assume_finite=self.handles_nan):
            for pos, t in enumerate(self.order):
                kind = self.kinds[t]
                y = Y[:, t] if kind == "regression" else Y[:, t].astype(np.int64)
                X_aug = np.hstack((X, feats[:, :pos]))
                base = self.regressor if kind == "regression" else self.classifier
                self.estimators_.append(clone(base).fit(X_aug, y))
                if pos == len(self.order) - 1:
                    break
                if self.cv is None:
                    feats[:, pos] = y
                else:
                    out = cross_val_predict(clone(base), X_aug, y, cv=self.cv,
                                            method="predict_proba" if kind == "binary" else "predict")
                    feats[:, pos] = out[:, 1] if kind == "binary" else out
        self.order_ = list(self.order)
        return self

    def predict_all(self, X: np.ndarray, n_classes: list[int]) -> list[np.ndarray]:
        feats = np.zeros((X.shape[0], len(self.order_)))
        out: list[np.ndarray | None] = [None] * len(self.order_)
        with config_context(assume_finite=self.handles_nan):
            for pos, est in enumerate(self.estimators_):
                t = self.order_[pos]
                X_aug = np.hstack((X, feats[:, :pos]))
                if self.kinds[t] == "regression":
                    out[t] = est.predict(X_aug).astype(float)
                else:
                    p = est.predict_proba(X_aug)
                    full = np.zeros((p.shape[0], n_classes[t]))
                    full[:, np.asarray(est.classes_, dtype=int)] = p
                    out[t] = full
                feats[:, pos] = self._output(est, X_aug, self.kinds[t])
        return out  # type: ignore[return-value]


SCATTER_MAX_POINTS = 1000
RESIDUAL_BINS = 30


def evaluate_regression(y: np.ndarray, pred: np.ndarray, seed: int = 42) -> tuple[dict[str, float], dict[str, Any]]:
    """Score a numeric target: RMSE, MAE, R² (when the actual values vary) and MAPE (when none is 0); details hold up to
    SCATTER_MAX_POINTS sampled (actual, predicted) pairs and a histogram of the errors (predicted − actual)."""
    err = pred - y
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}
    if y.size >= 2 and np.var(y) > 0:
        out["r2"] = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
    if not np.any(y == 0):
        out["mape"] = float(np.mean(np.abs(err) / np.abs(y)))
    idx = np.arange(y.size)
    if y.size > SCATTER_MAX_POINTS:
        idx = np.sort(np.random.default_rng(seed).choice(y.size, SCATTER_MAX_POINTS, replace=False))
    counts, edges = np.histogram(err, bins=RESIDUAL_BINS)
    details = {
        "kind": "regression", "n": int(y.size),
        "scatter": {"actual": [float(f"{v:.6g}") for v in y[idx]], "predicted": [float(f"{v:.6g}") for v in pred[idx]]},
        "residuals": {"edges": [float(f"{v:.6g}") for v in edges], "counts": counts.astype(int).tolist()},
    }
    return out, details
```

In `train()`:
- `kinds = [manifest["target_kinds"][t] for t in targets]`; `regression = [k == "regression" for k in kinds]`;
- `xy` returns `Y` as `float64` (`.astype(np.float64)`);
- class checks run only on class targets; a regression target with fewer than 2 distinct training values raises `ValueError(f"Target {t!r} has a single value in the training split; there is nothing to predict.")`;
- `n_classes = [0 if r else len(manifest["target_classes"][t]) for t, r in zip(targets, regression)]`;
- XGBoost refuses class labels with gaps: when `spec.key == "xgboost"` and `missing_from_train` is non-empty, raise `ValueError(f"XGBoost needs every class of a target in the training rows; missing: {...}")` naming target and classes (test: an export whose rare class lands only in validation, trained with xgboost, raises this message);
- when `any(regression)`: `if problem := regression_problem(spec, kinds): raise ValueError(problem)`; build `TargetChain(spec.build(merged, seed), spec.build_regressor({**spec.regressor_defaults, **regressor_params(spec, merged)}, seed), kinds, order, cv, spec.handles_nan)` where `merged = {**spec.default_params, **params}`; fit with `chain.fit(X_tr, Y_tr)`;
- else today's `build_chain(...)` and `chain.fit(X_tr, Y_tr.astype(np.int64))`;
- evaluation: a target chain uses `outs = chain.predict_all(X, n_classes)` and per target `evaluate_regression(Y[:, i], outs[i], seed)` or `evaluate(Y[:, i].astype(int), outs[i], labels)`; the classifier-chain path is unchanged (cast `Y` to int there);
- report adds `"target_tasks": {t: (manifest.get("target_tasks") or {}).get(t, "classes") for t in targets}` (the run's targets only), `"target_kinds": {t: manifest["target_kinds"][t] for t in targets}`, `"chain_kind": "target_chain" if any(regression) else "classifier_chain"`; `target_classes` only lists class targets.
- Module docstring: describe both chains in two short paragraphs; `main()`'s confusion print skips details with `kind == "regression"` (print RMSE/R² instead).

`routers/runs.py start_run`: after the order check, `kinds = [manifest["target_kinds"][t] for t in (req.order or manifest["targets"])]` and `if problem := regression_problem(spec, kinds): raise HTTPException(422, problem)`.

- [ ] **Step 4: Run** `"$PT" tests/test_train_regression.py tests/test_data_correctness.py tests/test_pipeline.py` → PASS.

- [ ] **Step 5: Commit** `"Train chains that mix classes and numbers, with regression metrics"`.

### Task 7: MLflow, run comparison, GPU

**Files:**
- Modify: `api/app/modeling/mlflow_log.py` (`run_params`)
- Modify: `api/app/experiments.py`, `api/app/routers/datasets.py` (one shared `HEADLINE_METRICS` with `r2`)
- Modify: `api/app/views.sql` (`v_fields` exposes `target_task`, `target_bins` from `meta.field_decisions`; `assessment.Source.fields_view` adds `dc.target_task, dc.target_bins` to match), `api/app/routers/query.py` (the "Compare model runs" example includes `'r2', 'rmse'`)
- Test: `api/tests/test_mlflow.py`, `api/tests/test_targets_and_lists.py`, `api/tests/test_target_tasks.py`

**Interfaces:**
- Produces: MLflow params `target.<t>.task`; `experiments.HEADLINE_METRICS = ("roc_auc", "roc_auc_ovr_macro", "r2", "accuracy")` imported by `routers/datasets.py`.

- [ ] **Step 1: Failing tests**

```python
# test_mlflow.py
def test_run_params_record_each_targets_task():
    from app.modeling.mlflow_log import run_params
    p = run_params({"model": "lightgbm", "order": ["g3"], "target_tasks": {"g3": "regression", "g1": "classes"}})
    assert p["target.g3.task"] == "regression" and p["target.g1.task"] == "classes"


# test_targets_and_lists.py
def test_headline_metric_uses_r2_for_a_regression_first_target():
    from app.routers.datasets import headline_metric
    assert headline_metric("done", ["price"], {"price": {"rmse": 3.0, "r2": 0.7}}) == {"metric": "r2", "value": 0.7}
```

And in `test_target_tasks.py`, an end-to-end run (LightGBM, `n_estimators` small) on the `PRICES` export that checks the run finishes with `metrics.valid.price.r2` and that `GET /api/datasets/{id}/experiments` (served by `experiments.list_for`) reports the experiment's `best.metric == "r2"`.

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**: in `run_params` after the estimator loop, `for t, task in (report.get("target_tasks") or {}).items(): out[_key("target", t, "task")] = task` (before the final mapping, as a plain key that the final `_key(k)` keeps); move `HEADLINE_METRICS` to `experiments.py` only and import it in `routers/datasets.py`.

- [ ] **Step 4: Run** the three files, then the full suite → PASS.

- [ ] **Step 5: Commit, PR B**: `"Log target tasks to MLflow; R² as a regression run's headline"`. Open PR B, CI green, merge, deploy; on dev-01 run XGBoost on the GPU with a regression target and confirm `device == "cuda"` in the report.

## PR C — UI, samples, docs

### Task 8: Predict as, in the decision panel and target pickers

**Files:**
- Modify: `ui/src/types.ts` (`Decision.target_task`, `target_bins`, `TargetBins`, `TargetTask`, catalog `target_rule`, review rows, `FieldResponse.baseline`)
- Modify: `ui/src/decision.ts` (`toInput` carries the task; new `defaultTask`, `taskProblem`, `bandsLabel`)
- Modify: `ui/src/api.ts` (`bands(fieldId, q)`)
- Modify: `ui/src/components/DecisionPanel.tsx` (Predict as block, band settings + preview)
- Modify: `ui/src/baselines.ts` (`eligibleTarget` follows the new rule; `defaultTaskLabel`)
- Modify: `ui/src/components/StartBaseline.tsx`, `ui/src/pages/DatasetsPage.tsx` (use `eligibleTarget`, show the default task)
- Modify: `ui/src/pages/DatasetTeamPage.tsx` (the lead's assessor comparison, lines 285-340: a *Predict as* row, and `target_task`/`target_bins` in `differing`)
- Modify: `ui/src/recipes.ts` (`SETTING_LABEL` gains `target_task: "Predict as"`, `target_bins: "Bands"`; the diff/adjustment value formatter renders a task with `taskLabel` and bins as "4 equal-sized" / "cut at 10, 25")
- Modify: `ui/src/experiments.ts` (`revertedInput`, `baselineDiff` compare the task)
- Test: `ui/src/decision.test.ts`, `ui/src/baselines.test.ts`, `ui/src/experiments.test.ts`

**Interfaces:**
- Consumes: API fields `target_task`, `target_bins`, `GET /api/fields/{id}/bands`, catalog `target_rule`.
- Produces: `type TargetTask = "classes" | "regression" | "bins"`; `type TargetBins = { k: number } | { edges: number[] }`; `defaultTask(type: string, nUnique: number | null): TargetTask | null`; `taskLabel(t: TargetTask): string` ("Classes" | "Number" | "Bands"); `parseEdges(text: string): number[] | string` (numbers, or the problem).

- [ ] **Step 1: Failing tests**

```ts
// decision.test.ts
import { defaultTask, parseEdges, taskLabel } from "./decision";

describe("target tasks", () => {
  it("defaults like the API", () => {
    expect(defaultTask("binary", 2)).toBe("classes");
    expect(defaultTask("numeric_continuous", 30)).toBe("regression");
    expect(defaultTask("numeric_discrete", 100)).toBe("classes");
    expect(defaultTask("numeric_discrete", 101)).toBe("regression");
    expect(defaultTask("text", 5)).toBeNull();
  });
  it("labels", () => expect(taskLabel("regression")).toBe("Number"));
  it("parses cut points", () => {
    expect(parseEdges("10, 25.5,80")).toEqual([10, 25.5, 80]);
    expect(parseEdges("5, 1")).toMatch(/ascending/);
    expect(parseEdges("a")).toMatch(/number/);
  });
});

// baselines.test.ts (replace the continuous case)
expect(eligibleTarget(f("x", { semantic_type: "numeric_continuous", n_unique: 5000 }))).toBeNull();
expect(eligibleTarget(f("t", { semantic_type: "text" }))).toMatch(/text/i);
expect(eligibleTarget(f("id", { semantic_type: "categorical", n_unique: 5000 }))).toMatch(/100/);

// experiments.test.ts
it("revertedInput restores the baseline's task", () => {
  const draft = { ...input, action: "target", target_task: "regression", target_bins: null } as DecisionInput;
  const out = revertedInput(draft, { action: "target", semantic_type: "numeric_continuous", notes: null,
    target_task: "bins", target_bins: { k: 4 }, settings: null });
  expect(out.target_task).toBe("bins");
  expect(out.target_bins).toEqual({ k: 4 });
});
```

(Use the fixtures those test files already define for `f(...)` and `input`.)

- [ ] **Step 2: Run** `cd ui && npm test` → FAIL.

- [ ] **Step 3: Implement**

`decision.ts`:

```ts
export type TargetTask = "classes" | "regression" | "bins";
const MAX_CLASSES = 100;

export function defaultTask(type: string, nUnique: number | null): TargetTask | null {
  if (type === "binary" || type === "categorical") return "classes";
  if (type === "numeric_continuous") return "regression";
  if (type === "numeric_discrete") return (nUnique ?? 0) <= MAX_CLASSES ? "classes" : "regression";
  return null;
}

export const taskLabel = (t: TargetTask) => ({ classes: "Classes", regression: "Number", bins: "Bands" })[t];

export function parseEdges(text: string): number[] | string {
  const parts = text.split(",").map((s) => s.trim()).filter(Boolean);
  const nums = parts.map(Number);
  if (!parts.length || nums.some((n) => !Number.isFinite(n))) return "Enter numbers separated by commas";
  if (nums.length > 19) return "At most 19 cut points";
  if (nums.some((n, i) => i > 0 && n <= nums[i - 1])) return "Cut points must be ascending, each different";
  return nums;
}
```

`toInput` adds `target_task: "target_task" in d ? d.target_task ?? null : null, target_bins: "target_bins" in d ? d.target_bins ?? null : null`. `DecisionInput` keeps them (they are part of `Decision`). `retype` clears both when the new type can't take the current task (`defaultTask(type, …) === null` or a number task on a non-numeric type) so the server's default applies.

`DecisionPanel.tsx`: under the action segment, when `isTarget`:

```tsx
{isTarget && (
  <div className={`setting${draft.target_task !== server.target_task ? " edited" : ""}`}>
    <div className="setting-row">
      <span>Predict as</span>
      <div className="segmented" role="radiogroup" aria-label="Predict as">
        {(["classes", "regression", "bins"] as const).map((t) => (
          <button key={t} type="button" role="radio" aria-checked={task === t} className={task === t ? "on" : ""}
            disabled={!taskAllowed(t)} title={taskHint(t)}
            onClick={() => update({ target_task: t, target_bins: t === "bins" ? draftRef.current.target_bins ?? { k: catalog.target_rule.default_k } : null }, 0)}>
            {taskLabel(t)}
          </button>
        ))}
      </div>
    </div>
    {task === "bins" && <BandSettings fieldId={f.id} bins={draft.target_bins} onChange={(b) => update({ target_bins: b })} />}
  </div>
)}
```

where `task = draft.target_task ?? defaultTask(draft.semantic_type, f.n_unique)`, `taskAllowed(t)` is `t === "classes" ? draft.semantic_type !== "numeric_continuous" && (f.n_unique ?? 0) <= catalog.target_rule.max_levels : NUMERIC.has(draft.semantic_type)`, and `taskHint` explains a disabled choice ("Only numeric fields can be predicted as a number", "More than 100 distinct values"). `BandSettings` (same file): a two-option segment *Equal-sized* (number input 2–20) / *Cut points* (text input parsed by `parseEdges`, error shown inline, committed only when valid), and a preview from `useQuery(["bands", fieldId, key], () => api.bands(fieldId, q))` rendered as a compact list `label · count` with a bar width proportional to the count, plus the note "Equal-sized bands are cut again on each export's training rows."

`baselines.ts eligibleTarget`: use `catalog`-free mirror of the API rule: not in `["binary","categorical","numeric_discrete","numeric_continuous"]` → "a {type} field can't be a target"; categorical/binary with `n_unique > 100` → "more than 100 distinct values". `StartBaseline.tsx`: after the field name, `<span className="muted small-text"> · {taskLabel(defaultTask(f.semantic_type, f.n_unique)!)}</span>` for eligible fields, and the legend hint becomes "· fixed for this baseline; experiments can change them; Number and Bands are set per field".

`DatasetsPage.tsx`: replace its local eligibility function with `eligibleTarget` from `baselines.ts`.

`DatasetTeamPage.tsx`: the assessor comparison builds rows with `line(...)` and marks differing ones with `differing([...])`; add a *Predict as* line (shown when any assessor made the field a target) whose value per assessor is `bandsText(a)` = `taskLabel(a.target_task)` plus ` · ${k} equal-sized` or ` · cut at ${edges.join(", ")}`, and include it in `differing`, so a disagreement on the task alone shows what differs. `ReviewPage.tsx` needs no change. Add a vitest for `bandsText` in `recipes.test.ts` (`{target_task:"bins", target_bins:{k:4}}` → `"Bands · 4 equal-sized"`).

`experiments.ts`: `revertedInput` copies `target_task`/`target_bins` from the baseline; `baselineDiff` adds `"target_task"` to the compared keys; the `BaselineLine` text includes the task when it differs.

- [ ] **Step 4: Run** `cd ui && npm run typecheck && npm test` → PASS.

- [ ] **Step 5: Commit** `"Predict as: classes, a number or bands, with a band preview"`.

### Task 9: Regression results in runs and run detail

**Files:**
- Modify: `ui/src/types.ts` (`RegressionEvaluation`, `ModelRun.report.target_kinds`, `ExportJob.summary.target_tasks`)
- Modify: `ui/src/runs.ts` (`RunMetric = "primary" | "secondary"`, regression aware)
- Modify: `ui/src/charts.ts` (`scatterOption`)
- Modify: `ui/src/components/RunDetail.tsx` (`RegressionCard`)
- Modify: `ui/src/pages/ExperimentWorkspace.tsx` (column headers, "Use from code" line)
- Modify: `ui/src/experiments.ts` (`metricLabel("r2") === "R²"`), `ui/src/pages/DatasetsPage.tsx` (use `metricLabel`)
- Test: `ui/src/runs.test.ts`, `ui/src/components/runDetail.test.ts`, `ui/src/experiments.test.ts`

**Interfaces:**
- Produces: `metricFor(r, split, target, m: "primary" | "secondary")` → AUC / F1 macro for classes, R² / RMSE for numbers; `bestRuns` stars the lowest RMSE; `regressionTiles(m)`; `scatterOption(t, actual, predicted)`; `columnLabels(kind?: string): [string, string]`.

- [ ] **Step 1: Failing tests**

```ts
// runs.test.ts
const reg = (id: number, r2: number, rmse: number) =>
  ({ id, metrics: { valid: { p: { r2, rmse } }, test: { p: { r2, rmse } } } }) as unknown as ModelRun;
it("stars the highest R² and the lowest RMSE", () => {
  const best = bestRuns([reg(1, 0.8, 3), reg(2, 0.7, 2)], ["p"]);
  expect(best["p:primary"]).toBe(1);
  expect(best["p:secondary"]).toBe(2);
});
it("reads R² and RMSE for a number target", () => {
  expect(metricFor(reg(1, 0.8, 3), "test", "p", "primary")).toBe(0.8);
  expect(metricFor(reg(1, 0.8, 3), "test", "p", "secondary")).toBe(3);
});

// runDetail.test.ts
it("regression tiles", () => {
  expect(regressionTiles({ rmse: 2, mae: 1, r2: 0.9 }).map(([l]) => l)).toEqual(["R²", "RMSE", "MAE", "MAPE"]);
  expect(regressionTiles({ rmse: 2, mae: 1, r2: 0.9 })[3][2]).toMatch(/0/);
});

// experiments.test.ts
it("labels R²", () => expect(metricLabel("r2")).toBe("R²"));
```

Update the existing `runs.test.ts` cases from `"auc"`/`"f1"` to `"primary"`/`"secondary"`.

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`runs.ts`:

```ts
export type RunMetric = "primary" | "secondary";
const isNumber = (x: Record<string, number>) => "r2" in x || "rmse" in x;

export function metricFor(r: ModelRun, split: Split, target: string, m: RunMetric): number | undefined {
  const x = r.metrics?.[split]?.[target];
  if (!x) return undefined;
  const v = isNumber(x) ? (m === "primary" ? x.r2 : x.rmse) : m === "primary" ? x.roc_auc ?? x.roc_auc_ovr_macro : x.f1_macro;
  return typeof v === "number" ? v : undefined;
}

/** RMSE is the one metric here where lower is better. */
const lowerIsBetter = (r: ModelRun, t: string, m: RunMetric) => m === "secondary" && isNumber(r.metrics?.valid?.[t] ?? {});
```

`bestRuns` loops `["primary", "secondary"]` and reduces with `lowerIsBetter(...) ? b.v < a.v : b.v > a.v`.

`ExperimentWorkspace.tsx`: `const kinds = e.summary?.target_kinds ?? {}`; headers `${t} ${kinds[t] === "regression" ? "R²" : "AUC"}` and `${t} ${kinds[t] === "regression" ? "RMSE" : "F1 (macro)"}`; cells call `cell(r, t, "primary" | "secondary")`. The "Use from code" line prints `t: number` for regression and `t: ${classes}` otherwise.

`charts.ts scatterOption(t, actual, predicted)`: value axes on a shared min/max, `scatter` series (symbolSize 4, 0.55 opacity), a dashed `line` series for the diagonal from min to max (silent), tooltip "actual a · predicted p" (escaped), axis names "Actual" / "Predicted".

`RunDetail.tsx`: `regressionTiles(m)` returns `[["R²", m.r2, "Share of the variance explained; 1 is perfect"], ["RMSE", m.rmse, "Root mean squared error, in the target's units"], ["MAE", m.mae, "Mean absolute error, in the target's units"], ["MAPE", m.mape, m.mape === undefined ? "Not defined: some actual values are 0" : "Mean absolute percentage error"]]`. `RegressionCard({target, m, ev})` shows the tiles, `scatterOption` ("Predicted vs actual · n rows, showing up to 1,000"), and the residual histogram through `histogramOption` with `{kind: "histogram", edges, counts}` and `xLabel: "Error (predicted − actual)"`. `RunDetail` picks `RegressionCard` when `ev?.kind === "regression"` or `"r2" in m`; the subtitle line reads "Class targets use the most likely class (0.5 threshold for binary targets)" when at least one card is a class card.

`experiments.ts metricLabel`: `metric === "r2" ? "R²" : metric.startsWith("roc_auc") ? "AUC" : metric`; `DatasetsPage.tsx` line with `roc_auc` uses `metricLabel`.

- [ ] **Step 4: Run** `npm run typecheck && npm test` → PASS.

- [ ] **Step 5: Commit** `"Regression run results: R², errors, predicted vs actual"`.

### Task 10: Samples — a numeric generated target and the Concrete dataset

**Files:**
- Modify: `api/app/sample_data.py` (`generate(..., numeric_target=False)`), `api/app/schemas.py` (`SampleRequest.numeric_target`), `api/app/routers/datasets.py` (pass it; the dataset's targets add `column_c`)
- Modify: `api/app/sample_library/__init__.py`, `api/app/sample_library/build.py`
- Create: `api/app/sample_library/data/concrete_strength.parquet`
- Modify: `ui/src/pages/DatasetsPage.tsx` (Generate tab checkbox), `ui/src/api.ts`/`types.ts` (`numeric_target`)
- Test: `api/tests/test_sample_limits.py` or `test_pipeline.py` (generator), `api/tests/test_sample_library.py`

**Interfaces:**
- Produces: `generate(n_rows, n_fields, seed, target_kind, numeric_target=False)` adds float `column_c` last, from the same latent factors, without changing any other column for the same seed; `LIBRARY["concrete_strength"]` with target `strength_mpa`.

- [ ] **Step 1: Failing tests**

```python
def test_numeric_target_adds_column_c_and_changes_nothing_else():
    from app.sample_data import generate
    a = generate(2000, 40, seed=5)
    b = generate(2000, 40, seed=5, numeric_target=True)
    assert b.columns[-1] == "column_c" and b.drop("column_c").equals(a)
    assert b["column_c"].dtype == pl.Float64 and b["column_c"].n_unique() > 1000
    assert abs(np.corrcoef(b["column_c"].to_numpy(), b["credit_score"].to_numpy())[0, 1]) > 0.2


def test_concrete_entry_is_a_regression_dataset():
    from app.sample_library import LIBRARY, path_of
    e = LIBRARY["concrete_strength"]
    df = pl.read_parquet(path_of("concrete_strength"))
    assert df.height == 1030 and e["targets"] == ["strength_mpa"] and df["strength_mpa"].dtype == pl.Float64
    assert "10.24432/C5PK67" in e["citation"]
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`generate`: after the existing frame is built, when `numeric_target`:

```python
    if numeric_target:
        # Its own generator, so the rest of the frame is the same as without it for the same seed.
        r2 = np.random.default_rng(seed + 1)
        c = 50 + 12 * Z[:, 0] + 8 * Z[:, 2] - 5 * Z[:, 5] + 6 * (a > 0) + 4 * r2.standard_normal(n)
        df = df.with_columns(pl.Series("column_c", np.round(c, 2)))
```

(Restructure the end of `generate` so the frame is a variable `df` before the return; `a` is the column_a code array already computed.) `SampleRequest.numeric_target: bool = False`; the cap check counts `req.fields + int(req.numeric_target)`; the generate endpoint passes it and appends `"column_c"` to the dataset's targets; the default name gets `-num` when set.

Library entry:

```python
    "concrete_strength": {
        "name": "Concrete strength",
        "description": "Concrete mixtures by their ingredients and age. Predict the compressive strength in MPa as a "
                       "number, or cut it into bands. Small and clean: 1,030 rows, eight numeric features.",
        "targets": ["strength_mpa"],
        "source_url": "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        "citation": "Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. "
                    "https://doi.org/10.24432/C5PK67",
    },
```

Update the module docstring ("All five come from…"). `build.py`: `ARCHIVES["concrete_strength"] = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"`; the branch reads `Concrete_Data.xls` with `pl.read_excel(io.BytesIO(z.read("Concrete_Data.xls")))` (needs `fastexcel`, a build-time dependency only: note it in the module docstring) and renames the nine columns to `cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age_days, strength_mpa` by position, casting all to Float64 except `age_days` (Int64). Build the parquet with the scratch venv (`pip install fastexcel` there), and commit it.

Generate tab in `DatasetsPage.tsx`: a checkbox "Add a numeric target (column_c)" bound to `numeric_target`.

- [ ] **Step 4: Run** the two test files and `npm test` → PASS.

- [ ] **Step 5: Commit** `"Samples: a numeric generated target and UCI concrete strength"`.

### Task 11: Docs, dev-01 verification, PR C, deploy

**Files:**
- Modify: `README.md` (targets section: Predict as, bands, regression metrics; Sample library table row; "On your own" walkthrough mentions Predict as), `docs/api.md` (decision fields, `/fields/{id}/bands`, manifest keys, run metrics, `numeric_target`), `docs/architecture.md` (target tasks + trigger, TargetChain, export bands)
- Modify: memory `rainforest-project.md` (one line)

- [ ] **Step 1:** Grep the docs for statements this work makes false: `grep -rn "Classification only\|classes first\|ClassifierChain\|at most 100\|class labels" README.md docs/*.md` and rewrite each to the current behaviour (current state only).

- [ ] **Step 2: Full verification**: API suite to the workspace log (all green), `cd ui && npm run typecheck && npm test && npm run build`.

- [ ] **Step 3: PR C**: push, open, read CI, merge, deploy (`DOCKER_CONTEXT=dev-01 make up`), preload Concrete via `POST /api/datasets/library/concrete_strength`.

- [ ] **Step 4: dev-01 checks (browser + API, as lead1):**
  1. Student grades: a new Trees baseline with `G3` as Number → finalize → experiment → export → LightGBM run: R² tile, scatter, residuals; ★ on the runs table.
  2. Same dataset, experiment adjusting `G3` to 4 Bands → export manifest shows 3 edges; run shows a 4-class confusion matrix; the adjustments card lists the task change.
  3. Chain `G1` (classes) → `G3` (number) in one run.
  4. Concrete strength as a number with XGBoost on the GPU.
  5. Two assessors disagreeing only on the task → the field appears in review.
  6. MLflow shows `target.G3.task` and `valid.G3.r2`.

- [ ] **Step 5: Commit docs** `"Docs: predicting numbers and bands"` (part of PR C before merge).

---

## Self-review notes

- Spec coverage: §1 → Tasks 1–3, 8; §2 → Task 4; §3 → Tasks 5–7; §4 → Tasks 8–10 (Explore in Task 4, MLflow in Task 7); build order → PR A/B/C; testing list → tests in each task plus Task 11 step 4.
- Types: `target_task`/`target_bins` (API, DB, snapshot, UI); manifest `target_tasks`/`target_bins`/`target_stats`; `TargetChain.predict_all`; `RunMetric = "primary" | "secondary"`, used consistently.
