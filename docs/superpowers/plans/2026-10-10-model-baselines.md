# Model Baselines and Experiment Adjustments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A team prepares a dataset as one baseline per model type (Trees, Linear, Neural), each assessed blind and finalized by a lead, and runs experiments that start from a finalized baseline and may adjust any field's role, type and prep settings, with every adjustment traceable in exports and MLflow.

**Architecture:** A baseline is a recipe row with a fixed `family`. The working tables (`field_decisions`, `field_settings`) keep serving the single baseline in progress; starting any baseline or version loads them from a snapshot first, so they never carry another baseline's choices. Experiment adjustments are a per-experiment copy of the baseline (`experiment_decisions` / `experiment_settings`), read and written through the existing `assessment.Source` abstraction, so Review and Bulk edit work on an experiment by passing `view=experiment:<id>`. A read-only `view=baseline:<recipe_id>` exposes any finalized snapshot the same way (Explore and browsing old baselines).

**Tech Stack:** FastAPI, psycopg 3, Postgres 16 (`meta` schema, numbered migrations + `views.sql`), polars, scikit-learn / LightGBM, MLflow 3.17; React 18 + react-query + TypeScript, vitest; Playwright (Python) for end-to-end checks on dev-01.

**Spec:** `docs/superpowers/specs/2026-10-10-model-baselines-design.md`

## Global Constraints

- Families: `tree` (label "Trees"), `linear` ("Linear"), `neural` ("Neural"); a baseline's family never changes once anyone is assigned to assess it.
- One baseline in progress (phase `assessing` or `review`) per dataset at a time (existing unique index `recipes_one_open_idx`).
- Labels: "Trees baseline v2" (`recipes.label`); team experiment default name "Team · Trees v2".
- Version numbers count per (dataset, family).
- Blind assessment is unchanged: no endpoint returns an assessor's choices to anyone else before review, including through the new `view` parameter.
- Only leads start baselines, start versions, assign assessors and finalize.
- Experiment adjustments: only the experiment's owner; never on default team experiments, personal (assessment) experiments or archived ones.
- Adjustable per field: action (`keep | drop | target`), `semantic_type`, every setting in `semantics.SETTING_KEYS`. Changing the type resets that field's settings to the suggestion for the new type.
- An experiment whose targets differ from its baseline's is marked `targets_changed` everywhere it is shown, and tagged `rainforest.targets_changed=true` in MLflow.
- Planned models (linear SVM, k-NN, CNN, RNN) are listed with `planned: true` and refused for training (422).
- LightGBM on the Trees baseline is the path that must work end to end and is exercised on dev-01.
- Docs describe the current state only (no history wording, no PR numbers). Comments sparse; no "honestly".
- CI must be green (API pytest with Postgres + MLflow services, UI typecheck/vitest/build, docker build) before each merge.

## Design decisions made while planning (fold into the spec in Task C4)

1. `meta.datasets.model_family` stays, maintained by the server as the family of the dataset's current baseline; clients can no longer set it (PATCH with `model_family` → 422). Every reader keeps working.
2. A new dataset still gets recipe v1 (`tree`, `assessing`, nobody assigned) on upload. That draft is **set up** in place by the Start dialog (family, start-from, targets, assessors). Family and targets change only while nobody is assigned and nothing is assessed.
3. Experiment adjustments are stored as a full copy of the baseline per experiment (seeded when the experiment is created), mirroring `assessments`; "adjusted fields" = `recipes.diff(baseline snapshot, experiment snapshot, family)`.
4. One `view` query parameter selects the decision source on field reads/writes, Bulk edit, the field finder and Explore: omitted (current baseline, as today), `experiment:<id>`, `baseline:<recipe_id>` (finalized, read-only).
5. Leakage for a snapshot's targets is computed at preview time from `meta.field_correlations` (`leak_names`), so a finalized baseline or an experiment with other targets gets the right warnings.

## Review Focus

1. A Trees v2 started after a Linear baseline was worked must start from Trees v1's decisions, not Linear's (working tables loaded from the parent snapshot). → Task A2 test `test_new_version_loads_its_own_parent_not_the_last_worked_baseline`.
2. `view=experiment:<id>` / `view=baseline:<id>` must not become a side door to an assessor's private choices or to another person's private experiment. → Task B2 adds both views to the no-leak matrix (`READS` in `test_team.py`).
3. Adjusting an experiment never changes its baseline, another experiment, or an export already built. → Task B3 test `test_adjustments_stay_in_their_experiment_and_exports_freeze_them`.
4. Retyping a field in an experiment (e.g. numeric → categorical) leaves settings that export without a "settings aren't valid" block. → Task B2 test `test_retype_in_experiment_resets_settings_to_valid_ones`.
5. Removing every target in an experiment, or making a free-text field a target, is refused or blocked with a clear message rather than failing in the export job. → Task B3 test `test_experiment_targets_are_validated`.

---

## File map

Backend (`api/app/`):
- `migrations/012_baselines.sql` (A1): `recipes.family`, per-family version uniqueness.
- `migrations/013_experiment_adjustments.sql` (B1): `experiment_decisions`, `experiment_settings`, `experiment_batches`.
- `recipes.py` (A2): family-aware creation, labels, `load_working`, family-scoped `snapshot`, `start_baseline`.
- `routers/baselines.py` (new, A3): `POST /datasets/{id}/baselines`, `GET /datasets/{id}/baselines`.
- `routers/recipes.py` (A3): versions per family, diff by recipe id, `new_version(family)`.
- `routers/datasets.py` (A4): model_family read-only, dataset rows carry `baseline` + `baselines`.
- `routers/exports.py`, `experiments.py` (A5, B3): family from the baseline, adjusted snapshots, leak warnings.
- `leakage.py` (new, A5): `leak_names(conn, dataset_id, targets)`.
- `reuse.py` (A6): family carried with a reused recipe.
- `modeling/registry.py` (A7): planned models.
- `assessment.py` (B2, C1): `Source` kinds `team | assessment | experiment | baseline`; `source_for(..., view=)`.
- `routers/fields.py`, `routers/query.py`, `routers/explore.py` (B2, C1): `view` parameter.
- `routers/experiments.py` (B3): adjustments summary, revert.
- `modeling/mlflow_log.py` (B4): baseline and adjustment tags/params/artifact.
- `views.sql` (C2): baseline family/version on metrics.

Frontend (`ui/src/`):
- `App.tsx` (A8): ModelSwitch → baseline chip.
- `pages/BaselinesPage.tsx` (A8, replaces `VersionsPage.tsx`), `components/StartBaseline.tsx` (A8).
- `pages/DatasetTeamPage.tsx` (A8): set-up prompt instead of "nobody assigned".
- `view.ts` (new, B5): `useView()`; `api.ts` view-aware calls.
- `pages/ReviewPage.tsx`, `pages/BulkEditPage.tsx`, `components/DecisionPanel.tsx` (B5): experiment mode.
- `pages/ExperimentWorkspace.tsx`, `pages/ExperimentsPage.tsx` (A8, B5): baseline labels, adjustments card, targets-changed badge.
- `pages/ExplorePage.tsx` (C1): decision source picker.
- `pages/DatasetsPage.tsx` (C2): baseline chips.

Docs: `README.md`, `docs/api.md`, `docs/architecture.md`, spec (C4).

---

## Review revisions (fresh-eyes review of this plan; each is binding on the task named)

- **A2 `load_working`**: keep the chosen settings whenever they validate for the field's *decided* type and the
  family (`validate_decision({**chosen, "action": "keep", "semantic_type": dec_type}, family, n_unique)`), else
  `suggest(dec_type, …)`. Import `suggest`, `validate_decision`. Signature
  `load_working(conn, dataset_id, snap, family, targets, user_id, reviewed=False) -> tuple[list[dict], list[int]]`.
  Extra test: right after `start_version`, `recipes.diff(parent_snap, recipes.snapshot(conn, ds, family=f), f)` has no
  fields, with a retyped and a customised field in the parent.
- **A2 ordering audit**: `latest_finalized` orders by `finalized_at DESC, id DESC`; `routers/recipes.py` lists order
  by `family, version DESC`; every lookup by `(dataset_id, version)` becomes by recipe id (`_version`, `processing.py`
  `version = 1` → the dataset's first recipe by id); `_pick_recipe` default = latest finalized of the dataset's current
  family, else any family.
- **A2 personal snapshots**: `assessment.personal_snapshot` takes carried (locked) fields from the version's parent
  snapshot, not the live working tables.
- **A3**: `POST /baselines` also accepts `start: "recipe"` with `source_recipe_id` of another dataset's finalized
  baseline (setup only): runs `reuse.apply`, sets the family from the source, assigns assessors, all in one
  transaction; requires the dataset `ready` (409). If reuse leaves nothing to assess (phase review), assessors are not
  assigned and the response says so. `GET /baselines` gets its own test. Diff by recipe ids: both must belong to the
  dataset (404); different families compare decisions only (no settings).
- **A4**: `SELECT DISTINCT ON (dataset_id, family) dataset_id, family, id, version, phase … ORDER BY dataset_id,
  family, id DESC`; `baselines[]` items carry `id` and `latest_finalized_id`.
- **A5 test**: replace the leak test with a target-set-specific one in B3: baseline target `y`; an experiment switching
  the target to `c` where a kept field leaks `c` warns, the baseline preview doesn't (wait for the association job).
- **A6**: `reuse._settings_problem` and the settings copy use only the source baseline's family; set `recipes.family`
  and `datasets.model_family` before `_active_family` and `set_targets`. Test: reusing a single-family snapshot carries
  fields.
- **A7**: `models_for_export` (`runs.py`) skips nothing but marks planned entries (`planned: true, compatible: false`)
  without indexing `MODELS`; `start_run` checks `PLANNED` before `MODELS.get`. Test GET `/exports/{id}/models`.
- **A8 labels**: replace every "Recipe v…" string in the UI (`experiments.ts`, `team.ts`, `ReuseCard.tsx`, Versions
  toasts, Team page) with the baseline label from the API (`label` field on recipe objects; add it to `RECIPE_COLS`
  consumers via `recipes.with_label(row)`). MLflow keeps `rainforest.recipe_version` and adds
  `rainforest.baseline` / `rainforest.baseline_family` (B4).
- **A8/A9 flow**: the start dialog's reuse option uses `start: "recipe"` (one call). A9 Playwright uses lead1 + ds1 +
  ds2 with lead1 assessing.
- **B2 family**: for `experiment`/`baseline` views the family is always `src.family` (a different `family` → 422);
  resolve the source before validating the body. For the team/assessment source, a `family` different from the
  current baseline's → 422 (one model type per baseline). The UI stops sending `family` (server default).
- **B2 `Source.kind`**: explicit `kind` field; `team` ≡ `kind == "team"`; every branch in `assessment.py`,
  `fields.py`, `datasets.list_fields`, `query.py` switches on `kind` (baseline → snapshot relation, read-only).
- **B2 view-correct badges**: in non-team views `list_fields`/`get_field` derive `is_target` from the view's action
  and `leakage` from `leakage.leak_names(view targets)`; `target_associations` uses the view's targets.
- **B2 can_adjust**: owner only (`x["owner"] == user["id"]`), team experiments only, not archived. Default team
  experiments (`is_default`) are not adjustable; the Experiments page offers "New experiment from this baseline".
- **B2 no-leak matrix**: new fixture state in `test_team.py`: a finalized Trees v1, a team experiment owned by A, then
  a Linear v1 being assessed where B makes secret choices. Assert 200 and no secret for fields / field / finder / bulk
  dry-run / explore under `view=experiment:<A's xp>` and `view=baseline:<trees v1>` for viewers lead, A, C; the
  `baseline` block equals the snapshot. `REFUSED` accepts a set of codes (`assert r.status_code in codes`).
- **B2/B3 changed targets**: making a field an experiment target queues `fields.target_associations_job(dataset_id,
  [field_id])` — correlate if its type has no association sample, else screen — which upserts correlations without
  `_sync_targets`; preview warns "Leak check for {names} is still running" while a target has no association rows.
- **B3 freeze**: `start_export` computes the snapshot once (locking the experiment row `FOR UPDATE`), validates it
  with the preview logic, and stores exactly it; experiment writes lock the experiment row `FOR SHARE`
  (`source_for(lock="share")` does both). A default team export (no experiment) freezes the recipe snapshot.
- **B3 tests**: the retype test moves here (it needs the adjusted snapshot), plus the target-set leak test (above).
  Shared helpers (`_finalized_trees`, `_adjust(client, ds, x, name, expect=200, **patch) -> Response`, `_export`,
  `KEYS`, `_start`, `_submit_and_finalize`) live in `api/tests/team_helpers.py`.
- **B4**: the run job writes `adjustments.json` into `model_dir` from the export config; `log_model_run`'s SELECT reads
  the export config so live, deferred and backfilled runs get the same tags.
- **C1 keys**: `key_owner` returns a user only for `assessment:` keys; `get_analysis` re-checks `get_visible` for
  `experiment:` keys; `from_key` handles all four kinds (tests for each).
- **C2**: `v_fields` and `v_field_profiles` gain `baseline_family`, `baseline_version` (the current baseline);
  `v_model_metrics` gains them per export. Startup `backfill_copies` skips `Imported` experiments whose recipe isn't
  finalized.

## Part A — Baselines (PR "Baselines: one per model type")

### Task A1: Migration — baseline family

**Files:**
- Create: `api/app/migrations/012_baselines.sql`
- Test: `api/tests/test_baselines.py` (new)

**Interfaces:**
- Produces: `meta.recipes.family TEXT NOT NULL CHECK (family IN ('tree','linear','neural'))`; unique `(dataset_id, family, version)`.

- [ ] **Step 1: Write the failing test** (module fixtures copied from `test_team.py`: `app_`, `people`, `_upload`, `_wait`, `_fields`, `_save`, `_review_all`)

```python
def test_recipes_have_a_family_and_versions_count_per_family(app_):
    from app.db import connection
    with connection() as conn:
        cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='meta' AND table_name='recipes'")}
        assert "family" in cols
        idx = conn.execute("SELECT indexdef FROM pg_indexes WHERE schemaname='meta' AND tablename='recipes'").fetchall()
        assert any("(dataset_id, family, version)" in r["indexdef"] for r in idx)
```

- [ ] **Step 2: Run** `cd api && RAINFOREST_TEST_DATABASE_URL=… pytest tests/test_baselines.py -q` → FAIL (no column).

- [ ] **Step 3: Write the migration**

```sql
-- Migration 012: a recipe version is a baseline for one model family. Versions count per family.
-- The default keeps 008's DO block (which inserts without a family) working when migrations re-run.
ALTER TABLE meta.recipes ADD COLUMN IF NOT EXISTS family TEXT NOT NULL DEFAULT 'tree';
ALTER TABLE meta.recipes DROP CONSTRAINT IF EXISTS recipes_family_check;
ALTER TABLE meta.recipes ADD CONSTRAINT recipes_family_check CHECK (family IN ('tree', 'linear', 'neural'));
-- Finalized rows are frozen by a trigger; their family is set once here, from the dataset's model family.
ALTER TABLE meta.recipes DISABLE TRIGGER recipes_frozen;
UPDATE meta.recipes r SET family = d.model_family FROM meta.datasets d
 WHERE d.id = r.dataset_id AND r.family <> d.model_family;
ALTER TABLE meta.recipes ENABLE TRIGGER recipes_frozen;
ALTER TABLE meta.recipes DROP CONSTRAINT IF EXISTS recipes_dataset_id_version_key;
CREATE UNIQUE INDEX IF NOT EXISTS recipes_family_version_idx ON meta.recipes (dataset_id, family, version);
-- Default team experiments are found by this flag, not by name, so they can be renamed.
ALTER TABLE meta.experiments ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT false;
UPDATE meta.experiments SET is_default = true
 WHERE owner IS NULL AND assessor_id IS NULL AND name LIKE 'Team · v%' AND NOT is_default;
UPDATE meta.experiments x
   SET name = 'Team · ' || CASE r.family WHEN 'tree' THEN 'Trees' WHEN 'linear' THEN 'Linear' ELSE 'Neural' END
              || ' v' || r.version
  FROM meta.recipes r WHERE r.id = x.recipe_id AND x.is_default AND x.name LIKE 'Team · v%';
```

(Check the trigger name in `009_recipe_versions.sql` — `CREATE TRIGGER … EXECUTE FUNCTION meta.recipes_frozen()` — and use it. Test: a finalized recipe with a snapshot gets its family; re-running migrations 008+ (as `test_existing_datasets_migrate_…` does) succeeds.)

- [ ] **Step 4: Run the test** → PASS. Run `tests/test_team.py tests/test_recipes.py` → still PASS (`create_first` keeps working through the default; `start_version` copies the parent's family explicitly).

- [ ] **Step 5: Commit** `git commit -m "Baselines: recipes carry a model family, versions count per family"`

### Task A2: recipes.py — family-aware baselines and the working copy

**Files:**
- Modify: `api/app/recipes.py`
- Test: `api/tests/test_baselines.py`

**Interfaces:**
- Produces:
  - `RECIPE_COLS` includes `family`.
  - `FAMILY_LABEL = {"tree": "Trees", "linear": "Linear", "neural": "Neural"}`
  - `label(recipe) -> str` → `"Trees baseline v2"`.
  - `create_first(conn, dataset_id, user_id, family="tree") -> dict`
  - `current(conn, dataset_id, lock=None) -> dict` — latest by `id` (not version).
  - `latest_finalized(conn, dataset_id, family: str | None = None) -> dict | None`
  - `next_version(conn, dataset_id, family) -> int`
  - `snapshot(conn, dataset_id, src=None, family=None, targets=None) -> dict` — settings for `family` only when given; `targets` overrides `target_order`.
  - `finalize(conn, recipe, user_id)` — snapshot with `family=recipe["family"]`.
  - `load_working(conn, dataset_id, snap: dict | None, family: str, targets: list[str], user_id: int) -> None` — overwrite `field_decisions` and `field_settings[family]` from `snap` (None = suggestions), every field unreviewed with source `suggested`/`user` per value, then `fields.set_targets(conn, dataset_id, targets, user_id)`, and `UPDATE datasets SET model_family = family`. Returns `(uncorrelated, added)` from `set_targets`.
  - `start_version(conn, dataset_id, user_id, family, phase="assessing", scope="all") -> dict` — parent = `latest_finalized(dataset, family)` (409 if none or if a baseline is in progress); inserts with `next_version`; calls `load_working(parent snapshot)`.
  - `is_setup(conn, recipe) -> bool` — v1, `assessing`, no assessors, no assessment rows, `reuse IS NULL`.

- [ ] **Step 1: Write the failing tests**

```python
def test_labels_and_per_family_versions(people):
    lead = people["lead"]
    ds = _upload(lead)
    from app import recipes
    from app.db import connection
    with connection() as conn:
        r = recipes.current(conn, ds["id"])
        assert r["family"] == "tree" and recipes.label(r) == "Trees baseline v1"
        assert recipes.next_version(conn, ds["id"], "linear") == 1
        assert recipes.next_version(conn, ds["id"], "tree") == 2


def test_snapshot_holds_only_the_baselines_family(people):
    lead = people["lead"]
    ds = _upload(lead)
    set_phase(ds["id"], "finalized")
    from app import recipes
    from app.db import connection
    with connection() as conn:
        snap = recipes.snapshot_of(conn, recipes.current(conn, ds["id"]))
    assert all(set(f["settings"]) <= {"tree"} for f in snap["fields"].values())


def test_new_version_loads_its_own_parent_not_the_last_worked_baseline(people):
    lead = people["lead"]
    ds = _upload(lead)
    f = _fields(lead, ds)
    # Trees v1 drops x (set up with the lead assessing, review opens on submit, finalize).
    _start(lead, ds, "tree", assessors=[lead])
    assert _save(lead, f["x"]["id"], action="drop", reviewed=True).status_code == 200
    _review_all(lead, ds); _submit_and_finalize(lead, ds)
    # Linear v1 from scratch keeps x.
    _start(lead, ds, "linear", assessors=[lead])
    assert _fields(lead, ds)["x"]["action"] == "keep"
    _review_all(lead, ds); _submit_and_finalize(lead, ds)
    # Trees v2 with nothing reopened starts from Trees v1: x is dropped again.
    r = lead.post(f"/api/datasets/{ds['id']}/recipes", json={"family": "tree", "reopen": []})
    assert r.status_code == 200, r.text
    assert r.json()["recipe"]["family"] == "tree" and r.json()["recipe"]["version"] == 2
    assert _fields(lead, ds)["x"]["action"] == "drop"
```

(`_start(lead, ds, family, assessors, start="scratch", source=None, targets=["y"])` posts to `/api/datasets/{id}/baselines` from Task A3; `_submit_and_finalize` posts `assessment/submit` then `finalize` with `accept_open: true`. Write these helpers at the top of `test_baselines.py`.)

- [ ] **Step 2: Run** → FAIL (`label`, `next_version`, endpoints missing).

- [ ] **Step 3: Implement** in `recipes.py`:

```python
FAMILY_LABEL = {"tree": "Trees", "linear": "Linear", "neural": "Neural"}
RECIPE_COLS = """id, dataset_id, family, version, phase, scope, parent_id, created_by, created_at, finalized_by,
                 finalized_at, schema_fingerprint"""
CURRENT_JOIN = """LEFT JOIN LATERAL (SELECT id, family, version, phase, scope FROM meta.recipes
                                       WHERE dataset_id = d.id ORDER BY id DESC LIMIT 1) rc ON true"""


def label(recipe: dict) -> str:
    return f"{FAMILY_LABEL[recipe['family']]} baseline v{recipe['version']}"


def next_version(conn, dataset_id: int, family: str) -> int:
    return conn.execute("SELECT coalesce(max(version), 0) + 1 AS v FROM meta.recipes WHERE dataset_id = %s AND family = %s",
                        (dataset_id, family)).fetchone()["v"]
```

`current` orders by `id DESC`. `snapshot(..., family=None, targets=None)` filters the settings query with `AND (%s::text IS NULL OR s.family = %s)` and uses `targets` for `target_order` when given. `finalize` passes `family=recipe["family"]`. `load_working`:

```python
def load_working(conn, dataset_id: int, snap: dict | None, family: str, targets: list[str], user_id: int):
    """Make the working tables the starting point of a baseline: `snap`'s decisions and `family` settings (settings
    that aren't valid for `family` take its suggestion), else the suggestions. Every field starts unreviewed."""
    from .routers.fields import STAT_KEYS, set_targets
    from .processing import upsert_settings
    fields = conn.execute("SELECT * FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal", (dataset_id,)).fetchall()
    for f in fields:
        sugg = (f["suggestion"] or {}).get(family) or suggest(f["inferred_type"], {k: f[k] for k in STAT_KEYS},
                                                              False, family)
        entry = (snap or {}).get("fields", {}).get(f["name"])
        dec = entry["decision"] if entry else {"action": sugg["action"], "semantic_type": sugg["semantic_type"],
                                                "notes": None}
        action = "keep" if dec["action"] == "target" else dec["action"]   # targets come from set_targets below
        conn.execute(
            """UPDATE meta.field_decisions SET action = %s, semantic_type = %s, notes = %s, reviewed = false,
                      source = %s, updated_at = now() WHERE field_id = %s""",
            (action, dec["semantic_type"], dec.get("notes"), "user" if entry else "suggested", f["id"]))
        chosen = (entry or {}).get("settings", {}).get(family)
        if chosen is None and entry:
            chosen = next(iter(entry["settings"].values()), None)          # populate from another family
        if chosen and dec["semantic_type"] == sugg["semantic_type"] and not validate_decision(
                {**chosen, "action": "keep", "semantic_type": dec["semantic_type"]}, family, n_unique=f["n_unique"]):
            upsert_settings(conn, dataset_id, f["id"], family, chosen, "user")
        else:
            upsert_settings(conn, dataset_id, f["id"], family,
                            sugg if sugg["semantic_type"] == dec["semantic_type"]
                            else suggest(dec["semantic_type"], {k: f[k] for k in STAT_KEYS}, False, family),
                            "suggested")
    conn.execute("UPDATE meta.datasets SET model_family = %s, updated_at = now() WHERE id = %s", (family, dataset_id))
    return set_targets(conn, dataset_id, targets, user_id)
```

`start_version(conn, dataset_id, user_id, family, phase="assessing", scope="all")`: lock `current`; 409 `"{label(cur)} is still {phase}; finalize it before starting another baseline."` when `cur["phase"] != "finalized"`; `parent = latest_finalized(conn, dataset_id, family)` (409 `"There's no finalized {Family} baseline to start a new version from."` when None); insert `(dataset_id, family, next_version, phase, scope, parent_id, created_by)`; `load_working(conn, dataset_id, snapshot_of(conn, parent), family, snapshot_targets(parent snap), user_id)` but keep `reviewed = true` for every field (carried over) — pass `reviewed=True` via a keyword on `load_working` (`reviewed: bool = False`). Return the new row plus the `(uncorrelated, added)` pair as `recipe["_target_work"]`.

`is_setup(conn, recipe)`:

```python
def is_setup(conn, recipe: dict) -> bool:
    """A dataset's first baseline before anyone works on it: family, start and targets can still change."""
    if recipe["version"] != 1 or recipe["phase"] != "assessing" or recipe["parent_id"] is not None:
        return False
    row = conn.execute(
        """SELECT reuse IS NULL AND NOT EXISTS (SELECT 1 FROM meta.dataset_assessors WHERE recipe_id = %(r)s)
                  AND NOT EXISTS (SELECT 1 FROM meta.assessments WHERE recipe_id = %(r)s) AS ok
             FROM meta.recipes WHERE id = %(r)s""", {"r": recipe["id"]}).fetchone()
    return bool(row["ok"])
```

Also update `tests/conftest.py::set_phase` to call `start_version(conn, dataset_id, None, r["family"], phase)`.

- [ ] **Step 4: Run** the three tests (the third waits for Task A3) and `tests/test_team.py tests/test_recipes.py tests/test_experiments.py` → fix label assertions that expected "Recipe v1" (update to "Trees baseline v1").

- [ ] **Step 5: Commit** `"Baselines: labels, per-family versions, snapshots and the working copy load from a snapshot"`

### Task A3: Starting baselines and versions (API)

**Files:**
- Create: `api/app/routers/baselines.py`; register in `api/app/main.py`
- Modify: `api/app/routers/recipes.py` (`versions` grouped data, `diff` by ids, `new_version` takes `family`), `api/app/schemas.py`
- Test: `api/tests/test_baselines.py`

**Interfaces:**
- Consumes: `recipes.is_setup`, `recipes.load_working`, `recipes.start_version`, `assessment.seed`, `routers.datasets.queue_target_work`, `routers.team._team_state`.
- Produces:
  - `POST /api/datasets/{id}/baselines` body `StartBaselineIn {family: str, start: "scratch" | "baseline", source_recipe_id: int | None, targets: list[str], assessors: list[int]}` → `TeamState` (200). Lead only.
  - `GET /api/datasets/{id}/baselines` → `[{family, label, versions: [RecipeVersion…], latest_finalized_id}]` for families that exist, Trees first.
  - `POST /api/datasets/{id}/recipes` body `NewVersionIn {family: str, reopen: list[int]}`.
  - `GET /api/datasets/{id}/recipes/diff?a=<recipe_id>&b=<recipe_id>` (ids, not versions; same family required, 422 otherwise).

Rules for `POST …/baselines`:
1. 422 unknown family; 422 `assessors` empty; 422 unknown/inactive user id; 422 any target failing `target_problem` (message lists each); 422 `targets` empty ("Choose at least one target").
2. `start == "baseline"`: `source_recipe_id` must be a finalized recipe of this dataset (404 otherwise).
3. If `current` is setup: update its `family`, `load_working(snap or None, family, targets)`, then assign + seed each assessor. Else: `current` must be finalized (409 as in A2), insert a new recipe `(family, next_version, 'assessing', 'all', parent = source or latest_finalized(family))`, `load_working`, assign + seed.
4. After commit run `queue_target_work(...)()` for the `(uncorrelated, added)` it returned.

- [ ] **Step 1: Write the failing tests**

```python
def test_start_sets_up_the_first_baseline_in_place(people):
    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    st = _start(lead, ds, "linear", assessors=[lead, a])
    assert st["recipe"]["family"] == "linear" and st["recipe"]["version"] == 1
    assert {x["user_id"] for x in st["assessors"]} == {lead.user["id"], a.user["id"]}
    assert lead.get(f"/api/datasets/{ds['id']}").json()["model_family"] == "linear"
    # Fixed once started.
    r = lead.post(f"/api/datasets/{ds['id']}/baselines", json={"family": "tree", "start": "scratch",
                  "targets": ["y"], "assessors": [lead.user["id"]]})
    assert r.status_code == 409


def test_start_validates_its_input(people):
    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    base = {"family": "tree", "start": "scratch", "targets": ["y"], "assessors": [lead.user["id"]]}
    for patch, code in (({"family": "svm"}, 422), ({"assessors": []}, 422), ({"targets": []}, 422),
                        ({"targets": ["z"]}, 422), ({"start": "baseline", "source_recipe_id": 999999}, 404)):
        assert lead.post(f"/api/datasets/{ds['id']}/baselines", json={**base, **patch}).status_code == code
    assert a.post(f"/api/datasets/{ds['id']}/baselines", json=base).status_code == 403


def test_populate_copies_decisions_and_only_valid_settings(people):
    lead = people["lead"]
    ds = _upload(lead)
    f = _fields(lead, ds)
    _start(lead, ds, "tree", assessors=[lead])
    assert _save(lead, f["w"]["id"], action="drop", reviewed=True).status_code == 200
    _review_all(lead, ds); st = _submit_and_finalize(lead, ds)
    trees = st["recipe"]["id"]
    _start(lead, ds, "linear", assessors=[lead], start="baseline", source=trees)
    now = _fields(lead, ds)
    assert now["w"]["action"] == "drop" and not now["w"]["reviewed"]           # copied, still to assess
    x = lead.get(f"/api/fields/{f['x']['id']}").json()["decision"]
    assert x["family"] == "linear" and x["imputation"] != "none"               # tree "leave missing" isn't valid for linear


def test_one_baseline_in_progress_at_a_time(people):
    lead = people["lead"]
    ds = _upload(lead)
    _start(lead, ds, "tree", assessors=[lead])
    r = lead.post(f"/api/datasets/{ds['id']}/baselines", json={"family": "linear", "start": "scratch",
                  "targets": ["y"], "assessors": [lead.user["id"]]})
    assert r.status_code == 409 and "finalize" in r.json()["detail"]
```

- [ ] **Step 2: Run** → FAIL (404 route).
- [ ] **Step 3: Implement** `routers/baselines.py` following the rules above (use `LeadUser`, `connection()`, `recipes.current(conn, id, lock="update")`; reuse `team._require_phase` messages style; seed with `assessment.seed`). Add `StartBaselineIn` to `schemas.py`:

```python
class StartBaselineIn(BaseModel):
    family: str
    start: Literal["scratch", "baseline"] = "scratch"
    source_recipe_id: int | None = None
    targets: list[str]
    assessors: list[int]
```

Update `routers/recipes.py::versions` to return `family` and order by `family, version DESC`; `diff` to take recipe ids and require the same family; `new_version` to call `recipes.start_version(conn, dataset_id, user["id"], body.family, …)`, then insert reopened fields and mark them unreviewed as today, and run the target work.

- [ ] **Step 4: Run** `pytest tests/test_baselines.py tests/test_recipes.py -q` → PASS (update `test_recipes.py` calls to send `family` and recipe ids for diff).
- [ ] **Step 5: Commit** `"Baselines: start one per model type (scratch or from another), versions per family"`

### Task A4: Datasets — family is the baseline's

**Files:**
- Modify: `api/app/routers/datasets.py`, `api/app/schemas.py` (keep `model_family` in `DatasetPatch` only to refuse it)
- Test: `api/tests/test_baselines.py`

**Interfaces:**
- Produces: dataset JSON gains `baseline: {id, family, version, phase, label}` (current) and `baselines: [{family, version, phase, label}]` (latest per family); `PATCH` with `model_family` → 422 "The model type comes from the baseline. Start a baseline for another type on the Baselines page."; target changes in setup still allowed via PATCH (`_check_targets_editable` unchanged).

- [ ] **Step 1: Test**

```python
def test_dataset_shows_its_baselines_and_refuses_family_switch(people):
    lead = people["lead"]
    ds = _upload(lead)
    d = lead.get(f"/api/datasets/{ds['id']}").json()
    assert d["baseline"]["label"] == "Trees baseline v1" and d["baselines"][0]["family"] == "tree"
    assert lead.patch(f"/api/datasets/{ds['id']}", json={"model_family": "linear"}).status_code == 422
```

- [ ] **Step 2–4:** implement (`SELECT_DATASET` adds `rc.family AS recipe_family`; build `baseline`/`baselines` in `get_dataset`/`list_datasets` with one query `SELECT DISTINCT ON (family) family, version, phase FROM meta.recipes WHERE dataset_id = ANY(%s) ORDER BY dataset_id, family, id DESC`), run, pass.
- [ ] **Step 5: Commit** `"Datasets: model type comes from the baseline"`

### Task A5: Exports and experiments follow the baseline; leakage per target set

**Files:**
- Create: `api/app/leakage.py`
- Modify: `api/app/routers/exports.py`, `api/app/experiments.py`
- Test: `api/tests/test_baselines.py`, existing `tests/test_experiments.py`

**Interfaces:**
- Produces: `leakage.leak_names(conn, dataset_id: int, targets: list[str]) -> set[str]` (non-target fields whose strongest association with any of `targets` ≥ `processing.LEAKAGE_THRESHOLD`); export family = `recipe["family"]` (a `family` query/body value that differs → 422); `experiments.COLS` adds `xr.family AS recipe_family`; `team_name(recipe)` → `"Team · Trees v1"`.

```python
def leak_names(conn, dataset_id: int, targets: list[str]) -> set[str]:
    rows = conn.execute(
        """SELECT DISTINCT o.name FROM meta.field_correlations c
             JOIN meta.fields t ON t.id IN (c.field_a, c.field_b) AND t.name = ANY(%(t)s)
             JOIN meta.fields o ON o.id IN (c.field_a, c.field_b) AND o.id <> t.id AND NOT o.name = ANY(%(t)s)
            WHERE c.dataset_id = %(ds)s AND c.strength >= %(th)s""",
        {"ds": dataset_id, "t": targets, "th": LEAKAGE_THRESHOLD}).fetchall()
    return {r["name"] for r in rows}
```

- [ ] **Step 1: Test**

```python
def test_exports_take_the_baselines_family_and_name_it(people):
    lead = people["lead"]
    ds = _upload(lead)
    _start(lead, ds, "linear", assessors=[lead]); _review_all(lead, ds); st = _submit_and_finalize(lead, ds)
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview").json()
    assert pv["family"] == "linear"
    assert lead.get(f"/api/datasets/{ds['id']}/export-preview?family=tree").status_code == 422
    e = lead.post(f"/api/datasets/{ds['id']}/exports", json={}).json()
    assert e["model_family"] == "linear" and e["experiment_name"] == "Team · Linear v1"


def test_preview_warns_about_leaks_for_the_snapshots_own_targets(people):
    lead = people["lead"]
    csv = "x,leak,y\n" + "".join(f"{i % 7},{'ab'[i % 2]}x,{'ab'[i % 2]}\n" for i in range(400))
    ds = _upload(lead, csv)
    _start(lead, ds, "tree", assessors=[lead], targets=["y"]); _review_all(lead, ds); _submit_and_finalize(lead, ds)
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview").json()
    assert any("leak" in w for w in pv["warnings"])
```

- [ ] **Step 2–4:** in `preview`, `fam = recipe["family"]` after `_plan` (validate a passed `family` equals it); replace `r["leakage"]` with `r["name"] in leak_names(conn, dataset_id, targets)`; `start_export` stores `pv["family"]`; `team_name` uses the label. Run `tests/test_baselines.py tests/test_experiments.py tests/test_team.py` → PASS.
- [ ] **Step 5: Commit** `"Exports follow their baseline's model type; leakage checked against the snapshot's targets"`

### Task A6: Reuse on new data carries the family

**Files:** Modify `api/app/reuse.py`, `api/app/routers/recipes.py::finalized_recipes`; Test `api/tests/test_recipes.py`

- [ ] **Step 1: Test** — extend the existing reuse test: the source is a finalized Linear baseline; after `POST /recipes/apply` the new dataset's current recipe has `family == "linear"` and `model_family == "linear"`; `GET /api/recipes` rows carry `family` and `label`.
- [ ] **Step 2–4:** `reuse.apply` sets `UPDATE meta.recipes SET family = source.family` and `datasets.model_family`; settings copied per family use `entry["settings"].get(source family)`; `_active_family` there becomes `source["family"]`. Run `tests/test_recipes.py` → PASS.
- [ ] **Step 5: Commit** `"Reuse on new data keeps the baseline's model type"`

### Task A7: Planned models

**Files:** Modify `api/app/modeling/registry.py`, `api/app/routers/runs.py`; Test `api/tests/test_validation.py`

**Interfaces:** `PLANNED = [{"key": "linear_svm", "label": "Linear SVM", "family": "linear"}, {"key": "knn", "label": "k-nearest neighbours", "family": "neural"}, {"key": "cnn", "label": "Convolutional network", "family": "neural"}, {"key": "rnn", "label": "Recurrent network", "family": "neural"}]`; `catalog()` appends them with `planned: True, available: False, compatible: False`; `POST /exports/{id}/runs` with a planned key → 422 `"<label> is planned and can't be trained yet."`.

- [ ] **Step 1: Test** `assert any(m["key"] == "linear_svm" and m["planned"] for m in catalog())`; API: training `knn` → 422 with "planned".
- [ ] **Step 2–4:** implement; `MODELS` unchanged so `train()` never sees planned keys. Run → PASS.
- [ ] **Step 5: Commit** `"Model list shows planned models (SVM, k-NN, CNN, RNN)"`

### Task A8: UI — baselines in the app

**Files:**
- Modify: `ui/src/App.tsx` (remove `ModelSwitch`; add `BaselineChip` linking to `/d/:id/baselines`), `ui/src/types.ts` (`Dataset.baseline`, `Dataset.baselines`, `RecipeVersion.family`, `Experiment.recipe_family`, `ModelOption.planned`), `ui/src/api.ts` (`startBaseline`, `baselines`, `newVersion(datasetId, family, reopen)`, `recipeDiff(datasetId, aId, bId)`), `ui/src/pages/DatasetTeamPage.tsx` (setup prompt), `ui/src/pages/ExperimentWorkspace.tsx` (family from the experiment; planned models disabled with "(planned)"), `ui/src/pages/ExperimentsPage.tsx` (baseline labels in the source picker), `ui/src/format.ts` (`familyShort` stays; add `baselineLabel(family, version)`).
- Create: `ui/src/pages/BaselinesPage.tsx` (from `VersionsPage.tsx`: one section per family with its versions, *New version* per family when nothing is in progress, *Start a baseline* button), `ui/src/components/StartBaseline.tsx` (dialog), route `/d/:id/baselines` (keep `/d/:id/versions` redirecting).
- Test: `ui/src/baselines.test.ts` (new) for pure helpers.

**Interfaces:**
- `api.startBaseline(datasetId, {family, start, source_recipe_id, targets, assessors}) → TeamState`
- `startBaselineDefaults(ds: Dataset, me: Me, users: TeamUser[], fields: FieldListItem[]) → {family, start, targets, assessors}` in `ui/src/baselines.ts` — first family without a baseline (Trees first), `start: "baseline"` with the latest finalized of another family when one exists else `"scratch"`, targets = current target fields, assessors = `[me.id]`.

*A recipe from another dataset* (setup only) calls the existing `api.applyRecipe` and then, if the version is still
assessing, `api.assign` for each ticked assessor; the dialog shows the reuse summary (carried / reopened / missing)
before closing.

Dialog content (one form): Model type (radio, families with an existing baseline say "new baseline replaces nothing; this will be Trees v2" — label `"{Family} · v{n}"`), Start from (Scratch / a finalized baseline of this dataset / for setup only: *A recipe from another dataset* which opens the existing reuse picker), Targets (checkbox list of eligible fields from `api.fields`, disabled with reason when `target_problem` would refuse — use the field's `semantic_type` in binary/categorical/numeric_discrete and `n_unique ≤ 100`), Assessors (active users, me ticked). Submit → `startBaseline` → invalidate team keys → navigate to Review.

- [ ] **Step 1: Test** `baselines.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { startBaselineDefaults } from "./baselines";

describe("startBaselineDefaults", () => {
  const me = { id: 7 } as any;
  it("offers the first type without a baseline and populates from the latest finalized one", () => {
    const ds = { baselines: [{ family: "tree", version: 2, phase: "finalized", id: 11 }] } as any;
    const d = startBaselineDefaults(ds, me, [], [{ name: "y", action: "target" }] as any);
    expect(d).toEqual({ family: "linear", start: "baseline", source_recipe_id: 11, targets: ["y"], assessors: [7] });
  });
  it("starts from scratch on a new dataset", () => {
    const ds = { baselines: [{ family: "tree", version: 1, phase: "assessing", id: 3 }] } as any;
    expect(startBaselineDefaults(ds, me, [], []).start).toBe("scratch");
  });
});
```

- [ ] **Step 2: Run** `cd ui && npx vitest run src/baselines.test.ts` → FAIL.
- [ ] **Step 3: Implement** the helper, dialog, page, chip, and wiring; delete `ModelSwitch` and its toast; `DecisionPanel`'s family badge shows `familyShort(ds.model_family)` (unchanged source of truth).
- [ ] **Step 4: Run** `npx tsc --noEmit -p . && npx vitest run && npm run build` → PASS.
- [ ] **Step 5: Commit** `"UI: baselines page and start dialog; the model type is the baseline's"`

### Task A9: Part A integration on dev-01, review, PR

- [ ] Full API suite against throwaway Postgres (tunnel to dev-01: `ssh -f -N -L 55432:127.0.0.1:55432 dev-01` after `docker run -d --rm --name rainforest-testpg -e POSTGRES_PASSWORD=test -p 127.0.0.1:55432:5432 postgres:16` on context dev-01), UI checks, `DOCKER_CONTEXT=dev-01 docker compose up -d --build api ui`.
- [ ] Playwright smoke (`scratchpad/baselines/a.py`): lead1 uploads a 5k sample, opens Start dialog (defaults: Trees, scratch, targets column_a+column_b, lead1 ticked), adds ds1, starts; both assess with *Accept suggestions*, submit; lead finalizes; Baselines page shows Trees v1 finalized; start Linear populated from Trees; screenshot each step; zero page errors.
- [ ] Fresh-eyes code review of the Part A diff by a reviewer agent (correctness, blind-data leaks, migration safety); fix findings.
- [ ] PR, CI green, merge.

---

## Part B — Experiment adjustments (PR "Experiments adjust their baseline")

### Task B1: Migration — experiment copies

**Files:** Create `api/app/migrations/013_experiment_adjustments.sql`; Test `api/tests/test_adjustments.py` (new; fixtures as in `test_baselines.py`)

```sql
-- Migration 013: each team experiment's own copy of its baseline, which its owner may adjust.
CREATE TABLE IF NOT EXISTS meta.experiment_decisions (
    experiment_id BIGINT NOT NULL REFERENCES meta.experiments(id) ON DELETE CASCADE,
    field_id      BIGINT NOT NULL REFERENCES meta.fields(id) ON DELETE CASCADE,
    action        TEXT NOT NULL, semantic_type TEXT NOT NULL, notes TEXT, reviewed BOOLEAN NOT NULL DEFAULT true,
    source        TEXT NOT NULL DEFAULT 'baseline', updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, field_id));
CREATE TABLE IF NOT EXISTS meta.experiment_settings (
    experiment_id BIGINT NOT NULL REFERENCES meta.experiments(id) ON DELETE CASCADE,
    field_id      BIGINT NOT NULL REFERENCES meta.fields(id) ON DELETE CASCADE,
    family TEXT NOT NULL, imputation TEXT NOT NULL, impute_value TEXT, add_missing_indicator BOOLEAN NOT NULL DEFAULT false,
    transform TEXT NOT NULL, scaling TEXT NOT NULL, encoding TEXT NOT NULL, params JSONB NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'baseline', updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, field_id, family));
CREATE TABLE IF NOT EXISTS meta.experiment_batches (
    id UUID PRIMARY KEY, experiment_id BIGINT NOT NULL REFERENCES meta.experiments(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES meta.users(id), summary JSONB NOT NULL, snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), undone_at TIMESTAMPTZ);
ALTER TABLE meta.experiments ADD COLUMN IF NOT EXISTS target_order TEXT[];
```

(Column types copied from `field_settings` in `001_initial.sql` — check `impute_value`'s type there and match it.)

- [ ] Test: tables exist; deleting an experiment cascades. Run → FAIL → write migration → PASS. Commit `"Experiments: their own copy of the baseline"`.

### Task B2: Experiment source — reads and writes through `view`

**Files:** Modify `api/app/assessment.py`, `api/app/routers/fields.py`, `api/app/routers/datasets.py::list_fields`, `api/app/routers/query.py`, `api/app/experiments.py` (`seed_copy`, `can_adjust`); Test `api/tests/test_adjustments.py`, `api/tests/test_team.py` (no-leak matrix)

**Interfaces:**
- `Source` gains `experiment: dict | None = None`; properties `team` (neither assessment nor experiment), `personal` (assessment) unchanged, `kind` in `"team" | "assessment" | "experiment" | "baseline"`, `family` (the recipe's family); `key` → `"experiment:<id>"` / `"baseline:<rid>"`.
- `decisions()` / `settings()` for experiments select from `meta.experiment_decisions` / `meta.experiment_settings` joined to `meta.fields` (same shape as today's personal branch). `fields_view()` joins settings on the literal `self.family`.
- `source_for(conn, dataset_id, user, recipe=None, lock=None, view: str | None = None) -> Source`:
  - `None` → as today.
  - `"experiment:<id>"` → `experiments.get_visible` (404 as for any hidden experiment), dataset must match (404), personal experiment → 409 `"A personal experiment follows your assessment; change your assessment instead."`; `denied` = 403 `"Only the experiment's owner (or a lead, for a team default) adjusts it."` unless `experiments.can_adjust(x, user)`, 409 when archived.
  - `"baseline:<rid>"` → finalized recipe of this dataset (404 otherwise); always `denied = (409, "<label> is finalized; its decisions can't change.")` (relation built from the snapshot in Task C1; until then 422 "not yet").
  - anything else → 422.
- `_decision_target` / `write_settings` / `restore` / batches: experiment rows keyed by `experiment_id`; batches in `meta.experiment_batches`.
- Every `if src.personal:` in `routers/fields.py` that means "not the team decision" becomes `if not src.team:`; the target branch for experiments: when an experiment write changes the target role, update `experiments.target_order` (append new target, drop removed) and, if `has_screen_fields`, `jobs.submit(screen_targets, dataset_id, [field_id])` after commit (associations are shared; `screen_targets` re-syncs dataset state from the working copy, which is harmless).
- Field endpoints take `view: str | None = None` (query string): `GET /datasets/{id}/fields`, `GET /fields/{id}`, `PUT /fields/{id}/decision`, `POST /datasets/{id}/decisions/bulk`, `POST …/bulk/{batch}/undo`, `POST /datasets/{id}/fields/query`. Default family = `src.family`. `GET /fields/{id}?view=experiment:…` also returns `baseline: {action, semantic_type, settings}` from the baseline snapshot for that field.
- `experiments.seed_copy(conn, x: dict) -> None` — insert the baseline snapshot into the copy tables (decisions by field name; settings for the recipe's family, falling back to the field's suggestion), `target_order = snapshot_targets(snap)`. Called from `experiments.create` and `team_for` for recipe-sourced experiments; migration backfill in `startup` hook: `experiments.backfill_copies(conn)` for team experiments without rows.
- `experiments.can_adjust(x, user) -> bool` — `x["assessor_id"] is None and x["archived_at"] is None and not x["is_default"] and user["id"] == x["owner"]`.

- [ ] **Step 1: Tests**

```python
def test_owner_adjusts_through_the_experiment_view(people):
    lead, a = people["lead"], people["a"]
    ds, rid = _finalized_trees(lead)                     # helper: upload, start tree with lead, finalize
    x = a.post(f"/api/datasets/{ds['id']}/experiments", json={"name": "A tries", "source": f"recipe:{rid}"}).json()
    view = f"experiment:{x['id']}"
    f = {r["name"]: r for r in a.get(f"/api/datasets/{ds['id']}/fields?view={view}").json()}
    cur = a.get(f"/api/fields/{f['w']['id']}?view={view}").json()
    assert cur["baseline"]["action"] == "keep"
    body = {**{k: cur["decision"][k] for k in KEYS}, "action": "drop"}
    assert a.put(f"/api/fields/{f['w']['id']}/decision?view={view}", json=body).status_code == 200
    assert {r["name"]: r for r in a.get(f"/api/datasets/{ds['id']}/fields?view={view}").json()}["w"]["action"] == "drop"
    assert _fields(lead, ds)["w"]["action"] == "keep"                       # the baseline didn't move
    assert people["b"].put(f"/api/fields/{f['w']['id']}/decision?view={view}", json=body).status_code == 403
    r = a.post(f"/api/datasets/{ds['id']}/decisions/bulk?view={view}", json={"field_ids": [f["x"]["id"]], "action": "drop"})
    assert r.status_code == 200
    assert a.post(f"/api/datasets/{ds['id']}/decisions/bulk/{r.json()['batch_id']}/undo?view={view}").status_code == 200


def test_retype_in_experiment_resets_settings_to_valid_ones(people):
    lead = people["lead"]
    ds, rid = _finalized_trees(lead)
    x = lead.post(f"/api/datasets/{ds['id']}/experiments", json={"name": "retype", "source": f"recipe:{rid}"}).json()
    view = f"experiment:{x['id']}"
    fid = _fields(lead, ds)["z"]["id"]
    cur = lead.get(f"/api/fields/{fid}?view={view}").json()["decision"]
    r = lead.put(f"/api/fields/{fid}/decision?view={view}",
                 json={**{k: cur[k] for k in KEYS}, "semantic_type": "categorical", "encoding": "ordinal"})
    assert r.status_code == 200
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert not any("aren't valid" in b for b in pv["blocking"])


def test_views_are_refused_where_they_must_be(people):
    lead, b = people["lead"], people["b"]
    ds, rid = _finalized_trees(lead)
    assert lead.get(f"/api/datasets/{ds['id']}/fields?view=nonsense").status_code == 422
    assert lead.get(f"/api/datasets/{ds['id']}/fields?view=experiment:999999").status_code == 404
```

Add to `READS` in `test_team.py`: `"fields via B's experiment view"` → `GET /api/datasets/{d}/fields?view=experiment:{b_xp}` (409 for a personal experiment, and 404 for viewers who can't see it — add to `REFUSED` as `{404, 409}` accepted) and `"field via baseline view of the version being assessed"` → `GET /api/fields/{f}?view=baseline:{recipe}` (404: not finalized).

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** per the interfaces. Keep `personal` meaning "assessment" so `team.py` adoption code is untouched.
- [ ] **Step 4: Run** `pytest tests/test_adjustments.py tests/test_team.py tests/test_bulk.py tests/test_targets_and_lists.py -q` → PASS.
- [ ] **Step 5: Commit** `"Experiments: adjust fields through view=experiment (Review, Bulk edit, finder)"`

### Task B3: Exports of adjusted experiments; adjustments summary; revert; targets changed

**Files:** Modify `api/app/routers/exports.py` (`_plan`), `api/app/experiments.py` (`snapshot_for`, `adjustments`), `api/app/routers/experiments.py`; Test `api/tests/test_adjustments.py`

**Interfaces:**
- `experiments.snapshot_for(conn, x) -> dict` — team experiment: `recipes.snapshot(conn, dataset_id, src=Source(recipe, experiment=x), family=recipe["family"], targets=x["target_order"])`; personal: `personal_snapshot` as today.
- `experiments.adjustments(conn, x) -> {"fields": diff["fields"], "targets": diff["targets"], "count": int, "targets_changed": bool}` using `recipes.diff(baseline_snap, snapshot_for(x), family)`.
- `GET /api/experiments/{id}` adds `adjustments` and `can_adjust` (bool, `experiments.can_adjust(x, user)`); list rows
  add `adjusted_count`, `targets_changed`. `experiments.COLS` adds `x.target_order` and `xr.family AS recipe_family`.
- `POST /api/experiments/{id}/adjustments/revert` body `{field_ids: list[int] | None}` (None = all) → copies baseline values back (decisions, settings, target_order); same permission as adjusting.
- `_plan` for a team experiment returns `experiments.snapshot_for`; `start_export` always freezes it in `exports.snapshot` and records `config.baseline = {recipe_id, family, version}`, `config.adjustments = {count, fields: [names], targets_changed}`.
- Preview: `blocking` gets "No target selected" (existing) when the experiment removed all targets; `warnings` gets `"Targets differ from {label}: {before} → {after}. Results aren't comparable with experiments on the baseline's targets."` when changed.
- Making a field a target in an experiment runs `target_problem` (422 as in assessment).

- [ ] **Step 1: Tests**

```python
def test_adjustments_stay_in_their_experiment_and_exports_freeze_them(people):
    lead, a = people["lead"], people["a"]
    ds, rid = _finalized_trees(lead)
    x1 = a.post(f"/api/datasets/{ds['id']}/experiments", json={"name": "one", "source": f"recipe:{rid}"}).json()
    x2 = a.post(f"/api/datasets/{ds['id']}/experiments", json={"name": "two", "source": f"recipe:{rid}"}).json()
    _adjust(a, ds, x1, "w", action="drop")
    e = _export(a, ds, x1)                                    # helper: start export, wait done
    _adjust(a, ds, x1, "x", action="drop")                     # after the export
    man = a.get(f"/api/exports/{e['id']}/files/manifest").json()
    assert "w" not in {f["name"] for f in man["fields"]} and "x" in {f["name"] for f in man["fields"]}
    assert e["config"]["adjustments"]["count"] == 1 and e["config"]["baseline"]["family"] == "tree"
    assert a.get(f"/api/experiments/{x2['id']}").json()["adjustments"]["count"] == 0
    assert a.get(f"/api/experiments/{x1['id']}").json()["adjustments"]["count"] == 2
    r = a.post(f"/api/experiments/{x1['id']}/adjustments/revert", json={"field_ids": None})
    assert r.status_code == 200 and a.get(f"/api/experiments/{x1['id']}").json()["adjustments"]["count"] == 0


def test_experiment_targets_are_validated(people):
    lead = people["lead"]
    ds, rid = _finalized_trees(lead)
    x = lead.post(f"/api/datasets/{ds['id']}/experiments", json={"name": "t", "source": f"recipe:{rid}"}).json()
    assert _adjust(lead, ds, x, "x", action="target", expect=422)          # continuous field can't be a target
    _adjust(lead, ds, x, "y", action="keep")
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert "No target selected" in pv["blocking"]
    _adjust(lead, ds, x, "c", action="target"); 
    info = lead.get(f"/api/experiments/{x['id']}").json()
    assert info["adjustments"]["targets_changed"] and info["adjustments"]["targets"]["after"] == ["c"]
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert any("Targets differ" in w for w in pv["warnings"])
```

- [ ] **Step 2–4:** implement; run `tests/test_adjustments.py tests/test_experiments.py tests/test_team.py` → PASS.
- [ ] **Step 5: Commit** `"Experiments: exports freeze baseline + adjustments; summary, revert, changed targets"`

### Task B4: Runs and MLflow record the preparation

**Files:** Modify `api/app/routers/runs.py` (run row carries export config baseline/adjustments), `api/app/modeling/mlflow_log.py`; Test `api/tests/test_mlflow.py`

**Interfaces:** `run_tags` adds `rainforest.baseline` (`"Trees baseline v1"`), `rainforest.baseline_family`, `rainforest.adjusted` (count as string), `rainforest.targets_changed` (`"true"`/`"false"`); params `prep.adjusted_fields` (count); artifact `adjustments.json` (the export's `config.adjustments` plus the per-field diff from `experiments.adjustments` at export time, stored in `exports.config.adjustments.fields_detail`).

- [ ] **Step 1: Test** in `test_mlflow.py` (runs against the MLflow service): train on an export of an experiment with one adjustment → the MLflow run's tags include `rainforest.adjusted == "1"` and `rainforest.baseline == "Trees baseline v1"`, and artifacts list contains `adjustments.json`.
- [ ] **Step 2–4:** implement; run `tests/test_mlflow.py` → PASS.
- [ ] **Step 5: Commit** `"MLflow: runs record their baseline and the experiment's adjustments"`

### Task B5: UI — experiment mode and adjustments

**Files:**
- Create: `ui/src/view.ts` — `useView(): { view: string | null; experimentId: number | null; withView(path: string): string }` reading `?view=` from the URL.
- Modify: `ui/src/api.ts` (optional `view` on `fields`, `field`, `saveDecision`, `bulk`, `bulkUndo`, `queryFields`; `experimentAdjustments` types; `revertAdjustments`), every `useQuery` key for those calls includes `view`, `ui/src/hooks.ts` (`useFields(id, view)`; `useAccess` returns experiment-mode access from the experiment: `canEdit` = `x.can_adjust`), `ui/src/pages/ReviewPage.tsx` and `BulkEditPage.tsx` (pass view; `ExperimentModeBar` component at top: "Adjusting experiment **{name}** (from {baseline label}). Changes apply only to this experiment. [Back to experiment]"), `ui/src/components/DecisionPanel.tsx` (when `data.baseline` exists and differs: line under each control "Baseline: drop" + *Revert to baseline* for the field), `ui/src/App.tsx` (`DatasetSteps` keeps `?view` on Review/Bulk edit links while in experiment mode; Team/Experiments links drop it), `ui/src/pages/ExperimentWorkspace.tsx` (*Adjust fields* button → `/d/:id/review?view=experiment:<id>`; **Adjustments** card listing changed fields with before → after and *Revert*; "targets changed" badge in the header), `ui/src/pages/ExperimentsPage.tsx` (adjusted count + badge per row), runs table group head shows the badge.
- Test: `ui/src/view.test.ts`, `ui/src/experiments.test.ts` (adjustment summary text helper).

**Interfaces:** `adjustmentLine(change: {key, family, before, after}, labels) → string` in `ui/src/experiments.ts` ("Imputation: median → none").

- [ ] **Step 1: Tests**

```ts
// view.test.ts
import { describe, expect, it } from "vitest";
import { parseView, withView } from "./view";
describe("view", () => {
  it("parses experiment and baseline views", () => {
    expect(parseView("experiment:12")).toEqual({ kind: "experiment", id: 12 });
    expect(parseView("baseline:3")).toEqual({ kind: "baseline", id: 3 });
    expect(parseView("x")).toBeNull();
  });
  it("adds the view to a path", () => {
    expect(withView("/d/1/review/5", "experiment:12")).toBe("/d/1/review/5?view=experiment%3A12");
    expect(withView("/d/1/review", null)).toBe("/d/1/review");
  });
});
```

- [ ] **Step 2–4:** implement; `npx tsc --noEmit -p . && npx vitest run && npm run build` → PASS.
- [ ] **Step 5: Commit** `"UI: experiment mode in Review and Bulk edit, adjustments card, targets-changed badge"`

### Task B6: Part B integration on dev-01, review, PR

- [ ] Full API + UI suites; deploy api+ui to dev-01.
- [ ] Playwright (`scratchpad/baselines/b.py`): ds1 creates experiment "Tuned" on Trees v1, *Adjust fields* → Review shows the mode bar; drops one field, retypes one, changes an imputation; Experiment page shows "3 fields adjusted"; export + LightGBM run; MLflow run page shows tags `rainforest.adjusted=3`; second experiment removes target column_b and adds none → preview shows the warning; trains single-target LightGBM; screenshots; zero page errors.
- [ ] Reviewer agent on the Part B diff (blind-data leaks through `view`, permission gaps, snapshot reproducibility); fix findings.
- [ ] PR, CI green, merge.

---

## Part C — Explore, datasets, views, docs (PR "Baselines everywhere")

### Task C1: Baseline view (read-only snapshot) and Explore source picker

**Files:** Modify `api/app/assessment.py` (baseline relation), `api/app/routers/explore.py` (`view` in analysis params → `decisions` key), `api/app/analysis/runner.py` (`from_key` handles `experiment:` and `baseline:`), `ui/src/pages/ExplorePage.tsx` (picker), `ui/src/api.ts`; Test `api/tests/test_explore.py`, `api/tests/test_baselines.py`

**Interfaces:**
- Baseline `Source.decisions()`:

```sql
(SELECT f.id AS field_id, f.dataset_id, e.value -> 'decision' ->> 'action' AS action,
        e.value -> 'decision' ->> 'semantic_type' AS semantic_type, e.value -> 'decision' ->> 'notes' AS notes,
        true AS reviewed, 'baseline' AS source, r.finalized_at AS updated_at
   FROM meta.recipes r CROSS JOIN LATERAL jsonb_each(r.snapshot -> 'fields') e
   JOIN meta.fields f ON f.dataset_id = r.dataset_id AND f.name = e.key
  WHERE r.id = <rid>)
```

  and `settings()` reads `e.value -> 'settings' -> <family>` with `jsonb_populate_record` columns (`imputation`, `impute_value`, `add_missing_indicator`, `transform`, `scaling`, `encoding`, `params`).
- Explore request params accept `view`; the stored `decisions` key is the source key; `from_key` rebuilds `experiment:`/`baseline:` sources (experiment visibility re-checked on read through `get_visible`).
- UI picker "Decisions from": *Current ({label} · {phase})* (default), each finalized baseline, and the user's adjustable/visible team experiments; only shown where decisions matter ("Kept only" on, or importance/correlation on raw data).

- [ ] Tests: browsing `GET /api/datasets/{id}/fields?view=baseline:<trees v1>` after Linear v1 started returns Trees decisions (w dropped) and `PUT` with that view → 409; Explore importance with `view=baseline:<rid>` and `kept: true` excludes `w`, and its `params.decisions == "baseline:<rid>"`.
- [ ] Implement, run `tests/test_explore.py tests/test_baselines.py`, commit `"Browse any finalized baseline read-only; Explore picks whose decisions it uses"`.

### Task C2: Datasets list chips and SQL views

**Files:** Modify `ui/src/pages/DatasetsPage.tsx` (chips: "Trees v2 · finalized", "Linear v1 · assessing" from `ds.baselines`), `api/app/views.sql` (`v_model_metrics` adds `baseline_family`, `baseline_version` via `exports.recipe_id`; `v_fields` unchanged), SQL console examples mention them; Test `api/tests/test_sql_console.py` (view has the columns).

- [ ] Test → implement → run → commit `"Datasets show their baselines; metrics view carries baseline family and version"`.

### Task C3: Existing E2E and regression

- [ ] Update `scratchpad/final/run.py` (22 checks) for the new flow: lead starts Trees baseline with ds1+ds2 (+lead), blind checks unchanged, review/finalize, team experiment "Team · Trees v1", LightGBM + XGBoost GPU runs, personal experiment privacy, MLflow deferral. All must pass on dev-01.
- [ ] Run the full API suite and UI suite once more.

### Task C4: Documentation (current state only)

**Files:** `README.md` (Team workflow → baselines: start dialog, one in progress, versions per type, populate; Experiments → adjustments, experiment mode, targets changed; Models → planned; remove Model switch mentions), `docs/api.md` (new endpoints and `view` parameter, changed bodies: `/recipes` `family`, diff ids, dataset `baseline(s)`, experiment `adjustments`, runs refusing planned models), `docs/architecture.md` (Source kinds, working copy rule, tables 012/013, leakage per target set), spec `2026-10-10-model-baselines-design.md` (fold in the five planning decisions).

- [ ] Verify `docs/api.md` routes == app routes (script from the docs audit: compare `app.routes` with the table), grep docs for "Model switch", "model family switch", "Recipe v", "used to", "now ", PR numbers.
- [ ] Commit `"Docs: baselines and experiment adjustments"`.

### Task C5: Final validation, review, deploy

- [ ] Whole-branch review agent against the spec and the requirements discussed (lead assesses, family fixed at start, populate or scratch, per-type baselines, experiments adjust type/targets/settings, LightGBM path, nothing private leaks); fix findings.
- [ ] PR, CI green, merge.
- [ ] Clean redeploy on dev-01 from `.env` (`make down && make up`), `make gpu-check`, run the updated E2E (C3) and the Part B Playwright script against the deployed stack; screenshots of the Baselines page, start dialog, experiment mode, adjustments card, MLflow run tags.
- [ ] Update memory (`rainforest-project.md`) with the shipped state.
