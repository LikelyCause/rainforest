# Baseline Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A baseline (one per model type per dataset) can have an optional name, set when it starts and renamed by a
lead, shown wherever the baseline is labelled.

**Architecture:** Names live in `meta.baseline_names (dataset_id, family, name)`; a SQL function
`meta.baseline_name(dataset_id, family)` supplies them to every query that reads a baseline, `recipes.RECIPE_COLS`
included. `recipes.label()` requires the `name` key, so a query that forgets it fails loudly instead of showing the
model type. The UI's `baselineLabel()` takes the name; every call site passes it.

**Tech Stack:** Postgres 16 migrations, FastAPI + psycopg3, React 18 + react-query, vitest, Playwright on dev-01.

**Spec:** `docs/superpowers/specs/2026-10-10-baseline-names-design.md`

## Global Constraints

- Name: trimmed, 1–60 characters, unique among the dataset's baselines ignoring case; blank means no name.
- Short label: unnamed "Trees v2", named "Dropout risk v2". Long label: unnamed "Trees baseline v2", named
  "Dropout risk v2 (Trees)".
- Renaming never touches a finalized version's snapshot; an export manifest and a new experiment's default name keep
  the label of the moment they were made.
- Rename: `PATCH /api/datasets/{id}/baselines/{family}` `{name}`, lead only; `404` without that baseline; `422` for
  a bad name.
- Docs state current behaviour only (no changelog wording). Comments sparse.
- Layout checked at 768, 1024 and 1440px (iPad to desktop; phones out of scope).
- Before `make up` on dev-01: no pending/running exports, runs or analyses.

## Review Focus

1. A name that differs only in case from another baseline's → refused, not silently allowed (Task 3 test).
2. Renaming a baseline whose versions are all finalized → works and the snapshot is byte-identical (Task 3 test).
3. Setting up a new dataset with a name while switching model type (placeholder Trees row becomes Linear) → the name
   lands on Linear, and Trees has none (Task 3 test).
4. A long name (60 characters) in the top bar chip and dataset list badge → cut with an ellipsis, full in the
   tooltip, no extra top-bar row at 1440px (Task 5 browser check).
5. A reused baseline's source label → shows the source's name as of reuse; old reuse records without one still
   read "Trees baseline v1" (Task 2 test + UI fallback).

---

### Task 1: Storage, name rules and the label

**Files:**
- Create: `api/app/migrations/016_baseline_names.sql`
- Modify: `api/app/recipes.py` (RECIPE_COLS, CURRENT_JOIN, `label`, new `clean_name`)
- Test: `api/tests/test_baseline_names.py`

**Interfaces:**
- Produces: `meta.baseline_names`, `meta.baseline_name(bigint, text) → text`; `recipes.clean_name(raw: str | None)
  -> str | None` (raises `HTTPException(422)`); `recipes.label(recipe: dict) -> str` reading `recipe["name"]`;
  `recipes.short_label(family: str, version: int, name: str | None) -> str`.

- [ ] **Step 1: Failing tests**

```python
def test_labels_use_the_name(people):
    from app import recipes
    assert recipes.label({"family": "tree", "version": 2, "name": None}) == "Trees baseline v2"
    assert recipes.label({"family": "tree", "version": 2, "name": "Dropout risk"}) == "Dropout risk v2 (Trees)"
    assert recipes.short_label("linear", 1, "Churn") == "Churn v1" and recipes.short_label("linear", 1, None) == "Linear v1"

def test_label_needs_the_name_key(people):
    import pytest
    from app import recipes
    with pytest.raises(KeyError):
        recipes.label({"family": "tree", "version": 1})

def test_name_rules(people):
    import pytest
    from fastapi import HTTPException
    from app import recipes
    assert recipes.clean_name("  Dropout risk ") == "Dropout risk" and recipes.clean_name("   ") is None
    assert recipes.clean_name(None) is None and recipes.clean_name("x" * 60) == "x" * 60
    with pytest.raises(HTTPException) as e:
        recipes.clean_name("x" * 61)
    assert e.value.status_code == 422

def test_recipe_rows_carry_the_name(people):
    from app import recipes
    from app.db import connection
    ds = upload(people["lead"])
    with connection() as conn:
        conn.execute("INSERT INTO meta.baseline_names (dataset_id, family, name) VALUES (%s, 'tree', 'Dropout risk')", (ds["id"],))
        assert recipes.current(conn, ds["id"])["name"] == "Dropout risk"
        assert recipes.label(recipes.current(conn, ds["id"])) == "Dropout risk v1 (Trees)"
```

- [ ] **Step 2:** Run `pt tests/test_baseline_names.py` → FAIL (no `clean_name`, no table).
- [ ] **Step 3: Migration**

```sql
-- Migration 016: a name per baseline (one per model family per dataset), outside the versions and their snapshots.
CREATE TABLE IF NOT EXISTS meta.baseline_names (
    dataset_id BIGINT NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    family     TEXT   NOT NULL,
    name       TEXT   NOT NULL CHECK (length(name) BETWEEN 1 AND 60 AND name = btrim(name)),
    PRIMARY KEY (dataset_id, family)
);
CREATE UNIQUE INDEX IF NOT EXISTS baseline_names_unique_idx ON meta.baseline_names (dataset_id, lower(name));
CREATE OR REPLACE FUNCTION meta.baseline_name(ds BIGINT, fam TEXT) RETURNS TEXT LANGUAGE sql STABLE AS
  $$ SELECT name FROM meta.baseline_names WHERE dataset_id = ds AND family = fam $$;
```

- [ ] **Step 4: recipes.py** — `RECIPE_COLS` gains `meta.baseline_name(dataset_id, family) AS name`; `CURRENT_JOIN`'s
  lateral select gains the same; then:

```python
NAME_MAX = 60

def clean_name(raw: str | None) -> str | None:
    name = (raw or "").strip()
    if len(name) > NAME_MAX:
        raise HTTPException(422, f"A baseline name is at most {NAME_MAX} characters")
    return name or None

def short_label(family: str, version: int, name: str | None) -> str:
    return f"{name or FAMILY_LABEL[family]} v{version}"

def label(recipe: dict) -> str:
    name, fam = recipe["name"], FAMILY_LABEL[recipe["family"]]
    return f"{name} v{recipe['version']} ({fam})" if name else f"{fam} baseline v{recipe['version']}"
```

- [ ] **Step 5:** Run the new tests → PASS; run the whole API suite → every `label()` site whose row lacks `name`
  fails with KeyError; fix each in Task 2.
- [ ] **Step 6: Commit** "Baseline names: storage, rules and labels".

### Task 2: Every payload carries the name

**Files:**
- Modify: `api/app/routers/datasets.py` (`with_baselines` latest + current queries),
  `api/app/routers/baselines.py` (list: `name`, `parent_name`; line `name`), `api/app/routers/recipes.py`
  (finalized list `name`; versions `name`, `parent_name`), `api/app/routers/exports.py` (list `recipe_name`),
  `api/app/experiments.py` (rows `recipe_name`; default name uses `short_label`), `api/app/reuse.py` (source `name`),
  `api/app/assessment.py` and any other `label()` site the suite flags.
- Test: `api/tests/test_baseline_names.py`

**Interfaces:**
- Consumes: `meta.baseline_name`, `recipes.label`, `recipes.short_label`.
- Produces (JSON): `baseline.name`, `baselines[].name`; team `recipe.name`; baselines list `name` per line and
  `versions[].name`, `versions[].parent_name`; `/api/recipes` rows `name`; versions `name`, `parent_name`; exports
  and experiments `recipe_name`; reuse `source.name`.

- [ ] **Step 1: Failing class test** — one dataset whose Trees baseline is named (inserted directly) and finalized,
  with a Team experiment and export; assert `"Dropout risk"` in: dataset list row `baseline.label` and
  `baselines[0].name`, dataset detail, team state `recipe.label`, `/baselines` line `name` and version `label`,
  `/recipes` (finalized list) `label`, `/datasets/{id}/recipes` `label`, diff `a.label`, export preview
  `recipe.label`, export row `recipe_name`, experiments `recipe_name`; a new default experiment is named
  "Team · Dropout risk v1"; reuse onto a new dataset records `source.name`.
- [ ] **Step 2:** Run → FAIL on the first missing site.
- [ ] **Step 3:** Add `meta.baseline_name(r.dataset_id, r.family) AS name` (or `AS recipe_name`, `AS parent_name`
  with `p.`) to each query; `experiments.default_name` → `f"Team · {recipes.short_label(recipe['family'],
  recipe['version'], recipe['name'])}"`; reuse source dict adds `"name": source["name"]`.
- [ ] **Step 4:** Whole API suite → green (no KeyError left).
- [ ] **Step 5: Commit** "Every baseline payload and label carries its name".

### Task 3: Naming at start, and rename

**Files:**
- Modify: `api/app/schemas.py` (`StartBaselineIn.name`, new `RenameBaselineIn`), `api/app/routers/baselines.py`
  (store at start; `PATCH /datasets/{id}/baselines/{family}`), `api/app/recipes.py` (`set_name(conn, dataset_id,
  family, name)` with the uniqueness check)
- Test: `api/tests/test_baseline_names.py`

**Interfaces:**
- Produces: `recipes.set_name(conn, dataset_id: int, family: str, name: str | None) -> None` (422 on a taken
  name); `POST …/baselines` `name`; `PATCH …/baselines/{family}` → the baselines list.

- [ ] **Step 1: Failing tests** — setup with `name` ("Dropout risk") and family linear → line linear named, tree
  none; second family with name; `name` for an existing family that differs → 422, equal → OK; a taken name in
  another case → 422; 61 chars → 422; blank → no name; rename by lead → 200 and label changes; member → 403; viewer →
  403; unknown family → 404; rename after finalize leaves `snapshot` unchanged (compare JSON before/after); clearing
  with `""` removes the row; deleting the dataset removes its names.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**

```python
def set_name(conn, dataset_id: int, family: str, name: str | None) -> None:
    if name is None:
        conn.execute("DELETE FROM meta.baseline_names WHERE dataset_id = %s AND family = %s", (dataset_id, family))
        return
    taken = conn.execute("SELECT family FROM meta.baseline_names WHERE dataset_id = %s AND family <> %s "
                         "AND lower(name) = lower(%s)", (dataset_id, family, name)).fetchone()
    if taken:
        raise HTTPException(422, f"Another baseline of this dataset ({FAMILY_LABEL[taken['family']]}) is called {name!r}")
    conn.execute("""INSERT INTO meta.baseline_names (dataset_id, family, name) VALUES (%s, %s, %s)
                    ON CONFLICT (dataset_id, family) DO UPDATE SET name = excluded.name""", (dataset_id, family, name))
```

  In `start_baseline`, after the family is settled: a family with no versions before this start (or `setup`) →
  `set_name(conn, dataset_id, recipe["family"], clean_name(body.name))`; an existing family with a different
  non-blank name → 422 "rename it on the Baselines page". The rename route takes `LeadUser`, locks the dataset,
  404s when the family has no versions, calls `set_name(…, clean_name(body.name))` and returns `list_baselines`.
- [ ] **Step 4:** Tests → PASS; full suite green.
- [ ] **Step 5: Commit** "Name a baseline when it starts; leads rename it".

### Task 4: UI labels everywhere

**Files:**
- Modify: `ui/src/format.ts` (`baselineLabel(family, version, short?, name?)`), `ui/src/types.ts` (the fields from
  Task 2), every call site: `App.tsx`, `experiments.ts`, `components/StartBaseline.tsx`,
  `components/ExperimentModeBar.tsx`, `components/ReuseCard.tsx`, `pages/DatasetsPage.tsx`,
  `pages/BaselinesPage.tsx`, `pages/ExperimentWorkspace.tsx`, `pages/ExperimentsPage.tsx`; `styles.css` (chip and
  phase badge ellipsis); `pages/DatasetTeamPage.tsx` (PhaseBadge title carries the full label)
- Test: `ui/src/baselines.test.ts`

- [ ] **Step 1: Failing test**

```ts
it("names a named baseline first, with its model type in the long form", () => {
  expect(baselineLabel("tree", 2, true, "Dropout risk")).toBe("Dropout risk v2");
  expect(baselineLabel("tree", 2, false, "Dropout risk")).toBe("Dropout risk v2 (Trees)");
  expect(baselineLabel("tree", 2, false, null)).toBe("Trees baseline v2");
});
```

- [ ] **Step 2:** `npx vitest run src/baselines.test.ts` → FAIL.
- [ ] **Step 3:** Implement `baselineLabel`; pass the name at every call site (`grep -n "baselineLabel(" ui/src`
  must show a fourth argument everywhere except where no name exists, e.g. a draft family in the start form).
  CSS: `.baseline-chip b, .badge.phase { max-width: 260px; overflow: hidden; text-overflow: ellipsis; }` with
  `display: inline-block` on the chip's `b`; the chip's and badge's `title` hold the full label.
- [ ] **Step 4:** typecheck, vitest, build → green.
- [ ] **Step 5: Commit** "UI: baseline names in every label".

### Task 5: Name field, Rename, docs, deploy

**Files:**
- Modify: `ui/src/baselines.ts` (`StartBaselineForm.name`), `ui/src/components/StartBaseline.tsx` (Name field when
  the family has no baseline or on setup; otherwise the current name and "Rename it on the Baselines page"),
  `ui/src/pages/BaselinesPage.tsx` (card heading: name + family tag; lead's *Rename* inline editor),
  `ui/src/api.ts` (`renameBaseline`), `ui/src/can.ts` (`renameBaseline`, lead only), `README.md`, `docs/api.md`,
  `docs/architecture.md`
- Test: `ui/src/can.test.ts`, browser script on dev-01

- [ ] **Step 1:** can test: `renameBaseline` true for lead only → FAIL → add to ACTIONS and LEAD_ONLY → PASS.
- [ ] **Step 2:** Build the field and the rename editor; typecheck, vitest, build.
- [ ] **Step 3:** Docs: README baseline section (name at setup, rename, how labels read), api.md (`name` on POST,
  the PATCH route, `name`/`recipe_name`/`parent_name` fields), architecture.md (`meta.baseline_names`,
  `meta.baseline_name`, label rule).
- [ ] **Step 4:** Check jobs, `DOCKER_CONTEXT=dev-01 make up`, browser at 768/1024/1440: set up a new dataset's
  baseline named "Dropout risk" (library sample, deleted afterwards), see it in list badge, top bar, Baselines page,
  team page; rename to a 60-character name → ellipsis in chip and badge with full tooltip, top bar stays one row at
  1440; rename Student dropout's baseline only if the owner asks (don't touch their data).
- [ ] **Step 5: Commit**, push, PR, final review, merge on CI green.
