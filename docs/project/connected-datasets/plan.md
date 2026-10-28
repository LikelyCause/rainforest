# Connected Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dataset can be created from one or more MongoDB collections of a named connection, with a `source` column and deduplication across collections, and refreshed later with decisions carried forward and drift reported per finalized baseline.

**Architecture:** A `sources` package (protocol + MongoDB implementation) streams documents, flattens them by fixed rules into chunked Parquet and concatenates into the dataset's usual `raw.parquet`, after which the existing `process_dataset` runs unchanged. A `refresh` module re-pulls into side files, profiles, then swaps data and reconciles `meta.fields` by name in one short transaction, keeping `field.id`. Connections are named in `RAINFOREST_MONGO_SOURCES`; the chart composes it from Secrets. UI adds a Connect dialog, a connected badge with *Refresh…*, a lock banner, and a per-family pull-summary card reusing `ReuseCard`.

**Tech Stack:** FastAPI + pydantic-settings, psycopg 3, polars 1.20 (`concat(how="diagonal_relaxed")`), `pymongo` (new), React/TanStack Query/vitest, helm-unittest, `mongo:7` for tests.

**Spec:** `docs/superpowers/specs/2026-10-25-connected-datasets-design.md`

## Global Constraints

- Settings reach the API by the exact names in `.env.example`; `api/tests/test_config_wiring.py` enforces that every `Settings` field is passed by the Compose `api` service with matching defaults. New settings: `RAINFOREST_MONGO_SOURCES` (default empty), `SOURCE_MAX_CELLS` (default `600000000`), `SOURCE_PULL_TIMEOUT_S` (default `3600`).
- A connection string, user name or password never appears in an API response, a log line, an error, the SQL console (`meta.dataset_sources` stores the name only) or an export manifest.
- `meta.datasets.file_format` gains `mongodb`; `meta.datasets.status` gains `refreshing`. Flattening rules are exactly the table in spec §4 *Flattening*; names follow `mongoexport --type=csv` (dotted paths, dots inside keys → `_`).
- Deduplication compares the columns shared by every chosen collection minus `_id`, `source` and identifier-like columns, or the explicit `match_on` list; within and across collections; ties keep the first-listed collection's row.
- The cell budget is the sum over collections of documents × projected fields, checked before the pull.
- A refresh keeps `field.id` for fields that persist; deletes fields that vanished (and derived fields whose inputs vanished); marks every `meta.analyses` row of the dataset stale; records drift per model family with a finalized baseline via `reuse.match`; is refused with `409` while a version is assessing / in review, while an export or run is running, and while another refresh runs. During a refresh the dataset is `refreshing`: every write, export, run and Explore computation on it gets `409`.
- Every export of a connected dataset records `pull_id` in its manifest `config` and as an MLflow param.
- Docs are current-state only (no "used to", no PR numbers). Comments explain why. Commit messages end with the attribution trailer the session gives.
- MongoDB-backed tests use `RAINFOREST_TEST_MONGO_URL` and skip without it; both CI pipelines run a `mongo:7` service.

## Review Focus

1. A key whose values change type across documents (`age: 41` then `age: "unknown"`) must land as one text column with a warning, never crash the concat. Pinned in Task 3.
2. A filter that isn't a JSON document (`{bad`) or uses an unknown extended-JSON wrapper must be a `422` naming the collection, before any job starts. Pinned in Task 2.
3. A pull that fails halfway through a refresh (connection dropped) must leave the old `raw.parquet`, `raw.ds_<id>`, fields and `status: ready` untouched, with the failure on the pull row. Pinned in Task 5.
4. A refresh whose new data lacks a current target column must drop it from `target_columns` with a warning, and an export of a baseline that still names it must be blocked in the preview rather than crash. Pinned in Tasks 5 and 6.
5. A pymongo error message contains the host and sometimes the URI; every error surfaced to the API or stored on a pull must be redacted (`mongodb://…@` → `mongodb://***@`, and the connection name instead of the host). Pinned in Task 1.

---

### Task 1: Settings, source registry, MongoDB flattening

**Files:**
- Create: `api/app/sources/__init__.py`, `api/app/sources/mongodb.py`
- Modify: `api/app/config.py`, `api/requirements.txt` (`pymongo==4.10.1`), `docker-compose.yml` (api env), `.env.example`, `docs/configuration.md`
- Test: `api/tests/test_sources.py`

**Interfaces:**
- Produces: `Settings.mongo_sources: str = Field("", validation_alias="RAINFOREST_MONGO_SOURCES")`, `Settings.source_max_cells: int = 600_000_000`, `Settings.source_pull_timeout_s: int = 3600`.
- `sources.parse_sources(text: str) -> dict[str, str]` (name → uri; `ValueError` naming the entry's *name or position*, never the URI, on a malformed entry, a bad name (`^[a-z0-9-]+$`), or a duplicate).
- `sources.registry() -> dict[str, Source]` (cached per process; empty when the setting is empty) and `sources.get(name) -> Source` (raises `KeyError`).
- `class Source(Protocol)`: `name: str`, `kind: str`; `list_databases() -> list[str] | None` (None when the account can't list), `list_collections(database) -> list[dict]` (`{name, documents}`), `preview(database, collections: list[CollectionSpec], sample=200) -> PreviewResult`, `pull(database, collections, out_dir: Path, budget_cells: int, timeout_s: int, progress) -> PullResult` (Task 3 implements `pull`; this task defines the dataclasses `CollectionSpec(name, label, filter: dict|None, projection: dict|None)`, `PreviewResult(columns: list[ColumnInfo], per_source: list[dict], estimated_rows: int, estimated_cells: int)`, `ColumnInfo(name, kind: "number"|"text"|"datetime"|"json"|"identifier"|"mixed", present_share: float)`).
- `mongodb.flatten_document(doc: dict) -> dict[str, Any]` (pure; the spec table) and `mongodb.redact(message: str, name: str) -> str` (URI userinfo and hosts replaced; prefix with the connection name).

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_sources_names_and_uris():
    assert parse_sources("vehicles=mongodb://u:p@h/db;archive=mongodb+srv://x@y/") == {"vehicles": "mongodb://u:p@h/db", "archive": "mongodb+srv://x@y/"}

@pytest.mark.parametrize("text,needle", [("vehicles=", "vehicles"), ("Bad Name=mongodb://h/", "entry 1"), ("a=mongodb://h/;a=mongodb://h/", "a")])
def test_parse_sources_rejects_without_echoing_the_uri(text, needle):
    with pytest.raises(ValueError) as e: parse_sources(text)
    assert needle in str(e.value) and "mongodb://" not in str(e.value)

def test_flatten_rules():
    doc = {"_id": ObjectId("65f0a1b2c3d4e5f6a7b8c9d0"), "a": {"b": 1, "c.d": "x"}, "arr": [1, {"k": 2}], "dec": Decimal128("1.5"), "when": datetime(2026, 1, 2), "n": None, "u": uuid.UUID(int=1)}
    assert flatten_document(doc) == {"_id": "65f0a1b2c3d4e5f6a7b8c9d0", "a.b": 1, "a.c_d": "x", "arr": '[1, {"k": 2}]', "dec": 1.5, "when": datetime(2026, 1, 2), "n": None, "u": "00000000-0000-0000-0000-000000000001"}

def test_redact_hides_credentials_and_hosts():
    assert "secret" not in redact("mongodb://user:secret@db1.example.com:27017/x timed out", "vehicles")
    assert redact("...", "vehicles").startswith("vehicles: ")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd api && pytest tests/test_sources.py -q` → FAIL (no module `app.sources`).

- [ ] **Step 3: Implement** `config.py` fields, `sources/__init__.py` (protocol, dataclasses, `parse_sources`, `registry`, `get`), `sources/mongodb.py` (`MongoSource(name, uri)` with a lazy `MongoClient(uri, serverSelectionTimeoutMS=10_000, readPreference="secondaryPreferred")`, `list_databases` via `list_database_names` catching `OperationFailure` → `None`, `list_collections` via `estimated_document_count`, `preview` via `$sample`-free `find(filter, projection).limit(sample)` per collection, `flatten_document`, `redact`). Startup: `db.bootstrap()` (or `main.py` startup) calls `parse_sources` once and exits with `SystemExit(f"RAINFOREST_MONGO_SOURCES: {exc}")` on error, like `RAINFOREST_USERS`.

- [ ] **Step 4: Run the tests and the config-wiring test**

Run: `pytest tests/test_sources.py tests/test_config_wiring.py -q` → PASS (after adding the three variables to `docker-compose.yml` and `.env.example`; `docs/configuration.md` gets three rows).

- [ ] **Step 5: Commit**

```bash
git commit -am "MongoDB sources: named connections from RAINFOREST_MONGO_SOURCES and the flattening rules"
```

### Task 2: Sources API and the MongoDB test fixture

**Files:**
- Create: `api/app/routers/sources.py`, `api/tests/mongo_helpers.py`
- Modify: `api/app/main.py` (register router), `api/tests/conftest.py` (fixture `mongo`), `Makefile` (`test-mongo`), `.github/workflows/ci.yml` (service `mongo:7`, env `RAINFOREST_TEST_MONGO_URL=mongodb://localhost:27017`), `.gitlab-ci.yml` (same service on `api-tests`), `docs/api.md`, `docs/development.md`
- Test: `api/tests/test_sources_api.py`

**Interfaces:**
- Consumes: Task 1's `registry`, `Source`, `CollectionSpec`, `PreviewResult`.
- Produces: `GET /api/sources` → `[{name, kind}]`; `GET /api/sources/{name}/databases` → `{databases: [...] | null, listable: bool}`; `GET /api/sources/{name}/databases/{db}/collections` → `[{name, documents}]`; `POST /api/sources/{name}/preview` body `{database, collections: [{name, filter?, projection?}]}` → `{columns, per_source, estimated_rows, estimated_cells, budget: {max_cells, fits}}`. Unknown name → `404`. Invalid filter/projection → `422` `"<collection>: filter is not a JSON document"` (parsed with `bson.json_util.loads`, must be a dict). All need a signed-in user (any role).
- Fixture `mongo` (session): skips without `RAINFOREST_TEST_MONGO_URL`; yields a `pymongo.database.Database` named `rf_test_<pid>` dropped at teardown; `mongo_helpers.seed(db)` inserts two collections: `curated` (300 docs: `_id`, `vehicle`, `engine`, `rpm.mean`, `rpm.max`, `tags: [..]`, `when`) and `massive` (500 docs: same keys plus `extra.depth.key`, 20 docs with `rpm.mean: "n/a"` (mixed type), and 50 docs identical to curated rows except `_id`). The test env sets `RAINFOREST_MONGO_SOURCES=test=<url>` through `monkeypatch` + `registry.cache_clear()`.

- [ ] **Step 1: Write the failing tests** (`client` fixture pattern from `tests/test_explore.py`):

```python
def test_sources_lists_names_only(client): assert client.get("/api/sources").json() == [{"name": "test", "kind": "mongodb"}]
def test_collections_and_counts(client, mongo): ... == [{"name": "curated", "documents": 300}, {"name": "massive", "documents": 500}]
def test_preview_flattens_and_estimates(client, mongo):
    r = client.post("/api/sources/test/preview", json={"database": mongo.name, "collections": [{"name": "curated"}, {"name": "massive"}]}).json()
    names = {c["name"]: c for c in r["columns"]}
    assert names["rpm.mean"]["kind"] == "mixed" and names["tags"]["kind"] == "json" and names["_id"]["kind"] == "identifier"
    assert r["estimated_rows"] == 800 and r["budget"]["fits"] is True
def test_preview_rejects_a_bad_filter(client, mongo):
    r = client.post(..., json={"database": mongo.name, "collections": [{"name": "curated", "filter": "{bad"}]})
    assert r.status_code == 422 and "curated: filter" in r.json()["detail"]
def test_errors_never_carry_the_uri(client, monkeypatch): # point the registry at mongodb://u:p@127.0.0.1:1/ and assert "u:p" not in the 502 detail
```

- [ ] **Step 2: Run to verify they fail**, then implement the router (driver errors → `502` with `redact(...)`), the fixture, Makefile target (`docker run -d --rm --name rainforest-testmongo -p 127.0.0.1:27017:27017 mongo:7`), CI services, docs.

- [ ] **Step 3: Run** `RAINFOREST_TEST_MONGO_URL=mongodb://127.0.0.1:27017 pytest tests/test_sources_api.py -q` → PASS; without the variable → skipped.

- [ ] **Step 4: Commit**

```bash
git commit -am "Sources API: list a connection's databases and collections and preview the flattened columns"
```

### Task 3: Migration and the pull

**Files:**
- Create: `api/app/migrations/023_connected_datasets.sql`, `api/app/sources/pull.py`
- Modify: `api/app/sources/mongodb.py` (`pull`), `api/app/ingest.py` (export `sanitize_identifiers` use unchanged)
- Test: `api/tests/test_pull.py`

**Interfaces:**
- Migration: `file_format` CHECK adds `'mongodb'`; `status` CHECK adds `'refreshing'`; tables `meta.dataset_sources` and `meta.dataset_pulls` exactly as spec §5, with `match_on TEXT[]` added to `dataset_sources`; `meta.exports.config` already JSONB (no change).
- `MongoSource.pull(database, collections, out_dir, budget_cells, timeout_s, progress) -> PullResult` where `PullResult(per_source: list[{label, documents}], flatten_notes: dict)`; writes `out_dir/chunk_<label>_<n>.parquet` of ≤ 50,000 flattened documents each (schema per chunk), with mixed-type detection per chunk (a column whose chunk dtype differs from an earlier chunk's is cast to `pl.String` in every chunk and listed under `flatten_notes["mixed"]`), arrays under `flatten_notes["arrays"]`, renamed keys under `flatten_notes["renamed"]`. Raises `BudgetExceeded(estimated, max)` before reading when the estimate exceeds the budget, `PullTimeout` past `timeout_s`.
- `pull.assemble(out_dir: Path, specs: list[CollectionSpec], add_source: bool, dedupe: bool, match_on: list[str] | None, dest: Path) -> dict`: `pl.concat(..., how="diagonal_relaxed")`, adds `source` (label) first when `add_source`, deduplicates per the Global Constraint (identifier-like columns detected with `profiling.infer_type` on the column), keeps first by `source` order via a stable `unique(subset, keep="first", maintain_order=True)` after sorting by source rank, writes `dest` with `row_group_size=100_000`, deletes the chunks, returns `{"rows": n, "removed_duplicates": {label: n}, "compared_on": [...]}`.
- `pull.run_connect(dataset_id: int) -> None`: the job for a new dataset: reads `dataset_sources`, inserts a `dataset_pulls` row (`running`), calls `pull` + `assemble` into `datasets/<id>/raw.parquet`, stores `per_source`/`flatten_notes`/`rows` on the pull, appends flatten notes to `datasets.warnings` as readable sentences, then `process_dataset(dataset_id, None, "parquet")`; on any exception marks the pull `failed` with `redact(...)` and the dataset `failed`.

- [ ] **Step 1: Write the failing tests** (need `mongo` fixture and `tmp_path`):

```python
def test_pull_writes_chunks_and_notes_mixed_and_arrays(mongo, tmp_path): r = src.pull(mongo.name, [spec("curated"), spec("massive")], tmp_path, 10**9, 60, None); assert "rpm.mean" in r.flatten_notes["mixed"] and "tags" in r.flatten_notes["arrays"]
def test_assemble_adds_source_and_dedupes_on_shared_columns(mongo, tmp_path): out = assemble(tmp_path, specs, True, True, None, dest); df = pl.read_parquet(dest); assert df["source"].unique().to_list() == ["curated", "massive"] and out["removed_duplicates"] == {"massive": 50} and "_id" not in out["compared_on"] and "extra.depth.key" not in out["compared_on"]
def test_assemble_match_on_overrides(…): compared_on == ["vehicle", "engine"]
def test_budget_refuses_before_reading(mongo, tmp_path): with pytest.raises(BudgetExceeded): src.pull(..., budget_cells=10, ...)
def test_run_connect_ends_ready_with_a_pull_row(database, mongo): …status == "ready", pulls[0]["status"] == "done", rows == 750, file_format == "mongodb"
def test_run_connect_failure_marks_dataset_failed_without_uri(database, monkeypatch): …
```

- [ ] **Step 2: Run to verify they fail; implement; run to pass**: `pytest tests/test_pull.py -q` with both test URLs set.

- [ ] **Step 3: Commit**

```bash
git commit -am "Pull MongoDB collections into a dataset's Parquet copy, with a source column and deduplication"
```

### Task 4: Connect endpoint and dataset source fields

**Files:**
- Modify: `api/app/routers/datasets.py` (`POST /connect`, `GET /{id}/pulls`, `DATASET_COLS`/`get_dataset` additions, `_create` accepting `fmt="mongodb"`), `api/app/schemas.py` (`ConnectRequest`), `docs/api.md`
- Test: `api/tests/test_connect_api.py`

**Interfaces:**
- Consumes: Task 3's `run_connect`, Task 1's `registry`, `CollectionSpec`.
- Produces: `POST /api/datasets/connect` (Member/Lead; body per spec §6 with `match_on?`; validates the connection exists (`404`), each filter/projection (`422`), the budget via `preview`'s estimate (`422` with `estimated`/`max`), then `_create(name, None, "mongodb", targets, user_id, None, recipe_id)`, inserts `dataset_sources`, `jobs.submit(run_connect, id)`, `202`). `GET /api/datasets/{id}` and the list gain `source: {kind, connection, database, collections: [{name, label}], add_source, dedupe, match_on} | null` and `last_pull: {id, status, finished_at, rows, has_drift} | null`. `GET /api/datasets/{id}/pulls` → pulls newest first with `per_source`, `flatten_notes`, `drift`, `error`, `created_by_name`.

- [ ] **Step 1: Write the failing tests**: connect two collections → `202`, dataset row has `source` without any URI, becomes `ready` with `source` field present and `row_count == 750`; viewer → `403`; unknown connection → `404`; bad filter → `422`; `recipe_id` path works (reuses `reuse.check_source`); pulls endpoint lists one `done` pull.

- [ ] **Step 2: Run to verify they fail; implement; run to pass**; `docs/api.md` rows.

- [ ] **Step 3: Commit**

```bash
git commit -am "Connected datasets: create one from MongoDB collections and show where it came from"
```

### Task 5: Refresh core

**Files:**
- Create: `api/app/refresh.py`
- Modify: `api/app/processing.py` (extract `_persist_profiles`'s per-field insert path for reuse; add `update_field(conn, field_id, p: FieldProfile)`), `api/app/analysis/runner.py` (`mark_stale(conn, dataset_id)`: set `data_version = 'stale'` on the dataset's done rows), `api/app/ingest.py` (`load_raw_table(..., table_suffix="")` so the new table is `raw.ds_<id>_new`)
- Test: `api/tests/test_refresh.py`

**Interfaces:**
- Consumes: Task 3's `pull`/`assemble`, `reuse.match`, `profiling.profile_parquet`, `derived.compute`, `refresh_target_state`, `compute_associations`/`upsert_correlations`.
- Produces: `refresh.run_refresh(dataset_id: int, pull_id: int) -> None` (the job): steps 1–5 of spec §4 *Refresh*. Reconciliation by field name: existing → `update_field` (stats, inferred type, charts, `physical_dtype`, `ordinal`), new → `insert_field` + `seed_decision`, vanished → `DELETE FROM meta.fields` (cascades); derived fields recomputed with `derived.compute` on the new frame and written into `raw.parquet` before the swap; a derived field whose `check_spec` fails is deleted and listed. `refresh.drift_by_family(conn, dataset_id, fields) -> dict[str, dict]`: `reuse.match(snapshot, fields)` for each family's latest finalized recipe. Targets no longer present are removed from `target_columns` with a warning `"Target column no longer in the data: <name>"`. The pull row gets `rows`, `per_source`, `flatten_notes`, `drift`, `status`.
- Failure: any exception before the swap transaction → old files/tables untouched, dataset back to `ready`, pull `failed` with the redacted error; the `.new` files are removed.

- [ ] **Step 1: Write the failing tests** (seed, connect, finalize a Trees baseline via `set_phase`, then mutate Mongo and refresh):

```python
def test_refresh_keeps_field_ids_and_reports_drift(database, mongo): # insert 200 massive docs with a new key "fresh" and null "engine" in 60% → field ids for "vehicle" unchanged, history rows kept, "fresh" present, drift["tree"]["reopened"] has {"name": "engine", "reason": "drift"} and {"name": "fresh", "reason": "new"}, analyses rows stale
def test_refresh_removes_vanished_fields_and_targets(…): # drop key "engine" everywhere → field gone, target_columns without it, warning present, drift["tree"]["missing"] == ["engine"]
def test_refresh_recomputes_derived_and_drops_orphans(…): # a formula on rpm.mean survives; a formula on a dropped key is deleted and listed
def test_refresh_failure_leaves_the_dataset_untouched(…): # monkeypatch pull to raise after one chunk → parquet mtime unchanged, raw table row count unchanged, status ready, pull failed, "mongodb://" not in error
```

- [ ] **Step 2: Run to verify they fail; implement; run to pass.**

- [ ] **Step 3: Commit**

```bash
git commit -am "Refresh a connected dataset: re-pull, keep decisions by field, report drift per finalized baseline"
```

### Task 6: Refresh endpoint, the lock, and export lineage

**Files:**
- Modify: `api/app/routers/datasets.py` (`POST /{id}/refresh`), `api/app/auth.py` or a shared dependency (`refuse_while_refreshing`), the write paths that check `status != "ready"` (fields, derived, explore, exports, runs — `grep -n "status.*ready"` lists them; they already refuse non-ready statuses, so verify each returns `409` for `refreshing` and add the check to `PATCH /datasets/{id}`, bulk edit, assessment submit, baseline start/finalize), `api/app/routers/exports.py` (`preview`: blocking `"Field in the baseline is no longer in the data: <name>"`, warning `"Baseline finalized before pull #<n>; consider a new version"`; manifest `config.pull_id`), `api/app/modeling/mlflow_log.py` (param `rainforest.pull_id`), `docs/api.md`, `docs/experiments-and-training.md`
- Test: `api/tests/test_refresh_api.py`

**Interfaces:**
- Produces: `POST /api/datasets/{id}/refresh` (Lead; `404` not connected; `409` with a message for: version assessing/in review, export or run `running`/`pending` on the dataset, a pull already `running`; else inserts the pull row, sets `status = 'refreshing'` with `status_message = 'Refreshing from <connection>'`, `jobs.submit(run_refresh, id, pull_id)`, `202` → the pull row). Export rows and `GET /api/exports/{id}` carry `pull_id`.

- [ ] **Step 1: Write the failing tests**: refresh → `202`, dataset `refreshing` then `ready`; member → `403`; during `refreshing`, `PATCH` a decision → `409`, `POST exports` → `409`, Explore compute → `409`; refresh while assessing → `409`; after a refresh that dropped a snapshot field, export preview `blocking` contains the field name; an export after a refresh has `config.pull_id == <latest pull>` and the warning when the baseline predates it.

- [ ] **Step 2: Run to verify they fail; implement; run to pass; docs.**

- [ ] **Step 3: Commit**

```bash
git commit -am "Refresh endpoint: lead-only, locks the dataset while it runs; exports record the pull they used"
```

### Task 7: UI — Connect dialog and connected badge

**Files:**
- Create: `ui/src/components/ConnectDialog.tsx`, `ui/src/sources.ts` (pure helpers: `flattenKindLabel`, `budgetMessage(estimated, max)`)
- Modify: `ui/src/api.ts` (`sources`, `sourceDatabases`, `sourceCollections`, `sourcePreview`, `connectDataset`, `datasetPulls`), `ui/src/types.ts` (`Dataset.file_format` adds `"mongodb"`, `Dataset.source`, `Dataset.last_pull`, `DatasetStatus` adds `"refreshing"`, new `SourceInfo`, `PreviewResult`, `Pull`), `ui/src/pages/DatasetsPage.tsx` (a *Connect ▾* button next to *Samples ▾*, shown when `/api/sources` is non-empty; a `connected` badge on the row with connection · collections and *pulled <relative time>*)
- Test: `ui/src/sources.test.ts`

**Interfaces:**
- Consumes: Task 2 and 4 endpoints.
- Produces: the dialog in spec §4 order (connection → database → collections with counts → per-collection label/filter/projection → options: name, *Add a `source` column* (locked on with 2+), *Leave out duplicates* (default on with 2+) with the *compared on* list from the preview and an optional *Match on* input → *Preview* → *Connect*). Budget overflow disables *Connect* and shows `budgetMessage`. Errors show the API's `detail`.

- [ ] **Step 1: Write the failing helper tests** (`budgetMessage(450_000_000, 600_000_000)` → `"450,000,000 of 600,000,000 cells"`; overflow wording; `flattenKindLabel("mixed")` → `"mixed types, read as text"`).

- [ ] **Step 2: Implement the dialog, API client, types and page changes; run** `npm run typecheck && npx vitest run && npx prettier --check src` → clean.

- [ ] **Step 3: Manual check** against `make up-mongo` (Task 9 provides it; until then, a local `mongo:7` seeded by `tests/mongo_helpers.seed` through a one-off script) — record what you saw in the report.

- [ ] **Step 4: Commit**

```bash
git commit -am "Connect dialog: pick a MongoDB connection, database and collections, preview the columns, connect"
```

### Task 8: UI — Refresh, the lock banner, and the pull summary

**Files:**
- Create: `ui/src/components/PullCard.tsx` (wraps `ReuseCard` per family with the pull's `drift[family]`, plus per-source counts, removed duplicates, flatten notes, and *Start a new version with these fields reopened* for leads → navigates to `/datasets/<id>/baselines?new=<family>&reopen=<names>`)
- Modify: `ui/src/pages/DatasetsPage.tsx` (row menu *Refresh…* for leads on connected datasets, with the confirmation text from spec §4), `ui/src/components/DatasetLayout.tsx` (status `refreshing`: pages render read-only with a banner "Refreshing from <connection>, started by <name>"), `ui/src/pages/DatasetTeamPage.tsx` and `ui/src/pages/BaselinesPage.tsx` (show `PullCard` for the latest pull with drift; BaselinesPage's *New version* form reads `reopen` from the URL to pre-select fields), `ui/src/can.ts` (refresh permission), `ui/src/types.ts`
- Test: `ui/src/pulls.test.ts` (pure helper `reopenedNames(drift, family)`), `ui/src/can.test.ts` (lead can refresh a connected dataset; member can't; nobody while refreshing)

- [ ] **Step 1: Write the failing tests; implement; run** typecheck/vitest/prettier → clean.

- [ ] **Step 2: Manual check**: refresh a connected dataset, watch the banner, see the card, start a version with the pre-selected fields.

- [ ] **Step 3: Commit**

```bash
git commit -am "Refresh from the dataset menu, a lock banner while it runs, and a pull summary per baseline"
```

### Task 9: Chart values, Compose profile, docs

**Files:**
- Create: `compose/mongo/seed.py` (or `api/app/sample_mongo.py`, run by a one-shot Compose service), `deploy/chart/rainforest/templates/_sources.tpl` or additions to `_helpers.tpl`, `deploy/chart/rainforest/tests/sources_test.yaml`
- Modify: `docker-compose.yml` (profile `mongo`: `mongo:7` + `mongo-seed`; api gets `RAINFOREST_MONGO_SOURCES` default pointing at it only under the profile — use `${RAINFOREST_MONGO_SOURCES:-}` and let `make up-mongo` export `RAINFOREST_MONGO_SOURCES=dev=mongodb://mongo:27017`), `Makefile` (`up-mongo`), `deploy/chart/rainforest/values.yaml` (`sources.mongodb: []`), `templates/api-deployment.yaml` (one env `MONGO_URI_<NAME>` per source from its Secret key `uri`, then `RAINFOREST_MONGO_SOURCES` as `name=$(MONGO_URI_NAME);…`), `deploy/chart/rainforest/README.md`, `deploy/README.md`, `deploy/argocd/teams/vehicles/values.yaml` (commented example), `docs/data-preparation.md` (connected datasets, the flattening table, dedupe, refresh), `docs/deployment.md`, `README.md` (one line in the datasets walkthrough)

- [ ] **Step 1: Chart tests first**: two sources → two `MONGO_URI_*` env entries from the named Secrets and `RAINFOREST_MONGO_SOURCES` = `vehicles=$(MONGO_URI_VEHICLES);archive=$(MONGO_URI_ARCHIVE)`; empty list → neither; a name that isn't a slug → `failedTemplate`. Run `bash deploy/chart/test.sh` → FAIL, implement, PASS; `helm lint -f ci/dev-values.yaml` clean.

- [ ] **Step 2: Compose profile**: `make up-mongo` starts the stack plus `mongo` and a seeder that writes `rf_dev.curated` and `rf_dev.massive` from `tests/mongo_helpers.seed`'s generator (move the generator into `app/sample_mongo.py` so tests and the seeder share it); verify *Connect ▾* appears and a connect succeeds.

- [ ] **Step 3: Docs**; grep for "used to"/"will be"; commit.

```bash
git commit -am "MongoDB sources in the chart and Compose, with a seeded dev profile and docs"
```

### Task 10: Branch finish

- [ ] Run everything: API suite with `RAINFOREST_TEST_DATABASE_URL` and `RAINFOREST_TEST_MONGO_URL`, ruff, UI typecheck/vitest/prettier, `bash ui/nginx/test.sh`, chart lint + unittest, `bash -n deploy/smoke.sh`.
- [ ] Re-read every touched doc for current-state wording; confirm no URI can reach a response (grep the routers for `uri` in returned dicts).
- [ ] Branch review, then the finishing menu.
