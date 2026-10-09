# Architecture

A developer's map of Rainforest: what runs where, how a request and a dataset move through the system, and where to
add things. For endpoints see [api.md](api.md); for usage see the [README](../README.md).

## Components

| Service | Code | Role |
|---|---|---|
| `ui` | `ui/` | React 18 + TypeScript, Vite, TanStack Query, ECharts. Built to static files and served by nginx (`ui/nginx.conf`), which proxies `/api/` to the API. |
| `api` | `api/app/` | FastAPI on uvicorn (`API_WORKERS` processes). Owns the schema, runs every background job in-process. |
| `db` | Postgres 16 | `meta.*` holds metadata, decisions, jobs and results; `raw.ds_<id>` holds each dataset as a typed table. |
| `/data` volume | `appdata` | `uploads/` (transient), `datasets/<id>/raw.parquet`, `exports/<id>/`, `models/run<id>_<model>_<ts>/`. |

### API modules

| Module | Responsibility |
|---|---|
| `main.py` | App, middleware, router registration, `/health`, `/api/options`. Startup loads the session secret, then runs `db.bootstrap()` and `db.mark_interrupted_jobs()`. |
| `auth.py` | Sign-in (`/api/auth/*`), the signed session cookie, the session secret, the login rate limiter, and `SecurityMiddleware`. |
| `config.py` | `Settings` (pydantic-settings, read from the environment). |
| `db.py` | Connection pools (the app's, and a small one for the console's read-only role), query helpers, migrations, read-only role setup, restart cleanup. |
| `jobs.py` | `submit(fn, *args)` onto a `ThreadPoolExecutor` of `JOB_WORKERS` threads. |
| `ingest.py` | CSV / Parquet → Parquet, identifier sanitising, `COPY` into `raw.ds_<id>`. |
| `profiling.py` | Type inference, statistics, charts, association sample vectors. |
| `correlations.py` | Pairwise associations (Spearman, Cramér's V, correlation ratio). |
| `semantics.py` | Model families, semantic types, allowed settings per family and type, suggestions, validation, the option catalog. |
| `processing.py` | The dataset pipeline (`process_dataset`), suggestion seeding, target-derived state (`refresh_target_state`), file cleanup. |
| `transform.py` | Export: split, fit preprocessing on train rows, chunked transform, manifest. |
| `modeling/registry.py`, `modeling/train.py` | Model catalog and compatibility; `ClassifierChain` training, evaluation, CLI. |
| `analysis/prep.py`, `analysis/compute.py`, `analysis/runner.py` | Explore jobs: sampling and typing, the numeric work, caching and job lifecycle. |
| `routers/*.py` | HTTP endpoints, one module per area (datasets, fields, query, exports, runs, explore). |
| `sample_data.py` | Synthetic dataset generator (also a CLI: `python -m app.sample_data`). |
| `schemas.py` | Request bodies and query models. |

### UI modules

| Path | Responsibility |
|---|---|
| `App.tsx` | Routes, the sign-in gate (`RequireAuth`), and the top bar (dataset switcher, workflow steps, Model switch, Sign out). |
| `api.ts`, `types.ts` | Typed fetch client for every endpoint; response types. Any `401` (fetch or the upload XHR) sends the user to `/login?next=<current path>`. |
| `hooks.ts` | Shared queries (`useDataset`, `useFields`, `useOptions`) and `useDebounced`. |
| `saveQueue.ts` | Review decision saves: one serial lane per field, latest draft wins, failures surface with Retry, flushes on page hide. |
| `pages/` | `LoginPage`, `DatasetsPage`, `ReviewPage`, `BulkEditPage`, `ExplorePage`, `ExportPage`. |
| `components/` | `DatasetLayout` (gates dataset pages on status), `FieldList`, `FieldDetailView`, `DecisionPanel`, `RunDetail`, `Chart`; `components/explore/` holds the four Explore views, their shared state (`state.tsx`) and chart options (`options.ts`). |
| `signals.ts`, `format.ts`, `theme.ts`, `charts.ts`, `toast.tsx` | Field signal thresholds, formatting and labels, chart theming, toasts. |

Routes: `/login`, `/` (datasets), `/d/:id/review/:fieldId?`, `/d/:id/bulk`, `/d/:id/explore/:view?`, `/d/:id/export`.

## Request and data flow

Every request first passes `auth.SecurityMiddleware`, before routing (so file downloads are covered too):

1. **Host.** Unless `RAINFOREST_ALLOWED_HOSTS` is `*`, the `Host` header must match it (`localhost` always does).
   `400` otherwise.
2. **Unsafe methods** (`POST`, `PUT`, `PATCH`, `DELETE`): `Origin`, or failing that `Referer`, must be same-origin with
   `Host` or `X-Forwarded-Host`, or be the Vite dev server (`403`); requests with neither header are not from a
   browser and pass. A body must be `application/json`, or `multipart/form-data` on `/api/datasets/upload` (`415`).
3. **Session.** Everything except `/health`, `/api/auth/login` and `/api/auth/logout` needs a valid `rf_session`
   cookie (`401`), and nothing but those is served while `RAINFOREST_ADMIN_PASSWORD` is unset (`503`).

The cookie is `base64(claims).HMAC`, with claims `{u, exp}`. The HMAC key is derived from the session secret plus the
admin user and password, so changing either invalidates every session. The secret comes from
`RAINFOREST_SESSION_SECRET` or is generated once into `DATA_DIR/session_secret` (mode 600; created with an atomic
link so concurrent workers agree). It is deliberately not in Postgres, where the console's read-only role could read
it. nginx forwards `Host` with its port and `X-Forwarded-Proto`, which the origin check and the cookie's `Secure`
flag rely on.

Requests are synchronous SQL against Postgres except where noted. Long work returns `202` with a row whose `status`
the UI polls (`pending → running → done | failed`, with `progress` and `status_message`).

1. **Upload / sample.** `POST /api/datasets/upload` streams the file to `uploads/`, inserts a `meta.datasets` row and
   submits `process_dataset`. `POST /api/datasets/sample` generates the frame first, then runs the same pipeline.
2. **Pipeline** (`processing.process_dataset`), with dataset `status` moving through `ingesting → profiling →
   correlating → ready`:
   * ingest: convert to `datasets/<id>/raw.parquet`, load `raw.ds_<id>` with `COPY`;
   * profile every column from the Parquet copy (exact counts, sampled distributions), write `meta.fields`, and seed
     `meta.field_decisions` plus one `meta.field_settings` row per family from the per-family suggestions;
   * compute pairwise associations into `meta.field_correlations`, then `refresh_target_state` derives
     `max_target_assoc`, leakage flags and target-dependent suggestions.
3. **Targets.** `PATCH /api/datasets/{id}` with `target_columns` calls `fields.set_targets`, which updates decisions
   and runs `_sync_targets` (keeps `is_target`, `target_columns` and target-derived state consistent, writes history
   for knock-on changes). Targets profiled as text or identifier have no association sample, so a `correlate_targets`
   job computes theirs.
4. **Review.** `GET /api/fields/{id}` returns the field, the shared decision merged with the active family's settings,
   and target associations. `PUT /api/fields/{id}/decision` validates against the family, writes the decision and
   settings, re-derives other families' settings on a type change, and appends `meta.decision_history`.
5. **Bulk edit.** The field filter compiles to a `WHERE` over `meta.v_fields` (`query.fields_where`, a whitelist of
   columns and operators). `POST …/decisions/bulk` resolves and locks the set in one statement, plans per-field
   changes, and either returns the plan (`dry_run`) or applies it, storing a pre-change snapshot in
   `meta.bulk_batches` for undo.
6. **SQL console.** `POST /api/sql` takes a connection from the `rainforest_ro` pool (`SQL_POOL_MAX`; `503` when
   busy), opens a `READ ONLY` transaction with `SET LOCAL statement_timeout`, and runs the text as a named
   server-side cursor. psycopg sends the `DECLARE` over the extended protocol, so Postgres rejects a second statement
   and `DECLARE` accepts only a query; a single `FETCH` pulls `SQL_MAX_ROWS + 1` rows to detect truncation. The
   transaction is always rolled back, and the pool runs `DISCARD ALL` before reusing the connection, so settings or
   advisory locks taken by a query don't carry over.
7. **Export.** `POST …/exports` checks the preview's blocking issues and submits `exports._run`, which calls
   `transform.run_export` and writes `exports/<id>/{model_ready.parquet, preprocessor.joblib, manifest.json}`.
8. **Training.** `POST /api/exports/{id}/runs` checks model compatibility and submits `runs._run`, which calls
   `modeling.train.train` and writes `models/run<id>_…/{classifier_chain.joblib, report.json}`; metrics and the
   report go to `meta.model_runs`.
9. **Explore.** See [Analyses](#analyses-explore-jobs).

## Background jobs

All jobs go through `jobs.submit`, run on a thread pool in the API worker that accepted the request, and keep their
state in Postgres, so any worker can serve the polling. On startup `db.mark_interrupted_jobs` marks datasets,
exports, runs and analyses that were mid-flight (not updated for 2 minutes) as failed; a dataset that was only
correlating new targets goes back to `ready` with a warning.

| Job | Entry point | State in |
|---|---|---|
| Dataset pipeline | `processing.process_dataset` (`datasets._generate_and_process` for samples) | `meta.datasets` |
| New-target associations | `datasets.correlate_targets` | `meta.datasets` |
| Export | `exports._run` | `meta.exports` |
| Model run | `runs._run` | `meta.model_runs` |
| Explore analysis | `analysis.runner.run_analysis` | `meta.analyses` |

### Analyses (Explore jobs)

`POST /api/datasets/{id}/analyses` with `{kind, params, recompute}`:

1. `AnalysisParams.normalised(kind)` keeps only the keys that affect that kind; its canonical JSON is `params_key`.
2. `runner.data_version`: for raw data, an md5 over every field's `(field_id, action, semantic_type)`, so any write
   path (single save, bulk, undo, target change) invalidates without hooking into it; for export data,
   `export:<id>`.
3. `runner.find_or_create` (under a lock on the dataset row) returns, in order: a done row for the same key and
   version; a pending or running one; a done row from an older version (returned with `stale: true`, no new job).
   Otherwise, or with `recompute`, it marks older pending/running rows of the same dataset and kind as failed
   (`Superseded`), inserts a row, prunes old results (keeps the current one per key plus the 3 newest others per
   dataset and kind) and submits `run_analysis`.
4. `run_analysis` prepares a sample (`prep.prepare_raw` or `prep.prepare_export`), calls `compute.run_map`,
   `run_tests` or `run_importance`, and stores `result`. Each progress update checks the row is still active and
   stops the job if it was superseded. `NotEnough` failures store `result.reason`
   (`no_target`, `too_few_fields`, `too_few_rows`, `single_class`) for the UI's explanation.

The UI (`components/explore/state.tsx`) POSTs whenever the URL params change and polls `GET /api/analyses/{id}`
every second while the row is active. The Correlations view is not a job: it reads `meta.field_correlations` directly.

## Schema and migrations

`api/app/migrations/NNN_name.sql` files run once each, in name order, on API startup under a Postgres advisory lock,
and are recorded in `meta.schema_migrations`. `api/app/views.sql` is re-run on every start, so views are always
recreated from that file. Bootstrap also (re)creates the `rainforest_ro` role and grants it `SELECT` on `meta` and
`raw`.

| Migration | Adds |
|---|---|
| `001_initial.sql` | `meta.datasets`, `fields`, `field_correlations`, `field_decisions`, `field_settings`, `decision_history`, `exports`, `model_runs` |
| `002_bulk_batches.sql` | `meta.bulk_batches`, `decision_history.batch_id` |
| `003_analyses.sql` | `meta.analyses` |

Child tables reference `meta.datasets` with `ON DELETE CASCADE`; `DELETE /api/datasets/{id}` removes the rows and then
the dataset's Parquet copy, raw table, export and model directories.

## Where to add things

**A model.** Add a `ModelSpec` to `MODELS` in `api/app/modeling/registry.py` (key, label, family, NaN support,
builder, default params, importance source, optional `requires` package). The CLI, `/api/models`,
`/api/exports/{id}/models` and the Train form pick it up. Compatibility rules are in `registry.compatibility`.

**A migration.** Add `api/app/migrations/NNN_description.sql` with the next number, written to be safe to run once
(`IF NOT EXISTS` where it helps). If a view needs the new column, edit `views.sql`. If the table holds job state,
add it to the loop in `db.mark_interrupted_jobs`.

**An analysis.**
1. Add the kind to the `CHECK` on `meta.analyses.kind` (new migration) and to `AnalysisRequest.kind` in `schemas.py`.
2. Add any parameters to `AnalysisParams` and include them in `normalised()` for that kind only.
3. Write `run_<kind>(p: Prepared, …, check)` in `analysis/compute.py`; raise `NotEnough(reason, "Not enough …")` for
   data that can't support it and call `check(fraction, message)` between steps.
4. Dispatch it in `runner.run_analysis`.
5. In the UI: add the kind to `AnalysisKind` and a result type in `types.ts`, a view in `components/explore/` that
   uses `useAnalysis` and `AnalysisState`, and an entry in `VIEWS` / `ViewBody` in `pages/ExplorePage.tsx`. A new
   `NotEnough` reason also needs a message in `NOT_ENOUGH` in `state.tsx`.
6. Tests go in `api/tests/test_explore.py` (unit tests on `compute` with synthetic data; API tests need the database).

**A UI page.** Add the component under `ui/src/pages/`, a `<Route>` under `/d/:datasetId` in `App.tsx` (it gets the
dataset from `useDatasetContext()`, and `DatasetLayout` handles loading, processing and failed states), a `NavLink` in
`DatasetSteps` if it's a workflow step, and its endpoints in `api.ts` / `types.ts`. Invalidate its query keys from
mutations that change what it shows (see the `invalidate` helper in `BulkEditPage.tsx`).

**An endpoint.** Add it to the matching `routers/*.py` module (or a new router registered in `main.py`), with request
bodies in `schemas.py`, and list it in [api.md](api.md).

## Testing and CI

`api/tests` uses pytest. `conftest.py` provides a generated sample Parquet (most tests are pure functions over it) and
the session-scoped `admin_login` fixture, which sets the admin credentials and signs a `TestClient` in. `test_auth.py`
drives the app without its startup hook, so it needs no database. The API tests in `test_explore.py` and
`test_sql_console.py` start the app against `RAINFOREST_TEST_DATABASE_URL` and skip without it. `make test`
runs the suite locally, `make test-db` runs it against a throwaway `postgres:16` container. CI
(`.github/workflows/ci.yml`, on pushes to `main` and pull requests) runs the API tests with a Postgres service, the UI
typecheck and build, and `docker compose build`.
