# Development

How to run Rainforest from source, test it, and change it. For how the code fits together, see
[Architecture](architecture.md). For the HTTP endpoints, see [API](api.md).

## Running the API and UI locally

Each block starts from the repository root.

### API

The API runs against the compose Postgres. Start it with `docker compose up -d db`. Use the user and password from
`.env`.

```bash
cd api && python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
DATABASE_URL=postgresql://rainforest:<POSTGRES_PASSWORD>@localhost:5432/rainforest DATA_DIR=./data \
  uvicorn app.main:app --reload
```

Outside Compose, `MLFLOW_TRACKING_URI` defaults to empty, so a local API doesn't log to MLflow unless you set it.

### UI

The dev server proxies `/api` to `:8000`. Set `API_URL` to point it elsewhere.

```bash
cd ui && npm ci && npm run dev    # http://localhost:5173
npm run typecheck && npm test
```

### A sample file

`make sample-csv` writes a 200k-row sample CSV to `./sample.csv`, for testing the upload path. It needs the stack
running.

## Tests

API tests live in `api/tests`. Most need no database. These include:

- the pick-mode sign-in and request-check tests in `test_auth.py`
- the OIDC redirect and ID-token checks in `test_oidc.py`, against an in-process provider (`tests/oidc_provider.py`)

The API tests that need a database run against an empty Postgres. They skip unless `RAINFOREST_TEST_DATABASE_URL` is
set. They use the fixtures in `tests/team_helpers.py`, and their module docstrings say "Needs
RAINFOREST_TEST_DATABASE_URL". They include the OIDC sign-in flows in `test_oidc.py` and the job tests in
`test_jobs.py`. They sign in with the `login_as(client, role)` fixture in `conftest.py`.

`test_mlflow.py` logs to a temporary file store, so it needs no MLflow server. Its database-backed tests follow the
same rule. One more test runs against a real tracking server when `MLFLOW_TEST_URI` is set. CI runs one.

| Command | What it runs |
|---|---|
| `make test` | the API suite locally; database-backed tests skip. Needs `pip install -r api/requirements-dev.txt`. |
| `make test-db` | the full API suite against a throwaway `postgres:16` container. Needs a local Docker daemon. |
| `RAINFOREST_TEST_DATABASE_URL=postgresql://user:pw@host:5432/db pytest` (from `api/`) | the full suite against your own empty database |
| `cd ui && npm test` | the UI tests (Vitest, no browser needed) |

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests. It runs:

- the API tests, with a Postgres service (so nothing skips) and an MLflow server container
- the UI typecheck, tests and build
- `docker compose build` for the images

## Schema changes

Add a migration as `api/app/migrations/NNN_name.sql`. Views go in `api/app/views.sql`. See
[SQL](sql.md#migrations-and-views) for how they are applied.

## Adding a model

See [Adding a model](experiments-and-training.md#adding-a-model).

## Rebuilding the sample library

The sample library's files are in `api/app/sample_library/data/`. Rebuild them from their sources with:

```bash
docker compose exec api python -m app.sample_library.build
```

You can also run the same module from `api/` locally. Concrete strength ships as an `.xls` file, so rebuilding it
needs `pip install fastexcel`. The app itself doesn't need it.
