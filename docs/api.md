# HTTP API

Every endpoint the FastAPI app serves (`api/app/main.py` and `api/app/routers/*.py`). Through the UI's nginx the API
is at `http://localhost:8080/api/…`; directly it is on `http://localhost:8000` (interactive docs at `/docs`).

Every path except `/health` and the sign-in endpoints marked *open* below needs the `rf_session` cookie that sign-in
sets, for a user who is still active; without it the answer is `401`. In `oidc` mode with a required setting missing,
everything but `/health` and the open paths answers `503` naming the setting. `POST`, `PUT`, `PATCH` and `DELETE`
from a browser must be same-origin (`403` otherwise), and a request body must be `application/json`
(`multipart/form-data` for the upload) or it gets `415`. A `Host` outside `RAINFOREST_ALLOWED_HOSTS` gets `400`.

Routes that need a lead answer `403` (`"Only a lead can do this."`) for members. Rows a user creates record them:
`created_by` on datasets, exports, model runs and analyses, `user_id` on decision history and bulk batches.

Errors are JSON `{"detail": …}`. Validation failures are `422`, sometimes with `{"message", "errors"}` in `detail`.
Long-running work returns `202` with a row to poll; its `status` is `pending`, `running`, `done` or `failed`
(datasets use `pending`, `ingesting`, `profiling`, `correlating`, `ready`, `failed`), with `progress` (0–1),
`status_message` and `error`.

## Meta

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness; runs `SELECT 1`. |
| GET | `/api/options` | Option catalog: semantic types, model families, setting labels, allowed settings per family × type, family notes, default encoder params, non-feature types, `param_rules` (per encoding: each parameter's `integer`, `min` and `max`) and `target_rule` (`types`, `max_levels`). |

## Sign-in (`auth.py`)

The session cookie is `HttpOnly`, `SameSite=Strict`, lasts 7 days and is `Secure` over HTTPS.

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/auth/config` | *Open.* What the sign-in page needs: `mode` (`pick`, `oidc`), `provider_name`, `lead_group`, `member_group`, and `problem` (why OIDC can't work yet, else null). | |
| GET | `/api/auth/me` | The signed-in user: `id`, `name`, `email`, `role` (`lead`, `member`), `mode`. | |
| GET, HEAD, POST, PUT, PATCH, DELETE | `/api/auth/check` | `204` when the session is valid, `401` otherwise, for nginx `auth_request` on `/mlflow/` (no body needed; a cross-origin unsafe method still gets `403`). | |
| POST | `/api/auth/logout` | *Open.* Clear the session cookie. Returns `{logout_url}`: in `oidc` mode the provider's end-session URL (with `client_id` and `post_logout_redirect_uri` = `<origin>/login`) for the browser to visit next, else null. | |
| GET | `/api/auth/users` | *Open, pick mode* (`404` otherwise). Active users for the picker: `id`, `name`, `role` only. | |
| POST | `/api/auth/pick` | *Open, pick mode* (`404` otherwise). Sign in as an active user (`404` for an unknown or deactivated one); returns it as `/me` does. | body: `user_id` |
| GET | `/api/auth/oidc/login` | *Open, oidc mode.* Start sign-in: `303` to the provider's authorization endpoint with `response_type=code`, `state`, `nonce` and a PKCE `S256` challenge, setting the short-lived `rf_oidc` state cookie (`SameSite=Lax`, path `/api/auth/oidc`, 10 minutes). `502` if the provider's discovery can't be read. | `next` (a same-app path to return to) |
| GET | `/api/auth/oidc/callback` | *Open, oidc mode.* The provider's redirect back. Checks `state` against the cookie, exchanges the code (with the PKCE verifier and client secret), verifies the ID token (signature from the provider's JWKS, issuer, audience, expiry, nonce), creates or refreshes the user and sets the session: `303` to `next`. Failures `303` to `/login?error=state` (missing, expired or mismatched state), `error=provider` (the provider returned an error) or `error=token` (exchange or verification failed; the API logs why); a user in neither group to `/forbidden?reason=no_group`, a deactivated one to `/forbidden?reason=deactivated`. | `code`, `state`, `error` |

## Users (`routers/users.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/users` | Every user: `id`, `name`, `email`, `role`, `active`, `sso` (signed in through OIDC), `created_at`, `last_seen_at`. Any signed-in user. | |
| POST | `/api/users` | *Lead.* Add a user (pick mode; `409` in oidc mode, where users are created on first sign-in). `201`. | `name`, `email`, `role` (default `member`) |
| PATCH | `/api/users/{user_id}` | *Lead.* Rename, change email or role, deactivate or reactivate. A role change or deactivation ends the user's sessions. `409` for demoting or deactivating the last active lead, and in oidc mode for anything but `active`. | `name`, `email`, `role`, `active` |

## Datasets (`routers/datasets.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets` | All datasets, newest first, with review counts, export and run counts, the latest export and the latest run's headline test metric. | |
| POST | `/api/datasets/upload` | Upload a file and start the pipeline. `202`. | multipart `file` (`.csv`, `.txt`, `.parquet`, `.pq`; `.tsv` is refused), optional `name`, optional `targets` (comma-separated) |
| POST | `/api/datasets/sample` | Generate a synthetic dataset and start the pipeline. Targets are `column_a`, `column_b`. `422` if `rows × fields` is over `SAMPLE_MAX_CELLS`. `202`. | `rows` (1,000–2,000,000, default 50,000), `fields` (30–400, default 150), `seed`, `target_kind` (`mixed`, `binary`, `multiclass`), `name` |
| GET | `/api/datasets/sample/limits` | `{max_cells}`: the cap on `rows × fields` for generated samples. | |
| GET | `/api/datasets/{dataset_id}` | One dataset. | |
| PATCH | `/api/datasets/{dataset_id}` | Rename, switch the active model family, or set the targets. Setting targets needs a `ready` dataset (`409` otherwise) and may start a job for targets with no associations yet. A target must be typed binary, categorical or discrete numeric with at most 100 distinct values (`422` naming each field otherwise). | `name`, `model_family` (`tree`, `linear`, `neural`), `target_columns` (exact list in chain order; `[]` for none) |
| DELETE | `/api/datasets/{dataset_id}` | *Lead.* Delete the dataset, its fields, exports, runs, analyses and files. `409` while it, or one of its exports or runs, is still running. Returns counts removed. | |
| GET | `/api/datasets/{dataset_id}/fields` | Lightweight field rows for the Review sidebar (type, confidence, missing, uniques, target association, leakage, decision, reviewed). | |

## Fields and decisions (`routers/fields.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/fields/{field_id}` | Field profile, the decision merged with one family's settings, that family's suggestion, and target associations. | `family` (default: the dataset's active family) |
| GET | `/api/fields/{field_id}/correlations` | The field's strongest associations. | `limit` (1–200, default 20), `exclude_targets` |
| GET | `/api/fields/{field_id}/history` | Last 50 decision changes, each with `user_id` and `user_name` (null for knock-on changes a background job made). | |
| PUT | `/api/fields/{field_id}/decision` | Save the shared decision and one family's settings; validated per family (numeric fill values must be finite, encoder params within `param_rules`). `source` becomes `user` only when keep / drop / target or the type changes, the settings' only when they differ. Retyping to datetime stores the column's detected date format. `action: target` follows the same target rule as `PATCH …/datasets/{id}`. A type change re-derives the other families' settings. | body: `family`, `action` (`keep`, `drop`, `target`), `semantic_type`, `imputation`, `impute_value`, `add_missing_indicator`, `transform`, `scaling`, `encoding`, `params`, `notes`, `reviewed` |
| POST | `/api/datasets/{dataset_id}/decisions/bulk` | Apply one change to many fields, or preview it. Never moves a field into or out of the target role. Returns matched / changed / unchanged counts, action transitions, per-field changes and a `batch_id`. | exactly one of `field_ids` or `filters`; `excluded_ids`; at least one of `accept_suggestion`, `action` (`keep`, `drop`), `reviewed`; `dry_run` |
| POST | `/api/datasets/{dataset_id}/decisions/bulk/{batch_id}/undo` | Restore every field a bulk batch changed (decision and all families' settings). `409` if already undone. | |

## Query (`routers/query.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| POST | `/api/datasets/{dataset_id}/fields/query` | Structured search over `meta.v_fields`. | `filters` (≤ 20 of `{column, op, value}`; ops `=`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `contains`, `is_null`, `not_null`; columns from a whitelist), `order_by`, `descending`, `limit` (≤ 5,000, default 500) |
| GET | `/api/sql/examples` | Example queries for the SQL console (`:dataset` is a placeholder the UI fills in). | |
| POST | `/api/sql` | Run one query (`SELECT`, `WITH … SELECT`, `VALUES`, `TABLE`) as the read-only role through a server-side cursor, in a read-only transaction with `SQL_TIMEOUT_MS`. Returns columns, rows (≤ `SQL_MAX_ROWS`), `truncated`, `elapsed_ms`. `400` with `{message, detail}` for anything else, including a second statement; `503` when all console connections are busy. | `sql` |

## Exports (`routers/exports.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets/{dataset_id}/export-preview` | What an export would contain now: kept / dropped / targets / customised counts, kept fields by type, warnings, blocking issues, and decisions changed since the last finished export for that family. | `family` |
| POST | `/api/datasets/{dataset_id}/exports` | Start an export. `409` if the dataset isn't ready, `422` if the preview has blocking issues. `202`. | `model_family`, `valid_size`, `test_size` (each < 0.5, sum < 0.9; default 0.15), `seed`, `fit_sample_rows` (default 500,000; lowered for wide exports to fit `EXPORT_FIT_MB`), `chunk_rows` (default 200,000; lowered to fit `EXPORT_CHUNK_MB`) |
| GET | `/api/datasets/{dataset_id}/exports` | The dataset's exports, newest first. | |
| GET | `/api/exports/{export_id}` | One export, with its summary (split counts, targets and classes, features by type, NaN flag, train command). | |
| GET | `/api/exports/{export_id}/files/{kind}` | Download a finished export's file. `410` if the files were removed. | `kind`: `data` (Parquet), `preprocessor` (joblib), `manifest` (JSON) |

## Models and runs (`routers/runs.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/models` | Registered models: key, label, family, NaN support, installed or not, description, `devices` (`cpu`, `cuda`). | |
| GET | `/api/system` | Training hardware: `gpu` is `null`, or `{available, name, memory_mb, driver, count, reason}`. Detected once per worker, in the background at startup (a request during detection waits for it). | |
| GET | `/api/exports/{export_id}/models` | Every model with whether it can train on this export, and why not. | |
| POST | `/api/exports/{export_id}/runs` | Train a `ClassifierChain` on a finished export. `202`. | `model` (default `lightgbm`), `order` (permutation of the export's targets), `cv` (0 chains the ground truth, or 2–10; default 5), `seed`, `params` (overrides of the model's defaults; `422` for a name the model doesn't have or a value its own checks reject), `device` (`cpu` default, or `cuda`: `422` if the model can't, `409` if no usable GPU) |
| GET | `/api/datasets/{dataset_id}/runs` | The dataset's runs, newest first, with metrics and report (without top features). Each has `mlflow_experiment_id`, `mlflow_run_id` and `mlflow_url` (`/mlflow/#/experiments/<id>/runs/<id>`), all `null` when the run wasn't logged; `report.mlflow_error` says why when logging failed. | |
| GET | `/api/runs/{run_id}` | One run, full report, with the same MLflow fields. | |
| GET | `/api/runs/{run_id}/files/{kind}` | Download a finished run's file. | `kind`: `model` (`classifier_chain.joblib`), `report` (`report.json`) |

## Explore (`routers/explore.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| POST | `/api/datasets/{dataset_id}/analyses` | Get or start an analysis. Returns the cached row for these params if current, else the running one, else an out-of-date result (`stale: true`), else a new `pending` row with a job started. `409` if the dataset isn't ready. | `kind` (`map`, `tests`, `importance`); `params`: `data` (`raw` or `export:<id>`), `kept`, `target`, `sample` (1,000–20,000, default 5,000), `method` (`tsne`, `umap`), `perplexity` (5–100, default 30), `n_neighbors` (5–100, default 15); `recompute` |
| GET | `/api/analyses/{analysis_id}` | Poll one analysis; includes `stale`. A run that can't proceed is `failed` with `error` starting "Not enough" and `result.reason` (`no_target`, `too_few_fields`, `too_few_rows`, `single_class`). | |
| GET | `/api/datasets/{dataset_id}/explore/correlations` | Stored associations for a capped, cluster-ordered set of fields. Returns `fields` (id, name, type, action), `order` (field ids) and `cells` (`a`, `b`, `method`, `value`, `strength`, `n`). | `kept`, `target`, `limit` (2–400, default 60), `add` (comma-separated field ids to include beyond the limit) |
