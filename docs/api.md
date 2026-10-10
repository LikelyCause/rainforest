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

**Whose decisions.** Every route that reads or writes field decisions works on the dataset's current recipe version
(see [Team](#team-routersteampy)). While it is `assessing`, an assigned assessor reads and writes only their own
assessment, and nobody else is ever shown it; everyone else reads the team decision (still the suggestions) and gets
`403` on writes. A submitted assessor gets `409` until they unsubmit. In `review` everyone reads the team decision and
only a lead writes it (`403` for members). A `finalized` version answers writes with `409`. On a version that
reopened some fields, the rest are carried over and locked: writes to them get `409` naming them, whoever asks, and
Bulk edit passes over them.

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
| GET | `/api/datasets` | All datasets, newest first, with review counts, export and run counts, the latest export and the latest run's headline test metric, the current recipe (`recipe_id`, `recipe_version`, `phase`, `scope`; `reopened`, the number of reopened fields, or null when every field is open) and `my_assessment` (`{reviewed, submitted}` when the viewer assesses the current version, else null; `reviewed` counts only the fields that version works on). | |
| POST | `/api/datasets/upload` | Upload a file and start the pipeline; the dataset starts at recipe v1, `assessing`. With `recipe_id`, processing applies that recipe before the dataset is ready, as `POST …/recipes/apply` does (a recipe that can't be used by then leaves a warning instead). `202`. | multipart `file` (`.csv`, `.txt`, `.parquet`, `.pq`; `.tsv` is refused), optional `name`, optional `targets` (comma-separated), optional `recipe_id` (a finalized recipe; leads only, `403` for members, `404`/`409` for a missing or unfinalized one) |
| POST | `/api/datasets/sample` | Generate a synthetic dataset and start the pipeline. Targets are `column_a`, `column_b`. `422` if `rows × fields` is over `SAMPLE_MAX_CELLS`. `202`. | `rows` (1,000–2,000,000, default 50,000), `fields` (30–400, default 150), `seed`, `target_kind` (`mixed`, `binary`, `multiclass`), `name` |
| GET | `/api/datasets/sample/limits` | `{max_cells}`: the cap on `rows × fields` for generated samples. | |
| GET | `/api/datasets/{dataset_id}` | One dataset, with `recipe_id`, `recipe_version` and `phase`. | |
| PATCH | `/api/datasets/{dataset_id}` | Rename, switch the active model family, or set the targets. Setting targets needs a `ready` dataset (`409` otherwise) and may start a job for targets with no associations yet. While assessing, targets change only before any assessor is assigned (`409` after); in review only a lead changes them; a finalized version refuses (`409`), and so does a change to a carried-over field's role. A target must be typed binary, categorical or discrete numeric with at most 100 distinct values (`422` naming each field otherwise). | `name`, `model_family` (`tree`, `linear`, `neural`), `target_columns` (exact list in chain order; `[]` for none) |
| DELETE | `/api/datasets/{dataset_id}` | *Lead.* Delete the dataset, its fields, exports, runs, analyses and files. `409` while it, or one of its exports or runs, is still running. Returns counts removed. | |
| GET | `/api/datasets/{dataset_id}/fields` | Lightweight field rows for the Review sidebar (type, confidence, missing, uniques, target association, leakage, decision, reviewed, `locked`: carried over into the current version), with the viewer's decisions. | |

## Fields and decisions (`routers/fields.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/fields/{field_id}` | Field profile, the viewer's decision merged with one family's settings, that family's suggestion, target associations, and `locked` (why the field can't change in the current version, or null). | `family` (default: the dataset's active family) |
| GET | `/api/fields/{field_id}/correlations` | The field's strongest associations. | `limit` (1–200, default 20), `exclude_targets` |
| GET | `/api/fields/{field_id}/history` | Last 50 changes to the team decision, each with `user_id` and `user_name` (null for knock-on changes a background job made). Assessments have no history. | |
| PUT | `/api/fields/{field_id}/decision` | Save the viewer's decision (their assessment, or the team decision for a lead in review) and one family's settings; validated per family (numeric fill values must be finite, encoder params within `param_rules`). `source` becomes `user` only when keep / drop / target or the type changes, the settings' only when they differ. Retyping to datetime stores the column's detected date format. `action: target` follows the same target rule as `PATCH …/datasets/{id}`. A type change re-derives the other families' settings. | body: `family`, `action` (`keep`, `drop`, `target`), `semantic_type`, `imputation`, `impute_value`, `add_missing_indicator`, `transform`, `scaling`, `encoding`, `params`, `notes`, `reviewed` |
| POST | `/api/datasets/{dataset_id}/decisions/bulk` | Apply one change to the viewer's decisions on many fields, or preview it (both need write access). Never moves a field into or out of the target role, and never touches carried-over (locked) fields. Returns matched / changed / unchanged counts, action transitions, per-field changes and a `batch_id`. | exactly one of `field_ids` or `filters`; `excluded_ids`; at least one of `accept_suggestion`, `action` (`keep`, `drop`), `reviewed`; `dry_run` |
| POST | `/api/datasets/{dataset_id}/decisions/bulk/{batch_id}/undo` | Restore every field a bulk batch changed (decision and all families' settings). An assessor undoes only their own batches (`404` for anyone else's). `409` if already undone, or if the batch changed fields now locked. | |

## Query (`routers/query.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| POST | `/api/datasets/{dataset_id}/fields/query` | Structured search over `meta.v_fields`, or the same columns over the viewer's own assessment while they assess. | `filters` (≤ 20 of `{column, op, value}`; ops `=`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `contains`, `is_null`, `not_null`; columns from a whitelist), `order_by`, `descending`, `limit` (≤ 5,000, default 500) |
| GET | `/api/sql/examples` | Example queries for the SQL console (`:dataset` is a placeholder the UI fills in). | |
| POST | `/api/sql` | Run one query (`SELECT`, `WITH … SELECT`, `VALUES`, `TABLE`) as the read-only role through a server-side cursor, in a read-only transaction with `SQL_TIMEOUT_MS`. The role can't read the assessment tables and sees only team results in `meta.analyses`. Returns columns, rows (≤ `SQL_MAX_ROWS`), `truncated`, `elapsed_ms`. `400` with `{message, detail}` for anything else, including a second statement; `503` when all console connections are busy. | `sql` |

## Exports (`routers/exports.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets/{dataset_id}/export-preview` | What an export of a recipe version would contain, read from its snapshot: kept / dropped / targets / customised (settings that differ from the suggestion) counts, kept fields by type, warnings, blocking issues, `recipe` (`id`, `version`, `phase`), `last_export` (the last finished export for that family, with its `recipe_id`) and `changed_since_export` (fields whose decision or settings for that family differ from that export's version). With no finalized version yet it shows the team decision as it stands, with the version's phase first among the blocking issues. | `family`, `recipe_id` (default: the latest finalized version) |
| POST | `/api/datasets/{dataset_id}/exports` | Start a team export of a finalized recipe version, built from its snapshot (never the decisions being worked on). `409` if the version isn't finalized (or there is none yet) or the dataset isn't ready, `404` for a recipe of another dataset, `422` if the preview has blocking issues. The export records `recipe_id`; its config (and manifest) `recipe_id` and `recipe_version`. `202`. | `source` (`recipe`, the default and only value so far), `recipe_id` (default: the latest finalized version), `model_family`, `valid_size`, `test_size` (each < 0.5, sum < 0.9; default 0.15), `seed`, `fit_sample_rows` (default 500,000; lowered for wide exports to fit `EXPORT_FIT_MB`), `chunk_rows` (default 200,000; lowered to fit `EXPORT_CHUNK_MB`) |
| GET | `/api/datasets/{dataset_id}/exports` | The dataset's exports, newest first, each with `recipe_id` and `recipe_version`. | |
| GET | `/api/exports/{export_id}` | One export, with `recipe_version` and its summary (split counts, targets and classes, features by type, NaN flag, train command). | |
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
| POST | `/api/datasets/{dataset_id}/analyses` | Get or start an analysis. Returns the cached row for these params if current, else the running one, else an out-of-date result (`stale: true`), else a new `pending` row with a job started. `409` if the dataset isn't ready. Raw-data params gain `decisions`, the decision source the result is computed from and cached under: `team`, or `assessment:<recipe_id>:<user_id>` for an assessor's *kept only* while assessing. A new request supersedes running ones of the same kind and source only. | `kind` (`map`, `tests`, `importance`); `params`: `data` (`raw` or `export:<id>`), `kept`, `target`, `sample` (1,000–20,000, default 5,000), `method` (`tsne`, `umap`), `perplexity` (5–100, default 30), `n_neighbors` (5–100, default 15); `recompute` |
| GET | `/api/analyses/{analysis_id}` | Poll one analysis; includes `stale`. Another person's assessment-based result is `404`. A run that can't proceed is `failed` with `error` starting "Not enough" and `result.reason` (`no_target`, `too_few_fields`, `too_few_rows`, `single_class`). | |
| GET | `/api/datasets/{dataset_id}/explore/correlations` | Stored associations for a capped, cluster-ordered set of fields (`kept` uses the viewer's own choices while they assess). Returns `fields` (id, name, type, action), `order` (field ids) and `cells` (`a`, `b`, `method`, `value`, `strength`, `n`). | `kept`, `target`, `limit` (2–400, default 60), `add` (comma-separated field ids to include beyond the limit) |

## Team (`routers/team.py`)

The process on a dataset's current recipe version, `assessing → review → finalized`. Responses that change it return
the team state, as `GET …/team` does.

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets/{dataset_id}/team` | The current `recipe` (with `scope`, `created_by_name`, `finalized_by_name`, `reopened`: how many fields it reopened or null when every field is open, and `reuse`: the summary when it started from a recipe), `assessors` (each `user_id`, `name`, `assigned_at`, `submitted_at`, `reviewed`, `total`, counting only the fields the version works on; progress only, never choices), `me` (`{submitted, reviewed, total}` when the viewer is assigned, else null) and, from review on, `review` (`fields`, `adopted`, `disagreements`, `open`). | |
| POST | `/api/datasets/{dataset_id}/assessors` | *Lead.* Assign an active user and seed their assessment from the team decision: the suggestions on a new dataset, the version before on a later one, with carried-over fields already reviewed (their earlier choices, if they were assigned before, stay). `409` unless assessing and the dataset is ready. | `user_id` |
| DELETE | `/api/datasets/{dataset_id}/assessors/{user_id}` | *Lead.* Unassign; their choices are kept but don't count. Opens review if everyone left has submitted. `409` unless assessing, `404` if not assigned. | |
| POST | `/api/datasets/{dataset_id}/assessment/submit` | Submit the viewer's assessment. `409` with the number left unless every field the version works on (all, or the reopened ones) is reviewed, `403` if not assigned. The last submission opens review: of those fields, the ones where every assessor chose the same action and type are adopted into the team decision (settings: the most common choice per family, a tie takes the suggestion) and recorded in history; the rest become the review queue (team decision unreviewed). | |
| POST | `/api/datasets/{dataset_id}/assessment/unsubmit` | Take the submission back. `409` once review has opened. | |
| GET | `/api/datasets/{dataset_id}/review` | The review queue: each disagreement with `team` (action, type), `resolved` (the team decision is reviewed), `comments` (count) and every assessor's `action`, `semantic_type` and `notes`; plus `summary`. `409` while assessing. | |
| GET | `/api/fields/{field_id}/review` | One field in review: `disagreement`, each assessor's decision, notes and `settings` for every family, and the `comments`. `409` while assessing. | |
| POST | `/api/fields/{field_id}/comments` | Add a comment (any user) during review; returns the field's comments. `409` outside review. `201`. | `body` (1–4,000 characters, not blank) |
| POST | `/api/datasets/{dataset_id}/finalize` | *Lead.* Freeze the team decision as the version's `snapshot` (per field name: decision, settings for every family, inferred type, `null_pct`, `n_unique`; plus `target_order`) and mark it finalized. `409` outside review, while the dataset is processing, or with open disagreements unless `accept_open`. | `accept_open` |

## Recipes (`routers/recipes.py`)

Versions of a dataset's decisions, and reusing a finalized recipe on new data (`reuse.py`). None of these return
assessments: a version still being worked appears as its team decision stands.

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/recipes` | Every finalized recipe version of every dataset, newest first: `id`, `dataset_id`, `dataset_name`, `version`, `field_count`, `finalized_by`, `finalized_by_name`, `finalized_at`. | |
| GET | `/api/datasets/{dataset_id}/recipes` | The dataset's versions, newest first: `id`, `version`, `phase`, `scope` (`all`, or `reopened`: only the reopened fields are worked, the rest are carried over and locked), `reopened` (count, null for `all`), `created_by`(`_name`), `created_at`, `finalized_by`(`_name`), `finalized_at`, `parent_id` with `parent_dataset_id`, `parent_dataset_name`, `parent_version` (what it was built from; null once that dataset is deleted), and `reuse`. | |
| GET | `/api/datasets/{dataset_id}/recipes/diff` | Fields whose team decision (`action`, `semantic_type`, `notes`) or settings (per family) differ between versions `a` and `b`: `fields` (each `name`, `change`: `changed`, `added` or `removed`, and `changes`: `key`, `family` (null for the decision), `before`, `after`) and `targets` (`{before, after}` when the targets or their order differ, else null). `404` for an unknown version. | `a`, `b` (version numbers), `family` (only that family's settings) |
| POST | `/api/datasets/{dataset_id}/recipes` | *Lead.* Start the next version from the latest one, which must be finalized (`409` otherwise). `reopen` lists the fields to assess again: the version starts `assessing` with scope `reopened` (reopened fields to do, the rest carried over and locked). With none, it starts in `review` with scope `all` and the lead edits the team decision directly, then finalizes. `422` for a field of another dataset. Returns the team state. | `reopen` (field ids; default none) |
| POST | `/api/datasets/{dataset_id}/recipes/apply` | *Lead.* Start a new dataset from a finalized recipe of any dataset. Only while its first version is assessing with nobody assigned and no assessment made, once (`409` otherwise), on a ready dataset. Fields match by name: a field with the same inferred type and no drift (missing share within 0.10, distinct values within 0.5–2×) whose decision and settings are still valid here is carried over and locked; every other field is reopened. The recipe's targets that are still valid targets become the dataset's targets. The version gets `parent_id` (the recipe), scope `reopened` and `reuse`: `source` (`recipe_id`, `dataset_id`, `dataset_name`, `version`), `carried`, `reopened` (each `name` and `reason`: `new`, `type_changed` with `before` / `after`, `drift` with `drift.null_pct` and/or `drift.n_unique` as `[recipe, new]`, `target` or `settings` with `problem`), `missing`, `targets`, `targets_not_carried` (`name`, `problem`). With nothing reopened it moves to `review`, for a lead to confirm by finalizing. Returns the team state. | `recipe_id` |
