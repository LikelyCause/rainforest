# HTTP API

Every endpoint the FastAPI app serves (`api/app/main.py` and `api/app/routers/*.py`). Through the UI's nginx the API
is at `http://localhost:8080/api/…`; directly it is on `http://localhost:8000` (interactive docs at `/docs`).

Every path except `/health` and the sign-in endpoints marked *open* below needs the `rf_session` cookie that sign-in
sets, for a user who is still active; without it the answer is `401`. In `oidc` mode with a required setting missing,
everything but `/health` and the open paths answers `503` naming the setting. `POST`, `PUT`, `PATCH` and `DELETE`
from a browser must be same-origin (`403` otherwise), and a request body must be `application/json`
(`multipart/form-data` for the upload) or it gets `415`. A `Host` outside `RAINFOREST_ALLOWED_HOSTS` gets `400`
(`localhost`, `127.0.0.1` and `::1` always pass).

**Private experiments.** A personal experiment (built from one assessor's assessment) is private to that assessor
while its baseline version is `assessing`. Every route answers for anyone else as if it, its exports and its runs
didn't exist: lists leave them out (and the Datasets list's counts and latest export / run don't include them), and
one asked for by id, a file download, a preview or an Explore analysis on its export is `404`. From `review` on it
is visible to everyone (see [Experiments](#experiments-routersexperimentspy)).

Routes that need a lead answer `403` (`"Only a lead can do this."`) for members. Rows a user creates record them:
`created_by` on datasets, exports, model runs and analyses, `user_id` on decision history and bulk batches.

**Whose decisions.** Every route that reads or writes field decisions works, by default, on the dataset's current
baseline version, the latest started (see [Team](#team-routersteampy)). While it is
`assessing`, an assigned assessor reads and writes only their own assessment, and nobody else is ever shown it;
everyone else reads the team decision (the baseline's starting point) and gets `403` on writes. A submitted assessor
gets `409` until they unsubmit. In `review` everyone reads the team decision and only a lead writes it (`403` for
members). A `finalized` version answers writes with `409`. On a version that reopened some fields, the rest are
carried over and locked: writes to them get `409` naming them, whoever asks, and Bulk edit passes over them.

**`view`.** The field routes (`GET …/datasets/{id}/fields`, `GET /api/fields/{id}`, `PUT …/decision`,
`POST …/decisions/bulk` and its undo, `POST …/fields/query`) and Explore take a `view` naming another decision
source. `experiment:<id>` is a team experiment's own copy of its finalized baseline (see
[Experiments](#experiments-routersexperimentspy)): anyone who can see the experiment reads it (`404` otherwise), and
only its owner writes it, while it isn't archived and isn't a baseline's default **Team ·** experiment (`403`; `409`
while archived). A personal experiment has no copy (it follows its assessor's assessment): `409`.
`baseline:<recipe_id>` is a finalized baseline's snapshot, read-only (`409` on writes; `404` for a version of
another dataset or one not finalized). A malformed `view` is `422`. Under either, field rows' targets and leak flags
follow that source's own targets, no field is locked, and nothing is written to history.

Settings are always the baseline's model family's (`tree`, `linear` or `neural`): a request naming another `family`
gets `422`.

Errors are JSON `{"detail": …}`. Validation failures are `422`, sometimes with `{"message", "errors"}` in `detail`.
Long-running work returns `202` with a row to poll; its `status` is `pending`, `running`, `done` or `failed`
(datasets use `pending`, `ingesting`, `profiling`, `correlating`, `ready`, `failed`), with `progress` (0–1),
`status_message` and `error`.

## Meta

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness; runs `SELECT 1`. |
| GET | `/api/options` | Option catalog: semantic types, model families, setting labels, allowed settings per family × type, family notes, default encoder params, `default_family`, non-feature types, `param_rules` (per encoding: each parameter's `integer`, `min` and `max`) and `target_rule` (`types`, `max_levels`). |

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
| GET | `/api/auth/oidc/login` | *Open, oidc mode.* Start sign-in: `303` to the provider's authorization endpoint with `response_type=code`, `state`, `nonce` and a PKCE `S256` challenge, setting the short-lived `rf_oidc` state cookie (`SameSite=Lax`, path `/api/auth/oidc`, 10 minutes). `502` if the provider's discovery can't be read, `503` while the OIDC settings have a problem. | `next` (a same-app path to return to) |
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
| GET | `/api/datasets` | All datasets, newest first, with review counts, export and run counts, `last_export` and `last_run` (`id`, `model`, `model_label`, `status`, and its headline test `metric` and `value`), the current baseline version (`recipe_id`, `recipe_version`, `phase`; `reopened`, the number of reopened fields, or null when every field is open), `baseline` and `baselines` (as in `GET …/datasets/{id}`), `model_family` (the current baseline's, kept by the server) and `my_assessment` (`{reviewed, submitted}` when the viewer assesses the current version, else null; `reviewed` counts only the fields that version works on). | |
| POST | `/api/datasets/upload` | Upload a file and start the pipeline; the dataset's first baseline is Trees v1, `assessing`, nobody assigned and still being set up (see `POST …/baselines`). With `recipe_id`, processing applies that baseline before the dataset is ready, as `POST …/recipes/apply` does (one that can't be used by then leaves a warning instead). `202`. | multipart `file` (`.csv`, `.tsv`, `.txt` with comma, semicolon, tab or pipe separators, detected; `.parquet`, `.pq`; `415` for any other extension, `400` for an empty file), optional `name`, optional `targets` (comma-separated), optional `recipe_id` (a finalized baseline of another dataset; leads only, `403` for members, `404`/`409` for a missing or unfinalized one) |
| POST | `/api/datasets/sample` | Generate a synthetic dataset and start the pipeline. Targets are `column_a`, `column_b`. `422` if `rows × fields` is over `SAMPLE_MAX_CELLS`. `202`. | `rows` (1,000–2,000,000, default 50,000), `fields` (30–400, default 150), `seed`, `target_kind` (`mixed`, `binary`, `multiclass`), `name` |
| GET | `/api/datasets/library` | The real datasets that ship with the app: `key`, `name`, `description`, `targets`, `rows`, `columns`, `source_url`, `citation`. | |
| POST | `/api/datasets/library/{key}` | Add a library dataset: a copy of its file goes through the upload pipeline with its targets set; the first baseline waits to be set up as for an upload. `404` for an unknown key. `202`. | |
| GET | `/api/datasets/sample/limits` | `{max_cells}`: the cap on `rows × fields` for generated samples. | |
| GET | `/api/datasets/{dataset_id}` | One dataset, with `recipe_id`, `recipe_version` and `phase`; `baseline`, the current version (`id`, `family`, `version`, `phase`, `label` like `Trees baseline v2`, and `setup`: a new dataset's first baseline nobody has started on), or null; and `baselines`, one per family that has one, Trees first (`family`, `id`, `version`, `phase` and `label` of its latest version, and `latest_finalized_id`). | |
| PATCH | `/api/datasets/{dataset_id}` | Rename or set the targets. `model_family` is the current baseline's: sending another is `422`. Setting targets needs a `ready` dataset (`409` otherwise) and may start a job for targets with no associations yet. While assessing, targets change only before any assessor is assigned (`409` after); in review only a lead changes them; a finalized version refuses (`409`), and so does a change to a carried-over field's role. A target must be typed binary, categorical or discrete numeric with at most 100 distinct values (`422` naming each field otherwise). | `name`, `target_columns` (exact list in chain order; `[]` for none) |
| DELETE | `/api/datasets/{dataset_id}` | *Lead.* Delete the dataset, its fields, exports, runs, analyses and files. `409` while it, or one of its exports or runs, is still running. Returns counts removed. | |
| GET | `/api/datasets/{dataset_id}/fields` | Lightweight field rows for the Review sidebar (type, confidence, missing, uniques, target association, leakage, decision, reviewed, `locked`: carried over into the current version, always false once it is finalized), with the viewer's decisions. | `view` |

## Fields and decisions (`routers/fields.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/fields/{field_id}` | Field profile, the viewer's decision merged with the baseline's family's settings, that family's suggestion, associations with the source's targets, and `locked` (why the field can't change in the current version, or null). Under `view=experiment:<id>` also `baseline`: the baseline's own decision and `settings` for the field (null when the baseline has no such field). | `family` (only the baseline's), `view` |
| GET | `/api/fields/{field_id}/correlations` | The field's strongest associations. | `limit` (1–200, default 20), `exclude_targets` |
| GET | `/api/fields/{field_id}/history` | Last 50 changes to the team decision, each with `user_id` and `user_name` (null for knock-on changes a background job made). Assessments have no history. | |
| PUT | `/api/fields/{field_id}/decision` | Save the viewer's decision (their assessment, the team decision for a lead in review, or with `view` an experiment's copy) and the baseline's family's settings; validated for that family (numeric fill values must be finite, encoder params within `param_rules`). A type change whose settings don't fit the new type takes the new type's suggested settings. `source` becomes `user` only when keep / drop / target or the type changes, the settings' only when they differ. Retyping the team decision or an experiment's copy to datetime stores the column's detected date format (never for an assessment, which stays private). `action: target` follows the same target rule as `PATCH …/datasets/{id}`; in an experiment it adds or removes the target from the experiment's `target_order`, and a new target gets the leak screen in the background. `422` for another `family`, `409` for a field with no decision. | `view`; body: `family`, `action` (`keep`, `drop`, `target`), `semantic_type`, `imputation`, `impute_value`, `add_missing_indicator`, `transform`, `scaling`, `encoding`, `params`, `notes`, `reviewed` |
| POST | `/api/datasets/{dataset_id}/decisions/bulk` | Apply one change to the viewer's decisions (or with `view` an experiment's copy) on many fields, or preview it (both need write access). Never moves a field into or out of the target role, never touches carried-over (locked) fields, and `action: keep` passes over non-feature types (identifier, constant). Returns `dry_run`, `matched`, `changed`, `unchanged`, `reviewed_changed`, `transitions` (`"keep→drop"`-style keys with counts), `changes` (each `field_id`, `name`, and `before` / `after` with `action`, `semantic_type`, `reviewed`) and `batch_id` (null for a dry run or when nothing changed). | `view`; exactly one of `field_ids` or `filters`; `excluded_ids`; at least one of `accept_suggestion`, `action` (`keep`, `drop`), `reviewed`; `dry_run` |
| POST | `/api/datasets/{dataset_id}/decisions/bulk/{batch_id}/undo` | Restore every field a bulk batch changed (decision and settings). An assessor undoes only their own batches, and an experiment's batches are undone through its `view` (`404` for any other). `409` if already undone, or if a team batch changed fields that are locked in the current version. Returns `restored` (count) and `batch_id` (the undo's own batch for a team batch, null for an assessor's or an experiment's). | `view` |

## Query (`routers/query.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| POST | `/api/datasets/{dataset_id}/fields/query` | Structured search over `meta.v_fields`, or the same columns over the viewer's own assessment while they assess, or over the source `view` names. | `view`; `filters` (≤ 20 of `{column, op, value}`; ops `=`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `contains`, `is_null`, `not_null`; columns from a whitelist), `order_by`, `descending`, `limit` (≤ 5,000, default 500) |
| GET | `/api/sql/examples` | Example queries for the SQL console (`:dataset` is a placeholder the UI fills in). | |
| POST | `/api/sql` | Run one query (`SELECT`, `WITH … SELECT`, `VALUES`, `TABLE`) as the read-only role through a server-side cursor, in a read-only transaction with `SQL_TIMEOUT_MS`. The role can't read the assessment tables, sees only team results in `meta.analyses`, and sees no private experiment, nor its exports, runs, metrics or Explore results (row-level security), whoever asks. Returns `columns`, `rows` (≤ `SQL_MAX_ROWS`), `row_count`, `truncated`, `elapsed_ms`. `400` with `{message, detail}` for anything else, including a second statement; `503` when all console connections are busy. | `sql` |

## Exports (`routers/exports.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets/{dataset_id}/export-preview` | What an export would contain: of an experiment's source with `experiment_id` (a team experiment: its copy of the baseline with its adjustments; a default **Team ·** one: the baseline's snapshot; a personal one: its assessor's assessment as it stands, unreviewed fields counted among them), else of a baseline version, read from its snapshot. The model family is the baseline's. Returns `family`, `recipe` (`id`, `family`, `version`, `phase`, `label`), `source` (`recipe`, `assessment`), `experiment_id`, the counts `kept`, `dropped`, `customized` (kept fields whose settings differ from the suggestion) and `unreviewed`, `targets` (names, in chain order), `kept_by_type`, `warnings` and `blocking` (lists of messages), `last_export` (the viewer's last visible finished export for that family, in that experiment when given: `id`, `created_at`, `status`, `recipe_id`; else null) and `changed_since_export` (the number of fields whose decision or settings for that family differ from what that export was built from; null without one). Leak warnings check the kept fields against the export's own targets; `warnings` also names targets with no stored associations (*No leak check has run for target(s) …*) and, for a team experiment, targets that differ from its baseline's (*Targets differ from …*). With no finalized version yet a preview shows the team decision as it stands, with the version's phase first among the blocking issues. `404` for a dataset with no profiled fields, `422` for a `family` other than the baseline's. | `family`, `recipe_id` (default: the latest finalized version of the current baseline's family, else of any), `experiment_id` |
| POST | `/api/datasets/{dataset_id}/exports` | Start an export in an experiment, prepared for the baseline's model family. A team export (`source: recipe`) is built from a finalized version's snapshot (never the decisions being worked on), in the experiment given or else that version's default Team experiment (`is_default`, named like **Team · Trees v2**; made on first use); in a team experiment the snapshot is its copy with its adjustments, frozen into the export. A personal export (`source: assessment`, or any export into a personal experiment) is built from the experiment's assessor's assessment as it stands now, frozen into the export so it rebuilds the same way; carried-over (locked) fields take the team's values. `409` if a team export's version isn't finalized (or there is none yet), the dataset isn't ready, or the experiment is archived; `403` adding to someone else's personal experiment; `404` for an unknown dataset, a baseline of another dataset or an experiment the viewer can't see; `422` for `source: assessment` without `experiment_id`, a `source` or `recipe_id` that doesn't match the experiment, or blocking issues in the preview (`{"message": "Cannot export", "errors": […]}`), or a `model_family` other than the baseline's. The export records `recipe_id`, `experiment_id` and the `snapshot` it is built from; its config (and manifest) `source`, `recipe_id`, `recipe_version`, `experiment_id`, `baseline` (`recipe_id`, `family`, `version`, `label`), `adjustments` (`count`, `fields`: names, `targets_changed`, `targets`: `{before, after}` or null, and `detail`: each field's changes as in `GET /api/experiments/{id}`) and, for a personal export, `assessor_id`. `202`. | `experiment_id`, `source` (`recipe`, `assessment`; default: the experiment's, else `recipe`), `recipe_id` (default: the experiment's, else the latest finalized version of the current baseline's family, else of any), `model_family` (only the baseline's), `valid_size`, `test_size` (each 0 to < 0.5, sum < 0.9; default 0.15), `seed`, `fit_sample_rows` (default 500,000; lowered for wide exports to fit `EXPORT_FIT_MB`), `chunk_rows` (default 200,000; lowered to fit `EXPORT_CHUNK_MB`) |
| GET | `/api/datasets/{dataset_id}/exports` | The dataset's exports the viewer may see, newest first, each with `recipe_id`, `recipe_family`, `recipe_version`, `experiment_id` and `experiment_name`. | `experiment_id` (only that experiment's) |
| GET | `/api/exports/{export_id}` | One export, with `recipe_family`, `recipe_version`, `experiment_id`, `experiment_name` and its summary (split counts, targets and classes, features by type, NaN flag, train command). | |
| GET | `/api/exports/{export_id}/files/{kind}` | Download a finished export's file. `404` for an unknown kind, `409` until the export is done, `410` if the files were removed. | `kind`: `data` (Parquet), `preprocessor` (joblib), `manifest` (JSON) |

## Models and runs (`routers/runs.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/models` | Registered models: key, label, family, NaN support, installed or not, description, `devices` (`cpu`, `cuda`), `planned` (false); then the planned models (`linear_svm`, `knn`, `cnn`, `rnn`) with `planned: true`, not installed and no devices. | |
| GET | `/api/system` | Training hardware: `gpu` is `null`, or `{available, name, memory_mb, driver, count, reason}`. Detected once per worker, in the background at startup (a request during detection waits for it). | |
| GET | `/api/exports/{export_id}/models` | Every model with whether it can train on this export, and why not (`planned` for a planned model). | |
| POST | `/api/exports/{export_id}/runs` | Train a `ClassifierChain` on a finished export; the run joins the export's experiment. `404` for an export the viewer can't see, `409` if it isn't finished or the experiment is archived, `403` on someone else's personal experiment, `422` for a planned, unknown or uninstalled model or one the export doesn't suit. `202`. | `experiment_id` (optional; `422` unless it is the export's), `model` (default `lightgbm`), `order` (the export's targets to train, each at most once, in chain order: one trains a single model, a subset leaves the rest out; default: all, in export order; `422` otherwise), `cv` (0 chains the ground truth, or 2–10; default 5), `seed`, `params` (overrides of the model's defaults; `422` for a name the model doesn't have or a value its own checks reject), `device` (`cpu` default, or `cuda`: `422` if the model can't, `409` if no usable GPU) |
| GET | `/api/datasets/{dataset_id}/runs` | The dataset's runs the viewer may see, newest first, with `experiment_id`, metrics and report (without top features). Each has `mlflow_experiment_id`, `mlflow_run_id` and `mlflow_url` (`/mlflow/#/experiments/<id>/runs/<id>`), all `null` when the run wasn't logged; `report.mlflow_error` says why when logging failed, and `report.mlflow_pending` marks a run of a private experiment, logged once its baseline version reaches review. | `experiment_id` (only that experiment's) |
| GET | `/api/runs/{run_id}` | One run, full report, with the same MLflow fields. | |
| GET | `/api/runs/{run_id}/files/{kind}` | Download a finished run's file. `404` for an unknown kind or an unfinished run, `410` if the files were removed. | `kind`: `model` (`classifier_chain.joblib`), `report` (`report.json`) |

## Explore (`routers/explore.py`)

| Method | Path | Purpose | Key params |
|---|---|---|---|
| POST | `/api/datasets/{dataset_id}/analyses` | Get or start an analysis. `export:<id>` data must be an export the viewer may see (`404` otherwise). Returns the cached row for these params if current, else the running one, else an out-of-date result (`stale: true`), else a new `pending` row with a job started. `409` if the dataset isn't ready. Raw-data params gain `decisions`, the decision source the result is computed from and cached under: `team` (the baseline in progress), `assessment:<recipe_id>:<user_id>` for an assessor's *kept only* while assessing, or with `view` `experiment:<id>` or `baseline:<recipe_id>`. A new request supersedes running ones of the same kind and source only. | `kind` (`map`, `tests`, `importance`); `view` (raw data: `experiment:<id>` or `baseline:<recipe_id>`, as on the field routes); `params`: `data` (`raw` or `export:<id>`), `kept`, `target`, `sample` (1,000–20,000, default 5,000), `method` (`tsne`, `umap`), `perplexity` (5–100, default 30), `n_neighbors` (5–100, default 15); `recompute` |
| GET | `/api/analyses/{analysis_id}` | Poll one analysis; includes `stale`. Another person's assessment-based result, a result on an experiment's decisions or an export the viewer can't see, is `404`. A run that can't proceed is `failed` with `error` starting "Not enough" and `result.reason` (`no_target`, `too_few_fields`, `too_few_rows`, `single_class`). | |
| GET | `/api/datasets/{dataset_id}/explore/correlations` | Stored associations for a capped, cluster-ordered set of fields (`kept` uses the viewer's own choices while they assess, or the source `view` names). Returns `fields` (id, name, type, action), `order` (field ids) and `cells` (`a`, `b`, `method`, `value`, `strength`, `n`). | `kept`, `view`, `target`, `limit` (2–400, default 60), `add` (comma-separated field ids to include beyond the limit) |

## Team (`routers/team.py`)

The process on a dataset's current baseline version, `assessing → review → finalized`. Responses that change it
return the team state, as `GET …/team` does.

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets/{dataset_id}/team` | The current baseline version as `recipe` (its row: `id`, `dataset_id`, `family`, `version`, `phase`, `scope`, `parent_id`, `created_by`, `created_at`, `finalized_by`, `finalized_at`, `schema_fingerprint`; plus `label` (`Trees baseline v2`), `setup` (a new dataset's first baseline nobody has started on: `POST …/baselines` sets it up in place), `created_by_name`, `finalized_by_name`, `reopened`: how many fields it reopened or null when every field is open, and `reuse`: the summary when it started from another dataset's baseline), `assessors` (each `user_id`, `name`, `assigned_by`, `assigned_at`, `submitted_at`, `reviewed`, `total`, counting only the fields the version works on; progress only, never choices), `me` (`{submitted, reviewed, total}` when the viewer is assigned, else null) and `review`: null while assessing, else the summary `{fields, adopted, disagreements, open}` (the fields the version works on, how many were adopted, how many disagreed, and how many of those the team decision hasn't settled). | |
| POST | `/api/datasets/{dataset_id}/assessors` | *Lead.* Assign an active user and seed their assessment from the team decision: the baseline's starting point (the suggestions, or the baseline it was populated from), or the version before on a later version, with carried-over fields already reviewed (their earlier choices, if they were assigned before, stay). `404` for an unknown user, `409` for a deactivated one, and unless assessing and the dataset is ready. | `user_id` |
| DELETE | `/api/datasets/{dataset_id}/assessors/{user_id}` | *Lead.* Unassign; their choices are kept but don't count. Opens review if everyone left has submitted. `409` unless assessing, `404` if not assigned. | |
| POST | `/api/datasets/{dataset_id}/assessment/submit` | Submit the viewer's assessment. `409` with the number left unless every field the version works on (all, or the reopened ones) is reviewed, `403` if not assigned. The last submission opens review: of those fields, the ones where every assessor chose the same action and type are adopted into the team decision (settings: the most common choice, a tie takes the suggestion) and recorded in history; the rest become the review queue (team decision unreviewed). | |
| POST | `/api/datasets/{dataset_id}/assessment/unsubmit` | Take the submission back. `403` if not assigned, `409` once review has opened. | |
| GET | `/api/datasets/{dataset_id}/review` | The review queue (also after finalizing): `recipe` (as in `…/team`), `summary` (`fields`, `adopted`, `disagreements`, `open`, as `review` in `…/team`) and `queue`, one item per disagreement in column order: `field_id`, `name`, `inferred_type`, `team` (`action`, `semantic_type`), `resolved` (the team decision is reviewed), `comments` (count) and `assessments` (each `user_id`, `user_name`, `action`, `semantic_type`, `notes`). `409` while assessing. | |
| GET | `/api/fields/{field_id}/review` | One field in review: `disagreement` (bool), `assessments` (each `user_id`, `user_name`, `action`, `semantic_type`, `notes` and `settings`: per family, `imputation`, `impute_value`, `add_missing_indicator`, `transform`, `scaling`, `encoding`, `params`, `source`; empty for a carried-over field) and `comments` (each `id`, `user_id`, `user_name`, `body`, `created_at`, oldest first). `404` for an unknown field, `409` while assessing. | |
| POST | `/api/fields/{field_id}/comments` | Add a comment (any user) during review; returns the field's comments, shaped as above. `409` outside review. `201`. | `body` (1–4,000 characters, not blank) |
| POST | `/api/datasets/{dataset_id}/finalize` | *Lead.* Freeze the team decision as the version's `snapshot` (per field name: decision, settings for the baseline's family, inferred type, `null_pct`, `n_unique`; plus `target_order`) and mark it finalized. `409` outside review, while the dataset is processing, or with open disagreements unless `accept_open` (`detail`: `{message, open}`). | `accept_open` |

## Baselines (`routers/baselines.py`, `routers/recipes.py`)

A baseline is a dataset's agreed decisions for one model family (`tree`, `linear`, `neural`), stored as rows of
`meta.recipes`, one per version; versions count per family, and a dataset has at most one baseline per family. At most
one version per dataset is in progress (`assessing` or `review`); starting another baseline or version while one is
gives `409`. Starting any of them loads the working decisions from its starting point. Reusing a finalized baseline
on new data is `reuse.py`. None of these return assessments: a version still being worked appears as its team
decision stands.

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets/{dataset_id}/baselines` | The dataset's baselines, one per family that has one, Trees first: `family`, `label` (`Trees`), `versions` (newest first: `id`, `family`, `version`, `phase`, `scope`, `label`, `parent_id`, `parent_family`, `parent_version`, `parent_dataset_id`, `created_at`, `created_by_name`, `finalized_at`, `finalized_by_name`) and `latest_finalized_id`. `404` for an unknown dataset. | |
| POST | `/api/datasets/{dataset_id}/baselines` | *Lead.* Start a baseline for `family` with its assessors, who are assigned and seeded at once. A new dataset's first baseline (`setup` in the team state) is set up in place: its family, starting point and targets change on that row. Otherwise this is a new version (`scope: all`, `assessing`) of that family's baseline, the first when the family has none, and the version in progress must be finalized first (`409`). `start`: `scratch` (the family's suggestions); `baseline` (a finalized baseline of this dataset, any family, as `parent_id`: its decisions and notes are copied, settings kept where valid for `family`, the rest suggested; every field is still assessed); `recipe` (a finalized baseline of another dataset, only for a new dataset's first baseline, `409` otherwise; as `POST …/recipes/apply` does, taking the source's family and targets; with nothing reopened it opens in `review` and nobody is assigned). `409` unless the dataset is ready; `404` for a `source_recipe_id` that isn't a finalized baseline of this dataset (`start: baseline`) or doesn't exist (`start: recipe`; `409` for an unfinalized one); `422` for an unknown family, no assessors, an unknown or deactivated assessor, no targets (unless `recipe`), no `source_recipe_id` (unless `scratch`), or `start: recipe` with this dataset's own baseline. Returns the team state. | `family`, `start` (default `scratch`), `source_recipe_id`, `targets` (chain order), `assessors` (user ids; the lead only when listed) |
| GET | `/api/recipes` | Every finalized baseline version of every dataset, newest first: `id`, `dataset_id`, `dataset_name`, `family`, `version`, `label`, `field_count`, `finalized_by`, `finalized_by_name`, `finalized_at`. | |
| GET | `/api/datasets/{dataset_id}/recipes` | The dataset's versions, by family (Trees, Linear, Neural) then newest first: `id`, `family`, `version`, `label`, `phase`, `scope` (`all`, or `reopened`: only the reopened fields are worked, the rest are carried over and locked), `reopened` (count, null for `all`), `created_by`(`_name`), `created_at`, `finalized_by`(`_name`), `finalized_at`, `parent_id` with `parent_dataset_id`, `parent_dataset_name`, `parent_family`, `parent_version` (what it was built from; null once that dataset is deleted), and `reuse`. | |
| GET | `/api/datasets/{dataset_id}/recipes/diff` | Fields whose team decision (`action`, `semantic_type`, `notes`) or settings differ between versions `a` and `b`: `a` and `b` (each `id`, `family`, `version`, `phase`, `label`), `fields` (each `name`, `change`: `changed`, `added` or `removed`, and `changes`: `key`, `family` (null for the decision), `before`, `after`) and `targets` (`{before, after}` when the targets or their order differ, else null). Versions of different families compare their decisions only. `404` for a version of another dataset. | `a`, `b` (version ids) |
| POST | `/api/datasets/{dataset_id}/recipes` | *Lead.* Start the next version of `family`'s baseline from its latest finalized version, whose snapshot becomes the working decisions (every field reviewed, its targets set). `409` while a version is in progress or when that family has no finalized version. `reopen` lists the fields to assess again: the version starts `assessing` with scope `reopened` (reopened fields to do, the rest carried over and locked). With none, it starts in `review` with scope `all` and the lead edits the team decision directly, then finalizes. `404` for an unknown dataset, `409` while it is processing, `422` for an unknown family or a field of another dataset. Returns the team state. | `family` (default: the current baseline's), `reopen` (field ids; default none) |
| POST | `/api/datasets/{dataset_id}/recipes/apply` | *Lead.* Start a new dataset's first baseline from a finalized baseline of any dataset; it takes that baseline's family. Only while its first version is assessing with nobody assigned and no assessment made, once (`409` otherwise), on a ready dataset; `404` for an unknown baseline, `409` for an unfinalized one. Fields match by name: a field with the same inferred type and no drift (missing share within 0.10, distinct values within 0.5–2×) whose decision and settings for the family are still valid here is carried over and locked; every other field is reopened. The source's targets that are still valid targets become the dataset's targets. The version gets `parent_id` (the source), scope `reopened` and `reuse`: `source` (`recipe_id`, `dataset_id`, `dataset_name`, `family`, `version`), `carried` (field names), `reopened` (each `name` and `reason`: `new`, `type_changed` with `before` / `after`, `drift` with `drift.null_pct` and/or `drift.n_unique` as `[source, new]`, `target` or `settings` with `problem`), `missing` and `targets` (field names), `targets_not_carried` (`name`, `problem`). With nothing reopened it moves to `review`, for a lead to confirm by finalizing. Returns the team state. | `recipe_id` |

## Experiments (`routers/experiments.py`)

Where exports and runs live (`experiments.py`). `source` is `recipe:<recipe_id>` (a finalized baseline version: a
team experiment) or `assessment:<recipe_id>:<user_id>` (one assessor's assessment: personal). A personal experiment is
`private` while its version is `assessing`: only its assessor sees it, and everyone else gets `404` for it, its
exports and its runs. Experiment rows carry `id`, `dataset_id`, `name`, `owner` (null for the default Team
experiments the app makes, `is_default`, named like **Team · Trees v2**, and for **Imported** ones, which collect
exports and runs that have no experiment) with `owner_name` (the owner's, else a personal experiment's assessor's),
`recipe_id`, `recipe_family`, `recipe_version`, `recipe_phase`, `assessor_id` (personal only), `source`, `notes`,
`created_at`, `archived_at`, `is_default`, `target_order` (a team experiment's targets in chain order) and `private`.

A team experiment has its own copy of its baseline (decisions and the baseline family's settings, made when the
experiment is), read and written through `view=experiment:<id>` on the field routes (see the top of this page). Its
**adjustments** are where the copy differs from the baseline's snapshot. Only its owner adjusts it (`can_adjust`):
not a default Team experiment, an Imported one, a personal one, or an archived one.

| Method | Path | Purpose | Key params |
|---|---|---|---|
| GET | `/api/datasets/{dataset_id}/experiments` | The experiments the viewer may see, archived last, newest first, each with `export_count`, `run_count`, `mlflow_experiment_id` (the dataset's MLflow experiment once one of its runs is logged), `best`: its best finished run on the validation split (`metric`, `value`, `run_id`, `target`: the first target in chain order, scored with `roc_auc`, else `roc_auc_ovr_macro`, else `accuracy`), or null; `adjusted_count`, `targets_changed` and `can_adjust`. | |
| POST | `/api/datasets/{dataset_id}/experiments` | Create an experiment; `owner` is the viewer. A team one needs a finalized version of this dataset (`409` otherwise) and gets its copy of it; a personal one only the viewer's own assessment of a version they assess (`403` otherwise). `404` for a baseline of another dataset, `422` for a malformed `source`. `201`. | `name`, `source`, `notes` |
| GET | `/api/experiments/{experiment_id}` | One experiment with `best`, `can_adjust`, `adjustments` (`fields`: each `name`, `change` and `changes` as in `…/recipes/diff`; `targets`: `{before, after}` when its targets or their order differ from the baseline's, else null; `count`; `targets_changed`; empty for a personal experiment), its `exports` and its `runs`, newest first, shaped as the lists above. | |
| POST | `/api/experiments/{experiment_id}/adjustments/revert` | Put fields of a team experiment back to its baseline: with `field_ids`, those fields (its target order keeps its own targets, then any the revert brings back); with null, every field and the targets. Its owner only, as for adjusting (`403`, or `409` for a personal or archived one); `404` for one the viewer can't see. Returns the experiment with `adjustments` and `can_adjust`. | `field_ids` (or null) |
| PATCH | `/api/experiments/{experiment_id}` | Rename, set the notes, archive or unarchive (`archived`). The owner, the assessor of a personal one, or a lead; `403` for anyone else. An archived experiment takes no new exports or runs (`409`). | `name`, `notes`, `archived` |
