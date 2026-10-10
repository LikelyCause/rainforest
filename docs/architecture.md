# Architecture

A developer's map of Rainforest: what runs where, how a request and a dataset move through the system, and where to
add things. For endpoints see [api.md](api.md); for usage see the [README](../README.md).

## Components

| Service | Code | Role |
|---|---|---|
| `ui` | `ui/` | React 18 + TypeScript, Vite, TanStack Query, ECharts. Built to static files and served by nginx (`ui/nginx.conf`), which proxies `/api/` to the API and `/mlflow/` to MLflow (behind `auth_request`). |
| `mlflow` | `mlflow/` | MLflow 3.17 tracking server (the official image plus psycopg). Backend store: database `mlflow` in `db`, created by `mlflow/start.py` on every start if missing. Artifacts on the `mlartifacts` volume, served through the server (`--serve-artifacts`). Everything, REST API included, is under `--static-prefix /mlflow`. No published port. |
| `api` | `api/app/` | FastAPI on uvicorn (`API_WORKERS` processes). Owns the schema, runs every background job in-process. |
| `db` | Postgres 16 | `meta.*` holds metadata, decisions, jobs and results; `raw.ds_<id>` holds each dataset as a typed table. |
| `keycloak` | `keycloak/` | Optional (`keycloak` Compose profile): a dev-mode Keycloak with the `rainforest` realm baked in, as a reference OIDC provider; its database `keycloak` in `db` is created by the one-shot `keycloak-db` service. |
| `/data` volume | `appdata` | `uploads/` (transient), `datasets/<id>/raw.parquet`, `exports/<id>/`, `models/run<id>_<model>_<ts>/` (CLI runs: `models/export<id>_<model>_<ts>/`), `session_secret` when generated. |

Compose reads `.env` for every setting, including `COMPOSE_FILE` (`docker-compose.gpu.yml` adds the CUDA build of the
API and the host's GPUs) and `COMPOSE_PROFILES` (`keycloak`); the Makefile's `up-gpu` and `up-keycloak` add those
for one run.

### API modules

| Module | Responsibility |
|---|---|
| `main.py` | App, middleware (gzip for API responses, not file downloads; CORS for the Vite dev server), router registration, `/health`, `/api/options`. Startup loads the session secret, runs `db.bootstrap()` and `jobs.start()`, removes orphaned uploads, queues `experiments.startup_flush` (MLflow logging of runs that waited for review) and starts GPU detection. |
| `auth.py` | Sign-in routes (`/api/auth/*`: config, me, check, logout, pick mode's users and pick, and the OIDC login / callback), the signed session cookie, the session secret, `SecurityMiddleware`, and the `current_user` / `require_lead` dependencies. |
| `oidc.py` | The OIDC relying party: discovery and JWKS (cached; an unknown key id refetches), the authorization URL with PKCE, the code exchange, ID token verification, role from the groups claim. Uses httpx and PyJWT. |
| `recipes.py` | Baselines (rows of `meta.recipes`, one per version of one model family): `FAMILY_LABEL` and `label` (`Trees baseline v2`), the current version (the latest started, with the row lock phase changes and decision writes take), the latest finalized one (of a family), creating the first and the next version (`create_first`, `start_version`, `next_version`), the one-in-progress rule (`require_none_in_progress`, serialised by `lock_dataset`), `is_setup`, `load_working` (makes the working tables a baseline's starting point), which fields a version works on (`open_field_ids`) and the lock on the rest (`locked_problem`), the snapshot and schema fingerprint, finalizing, snapshotting versions migrated without one, the export spec rows read from a snapshot (`spec_rows`, `snapshot_targets`) and the diff between two snapshots. |
| `reuse.py` | Reusing a finalized baseline on new data: the drift rule, matching by name (`match`), and `apply`, which gives the new baseline the source's family, carries fields into the team decision, sets the targets and records the reopened fields and the summary. |
| `experiments.py` | Experiments and who sees them: `visible_ids(user_id)` (the SQL every list and lookup of experiments, exports and runs filters on), `get_visible` (`404` for a private one), the add / edit / adjust rules (`adjust_denied`, `can_adjust`), creating one, the per-version default Team experiment (`team_for`), a team experiment's copy of its baseline (`seed_copy`, `backfill_copies`), what an export of it is built from (`snapshot_for`), its `adjustments` and `revert`, the list with counts, adjustments and the best run, and MLflow deferral (`run_is_private`, `flush_deferred`). |
| `assessment.py` | Decision sources and blind assessment: `Source` (whose decisions a request sees, as SQL relations shaped like `field_decisions`, `field_settings` and `v_fields`, plus whether it may write; see [Team process](#team-process)), `source_for` (with `view` for an experiment or a finalized baseline), `from_key` / `key_owner` for stored source keys, the write helpers every writable kind shares, seeding assessors, progress, the adoption rule (`adopt`, `disagreements`), and `personal_snapshot` (one assessor's assessment shaped like a snapshot, carried fields from the version's own values). |
| `leakage.py` | `leak_names`: the fields whose stored association with any of a given set of targets reaches the leakage threshold, for sources whose targets aren't the working copy's (export previews, experiment and baseline views). |
| `users.py` | User queries: the per-request session lookup, OIDC provisioning, Team edits (with the last-lead guard and session-version bumps). |
| `config.py` | `Settings` (pydantic-settings, read from the environment). |
| `db.py` | Connection pools (the app's, and a small one for the console's read-only role), query helpers, migrations, read-only role setup. |
| `jobs.py` | `submit(fn, *args)` onto a `ThreadPoolExecutor` of `JOB_WORKERS` threads; job ownership, the restart sweep and the global job slots. |
| `ingest.py` | CSV / Parquet → Parquet, identifier sanitising, `COPY` into `raw.ds_<id>`. |
| `profiling.py` | Type inference, statistics, charts, association sample vectors. |
| `correlations.py` | Pairwise associations (Spearman, Cramér's V, correlation ratio). |
| `semantics.py` | Model families, semantic types, allowed settings per family and type, suggestions, validation, the option catalog. |
| `processing.py` | The dataset pipeline (`process_dataset`), suggestion seeding, target-derived state (`refresh_target_state`), file cleanup. |
| `transform.py` | Export: split, fit preprocessing on train rows, chunked transform, manifest. |
| `parquet_rows.py` | `read_rows`: selected rows of a Parquet file, read a few column chunks at a time (export fit sample, Explore samples). |
| `modeling/registry.py`, `modeling/train.py` | Model catalog, compatibility and device params; `ClassifierChain` training, evaluation, CLI. |
| `modeling/mlflow_log.py`, `modeling/mlflow_backfill.py` | Logging a finished model run to MLflow (experiment per dataset, params, metrics, tags, artifacts), and the CLI that logs runs that have no MLflow run yet. |
| `gpu.py` | NVIDIA GPU detection (`nvidia-smi` plus an XGBoost CUDA test fit in a subprocess), started in a background thread at startup and cached per process. |
| `analysis/prep.py`, `analysis/compute.py`, `analysis/runner.py` | Explore jobs: sampling and typing, the numeric work, caching and job lifecycle. |
| `routers/*.py` | HTTP endpoints, one module per area (datasets, fields, query, exports, runs, experiments, explore, users, team, recipes, baselines). `routers/team.py` holds the phase transitions: assigning, submitting, opening review (adoption), the queue, comments and finalizing; `routers/baselines.py` listing a dataset's baselines and starting one (scratch, another baseline, another dataset's); `routers/recipes.py` the finalized baselines across datasets, the version history, diffs, starting a new version and applying another dataset's baseline. |
| `sample_data.py` | Synthetic dataset generator (also a CLI: `python -m app.sample_data`). |
| `schemas.py` | Request bodies and query models. |

### UI modules

| Path | Responsibility |
|---|---|
| `App.tsx` | Routes, the sign-in gate (`RequireAuth`), and the top bar (dataset switcher, workflow steps, the baseline chip, which names the experiment instead in experiment mode, Team, the user and role, Sign out). |
| `api.ts`, `types.ts` | Typed fetch client for every endpoint; response types. Any `401` (fetch or the upload XHR) sends the user to `/login?next=<current path>`. |
| `hooks.ts` | Shared queries (`useMe`, `useDataset`, `useFields` (per view), `useOptions`, `useTeam`, `useExperiment`), `useAccess(ds, view)` (wraps `accessOf`; in an experiment or a finalized baseline it answers from the view), `PROCESSING` (the dataset statuses still processing) and `useDebounced`. |
| `saveQueue.ts` | Review decision saves: one serial lane per field and view, latest draft wins, failures surface with Retry, flushes on page hide. |
| `view.ts` | The decision source Review, Bulk edit and the finder work on, carried as `?view=` (`experiment:<id>`, `baseline:<id>`; none for the baseline in progress): `useView`, `parseView`, `withView`, and the query-key part `viewKey`. |
| `pages/` | `LoginPage` (user picker or *Sign in with …*), `ForbiddenPage` (OIDC user in neither group, or deactivated), `TeamPage` (users), `DatasetsPage`, `ReviewPage`, `BulkEditPage`, `ExplorePage`, `ExperimentsPage` (the list with its filters and the *New experiment* dialog; `ExperimentPage`, one experiment's header, notes and archive, around `ExperimentWorkspace`), `ExperimentWorkspace` (the export build panel, exports and runs of one experiment), `DatasetTeamPage` (setting up a new dataset's first baseline; a dataset's assessors and your assessment while assessing; the review queue, side-by-side comparison, comments, team decision and Finalize from review on), `BaselinesPage` (one card per baseline with its version history, starting a baseline, starting the next version with its field picker, comparing two versions). `ExperimentPage` also holds the Adjustments card (per-field revert). |
| `components/` | `DatasetLayout` (gates dataset pages on status), `FieldList`, `FieldDetailView`, `DecisionPanel` (in an experiment, what differs from the baseline and *Revert to baseline*), `StartBaseline` (the start / set-up form: model type, start from, targets, assessors), `ExperimentModeBar` (the bar Review and Bulk edit show under a `view`), `JoinAssessment` (*Assess it myself*), `RunDetail`, `Chart`, `ReuseCard` (the reuse summary), `ui.tsx` (error alert, empty state, status glyphs, review meter), `Logo`; `components/explore/` holds the four Explore views, their shared state (`state.tsx`) and chart options (`options.ts`). |
| `signals.ts`, `format.ts`, `theme.ts`, `charts.ts`, `toast.tsx` | Field signal thresholds, formatting and labels, chart theme, shared chart options, toasts. |
| `main.tsx`, `styles.css` | Entry point (query client, providers) and global styles. |
| `decision.ts`, `finder.ts`, `runs.ts`, `team.ts`, `recipes.ts`, `baselines.ts`, `experiments.ts`, `components/explore/params.ts` | Pure logic behind the pages: retyping a draft and encoder-param checks, Bulk edit's presets and URL state, run ranking, what the viewer may edit in each phase (`accessOf`, used through `useAccess`) and the review queue's helpers, the reuse summary's grouping and reasons, diff formatting and the reopen picker's filter, the start form's defaults and target rule, the experiment list filters, source options and labels, privacy notes, who may edit or add, and a field's differences from its baseline (`baselineDiff`, `revertedInput`), Explore's URL state. |

Routes: `/login`, `/forbidden`, `/` (datasets), `/team`, `/d/:id/review/:fieldId?`, `/d/:id/bulk`,
`/d/:id/explore/:view?` (`?decisions=` picks the decision source), `/d/:id/team` (`?field=` selects a queue item),
`/d/:id/baselines`, `/d/:id/experiments`, `/d/:id/experiments/:experimentId`. Review and Bulk edit take `?view=`.
`/d/:id` redirects to Review, `/d/:id/query` to Bulk edit, `/d/:id/versions` to Baselines and `/d/:id/export` to the
experiments list; anything else shows *Page not found*. The top bar's step is *Experiments* rather than a default
experiment's *Export & train*: while a dataset's first baseline is assessing there is no finalized baseline and so no
team experiment to land in, and the experiments an assessor can use then are personal ones that only the list shows.

Review, Bulk edit and the decision panel don't know about phases beyond `useAccess(ds, view)`: it says whether the
viewer works on their own assessment, the team decision, an experiment's copy or a finalized baseline, and whether
they can edit, and the panel turns read-only with the reason when they can't. Every query and save they make passes
the view on. A carried-over field arrives with `locked` (the reason) on its field detail, which the Review page passes
to the panel as its read-only note.

## Request and data flow

Every request first passes `auth.SecurityMiddleware`, before routing (so file downloads are covered too):

1. **Host.** Unless `RAINFOREST_ALLOWED_HOSTS` is `*`, the `Host` header must match it (`localhost`, `127.0.0.1` and
   `::1` always do). `400` otherwise. `/health` skips every check.
2. **Unsafe methods** (`POST`, `PUT`, `PATCH`, `DELETE`): `Origin`, or failing that `Referer`, must be same-origin with
   `Host` or `X-Forwarded-Host`, or be the Vite dev server (`403`); requests with neither header are not from a
   browser and pass. A request with a body must send `application/json`, or `multipart/form-data` on
   `/api/datasets/upload` (`415`); `/api/auth/check` is exempt.
3. **Session.** Everything except `/health` and the open sign-in paths (`auth.OPEN_PATHS`) needs a valid `rf_session`
   cookie naming an active user whose session version still matches (one indexed lookup, run off the event loop;
   `401`). The user `{id, name, email, role}` is left in `request.state.user`. In `oidc` mode with settings missing,
   nothing but those paths is served (`503`).

Routes take the user through the `CurrentUser` / `LeadUser` dependencies (`Annotated` aliases of `current_user` and
`require_lead` in `auth.py`); `LeadUser` answers members with `403`. Writes pass `user["id"]` to the row they create.

The cookie is `base64(claims).HMAC`, with claims `{u, v, exp}`: user id, session version, expiry. Deactivating a user
or changing their role increments `meta.users.session_version`, which ends their sessions on the next request. Each
use of the signing key has its own derived key (`session`, `oidc-state`, `pkce`, `nonce`), so a value signed for one
can't pass as another. The secret comes from `RAINFOREST_SESSION_SECRET` or is generated once into
`DATA_DIR/session_secret` (mode 600; created with an atomic link so concurrent workers agree). It is deliberately not
in Postgres, where the console's read-only role could read it. nginx forwards `Host` with its port and
`X-Forwarded-Proto`, which the origin check and the cookie's `Secure` flag rely on.

**OIDC sign-in.** `/api/auth/oidc/login` makes a random `state`, puts it with the return path in the signed `rf_oidc`
cookie, derives the PKCE verifier and the nonce from it (HMAC with the secret), and redirects to the provider. The
callback checks `state` against the cookie, exchanges the code at the token endpoint with the verifier and client
secret, verifies the ID token (`oidc.verify_id_token`) and calls `users.sign_in_oidc`, which finds the user by subject,
creates or refreshes them, and refuses deactivated users and users in neither group. Nothing from the provider is kept
but the subject, name, email and role. The state cookie is `SameSite=Lax` because the provider's redirect back is a
cross-site navigation, which doesn't carry `Strict` cookies; the session cookie stays `Strict`, since the callback sets
it and only same-site requests from the app need it. Cross-site writes are refused by the origin check either way.

Discovery is read from `RAINFOREST_OIDC_DISCOVERY_URL` when set (else the issuer's `/.well-known` URL) and its
`issuer` must equal `RAINFOREST_OIDC_ISSUER`. That split exists for Docker: browsers and token issuers use the
external URL, while the API container reaches the provider on an internal name. In the Keycloak profile, Keycloak's
`KC_HOSTNAME` is the external URL and `KC_HOSTNAME_BACKCHANNEL_DYNAMIC` makes the discovery it serves on
`http://keycloak:8080` list the token and JWKS endpoints on that name, while the issuer and the authorization and
end-session endpoints stay external.

Requests are synchronous SQL against Postgres except where noted. Long work returns `202` with a row whose `status`
the UI polls (`pending → running → done | failed`, with `progress` and `status_message`).

1. **Upload / sample.** `POST /api/datasets/upload` streams the file to `uploads/`, inserts a `meta.datasets` row
   (recording `upload_path`) and submits `process_dataset`, which removes the upload when it ends however it ends.
   Startup removes uploads older than 10 minutes that no queued or ingesting dataset needs
   (`processing.sweep_uploads`). `POST /api/datasets/sample` generates the frame first, then runs the same pipeline.
2. **Pipeline** (`processing.process_dataset`), with dataset `status` moving through `ingesting → profiling →
   correlating → ready`:
   * ingest: convert to `datasets/<id>/raw.parquet`, load `raw.ds_<id>` with `COPY` (strings quoted, NUL bytes
     removed with a warning);
   * profile every column from the Parquet copy (exact counts, sampled distributions), write `meta.fields`, and seed
     `meta.field_decisions` plus one `meta.field_settings` row per family from the per-family suggestions;
   * compute pairwise associations into `meta.field_correlations`, replace the capped values of text and
     high-cardinality fields against targets with the uncapped leak screen (`correlations.screen_associations`), then
     `refresh_target_state` derives `max_target_assoc`, leakage flags and target-dependent suggestions.
3. **Targets.** `PATCH /api/datasets/{id}` with `target_columns` calls `fields.set_targets`, which updates decisions
   and runs `_sync_targets` (keeps `is_target`, `target_columns` and target-derived state consistent, writes history
   for knock-on changes). Targets profiled as text or identifier have no association sample, so a `correlate_targets`
   job computes theirs. Added targets also get the leak screen against text and high-cardinality fields: inside
   `correlate_targets` when that job runs, otherwise as a `fields.screen_targets` job that leaves the dataset
   `ready`. A target chosen in Review (`PUT …/decision`) gets the same job.
4. **Review.** `GET /api/fields/{id}` returns the field, the viewer's decision merged with the baseline family's
   settings, and target associations. `PUT /api/fields/{id}/decision` validates against the family (a type change
   whose settings don't fit takes the new type's suggestion), writes the decision and settings and, for the team
   decision only, appends `meta.decision_history` and keeps the target state in step. See
   [Team process](#team-process) for whose decision that is.
5. **Bulk edit.** The field filter compiles to a `WHERE` over the viewer's `fields_view()` (`meta.v_fields` for the
   team decision; `query.fields_where`, a whitelist of columns and operators). `POST …/decisions/bulk` resolves and locks the set in one statement, plans per-field
   changes, and either returns the plan (`dry_run`) or applies it, storing a pre-change snapshot in
   `meta.bulk_batches` (an assessor's in `meta.assessment_batches`, an experiment's in `meta.experiment_batches`) for
   undo.
6. **SQL console.** `POST /api/sql` takes a connection from the `rainforest_ro` pool (`SQL_POOL_MAX`; `503` when
   busy), opens a `READ ONLY` transaction with `SET LOCAL statement_timeout`, and runs the text as a named
   server-side cursor. psycopg sends the `DECLARE` over the extended protocol, so Postgres rejects a second statement
   and `DECLARE` accepts only a query; a single `FETCH` pulls `SQL_MAX_ROWS + 1` rows to detect truncation. The
   transaction is always rolled back, and the pool runs `DISCARD ALL` before reusing the connection, so settings or
   advisory locks taken by a query don't carry over.
7. **Export.** `POST …/exports` resolves the experiment (given, or the version's default Team experiment) and the
   snapshot it builds from (`experiments.snapshot_for`: a finalized version's stored snapshot, a team experiment's copy
   with its adjustments, or `assessment.personal_snapshot` for a personal experiment), stores it in
   `exports.snapshot` with the baseline and adjustments in `exports.config`, checks the preview's blocking issues and
   submits `exports._run`, which reads that snapshot, calls
   `transform.run_export` and writes `exports/<id>/{model_ready.parquet, preprocessor.joblib, manifest.json}`.
8. **Training.** `POST /api/exports/{id}/runs` checks model compatibility and the requested device (see `gpu.py`) and submits `runs._run`, which calls
   `modeling.train.train` and writes `models/run<id>_…/{classifier_chain.joblib, report.json}`; metrics and the
   report go to `meta.model_runs`. With `MLFLOW_TRACKING_URI` set, `mlflow_log.log_model_run` then logs the run
   before it is marked done (status message *Logging to MLflow*), unless its experiment is private: then the run is
   marked done with `report.mlflow_pending` instead, in the same transaction that checked (see
   [Experiments](#experiments)). It never raises: a failure, bounded by short HTTP
   timeouts and two retries, deletes any half-logged MLflow run and lands in the report as `mlflow_error`; success
   stores `mlflow_experiment_id` and `mlflow_run_id`, from which the runs endpoints derive `mlflow_url`.
9. **MLflow UI.** nginx sends each `/mlflow/` request through `auth_request` to `GET /api/auth/check` (always as a
   `GET`, without the body), which the middleware answers `401` without a valid session and the route `204`. A `401`
   becomes a relative redirect to `/login?next=/mlflow/`, and the login page does a full page load for that path
   (`isServerPath` in `api.ts`) since it is outside the single-page app. nginx resolves `api` and `mlflow` per request
   through Docker's DNS (`resolver 127.0.0.11`), so it starts without them and follows a recreated container. Toward MLflow, nginx sends `Host: mlflow`
   (its `--allowed-hosts`) and drops `Origin`, as MLflow's own cross-origin check only knows localhost; cross-site
   requests can't get through anyway, because the `SameSite=Strict` cookie is what `auth_request` checks.
10. **Explore.** See [Analyses](#analyses-explore-jobs).

## Team process

A dataset's decisions are **baselines** (`meta.recipes`, `recipes.py`): one line per model family (`family`: `tree`,
`linear`, `neural`), each versioned (`version` counts per family, unique per dataset and family). Each version moves
`assessing → review → finalized`, and at most one version per dataset is in progress (`assessing` or `review`):
starting a baseline or a version checks the current one is finalized (`recipes.require_none_in_progress`, under a lock
on the dataset row). The current version is the latest started (highest id).

**The working copy.** `meta.field_decisions` / `meta.field_settings` hold the team decision of the current version
only. Starting any baseline or version first loads them from its starting point with `recipes.load_working`: the
source snapshot's decisions and its settings for the new family where they are valid (else the family's suggestion),
or the suggestions from scratch, plus the targets; it also sets `meta.datasets.model_family`, which the server keeps
as the current baseline's family (a `PATCH` naming another is refused). So the working tables never carry another
baseline's choices, and everything finalized is read from snapshots. A snapshot stores the settings of its own
family only. A new dataset's first baseline (Trees v1, created by the upload) is *set up* while nobody has worked on
it (`recipes.is_setup`: v1, assessing, no parent, no reuse, no assessors or assessments, no other version):
`POST …/baselines` then changes its family, starting point and targets on that row instead of starting a new one.

**Sources.** Every route that touches decisions starts from `assessment.source_for(conn, dataset_id, user, view=…)`,
which returns a `Source` of one of four kinds, each a set of SQL relations shaped like `field_decisions`,
`field_settings` and `v_fields`, a stored key, and the reason writes are refused (or none):

* **team** (`team`): the working tables. While **assessing** it is read-only for everyone but is still the starting
  point plus the targets, so reading it reveals nobody's choices; in **review** leads write it; **finalized**:
  read-only.
* **assessment** (`assessment:<recipe_id>:<user_id>`): while assessing, for an assigned viewer, `meta.assessments` /
  `meta.assessment_settings` for that version and user. Reads and writes go there; no history, no target sync, no
  stored date format (those would show the choice to others). A submitted assessor's source refuses writes.
* **experiment** (`experiment:<id>`, from `view`): a team experiment's own copy (`meta.experiment_decisions` /
  `meta.experiment_settings`), writable by its owner only (`experiments.adjust_denied`). No history, no target sync on
  the dataset; a target change updates the experiment's `target_order`.
* **baseline** (`baseline:<recipe_id>`, from `view`): a finalized version's snapshot, read through `jsonb_each`;
  never writable.

Without a `view`, the team or assessment source follows from the phase and the viewer, as above. Experiment and
baseline sources take their targets from their own decisions, and their leak flags from `leakage.leak_names` against
those targets, rather than the dataset's stored flags.

Writers lock the recipe row `FOR SHARE` and phase changes take it `FOR UPDATE`, so a save can't land in a phase it
wasn't checked against. Assigning seeds the assessor from the team decision (`assessment.seed`). The submit that
completes the set (or the unassign that does) calls `team.open_review`: `assessment.adopt` decides each field (same
action and type from everyone → adopted, settings by most common choice per family with ties going to the
suggestion), adopted fields are written to the team decision as reviewed with history attributed to the actor, the
rest are marked unreviewed and are the queue, and target changes go through `_sync_targets` and the leak screen.
Finalizing writes `recipes.snapshot` (the baseline's family's settings only) and the schema fingerprint; a trigger
refuses any later update of a finalized row that has a snapshot.

Blindness outside the routes: the console role is denied every table in `db.PRIVATE_TABLES` (bootstrap grants
`SELECT` on all of `meta` and then revokes these, every start), and `meta.analyses` has row-level security so that
role sees only team results. Explore keys raw-data results by decision source (`params.decisions`), so the cache never
returns one person's result to another, and `GET /api/analyses/{id}` hides others' personal results and results on
an experiment the viewer can't see.

**Starting.** `POST …/baselines` (`routers/baselines.py`) starts or sets up a baseline from scratch, from a finalized
baseline of the same dataset (recorded as `parent_id`; scope `all`, every field assessed), or for a set-up first
baseline from another dataset's (`reuse.apply`, below), then assigns and seeds the chosen assessors.

**Versions.** `POST …/recipes` calls `recipes.start_version` for vN+1 of a family from its latest finalized version,
whose snapshot `load_working` puts into the working tables with every field reviewed. A version's `scope` says which
fields it works on: `all`, or `reopened` with the fields in `meta.recipe_reopened` (`recipes.open_field_ids`).
Everything else is carried over: the team decision holds vN's values, and every write path refuses carried fields
through `recipes.locked_problem` (single saves, bulk edits, which filter them out, bulk undo and target changes),
whoever writes. Assessments are keyed by recipe id, so a new version starts clean; `assessment.seed` copies the team
decision with carried fields already reviewed, and progress, the submit rule, adoption, the queue and the review
summary all count only the open fields. A version with nothing reopened starts in `review` with scope `all`: the lead
edits the team decision (history records it) and finalizes. Diffs (`recipes.diff`) compare snapshots (decisions only
between families); a version still being worked is compared through `recipes.snapshot_of`, which builds the snapshot
from the team decision, never from assessments.

**Reuse.** `reuse.apply` matches a finalized snapshot onto a new dataset's fields by name (`reuse.match`, drift in
`reuse.drift`), gives the dataset's first baseline the source's family, writes carried fields' decisions, settings
(that family's) and notes into the team decision (with history), sets the still-valid targets through
`fields.set_targets`, records the reopened fields and the summary (`recipes.reuse`) and sets `parent_id` to the source
baseline. With nothing reopened the version moves to `review`. It runs from `POST …/recipes/apply`, `POST …/baselines`
with `start: recipe`, or inside `processing.process_dataset` before the dataset turns ready when the upload named a
baseline (`recipe_id`, stored as the new v1's `parent_id` until then). `parent_id` is `ON DELETE SET NULL` so a source
dataset can be deleted; the frozen-row trigger lets exactly that change through, and the summary keeps the provenance.

**Exports** read a snapshot through `recipes.spec_rows` (decision and the baseline family's settings per field, with
the dataset's stored date formats) in both the preview and `exports._run`, always for the baseline's own family:
the preview from `experiments.snapshot_for` (or a version's snapshot: the requested one, else the latest finalized of
the current family, else of any), the export from the snapshot frozen into its row when it was requested. Nothing on
that path reads `field_decisions` / `field_settings`. The preview checks leakage against the snapshot's own targets
(`leakage.leak_names`), names targets with no stored associations, and for a team experiment warns when its targets
differ from its baseline's. MLflow runs are tagged with the export's baseline (id, version, label, family), its
experiment and its adjustments (`rainforest.adjusted`, `rainforest.targets_changed`; the param
`prep.adjusted_fields` and the artifact `adjustments.json`), read from `exports.config`.

## Experiments

`meta.experiments` (`experiments.py`) holds a dataset's experiments; `exports.experiment_id` and
`model_runs.experiment_id` place every export and run in one (a run in its export's). The source is stored as
`recipe_id` plus `assessor_id` (null for a team experiment), and as the `source` text `recipe:<id>` or
`assessment:<recipe_id>:<user_id>`, which a check constraint keeps in step. A finalized version gets one default Team
experiment (`is_default`, named like *Team · Trees v2*) from `team_for` on first use, found by the flag so it can be
renamed.

**Copies and adjustments.** A team experiment gets its own copy of its baseline when it is created (`seed_copy`;
bootstrap's `backfill_copies` seeds any that lack one): every field's decision in `meta.experiment_decisions`, the
baseline family's settings in `meta.experiment_settings`, and the baseline's target order in
`experiments.target_order`. Review, Bulk edit and the finder read and write it through the experiment source
(`view=experiment:<id>`), with bulk batches in `meta.experiment_batches`. The adjustments are not stored separately:
`experiments.adjustments` diffs the baseline's snapshot against `snapshot_for` (the copy shaped as a snapshot, with
the experiment's targets), and `revert` deletes the copy's rows for some or all fields and reseeds them. Only the
owner of a non-default, unarchived team experiment writes the copy (`adjust_denied`); a default Team experiment and a
personal one export from the baseline's snapshot and the assessment respectively. A field an experiment makes a target
gets the leak screen against text and high-cardinality fields (`fields.target_associations_job`), which adds shared
associations without touching the baseline's target state. Writes to the copy hold the experiment's row `FOR SHARE`
and an export takes it `FOR UPDATE`, so an export is exactly the copy that was checked.

**Visibility** is one rule: an experiment is visible to a user unless it is personal, its version is `assessing`, and
the user isn't its assessor. Phases only move forward, so once visible it stays so. The API applies it in SQL
(`experiments.visible_ids`) on every list and lookup of experiments, exports and runs (the Datasets list's counts
and latest export / run too), and `get_visible` answers `404`. Everything reached through an export or run id
(files, the models listing, training, the preview, Explore `export:<id>` data and its cached results) goes through
those lookups. For the SQL console, which has no user, migration 010 adds row-level security: `meta.experiments`
shows only non-private rows, `meta.exports` and `meta.model_runs` show a row only when its experiment is visible
(`experiment_id IN (SELECT id FROM meta.experiments)`, which applies the experiments policy in turn), and
`meta.analyses` shows a result on an export only when that export is visible (`… IN (SELECT id FROM meta.exports)`). The app's role owns these tables, so the policies don't apply to it.
`meta.v_model_metrics` is `security_invoker`, so the console reads it under those policies. Rows with no
experiment are visible to nobody; `meta.import_orphan_experiments()` (run by migration 010 and by bootstrap) moves
them into the dataset's **Imported** experiment.

**MLflow** has no access control, so a private experiment's runs are kept out of it. `runs._run` checks
`experiments.run_is_private` holding the version's row `FOR SHARE` and, if private, marks the run done with
`report.mlflow_pending` in that transaction. Opening review takes the row `FOR UPDATE`, so a run either sees the new
phase and logs at once, or was marked before review opened and is picked up by `experiments.flush_deferred`, which
the submit or unassign that opened review queues after it commits (and startup runs for anything left). Flushing
goes through `mlflow_backfill.backfill`, whose `pending` never includes a run of a still-private experiment.

## Background jobs

All jobs go through `jobs.submit`, run on a thread pool in the API worker that accepted the request, and keep their
state in Postgres, so any worker can serve the polling.

* **Ownership.** Each API process holds a session advisory lock on a random owner id (class `jobs.OWNER_LOCK`) on a
  dedicated connection for its lifetime, and stamps that id in the `owner` column of every job row it creates. On
  startup `jobs.start()` runs `jobs.sweep()`, which fails datasets still processing and pending or running exports,
  runs and analyses whose owner doesn't hold its lock (found in `pg_locks`), however recently they reported progress. A sibling
  worker's jobs are never touched. A dataset that was only correlating new targets goes back to `ready` with a
  warning.
* **Slots.** A queued job starts only once it holds one of `JOB_SLOTS` advisory locks (class `jobs.SLOT_LOCK`, one by
  default), shared by every process, and runs with its native thread pools limited to `jobs.cpu_threads()`
  (usable CPUs `// JOB_SLOTS`, at least 1). Model runs pass that as `n_jobs` to estimators that take one; Explore's t-SNE and
  mutual information use it too.
* **Queue order.** `jobs.submit` takes a ticket from the `meta.job_ticket` sequence and locks it (class
  `jobs.TICKET_LOCK`) on the owner connection. A waiting job takes a free slot only when no lower ticket is locked, then
  releases its ticket, so jobs start in submission order across every process. A process that dies releases its
  tickets with its connection.

| Job | Entry point | State in |
|---|---|---|
| Dataset pipeline | `processing.process_dataset` (`datasets._generate_and_process` for samples) | `meta.datasets` |
| New-target associations | `datasets.correlate_targets` | `meta.datasets` |
| Leak screen for added targets | `fields.screen_targets` | `meta.field_correlations` (no status; a failure adds a dataset warning) |
| Export | `exports._run` | `meta.exports` |
| Model run | `runs._run` | `meta.model_runs` |
| Explore analysis | `analysis.runner.run_analysis` | `meta.analyses` |

### Analyses (Explore jobs)

`POST /api/datasets/{id}/analyses` with `{kind, params, recompute}`:

1. `AnalysisParams.normalised(kind)` keeps only the keys that affect that kind; for raw data the route adds the
   decision source (`decisions`). The canonical JSON of the result is `params_key`.
2. `runner.data_version`: for raw data, an md5 over every field's `(field_id, action, semantic_type)` in the params'
   decision source (`decisions`: `team`, `assessment:<recipe>:<user>` for an assessor's *kept only*, or from the
   request's `view` `experiment:<id>` or `baseline:<recipe>`), so any write
   path (single save, bulk, undo, target change) invalidates without hooking into it; for export data,
   `export:<id>`.
3. `runner.find_or_create` (under a lock on the dataset row) returns, in order: a done row for the same key and
   version; a pending or running one; a done row from an older version (returned with `stale: true`, no new job).
   Otherwise, or with `recompute`, it marks older pending/running rows of the same dataset, kind and decision source as
   failed (`Superseded`), inserts a row, prunes old done and failed results (keeping, per dataset and kind, the newest
   done row per key on the current data version or on an export, plus the 3 newest others) and submits
   `run_analysis`.
4. `run_analysis` prepares a sample (`prep.prepare_raw` or `prep.prepare_export`), calls `compute.run_map`,
   `run_tests` or `run_importance`, and stores `result`. Each progress update checks the row is still active and
   stops the job if it was superseded. `NotEnough` failures store `result.reason`
   (`no_target`, `too_few_fields`, `too_few_rows`, `single_class`) for the UI's explanation.

The UI (`components/explore/state.tsx`) POSTs whenever the URL params change and polls `GET /api/analyses/{id}`
every second while the row is active. The Correlations view is not a job: it reads `meta.field_correlations` directly.

## Schema and migrations

`api/app/migrations/NNN_name.sql` files run once each, in name order, on API startup under a Postgres advisory lock,
and are recorded in `meta.schema_migrations`. `api/app/views.sql` is re-run on every start, so views are always
recreated from that file. Bootstrap also snapshots finalized versions that have none, moves orphaned exports and runs
into *Imported* experiments, seeds team experiments' missing copies, and (re)creates the `rainforest_ro` role and
grants it `SELECT` on `meta` and `raw`, except the tables in `db.PRIVATE_TABLES`.

| Migration | Adds |
|---|---|
| `001_initial.sql` | `meta.datasets`, `fields`, `field_correlations`, `field_decisions`, `field_settings`, `decision_history`, `exports`, `model_runs` |
| `002_bulk_batches.sql` | `meta.bulk_batches`, `decision_history.batch_id` |
| `003_analyses.sql` | `meta.analyses` |
| `004_job_owner.sql` | `owner` on `meta.datasets`, `exports`, `model_runs` and `analyses` |
| `005_upload_path.sql` | `upload_path` on `meta.datasets` |
| `006_mlflow_run_id.sql` | `mlflow_experiment_id`, `mlflow_run_id` on `meta.model_runs` |
| `007_users.sql` | `meta.users` (with the first lead, *Admin*); `user_id` on `decision_history` and `bulk_batches`, `created_by` on `datasets`, `exports`, `model_runs` and `analyses`, backfilled to the first lead |
| `008_team_assessment.sql` | `meta.recipes` (with the finalized-snapshot trigger), `dataset_assessors`, `assessments`, `assessment_settings`, `assessment_batches`, `review_comments`, `exports.recipe_id`, row-level security on `analyses`; every existing dataset gets version 1, finalized by the first lead (bootstrap writes the snapshots) and its exports point at it |
| `009_recipe_versions.sql` | `scope` and `reuse` on `meta.recipes`, `meta.recipe_reopened`; `recipes.parent_id` becomes `ON DELETE SET NULL` and the frozen-row trigger allows only that change on a finalized row |
| `010_experiments.sql` | `meta.experiments`; `experiment_id` on `exports` and `model_runs`, `exports.snapshot`; `meta.import_orphan_experiments()`, which moves existing exports and runs into one *Imported* experiment per dataset; row-level security on `experiments`, `exports` and `model_runs`, and the `analyses` policy extended to results on exports |
| `011_job_ticket.sql` | `meta.job_ticket`, the sequence that orders queued background jobs |
| `012_baselines.sql` | `family` on `meta.recipes` (`tree`, `linear`, `neural`; existing versions take their dataset's `model_family`), versions unique per dataset and family; `experiments.is_default`, set on the existing default Team experiments, which are renamed *Team · Trees vN* style |
| `013_experiment_adjustments.sql` | `meta.experiment_decisions`, `experiment_settings` and `experiment_batches` (a team experiment's copy of its baseline and its bulk edits); `experiments.target_order` |

Child tables reference `meta.datasets` with `ON DELETE CASCADE`; `DELETE /api/datasets/{id}` removes the rows and then
the dataset's Parquet copy, raw table, export and model directories.

## Where to add things

**A model.** Add a `ModelSpec` to `MODELS` in `api/app/modeling/registry.py` (key, label, family, NaN support,
builder, default params, importance source, optional `requires` package, `cuda_params` if it can train on a GPU). The CLI, `/api/models`,
`/api/exports/{id}/models` and the Train form pick it up. Compatibility rules are in `registry.compatibility`.

**A migration.** Add `api/app/migrations/NNN_description.sql` with the next number, written to be safe to run once
(`IF NOT EXISTS` where it helps). If a view needs the new column, edit `views.sql`. If the table holds job state,
stamp `owner = jobs.owner_id()` where its rows are created and add it to `jobs.sweep`.

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

**Anything that reads field decisions.** Take the user (and a `view` query parameter where a page can work on an
experiment or a finalized baseline) and read through `source_for(..., view=view).decisions()` / `.settings()` /
`.fields_view()` rather than naming `meta.field_decisions` directly, so an assessor sees their own choices and nobody
else's and an experiment its own copy; write through `write_decision` / `write_settings` after `require_write()`.
Settings are the source's family's (`src.family`). Add the
endpoint to `READS` in `api/tests/test_team.py`, which checks every read path for leaks. A new table holding someone's
assessment goes in `db.PRIVATE_TABLES`.

**An endpoint.** Add it to the matching `routers/*.py` module (or a new router registered in `main.py`), with request
bodies in `schemas.py`, and list it in [api.md](api.md). Take `user: CurrentUser` to know who is acting (and record
`user["id"]` on rows it creates), or `_: LeadUser` for lead-only actions.

## Testing and CI

`api/tests` uses pytest. `conftest.py` provides a generated sample Parquet (most tests are pure functions over it) and
the session-scoped `login_as(client, role="lead", name=None)` fixture, which signs a `TestClient` in through pick mode
as the built-in Admin (a lead) or a user with that role, created on first use. `test_auth.py` drives the app without
its startup hook and with an in-memory user store, so it needs no database. `test_oidc.py` runs OIDC against an
in-process provider (`tests/oidc_provider.py`, served through `httpx.MockTransport` by setting `oidc.transport`): the
redirect and token checks need no database, the provisioning flows do. The API tests in `test_explore.py`,
`test_sql_console.py`, `test_data_correctness.py`, `test_api_cleanup.py`, `test_users.py`, `test_team.py`,
`test_experiments.py`, `test_recipes.py`, `test_baselines.py` and `test_adjustments.py` (helpers in
`team_helpers.py`) start the app against `RAINFOREST_TEST_DATABASE_URL` and skip without it;
`test_jobs.py` (job ownership, the restart sweep and the slots) needs that database too. `test_mlflow.py` logs to a temporary MLflow file store
(always runs), drives `runs._run` and the backfill against the database (skips without it), and has one test against a
real tracking server that runs when `MLFLOW_TEST_URI` is set. `make test` runs the suite locally, `make test-db` runs it
against a throwaway `postgres:16` container.

`test_team.py` covers phases and transitions, role checks, adoption, finalizing, exports, the migration and the
console role's grants, and runs every read endpoint as another assessor, the lead and a bystander to check that an
assessor's choices never show (`READS`). `test_recipes.py` covers the drift, matching and diff rules, new versions
with and without reopened fields (locked carried fields, the reopened-only submit rule and review), version history
and diffs (including that they show no assessment while assessing), reuse on new data (carried / reopened / missing,
targets, provenance, applying at upload) and that exports come from a version's snapshot. `READS` also covers a
private personal experiment of another assessor: its listing, detail, edits, exports, files, models, preview, runs,
run files, Explore on its export, and the console's experiments, exports, runs and metrics. `test_experiments.py`
covers the visibility matrix (owner, member, lead × assessing, review, finalized, through the API and the console),
what a personal export contains (carried fields included) and that it rebuilds the same way after edits, the Team
experiment, role checks, the Imported migration and MLflow logging deferred until review opens.
`test_baselines.py` covers families and per-family versions, labels, snapshots holding only their family, setting up
and starting baselines (scratch, populated, from another dataset), the one-in-progress rule, new versions loading
their own parent, diffs across families, the dataset's family being refused, exports taking the baseline's family,
leak checks per target set, planned models, browsing a finalized baseline, Explore's decision source and the SQL
views. `test_adjustments.py` covers experiment copies, owner-only adjusting through the view, refused views, type and
target changes, leak warnings for an experiment's targets, that adjustments never reach the baseline, and that
exports freeze them. Tests that need team decisions to be editable or exportable move the recipe straight to a phase
with `conftest.set_phase`.

`ui` uses Vitest (`npm test`) for the pure logic: field signals, formatting, the save queue (with the API mocked),
retype rules, the Bulk edit and Explore URL state, error parsing, run ranking, phase access, the experiment
helpers (including a field's differences from its baseline), the start form's defaults and target rule, the view
URL helpers, and the reuse summary, diff formatting and reopen picker. Tests sit next to the code as
`*.test.ts`. CI (`.github/workflows/ci.yml`, on pushes to `main` and pull requests) runs the API tests with a
Postgres service and an MLflow server container, the UI typecheck, tests and build, and `docker compose build`.
