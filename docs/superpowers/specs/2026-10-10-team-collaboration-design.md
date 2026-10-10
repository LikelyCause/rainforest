# Team collaboration: users, blind assessment, recipes, experiments

Goal: a data science team explores an uploaded dataset together. Each assigned person independently decides how every
field should be prepared; the team reviews only the fields where they disagreed; a lead finalizes the result as a
versioned recipe that exports, code and new uploads can reuse. Model exploration is never blocked and is tracked in
MLflow.

Decisions made with the owner (2026-10-10):
- One team per deployment (schema leaves room for more later).
- Assessments are blind until the dataset reaches Pending Team Review.
- Leads assign assessors per dataset. Review opens automatically when every assigned assessor has submitted.
- Only a lead finalizes; the lead's decision is enough.
- Disagreement = the core decision differs: keep / drop / target, or semantic type. Prep settings (imputation,
  indicator, transform, scaling, encoding, encoder params, per model family) follow, and the lead can adjust them.
- Model parameters are not part of the team process; they belong to experiments and are tracked in MLflow.
- Finalized recipes are versioned and frozen. Changes produce a new version that goes back through assessment.
- A finalized recipe can be applied to a new upload with the same columns; only new or changed fields need assessing.
- Standard OIDC sign-in with Keycloak as the reference identity provider; a "pick a user" mode for dev and demos.
- Backups are out of scope.

Built as four sub-projects, in order, each its own PR: (1) users and sign-in, (2) blind assessment and team review,
(3) recipes, (4) experiments and MLflow. MLflow infrastructure may land alongside (1).

## 1. Users and sign-in

Tables: `meta.users` (id, name, email, role `lead | member`, active, oidc_subject unique nullable, created_at,
last_seen_at). All writes record the acting user: `decision_history.user_id`, `bulk_batches.user_id`,
`exports.created_by`, `model_runs.created_by`, `analyses.created_by`, `datasets.created_by`.

`RAINFOREST_AUTH_MODE`:
- `oidc` (team deployments): authorization-code flow with PKCE, handled by the API (`/api/auth/oidc/login`,
  `/api/auth/oidc/callback`). Settings: issuer URL, client id, client secret, redirect URL, groups claim name,
  `RAINFOREST_LEAD_GROUP` (default `rainforest-lead`), `RAINFOREST_MEMBER_GROUP` (default `rainforest-member`).
  Users are created on first sign-in; name, email and role are refreshed from the token on every sign-in. Someone in
  neither group gets a clear 403 page. Sign-out also ends the provider session (end_session_endpoint).
- `pick` (dev, demos, trusted single-team networks): the login page lists active users; choosing one signs in as them.
  Leads manage users on a Team page (add, rename, set role, deactivate). First start creates one lead, "Admin".
  The UI says plainly that this mode does not authenticate.
The existing signed-cookie session carries the user id instead of "admin". The single admin password login is removed.
CSRF / Origin / Host / content-type checks stay as they are.

Keycloak reference setup: compose profile `keycloak` (Keycloak in dev-file or prod mode on :8081, its own database
`keycloak` in the existing Postgres), with a realm import containing client `rainforest`, groups `rainforest-lead` and
`rainforest-member`, and test users (`lead1`, `ds1`, `ds2`; passwords documented as dev-only). `make up-keycloak`.

Roles: members edit their own assessments, comment in review, run Explore and their own experiments, upload datasets.
Leads also assign assessors, settle disagreements, finalize, start new recipe versions, delete datasets, manage users
(pick mode).

The SQL console runs as the read-only role, which gets no access to assessment tables (see 2), so blind data can't leak
through it.

## 2. Blind assessment and team review

Tables:
- `meta.dataset_assessors` (recipe_id, user_id, assigned_by, assigned_at, submitted_at).
- `meta.assessments` (recipe_id, field_id, user_id, action, semantic_type, notes, reviewed, updated_at) and
  `meta.assessment_settings` (recipe_id, field_id, user_id, family, the same columns as `field_settings`).
- `meta.review_comments` (id, recipe_id, field_id, user_id, body, created_at).
Assessment rows are keyed by recipe version (3) so a new version starts clean.
The existing `field_decisions` / `field_settings` hold the team's working decision for the dataset's current version.

Phases (on the recipe version being worked, shown on the dataset): `assessing → review → finalized`.
- Assessing: assigned assessors use the same Review and Bulk edit UI, but read and write only their own assessments,
  seeded from the suggestions. The API never returns another user's assessment in this phase (enforced in queries, with
  tests). Leads see progress per assessor (fields reviewed / total, submitted or not), never choices. "Submit
  assessment" requires every field to be reviewed; an assessor can unsubmit until review opens. A lead can unassign
  someone so review can open. Unassigned users can browse the dataset, Explore, and run their own experiments.
- Review opens automatically when the last assigned assessor submits. Fields where every assessor agrees on action and
  type are adopted into the team decision; their prep settings take the most common choice per family, ties broken by
  the suggestion. Disagreements form the review queue: each assessor's choice, settings and notes side by side, plus a
  comment thread. Everyone can view and comment; only a lead sets the team decision. Bulk edit in review is lead-only
  and acts on team decisions. A lead finalizes when the queue is empty (or confirms leaving fields as adopted).
- Finalized: see 3.
The read-only SQL role is granted nothing on `assessments`, `assessment_settings` or `dataset_assessors`.

Explore: shared. Raw-data analyses use the team decision, except that during assessing the "Kept only" toggle uses the
viewer's own assessments; the cache key includes the decision source so results never mix users.

## 3. Recipes

`meta.recipes` (id, dataset_id, version, phase `assessing | review | finalized`, parent_id, created_by, created_at,
finalized_by, finalized_at, schema_fingerprint, snapshot JSONB). A dataset has exactly one non-finalized version at a
time at most. Finalizing writes `snapshot`: per field name, the decision and the settings for every model family, plus
the field's inferred type and key stats (null_pct, n_unique) for drift checks. Finalized snapshots are immutable.

New version: a lead starts vN+1 from vN and picks which fields to reopen (default: none selected, with search and
filters). Reopened fields go through blind assessment and review with the assigned assessors; the rest are carried over
and locked. With nothing reopened, the lead may edit directly and finalize (small fixes), recorded in history.

Reuse on new data: on upload (or later, while nothing has been assessed), choose "Start from recipe" (any finalized
recipe of any dataset). Fields match by name. A matched field whose inferred type is unchanged and whose stats haven't
drifted (null_pct change ≤ 10 points, n_unique ratio within 0.5–2×) is carried over and locked; matched-but-changed and
unmatched fields are reopened for assessment; recipe fields missing from the upload are listed. Targets carry over.
The new dataset starts at v1 in assessing (or finalized directly if nothing needs assessing, after a lead confirms).

Exports: built from a recipe version (finalized: team exports, visible to all) or from a person's own assessments
(personal, belongs to an experiment, see 4). The manifest records recipe id and version.

Existing data migrates as: every existing dataset gets recipe v1 finalized from its current decisions, created by the
first lead; existing exports point at it.

## 4. Experiments and MLflow

`meta.experiments` (id, dataset_id, name, owner, source `recipe:<id>` or `assessment:<recipe_id>:<user_id>`,
created_at, notes). Exports and model runs belong to an experiment. Visibility: an experiment on a finalized recipe is
visible to everyone; one on personal assessments is private to its owner while that recipe is assessing and visible to
the team from review onwards. Existing exports and runs move into one "Imported" experiment per dataset.

MLflow: compose service (official image) with its backend store in Postgres database `mlflow` and artifacts on a volume,
served through the existing nginx at `/mlflow/` behind the Rainforest session (nginx `auth_request` to the API), so the
tracking UI and API are not open on the LAN. One MLflow experiment per Rainforest dataset. Each training run logs
params (model, chain order, cv, device, every estimator param), metrics per target and split, the trained model as an
artifact, the run report, and tags: dataset, recipe id and version, export id, experiment id, user, git sha.
`model_runs.mlflow_run_id` links both ways; the runs table links to the MLflow run. If MLflow is unreachable the run
still completes and records the logging error (no silent loss).

## Testing

Per sub-project unit and DB-backed API tests, including: no endpoint leaks another user's assessment during
assessing (parametrized over every read endpoint); phase transitions; role checks (403 for members on lead actions);
recipe immutability; reuse matching and drift; experiment visibility; OIDC flow against a mocked provider and once
against the Keycloak profile; MLflow logging against a real tracking server in CI (service container). Playwright on
dev-01: a three-user flow (lead + two members) from assignment through blind assessment, review, finalize, reuse on a
new upload, and an MLflow-logged run.
