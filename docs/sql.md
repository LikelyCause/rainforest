# SQL

Rainforest keeps everything in Postgres, so you can query it with SQL. Terms are defined in
[Concepts](concepts.md), including the code names (a baseline version is a `recipe`, a model type is a `family`).

## Connecting

You have two options:

- The *SQL* tab in *Bulk edit*: a read-only console. Members and leads can use it.
- Any SQL client, connected to `localhost:5432` on the Docker host (`DB_PORT`), with the user and password from
  `.env`. `make psql` opens psql as the database owner.

## Console rules

- The console runs one query: `SELECT`, `WITH … SELECT`, `VALUES` or `TABLE`.
- It runs as the SELECT-only `rainforest_ro` role, in a read-only transaction with a statement timeout
  (`SQL_TIMEOUT_MS`).
- Postgres parses the text through a server-side cursor. So a second statement is refused however it's written.
- Only the rows shown are fetched, up to `SQL_MAX_ROWS`, not the whole result.
- `:dataset` is replaced with the current dataset's id.

## Example queries

```sql
-- all text fields
SELECT name, null_pct, avg_length FROM meta.v_fields WHERE dataset_id = 1 AND inferred_type = 'text';

-- fields with more than 20% missing that the team keeps in the baseline in progress
SELECT name, null_pct, imputation FROM meta.v_fields WHERE dataset_id = 1 AND null_pct > 0.2 AND action = 'keep';

-- how the baseline in progress prepares one field
SELECT s.family, imputation, transform, scaling, encoding, source
FROM meta.v_field_settings s JOIN meta.datasets d ON d.id = s.dataset_id
WHERE s.field_id = 42 AND s.family = d.model_family;

-- compare trained models on the test split, by baseline
SELECT baseline_family, baseline_version, experiment_id, model, target, metric, value FROM meta.v_model_metrics
WHERE dataset_id = 1 AND split = 'test' AND metric IN ('roc_auc', 'roc_auc_ovr_macro', 'r2', 'rmse')
ORDER BY target, metric, value DESC;
```

## Schema

| Table / view | Contents |
|---|---|
| `meta.datasets` | uploads: status, progress, counts, targets, `model_family` (the current baseline's) |
| `meta.fields` | inferred type, confidence and reasons, typed statistics, `leakage_flag`, `stats` / `charts` / per-family `suggestion` JSONB |
| `meta.field_decisions` | the team decision per field of the baseline in progress (keep / drop / target, `target_task` and `target_bins`, type, notes, reviewed) |
| `meta.field_settings` | preprocessing per field **per model family** |
| `meta.decision_history` | append-only audit log of every change (with family, `user_id` and, for bulk edits, `batch_id`) |
| `meta.bulk_batches` | each applied bulk edit: summary and a pre-change snapshot used by undo |
| `meta.field_correlations` | pairwise associations (one row per unordered pair) |
| `meta.users` | users: name, email, role (`lead` / `member` / `viewer`), active, `key` (an account from `RAINFOREST_USERS`), `avatar` (the picture's hash), OIDC subject, last seen |
| `meta.recipes` | baseline versions per dataset: `family` (`tree` / `linear` / `neural`), `version` (counted per family), phase (`assessing` / `review` / `finalized`), `scope` (`all` / `reopened`), who finalized it, the frozen `snapshot`, and `reuse` (the summary when it started from another dataset's baseline) |
| `meta.recipe_reopened` | the fields a `reopened` version works on |
| `meta.baseline_names` | a baseline's name: one per dataset and model family (`meta.baseline_name(dataset_id, family)` reads it) |
| `meta.experiments` | experiments per dataset: name, owner, `source` (`recipe:<id>` or `assessment:<recipe id>:<user id>`), `is_default` (a baseline version's *Team ·* experiment), `target_order`, notes, archived |
| `meta.experiment_decisions`, `meta.experiment_settings`, `meta.experiment_batches` | each team experiment's own copy of its baseline (decisions, its model family's settings) and its bulk edits; fields that differ from the baseline's snapshot are its adjustments |
| `meta.exports`, `meta.model_runs` | export jobs; training runs with metrics and report; each with its `experiment_id` |
| `meta.analyses` | *Explore* jobs (map, tests, importance): params, data version, status, result JSONB |
| `meta.v_field_profiles`, `meta.v_fields`, `meta.v_field_settings`, `meta.v_correlations`, `meta.v_model_metrics` | query-friendly views; see below |
| `raw.ds_<id>` | the uploaded data (sanitized column names; see `meta.fields.db_column`) |

About the views:

- `v_fields` and `v_field_settings` show the team decision of the baseline in progress. `v_fields` includes
  `target_task` and `target_bins`.
- `v_field_profiles` and `v_fields` carry the baseline's `baseline_family` and `baseline_version`.
- `v_model_metrics` carries each run's `experiment_id`, and the `baseline_family` and `baseline_version` its export
  was built from. Its `split` is `valid`, `test` or `cv` (the cross-validation means).

## Privacy

The console can't see work that is private to one person.

- Assessments are private while a dataset is assessed. The console's role can't read `meta.assessments`,
  `meta.assessment_settings`, `meta.assessment_batches`, `meta.dataset_assessors` or `meta.review_comments`.
- In `meta.analyses`, the console sees only the team's *Explore* results.
- Personal experiments are private while assessing. Row-level security hides them in `meta.experiments`, along with
  their rows in `meta.exports`, `meta.model_runs`, `meta.v_model_metrics` and `meta.analyses`. The console doesn't
  know who is asking, so this hides them from their owner too.

## Migrations and views

Schema changes go in `api/app/migrations/NNN_name.sql`. The API applies new migrations once on startup, under an
advisory lock, and records them in `meta.schema_migrations`.

Views live in `api/app/views.sql`. The API recreates them on every start.
