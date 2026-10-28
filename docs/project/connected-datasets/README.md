# Connected datasets: review guide

This branch (`connected-datasets`, off `main` at `117bd62`) lets a dataset be created from one or more MongoDB
collections instead of an uploaded file, and refreshed later with the team's field decisions carried forward. It is the
second of three agreed projects (Kubernetes deployment, already on `main`; connected datasets; verification modeling,
not started).

## What is in this folder

| File | What it is |
|---|---|
| [design.md](design.md) | The design the work was built from (decisions made with the owner, goals, non-goals, behaviour, data model, API). |
| [plan.md](plan.md) | The implementation plan: ten tasks with their interfaces and tests. |
| [review-log.md](review-log.md) | The working log: every task, every review round, every finding deferred as minor, and every ruling made on the owner's behalf (*Ruling N*), with what it costs if wrong. |
| [final-review-findings.md](final-review-findings.md) | The findings of the whole-branch review that survived refutation, as handed to the last fix round. |

## What to review

The feature in one pass:

1. **Sources** (`api/app/sources/`): named MongoDB connections from `RAINFOREST_MONGO_SOURCES` (the chart builds it
   from Secrets), listing, preview, the flattening rules (`flatten_document`), error redaction, the pull to chunked
   Parquet and the assembly with a `source` column and deduplication (`pull.py`).
2. **Connect** (`POST /api/datasets/connect`, `routers/datasets.py`): validation before any dataset exists (connection,
   labels, filters, budget, Match on), then the normal processing path.
3. **Refresh** (`api/app/refresh.py`, `POST /api/datasets/{id}/refresh`): re-pull into side files, one short swap
   transaction, fields reconciled by name so ids, decisions and history survive, drift reported per finalized
   baseline, the dataset locked (`status: refreshing`) while it runs, failures leave the dataset as it was.
4. **Exports after a refresh** (`routers/exports.py`): the gone-field block, the pull recorded on each export and in
   MLflow (`rainforest.pull_id`).
5. **UI**: the Connect dialog (`ui/src/components/ConnectDialog.tsx`), the connected badge and *Refresh…* on the
   Datasets page, the lock banner (`DatasetLayout.tsx`), the pull summary (`PullCard.tsx`).
6. **Delivery**: chart values `sources.mongodb`, the Compose `mongo` profile and `make up-mongo`, MongoDB services in
   both CI pipelines, migrations `023` and `024`.
7. **Docs**: `docs/data-preparation.md` (Connected datasets), `docs/api.md`, `docs/configuration.md`,
   `docs/deployment.md`, `deploy/README.md`.

## Known open issues

These are confirmed and being fixed; check `git log` for commits after `0e4423e` to see which have landed.

- **Exports after a refresh judge the wrong decisions for non-default experiments** (`routers/exports.py`
  `used_fields`). A refresh deletes a vanished field, and the experiment's own decision for it goes with it (cascade),
  so the check falls back to the baseline's decisions. Two effects: an experiment that dropped a field the baseline
  keeps is blocked for good once the field leaves the data; an experiment that kept a field the baseline drops exports
  with that feature silently missing. The fix records, at removal time, which fields each experiment and assessment
  used (Ruling 21).
- A retried failure write in `refresh._record_failure` can unlock a newer refresh.
- `missing_collections` lists collections without `authorizedCollections`, so an account with `find` on specific
  collections but no database-level `read` can't refresh.
- Renamed column names (`_empty`/`_empty_2`, dotted-key collisions) depend on document order across pulls, so a refresh
  can swap their meanings under the same field ids (Ruling 22 seeds them from the previous pull).

Deferred minors are listed in the review log (lines with `minor (deferred)`); none was judged to block a merge.

## Running it

- API tests need a Postgres and a MongoDB: `make test-db` for Postgres; set
  `RAINFOREST_TEST_MONGO_URL=mongodb://127.0.0.1:27017` with a `mongo:7` running (`make test-mongo` starts one) to
  run the MongoDB-backed tests, which skip otherwise.
- Try it end to end: `make up-mongo` starts the stack with a seeded MongoDB (`rf_dev.curated`, `rf_dev.massive`) and a
  `dev` connection; *Datasets → Connect ▾*.
- Chart: `bash deploy/chart/test.sh`; `helm lint deploy/chart/rainforest -f deploy/chart/rainforest/ci/dev-values.yaml`.

## State at the last full run (`0e4423e`)

API 1085 passed, 1 skipped. UI typecheck clean, 254 passed plus the 4 `theme.test.ts` failures that are also on
`main`. Chart 174/174, lint clean. ESLint 0 errors (20 warnings, also on `main`).
