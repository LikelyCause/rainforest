# Connected datasets: MongoDB collections as a source, with refresh

A dataset can be created from one or more MongoDB collections instead of an uploaded file. Rainforest pulls the
documents into its usual Parquet copy and raw table, profiles them like an upload, remembers where they came from, and
can pull again on request, carrying the team's field decisions forward.

Decisions made with the owner (2026-10-24):

- **The data lives in MongoDB**, 200k–500k documents and 250–300 fields per collection. At that size a copy is cheap,
  so Rainforest's internal model stays "a dataset is a Parquet file plus a raw table"; what changes is how the data gets
  there.
- **Some collections are fixed, others change** and need a *Refresh* that re-pulls and carries decisions forward.
- **Several collections can form one dataset**, with a `source` column naming the collection each row came from. This
  is how the vehicle team assembles curated + mining + massive and deduplicates across them.
- **Connections come from the deployment** (a Kubernetes Secret or `.env`), chosen by name in the UI; nobody types a
  connection string into Rainforest.

## 1. Intent

The vehicle-engine team keeps its three datasets in MongoDB. Today each pass through Rainforest means exporting a
collection to CSV and uploading it. With connected datasets a lead picks the collections once; Rainforest pulls them,
and when the "mining" collection grows, *Refresh* pulls it again and tells the team which fields changed enough to
need another look. Nothing downstream (review, baselines, Explore, exports, training) knows or cares that the data
came from MongoDB.

### Goals

- Create a dataset from one or more collections of a named MongoDB connection, with an optional filter and projection
  per collection, through the UI and the API.
- Flatten documents into columns deterministically and tell the user what was flattened, stringified or dropped.
- A `source` column on multi-collection datasets; optional exact-duplicate removal across collections.
- *Refresh* (leads only): re-pull, re-profile, replace the raw data, and report what changed against the current
  finalized baseline using the existing drift rules, so the lead can start a new version with the right fields
  reopened.
- Connections configured per deployment, the same way as other secrets; the chart exposes them as values.
- Everything that works on an uploaded dataset works unchanged on a connected one.

### Non-goals

- Other source kinds (S3/object storage, SQL warehouses, Parquet directories). The code gets a small source interface
  so one can be added later; only MongoDB is implemented.
- Incremental or change-stream refresh. A refresh is a full re-pull.
- Scheduled or automatic refresh. A lead presses the button.
- Querying MongoDB at analysis time instead of copying. The copy *is* the dataset.
- Writing anything back to MongoDB.
- Letting users define connections in the UI.

### Anti-goals

- A connection string, user name or password visible in the UI, the API, logs, the SQL console or an export manifest.
- A refresh that silently changes what an existing export or run was built from. Exports keep their own Parquet; a
  refresh touches only the dataset's raw data and fields.
- A refresh that discards team decisions without saying so. Every field whose decision is lost (because the field
  vanished) is listed.

## 2. Constraints

- `meta.datasets.file_format` is `csv | parquet | generated` with a CHECK constraint; `process_dataset(dataset_id,
  upload_path, fmt)` converts an upload to `datasets/<id>/raw.parquet`, loads `raw.ds_<id>`, profiles, correlates, then
  applies a pending recipe. Generated samples already skip the upload step by writing `raw.parquet` themselves and
  calling `process_dataset(id, None, "parquet")`; a MongoDB pull does the same.
- `ingest._normalize` drops nested/binary Parquet columns with a warning. Flattening therefore happens *before* the
  Parquet is written, in the puller, so the canonical copy is already flat.
- Fields are rows in `meta.fields` keyed by dataset and name; decisions, history, settings and correlations hang off
  `field.id`. A refresh must keep `field.id` stable for fields that persist, so decisions and history survive.
- `reuse.match(snapshot, fields)` already classifies fields as carried / reopened (`new`, `type_changed`, `drift`) /
  missing from a finalized snapshot and new field stats; refresh reports with the same function and the same
  thresholds (`NULL_PCT_DRIFT` 0.10, `UNIQUE_RATIO` 2).
- Derived fields (`meta.fields.derived`) are computed from other columns; a refresh recomputes them after the pull
  with the same formulas, and a derived field whose inputs vanished is reported and removed.
- Explore results in `meta.analyses` are keyed by the decisions they were computed from, not by the data; a refresh
  must invalidate every analysis of the dataset.
- Background work goes through `jobs.submit` (one slot by default, ordered tickets); a pull is a job like an upload's
  processing and reports progress through the dataset's `status`/`progress`/`status_message`.
- Viewer / Member / Lead rules: creating a connected dataset needs the same role as uploading (Member or Lead);
  *Refresh* is a Lead action, like starting a baseline version, because it can reopen fields.
- Row and cell budgets: the pull honours a new `SOURCE_MAX_CELLS` (default 600,000,000; see *Size*) and refuses to
  start a pull whose estimated size exceeds it.

## 3. Configuration

Connections are named and come from the environment:

```
RAINFOREST_MONGO_SOURCES=vehicles=mongodb+srv://user:pass@cluster.example.com/?authSource=admin;archive=mongodb://…
```

`name=uri;name=uri`, the same shape as `RAINFOREST_USERS`. Names are slugs (`[a-z0-9-]+`). A malformed entry stops the
API at startup with a message naming the entry (never echoing the URI). The URI is used as given (TLS, auth source,
replica set options all live in it). An empty variable means the *Connect* menu is hidden.

Chart values:

```yaml
sources:
  mongodb:
    - name: vehicles
      existingSecret: rainforest-vehicles-mongo   # key `uri`
    - name: archive
      existingSecret: rainforest-archive-mongo
```

The chart renders `RAINFOREST_MONGO_SOURCES` from the Secrets with `$(VAR)` expansion of one env entry per source
(`MONGO_URI_VEHICLES` from the Secret key `uri`), so the URI never lands in the ConfigMap. The chart README and
`deploy/README.md` describe it; the per-team values example gets a commented block.

Compose: `RAINFOREST_MONGO_SOURCES` in `.env.example` (commented), passed through to the `api` service. A `mongo`
Compose profile (`make up-mongo`, like `make up-keycloak`) runs `mongo:7` with a seeder that loads two collections
generated by `app.sample_data` (one in-scope, one "massive" with extra fields and nested documents) so a developer can
try the feature without a real cluster. My call.

## 4. Behaviour

### Creating a connected dataset

*Datasets → Connect ▾ → MongoDB collection…* (shown when the API lists at least one source). The dialog:

1. **Connection** (a select of names from `GET /api/sources`).
2. **Database** (a select from `GET /api/sources/{name}/databases` when the connection may list them, else a text
   field) and **Collections** (multi-select from `GET /api/sources/{name}/databases/{db}/collections`, each with its
   document count).
3. Per collection, optional **Filter** and **Projection** as JSON (validated as JSON documents; MongoDB's extended
   JSON for ObjectId and dates is accepted).
4. **Preview**: `POST /api/sources/{name}/preview` samples 200 documents per selected collection and shows the
   flattened columns (name, inferred type, share present in the sample), flagged columns (arrays, mixed types,
   dropped), and the estimated row and cell counts against the budget.
5. **Name**, **Add a `source` column** (on and locked when more than one collection is chosen; optional for one),
   **Leave out exact duplicate rows across collections** (default on for several collections; ties keep the row
   from the collection listed first, so list the most trusted first — the dialog says so).

*Connect* calls `POST /api/datasets/connect` (202) and returns the dataset row at `status: pending`. The job pulls,
writes `raw.parquet`, then runs the normal `process_dataset`. The Datasets list shows a **connected** badge with the
connection and collection names and *pulled 3 minutes ago*.

### Flattening

Deterministic rules, applied per document, documented in `docs/data-preparation.md`:

| Value | Becomes |
|---|---|
| nested document `{a: {b: 1}}` | column `a.b` (dots in keys are replaced by `_`; depth unlimited) |
| array of scalars or documents | one string column holding the array as JSON (reviewable as text, droppable) |
| `ObjectId`, `UUID`, `Binary` | string; `_id` is kept (it will profile as an identifier) |
| `Decimal128`, `Int64`, `Double`, `Int32` | number (Decimal128 → float64) |
| `Date` | datetime |
| `true` / `false` | boolean (profiling reads it as a binary field) |
| `null`, missing key | missing |
| a key whose values have mixed types across documents | the column is read as text and listed in the warnings |

The column set is the union of keys over every pulled document, in first-seen order. The puller streams documents in
batches of 50,000 into chunked Parquet files and finishes with a relaxed diagonal concatenation into `raw.parquet`, so
the budget is memory-bounded and a key that first appears in document 400,000 still becomes a column. Column names
then go through the existing `sanitize_identifiers` for the raw table.

### `source` and deduplication

With several collections, each row gets `source` = the collection name (or the per-collection **label** typed in the
dialog, default the collection name). It is an ordinary categorical field, reviewed and droppable like any other; the
spec for project 3 uses it as a target for the source-fingerprint experiment.

Deduplication compares rows on the **columns the chosen collections share**, excluding `_id`, `source` and any column
the preview flags as an identifier (the same `identifier` inference profiling uses: unique-looking keys such as
`…_id`, UUIDs). Columns present in only some collections never take part, so a "massive" row with extra fields can
still match a "curated" row. The dialog shows the comparison columns and lets the user replace them with an explicit
**Match on** list (e.g. the record's natural key). Duplicates are removed within a collection as well as across them;
ties keep the row from the collection listed first (the dialog says to list the most trusted first). The counts
removed per collection are stored on the pull and shown with the dataset's warnings. (My call: a rule that excluded
nothing would remove nothing, since each collection has its own `_id`s.)

### Size

The budget `SOURCE_MAX_CELLS` applies to the whole dataset, estimated before deduplication as the sum over collections
of documents × projected fields. Its default is 600,000,000 so curated + mining + massive (about 450M cells) fits out
of the box; the Compose and chart docs say what memory that needs (the pull streams in chunks; profiling samples
`PROFILE_SAMPLE_ROWS`; the raw-table load uses `COPY_BATCH_ROWS` as uploads do). A refresh always re-pulls every
collection of the dataset: at this size a full pull is minutes, and re-pulling one collection would need per-collection
chunk storage and a re-run of deduplication against stored chunks, which is not worth it yet. (My call.)

### Refresh

*Datasets row menu → Refresh…* (leads). The confirmation lists the collections, the last pull time, and what will
happen: raw data replaced, fields re-profiled, Explore results marked out of date, decisions kept for fields that
remain, new and drifted fields listed for a new baseline version. It is refused (`409`) while a baseline version is
being assessed or is in team review (the same rule derived fields follow), and while an export or run of the dataset
is running.

While it runs the dataset is **locked**: a new status `refreshing` (added to the CHECK constraint) under which the
dataset's pages open read-only with a banner saying a refresh is running and who started it, and the API refuses
every write to the dataset and every new export, run or Explore computation on it with `409`. (My call: letting work
continue during a pull that can take minutes means edits to fields the refresh then deletes, and exports reading a
Parquet file mid-swap; a lock is simpler and the refresh is rare.)

The job:

1. Pulls into `raw.parquet.new`, chunked as above, and loads it into `raw.ds_<id>_new` with the usual batched COPY,
   both outside any long transaction.
2. Profiles the new Parquet (`profiling.profile_parquet`, which samples) and recomputes derived fields.
3. In one short transaction: renames `raw.parquet.new` over `raw.parquet` and `raw.ds_<id>_new` over `raw.ds_<id>`,
   updates each existing field's stats and inferred type by name (keeping `field.id`), inserts new fields with seeded
   suggestions, deletes fields no longer present (their decisions and history go with them; derived fields whose
   inputs vanished too), recomputes correlations, refreshes target state (a target that vanished is removed from
   `target_columns` with a warning), and marks every `meta.analyses` row of the dataset out of date.
4. For **each model family with a finalized baseline**, compares the new fields with that family's latest finalized
   snapshot using `reuse.match` and stores the results as the pull's **drift summary** keyed by family: carried,
   reopened (new / type changed / drift with the numbers), removed.
5. Sets `status: ready` and records the pull in `meta.dataset_pulls`.

The **Team** page and each baseline's card on the **Baselines** page show the latest pull's summary for that family
with a *Start a new version with these fields reopened* button for leads, which opens the existing *New version* form
with the reopened fields pre-selected. With no finalized baseline yet (the dataset is still being set up), the summary
is informational and the current working decisions simply follow the new stats. Experiments' adjusted copies lose
adjustments on removed fields; the experiment's Adjustments card says so.

**Exports after a refresh** build on the new data, since that is the dataset now. The export preview gains a blocking
issue when a field in the baseline snapshot no longer exists in the data, and a non-blocking warning when the baseline
was finalized before the latest pull ("finalized before pull #3; consider a new version"). Every export of a connected
dataset records `pull_id` in its manifest and MLflow params, and the exports list shows *pull #3* on the row, so
lineage stays truthful. (My call: blocking exports until a new version is finalized would stop a team whose data
changed in ways the baseline doesn't care about.)

**Field names and existing baselines.** The flattening rules produce the same names `mongoexport --type=csv` does
(dotted paths for nested documents), so a finalized baseline built on a CSV export of the same collection matches by
name through the ordinary `recipe_id` reuse path. (My call; the deployment is greenfield so nothing depends on it.)

A fixed collection is one nobody refreshes; nothing distinguishes it otherwise.

### Failure and safety

- A pull that fails (connection refused, auth, budget exceeded mid-way, a document that can't be flattened) leaves the
  dataset exactly as it was: new datasets go to `failed` with the reason (never the URI); a refresh keeps the old data,
  fields and status `ready`, and the failed pull is recorded with its error.
- Connection errors are reported with the connection *name* and the driver's message with any URI redacted.
- Pulls run with `serverSelectionTimeoutMS` 10 s, read preference `secondaryPreferred`, `batch_size` 5,000, and a
  per-pull wall-clock limit (`SOURCE_PULL_TIMEOUT_S`, default 3,600) after which the job fails.

## 5. Data model

```sql
ALTER TABLE meta.datasets DROP CONSTRAINT ...; -- file_format gains 'mongodb'
CREATE TABLE meta.dataset_sources (
    dataset_id   BIGINT PRIMARY KEY REFERENCES meta.datasets ON DELETE CASCADE,
    kind         TEXT NOT NULL CHECK (kind = 'mongodb'),
    connection   TEXT NOT NULL,            -- the name; never the URI
    database     TEXT NOT NULL,
    collections  JSONB NOT NULL,           -- [{name, label, filter, projection}]
    add_source   BOOLEAN NOT NULL,
    dedupe       BOOLEAN NOT NULL,
    created_by   BIGINT REFERENCES meta.users
);
CREATE TABLE meta.dataset_pulls (
    id           BIGSERIAL PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.datasets ON DELETE CASCADE,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    rows         BIGINT,
    per_source   JSONB,                    -- [{label, documents, removed_duplicates}]
    flatten_notes JSONB,                   -- arrays stringified, mixed-type columns, renamed keys
    drift        JSONB,                    -- {family: reuse.match output} per finalized baseline, or null
    error        TEXT,
    created_by   BIGINT REFERENCES meta.users
);
```

`GET /api/datasets/{id}` gains `source` (the `dataset_sources` row without secrets) and `last_pull` (status,
finished_at, rows, whether a drift summary exists). `GET /api/datasets/{id}/pulls` lists pulls with their summaries.

## 6. API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/sources` | `[{name, kind}]`; empty when none configured. Any signed-in user. |
| GET | `/api/sources/{name}/databases` | names, or `403`-style `{listable: false}` when the account can't list. |
| GET | `/api/sources/{name}/databases/{db}/collections` | `[{name, documents}]` (estimated counts). |
| POST | `/api/sources/{name}/preview` | body `{database, collections: [{name, filter, projection}]}` → flattened columns per collection and merged, estimated rows/cells, budget verdict. |
| POST | `/api/datasets/connect` | body `{name, source: {connection, database, collections: [{name, label, filter, projection}], add_source, dedupe, match_on?}, targets?, recipe_id?}` → `202`, the dataset row. Member or Lead. `recipe_id` works as on upload. `match_on` (optional column list) replaces the default comparison columns for deduplication. |
| POST | `/api/datasets/{id}/refresh` | Lead. `202`; `409` while assessing/in review or while an export/run is running; `404` for a dataset that isn't connected. |
| GET | `/api/datasets/{id}/pulls` | pull history with summaries. |

Filters and projections are validated as JSON documents before any job starts; operator keys are passed to the driver
as given (the account's privileges bound what they can do — the connection is expected to be read-only, and the docs
say so).

## 7. Code layout

- `api/app/sources/__init__.py`: the `Source` protocol (`list_databases`, `list_collections`, `preview`, `pull`) and a
  registry keyed by name, built from `Settings.mongo_sources`.
- `api/app/sources/mongodb.py`: `pymongo` client per connection (lazy, one per process), listing, sampling,
  streaming pull → chunked Parquet, flatten rules in a pure function `flatten_document(doc) -> dict` with its own
  tests.
- `api/app/sources/pull.py`: the job: budget check, pull to chunks, diagonal concat, dedupe, `source` column, then
  either `process_dataset(id, None, "parquet")` for a new dataset or `refresh_dataset(id)` for a refresh.
- `api/app/refresh.py`: steps 2–5 above (profiling into memory, the swap transaction, field reconciliation by name,
  derived recompute, analysis invalidation, drift summary via `reuse.match`).
- `api/app/routers/sources.py`, additions to `routers/datasets.py`.
- Migration `023_connected_datasets.sql`.
- UI: `ConnectDialog.tsx` (connection → database → collections → options → preview), a connected badge and *Refresh…*
  in the dataset row menu, a *Latest pull* card on the Team and Baselines pages reusing the reuse-summary component.
- Chart: `sources.mongodb[]` values → env from Secrets; chart README + `deploy/README.md`.
- Compose: `mongo` profile with seeder; `make up-mongo`.
- Docs: `docs/data-preparation.md` (connected datasets, flattening table, refresh), `docs/api.md`,
  `docs/configuration.md`, `docs/deployment.md`.

## 8. Testing

- Unit: `flatten_document` (every row of the table, dotted-key collisions, depth, mixed types); source-string parsing;
  budget arithmetic; dedupe keeps the first collection's row.
- Integration against a real MongoDB (`RAINFOREST_TEST_MONGO_URL`, `mongo:7` via Docker locally and as a CI service in
  both pipelines; tests skip without it): connect two seeded collections with a filter and a projection → dataset ready
  with the `source` column and the right counts; a refresh after inserting documents with a new key and shifting a
  column's null share → the new field appears, the drifted field is in the summary, decisions on untouched fields keep
  their `field.id` and history, Explore analyses are out of date; a refresh with a dropped collection key → field
  removed and listed; a failing pull leaves a `ready` dataset untouched; a URI never appears in any API response or
  error.
- Chart: `sources.mongodb` renders one env entry per source from the named Secret and the composed
  `RAINFOREST_MONGO_SOURCES`; absent when the list is empty.
- The kind smoke test is unchanged (no MongoDB there); `make up-mongo` is the manual check.

## 9. Left to the builder

- Exact dialog layout and the shape of the pull-summary card (reuse the existing reuse-summary component).
- Whether `preview` samples with `$sample` or the first N documents (either; say which).
- Chunk sizes and the exact relaxed-concat call; how mixed-type columns are detected across chunks.
- Whether listing databases uses `listDatabaseNames` and how the "can't list" case is detected.
- The Compose seeder's exact collections.
