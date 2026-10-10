# 🌲 Rainforest

Rainforest prepares wide, mixed-type data for **tree-based models**. Ingest a dataset, pick its targets, review it
**one field at a time** in a fast UI, store keep / drop / imputation / encoding decisions in Postgres, explore the data
with correlations, embeddings and per-field tests, export a **model-ready** dataset, and train a **`ClassifierChain`**
of LightGBM (or another tree ensemble) over your targets.

Trees are the default and the main path. Rainforest knows that tree ensembles split on value *order*, so it never
suggests scaling or log transforms for them, and it leaves missing values for the model to route. Linear and neural
model families are also available. When you switch to one, it gets its own preprocessing settings (scaling,
imputation, one-hot encoding) without touching the tree settings.

```
┌────────────┐  /api/*   ┌──────────────────────────┐        ┌────────────────────────────────┐
│  ui        │ ────────► │  api (FastAPI)           │ ─────► │  db (Postgres 16)              │
│  React +   │  nginx    │  ingest · profile ·      │  SQL   │  meta.*  fields, stats,        │
│  ECharts   │  proxy    │  correlate · explore ·   │        │          decisions, settings,  │
│  :8080     │           │  export · train chains   │        │          exports, model runs,  │
└────────────┘           └─────────┬────────────────┘        │          analyses              │
                                   │ /data volume             │  raw.ds_<id>  your data        │
                                   │ Parquet copies, exports, └────────────────────────────────┘
                                   │ trained models
```

Developer docs: [docs/architecture.md](docs/architecture.md) (how the pieces fit, where to add things) and
[docs/api.md](docs/api.md) (every HTTP endpoint).

## Quick start

```bash
cp .env.example .env          # optional — every value has a default
docker compose up -d --build
open http://localhost:8080
```

1. **Datasets** → upload a CSV / Parquet file, or use **Generate sample ▾** (1k–2M rows, 30–400 fields, at most
   60M cells, with `column_a` / `column_b` targets that are binary + 3-class, both binary or both multiclass). On an empty install,
   *Open the 50k × 150 demo* does the same in one click. The upload is converted to Parquet, loaded into
   `raw.ds_<id>`, profiled and correlated, with live progress.
   Then **Choose target ▾** on the dataset's row: pick one or more columns, or *No target (unsupervised)*. Saving
   refreshes the target associations and leakage flags. Generated samples already have their two targets set. The row menu also
   renames or deletes a dataset (deleting removes its fields, exports, model runs and their files).
2. **Review** → walk the fields. Each field shows its signals (missing, skew, outliers, levels, target association
   with a leakage flag), its distribution and quantile strip, top tokens or weekdays, and its strongest
   **associations** (click a bar to jump to that field). Decisions autosave. The sidebar filters by status
   (*To do* · *Kept* · *Dropped* · *All*), by type, and by *Needs attention* (leakage, mostly missing or constant, low
   type confidence, failed saves).
3. **Bulk edit** → find fields with presets (*Unreviewed*, *Missing > 20%*, *Skewed numerics*, *Target-associated*,
   *High-card categoricals*, *Text*, *Possible leaks*, *Edited by you*) or your own filters, then **Accept
   suggestions**, **Keep**, **Drop** or **Mark reviewed**. Each bulk action shows a preview of what will change before
   you apply it, and can be undone from the toast. *Review these N →* (`R`) opens the matching fields as a working
   set in Review. The **SQL** tab is a read-only console with examples.
4. **Explore** → correlations heatmap, a 2-D map of rows, per-field statistical tests against a target, and feature
   importance. See [Explore](#explore).
5. **Export & train** → fit the preprocessing on the training split, write the model-ready Parquet, and train
   chains with any compatible model. Runs are compared side by side, and the best result in each column is starred.

The top bar shows the workflow steps **Review · Bulk edit · Explore · Export & train** and the **Model** switch
(*Trees* · *Linear* · *Neural*).

### Keyboard shortcuts (Review)

| Key | Action |
|---|---|
| `J` / `→` · `K` / `←` | next / previous field |
| `U` | next unreviewed field (also jumps into Review from Bulk edit, Explore and Export) |
| `X` | toggle keep / drop |
| `Enter` | confirm and go next |
| `A` | use the suggestion and confirm |
| `Z` | undo the last confirm |
| `N` | add a note |
| `/` | search fields |
| `Esc` | leave a control, back to review |
| `?` | show or hide the shortcut list |

## Decisions: shared vs per model family

| Shared by every family | Per family (`meta.field_settings`) |
|---|---|
| keep / drop / target, semantic type, notes, reviewed | imputation, missing-indicator, transform, scaling, encoding, encoder params |

The top-bar **Model** switch selects which family's settings you're reviewing. It defaults to *Trees*. Each family has
its own suggestions, so switching never overwrites another family's choices. If you change a field's type, the other
families' settings are re-derived for the new type.

| | **Tree ensembles** (default) | Linear models | Neural / distance |
|---|---|---|---|
| Models | LightGBM, XGBoost\*, histogram GBM, random forest | logistic regression | MLP |
| Missing numerics | left as NaN; the model learns where to send them | median + "was missing" flag | median + flag |
| Skew / outliers | nothing: splits only depend on order | signed log1p / winsorize | log1p |
| Scaling | **not offered**: it can't change a split | standard | standard / quantile-normal |
| Categoricals | ≤ 8 levels one-hot, ≤ 1,000 ordinal codes, else frequency | one-hot (top 30 kept), else frequency | same as linear |
| Text | TF-IDF → 32 SVD components | same | same |
| Dates | year, month, day, weekday, hour, day-of-year, epoch days | + median impute + scaling | same |

\* XGBoost is optional: `API_EXTRA_PIP=xgboost docker compose build api` (or set it in `.env`).

In the UI, settings a family doesn't use appear as a locked line with the reason, for example *"Not used for tree
models — they split on value order, so rescaling changes nothing."* The API rejects them too
(`scaling='standard' is not valid … with tree models`).

**Type inference** (`api/app/profiling.py`) proposes `numeric_continuous`, `numeric_discrete`, `categorical`,
`binary`, `text`, `datetime`, `identifier` or `constant`, with a confidence score and reasons. It detects numbers stored
as text (`"1,234.50"`), dates stored as text (13 formats), integer and UUID keys, and free text compared with short
categorical strings. Fields with a ≥ 0.95 association to a target are flagged as **possible leakage** and suggested
for dropping.

**Associations** use a seeded 100k-row sample:
- Spearman ρ for numeric pairs (pairwise-complete, vectorised).
- Bias-corrected Cramér's V for categorical pairs, with missing as its own level.
- Correlation ratio η for numeric × categorical pairs.

## Explore

Four views over one dataset, sharing a toolbar whose state lives in the URL:

* **Data:** *Raw fields* (with a *Kept only* toggle) or a finished export (its train split, features as encoded).
* **Target:** one of the dataset's targets. Without one, the views that need it say so and link to the target picker.
* **Sample:** 1k / 5k (default) / 10k / 20k rows, seeded, stratified on a categorical target. Rows with no target
  value are left out.

In raw mode the fields used are the kept fields (*Kept only*) or every non-target field, typed by their review
decision. Text, identifier and constant fields, and fields constant in the sample, are skipped and listed.
Categoricals keep their top 20 levels plus *Other* and *(missing)*; dates become epoch seconds. A numeric target with
more than 20 distinct values is treated as numeric, otherwise as classes.

| View | What it computes |
|---|---|
| **Correlations** | Clustered heatmap of the stored associations (no job, raw data only): the 60 fields most associated with the target, or with the strongest associations overall, ordered by average-linkage clustering on 1 − strength. A slider hides weak cells, you can add fields beyond the 60, and clicking a cell shows the pair with links to Review. |
| **Map** | 2-D embedding of the sampled rows, coloured by the target. **t-SNE** (default; PCA to 50 components first, perplexity 5–100, default 30) or **UMAP** (neighbours 5–100, default 15), fixed seed. Numerics are median-filled and standardised, categoricals one-hot. Hover shows the 5 fields that spread the rows most (largest weight in the first two principal components). |
| **Feature tests** | One test per field against the target, with Benjamini–Hochberg adjusted p-values (the UI marks adjusted p < 0.05). Numeric field × class target: one-way ANOVA (effect η²) plus Kruskal–Wallis, box plots per class. Categorical field × class target: χ² (effect Cramér's V), level counts per class. Numeric target: Spearman (effect \|ρ\|) plus Pearson, a scatter of up to 500 points; or ANOVA of the target across the field's levels. Classes with fewer than 5 sampled rows are excluded and listed. |
| **Importance** | Mutual information of each field with the target (needs a target), and PCA: explained variance for up to 20 components with the cumulative line, plus loadings of the first 5 components for the top 20 features. |

Map, Feature tests and Importance run as background jobs and are cached in `meta.analyses` per parameter set. A
result is tied to the decisions it was computed from (every field's keep / drop / target and type; for export data,
the export itself). After you change a decision, the cached result is shown as **Out of date** with a *Recompute*
button rather than recomputed automatically. A new request for the same view supersedes one still running. When the
sample can't support an analysis (no target, too few rows or fields, a single class), the view explains why instead
of failing.

## Export → ClassifierChain

* **Leakage-safe split:** the train / valid / test split is assigned first, stratified on the target combination.
  Every statistic is then fitted on training rows only (up to 500k of them by default), and the full file is
  transformed in chunks.
* **Memory:** both steps are sized by output width. A transform chunk holds at most `EXPORT_CHUNK_MB` of dense
  output (rows × features × 8 bytes, 256 MB by default), and the fit sample at most `EXPORT_FIT_MB` (512 MB), so a
  wide export fits on fewer rows (the manifest records `fit_sample_rows` and `chunk_rows`). Peak memory runs at a
  few times these budgets: a 300k-row export peaks at about 1.3–2 GB.
* **Output:** float32 features, `__split`, `__row_id` (joins back to `raw.ds_<id>`), label-encoded targets, the fitted
  `preprocessor.joblib`, and `manifest.json` (feature → source field, class mappings, model family, NaN report).
* **Before you export**, the page summarises what will go in (kept / dropped / targets / customised fields, kept
  fields by type) and warns about unreviewed fields, kept leakage flags and a target count other than two. No target
  or no kept features blocks the export. Once an export exists, it shows how many decisions changed since, with
  *Re-export with same split*. Validation and test share, seed and the fit-row cap are adjustable.
* **Compatibility:** tree models train on any export. Linear and neural models need an export prepared for their
  family, and they're refused if the export contains NaN. The UI disables incompatible models and shows the reason.
* **Chain:** `ClassifierChain(model, order=<targets>, cv=5)`, in the dataset's target order unless you choose another
  (the UI offers both orders when there are two targets).
  * The chained feature is **out-of-fold** during training, so a later target's model learns from realistic
    predictions of the earlier ones rather than the ground truth.
  * All targets binary → P(class 1) is passed down the chain; any multiclass target → the predicted class code.
  * NaN-capable models are wrapped so sklearn's finiteness check doesn't reject the NaN they handle natively.
* **Metrics** per target on the valid and test splits: accuracy, balanced accuracy, log-loss, ROC-AUC (one-vs-rest
  macro for multiclass), precision / recall / F1 (macro and weighted, plus positive-class and specificity for binary
  targets), confusion matrices and a per-class table, plus the top importances per chain link.
* **On the page:** each export has *Train ▾* (model and chain order), *Files ▾* (Parquet, preprocessor, manifest) and
  *Use from code* (the CLI command and class mappings). The runs table compares test-split AUC and macro-F1 per
  target, stars the best within each export, and expands a run into its full metrics and confusion matrices.

```bash
make train EXPORT=1                          # LightGBM
make train EXPORT=1 MODEL=random_forest
docker compose exec api python -m app.modeling.train --export-id 1 --model xgboost \
    --order column_b,column_a --param n_estimators=1000 --param learning_rate=0.03
make models                                  # what's registered / installed
```

**Adding a model** means adding one `ModelSpec` to `api/app/modeling/registry.py`: key, family, NaN support,
builder, default parameters and how to read importances. It then appears in the CLI, the API and the UI.

## Querying

Connect any SQL client to `localhost:5432` (user and password from `.env`), or use the SQL tab in Bulk edit.
The console runs as the SELECT-only `rainforest_ro` role, in a read-only transaction with a statement timeout.
`:dataset` in the console is replaced with the current dataset id.

```sql
-- all text fields
SELECT name, null_pct, avg_length FROM meta.v_fields WHERE dataset_id = 1 AND inferred_type = 'text';

-- fields with more than 20% missing that I'm keeping
SELECT name, null_pct, imputation FROM meta.v_fields WHERE dataset_id = 1 AND null_pct > 0.2 AND action = 'keep';

-- how one field is prepared for each model family
SELECT family, imputation, transform, scaling, encoding, source FROM meta.v_field_settings WHERE field_id = 42;

-- compare trained models on the test split
SELECT model, target, metric, value FROM meta.v_model_metrics
WHERE dataset_id = 1 AND split = 'test' AND metric IN ('roc_auc', 'roc_auc_ovr_macro') ORDER BY target, value DESC;
```

| Table / view | Contents |
|---|---|
| `meta.datasets` | uploads: status, progress, counts, targets, active `model_family` |
| `meta.fields` | inferred type, confidence and reasons, typed statistics, `leakage_flag`, `stats` / `charts` / per-family `suggestion` JSONB |
| `meta.field_decisions` | shared decision per field (keep / drop / target, type, notes, reviewed) |
| `meta.field_settings` | preprocessing per field **per model family** |
| `meta.decision_history` | append-only audit log of every change (with family and, for bulk edits, `batch_id`) |
| `meta.bulk_batches` | each applied bulk edit: summary and a pre-change snapshot used by undo |
| `meta.field_correlations` | pairwise associations (one row per unordered pair) |
| `meta.exports`, `meta.model_runs` | export jobs; training runs with metrics and report |
| `meta.analyses` | Explore jobs (map, tests, importance): params, data version, status, result JSONB |
| `meta.v_fields`, `meta.v_field_settings`, `meta.v_correlations`, `meta.v_model_metrics` | query-friendly views |
| `raw.ds_<id>` | the uploaded data (sanitized column names; see `meta.fields.db_column`) |

**Schema changes** go in `api/app/migrations/NNN_name.sql`. The API applies new migrations once on startup, under an
advisory lock, and records them in `meta.schema_migrations`. Views live in `api/app/views.sql` and are recreated on
every start.

## Configuration

Compose settings, read from `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | rainforest | database name and owner |
| `READONLY_PASSWORD` | rainforest_ro | password for the SQL console's `rainforest_ro` role |
| `UI_PORT`, `API_PORT`, `DB_PORT` | 8080, 8000, 5432 | published ports (`API_PORT` and `DB_PORT` bind to 127.0.0.1) |
| `API_WORKERS` / `JOB_WORKERS` | 2 / 2 | uvicorn workers / background job threads per worker |
| `JOB_SLOTS` | 2 | background jobs running at once across all workers; each uses `cpu_count // JOB_SLOTS` threads |
| `LOG_LEVEL` | info | API log level |
| `API_EXTRA_PIP` | – | extra packages baked into the API image (e.g. `xgboost`) |

The API reads further settings from its environment (`api/app/config.py`). Compose doesn't pass these through from
`.env`, so set them under the `api` service's `environment:`:

| Variable | Default | Purpose |
|---|---|---|
| `PROFILE_SAMPLE_ROWS` | 500000 | rows for quantiles, histograms and text stats (counts are always exact) |
| `CORR_SAMPLE_ROWS` | 100000 | rows for associations |
| `CORR_MAX_LEVELS` | 50 | categorical levels kept for associations |
| `COPY_BATCH_ROWS` | 100000 | rows per batch when loading `raw.ds_<id>` |
| `EXPORT_CHUNK_MB`, `EXPORT_FIT_MB` | 256, 512 | export memory budgets (dense output per transform chunk, and for the fit sample) |
| `SAMPLE_MAX_CELLS` | 60000000 | cap on rows × fields for generated samples (about 2 GB peak) |
| `DB_POOL_MAX` | 10 | connections per API worker |
| `SQL_TIMEOUT_MS`, `SQL_MAX_ROWS` | 15000, 2000 | SQL console guard rails |

## Development

```bash
# API (needs Postgres: `docker compose up -d db`)
cd api && python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
DATABASE_URL=postgresql://rainforest:rainforest@localhost:5432/rainforest DATA_DIR=./data uvicorn app.main:app --reload

# UI (proxies /api to :8000; set API_URL to point elsewhere)
cd ui && npm ci && npm run dev    # http://localhost:5173
npm run typecheck

# A sample file to test the upload path
make sample-csv
```

**Tests** live in `api/tests` (105 tests). Most need no database; the 13 in `test_explore.py` and `test_jobs.py` that
do run against an empty Postgres and skip unless `RAINFOREST_TEST_DATABASE_URL` is set.

```bash
make test                         # API suite locally; DB-backed tests skip
make test-db                      # full suite against a throwaway postgres:16 container (needs a local Docker daemon)
RAINFOREST_TEST_DATABASE_URL=postgresql://user:pw@host:5432/db pytest   # from api/, against your own empty database
```

**CI** (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull requests: the API tests with a Postgres
service (so nothing skips), the UI typecheck and build, and `docker compose build` for the images.

## Operational notes and limits

* **Background jobs** (ingest, target associations, export, training, Explore analyses) run in a thread pool inside
  the API process, and their state lives in Postgres. At most `JOB_SLOTS` run at once across all workers; the rest
  wait as *Queued*. Each job row records the process that runs it, and when a worker starts it fails the queued and
  running jobs of processes that are gone, leaving its siblings' jobs alone. For multi-node deployments, swap
  `app/jobs.py` for a queue (RQ, Celery, Arq).
* **Disk:** uploads stream to the `/data` volume, and Postgres keeps a full typed copy (`raw.*`). Profiling, Explore
  and export read the Parquet copy. Budget roughly 2–3× the file size.
* **Samples:** text-field and chart statistics come from a sample, and the UI labels them as such. If you change a
  field's type, the stored statistics still describe the inferred type (the UI says so). The export uses your type.
* **Security:** there's no authentication. Keep the ports bound to localhost (the default for `db` and `api`) or put
  the stack behind your SSO proxy.
