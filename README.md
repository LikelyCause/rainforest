# 🌲 Rainforest

Rainforest prepares wide, mixed-type data for **tree-based models**. Ingest a dataset, review it **one field at a time**
in a fast UI, store keep / drop / imputation / encoding decisions in Postgres, export a **model-ready** dataset, and
train a **`ClassifierChain`** of LightGBM (or another tree ensemble) that predicts `column_a` and `column_b`.

Trees are the default and the main path. Rainforest knows that tree ensembles split on value *order*, so it never
suggests scaling or log transforms for them, and it leaves missing values for the model to route. Linear and neural
model families are also available. When you switch to one, it gets its own preprocessing settings (scaling,
imputation, one-hot encoding) without touching the tree settings.

```
┌────────────┐  /api/*   ┌──────────────────────────┐        ┌────────────────────────────────┐
│  ui        │ ────────► │  api (FastAPI)           │ ─────► │  db (Postgres 16)              │
│  React +   │  nginx    │  ingest · profile ·      │  SQL   │  meta.*  fields, stats,        │
│  ECharts   │  proxy    │  correlate · export ·    │        │          decisions, settings,  │
│  :8080     │           │  train ClassifierChain   │        │          exports, model runs   │
└────────────┘           └─────────┬────────────────┘        │  raw.ds_<id>  your data        │
                                   │ /data volume             └────────────────────────────────┘
                                   │ Parquet copies, exports, trained models
```

## Quick start

```bash
cp .env.example .env          # optional — every value has a default
docker compose up -d --build
open http://localhost:8080
```

1. **Datasets** → *Generate a sample dataset* (50k rows × 150 fields), or upload your own CSV / Parquet.
   The upload is converted to Parquet, loaded into `raw.ds_<id>`, profiled and correlated, with live progress.
2. **Review** → walk the fields. Each field shows its statistics, distribution, quantile strip, top tokens or weekdays,
   the **top-20 associated fields** (click a bar to jump to that field), and its association with each target.
   Decisions autosave.
3. **Query** → find fields with filters (*text fields*, *missing > 20%*, *most target-associated*, …) and apply bulk
   actions, or use the read-only **SQL console**.
4. **Export & train** → fit the preprocessing on the training split, write the model-ready Parquet, and train
   chains with any compatible model. Runs are compared side by side, and the best result in each column is starred.

### Keyboard shortcuts (Review)

| Key | Action |
|---|---|
| `J` / `→` · `K` / `←` | next / previous field |
| `U` | next unreviewed field |
| `X` | toggle keep / drop |
| `A` | accept the suggestion, mark reviewed, next |
| `Enter` | mark reviewed, next |
| `/` | search fields |

## Decisions: shared vs per model family

| Shared by every family | Per family (`meta.field_settings`) |
|---|---|
| keep / drop / target, semantic type, notes, reviewed | imputation, missing-indicator, transform, scaling, encoding, encoder params |

The top-bar **Preparing for** switch selects which family's settings you're reviewing. It defaults to tree ensembles.
Each family has its own suggestions, so switching never overwrites another family's choices. If you change a field's
type, the other families' settings are re-derived for the new type.

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

## Export → ClassifierChain

* **Leakage-safe split:** the train / valid / test split is assigned first, stratified on the target combination.
  Every statistic is then fitted on training rows only (up to 500k of them), and the full file is transformed in
  200k-row chunks.
* **Output:** float32 features, `__split`, `__row_id` (joins back to `raw.ds_<id>`), label-encoded targets, and
  `manifest.json` (feature → source field, class mappings, model family, NaN report).
* **Compatibility:** tree models train on any export. Linear and neural models need an export prepared for their
  family, and they're refused if the export contains NaN. The UI disables incompatible models and shows the reason.
* **Chain:** `ClassifierChain(model, order=[column_a, column_b], cv=5)`.
  * The chained feature is **out-of-fold** during training, so `column_b`'s model learns from realistic
    `column_a` predictions rather than the ground truth.
  * Both targets binary → P(class 1) is passed down the chain; any multiclass target → the predicted class code.
  * NaN-capable models are wrapped so sklearn's finiteness check doesn't reject the NaN they handle natively.
* **Metrics:** accuracy, balanced accuracy, macro-F1, log-loss and ROC-AUC per target on the valid and test splits,
  plus importances per chain link.

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

Connect any SQL client to `localhost:5432` (user and password from `.env`), or use the SQL console in the UI.
The console runs as the SELECT-only `rainforest_ro` role, in a read-only transaction with a statement timeout.

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
| `meta.decision_history` | append-only audit log of every change (with family) |
| `meta.field_correlations` | pairwise associations (one row per unordered pair) |
| `meta.exports`, `meta.model_runs` | export jobs; training runs with metrics and report |
| `meta.v_fields`, `meta.v_field_settings`, `meta.v_correlations`, `meta.v_model_metrics` | query-friendly views |
| `raw.ds_<id>` | the uploaded data (sanitized column names; see `meta.fields.db_column`) |

**Schema changes** go in `api/app/migrations/NNN_name.sql`. The API applies new migrations once on startup, under an
advisory lock, and records them in `meta.schema_migrations`. Views live in `api/app/views.sql` and are recreated on
every start.

## Configuration

Environment variables, set in `.env` or on the `api` service:

| Variable | Default | Purpose |
|---|---|---|
| `API_EXTRA_PIP` | – | extra packages baked into the API image (e.g. `xgboost`) |
| `PROFILE_SAMPLE_ROWS` | 500000 | rows for quantiles, histograms and text stats (counts are always exact) |
| `CORR_SAMPLE_ROWS` | 100000 | rows for associations |
| `JOB_WORKERS` / `API_WORKERS` | 2 / 2 | background job threads per worker / uvicorn workers |
| `SQL_TIMEOUT_MS`, `SQL_MAX_ROWS` | 15000, 2000 | SQL console guard rails |

## Development

```bash
# API (needs Postgres: `docker compose up -d db`)
cd api && python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
DATABASE_URL=postgresql://rainforest:rainforest@localhost:5432/rainforest DATA_DIR=./data uvicorn app.main:app --reload
pytest                            # 34 tests, no database needed

# UI (proxies /api to :8000)
cd ui && npm ci && npm run dev    # http://localhost:5173

# A sample file to test the upload path
make sample-csv
```

## Operational notes and limits

* **Background jobs** (ingest, export, training) run in a thread pool inside the API process, and their state lives
  in Postgres. A restart marks in-flight jobs as failed. For multi-node deployments, swap `app/jobs.py` for a queue
  (RQ, Celery, Arq).
* **Disk:** uploads stream to the `/data` volume, and Postgres keeps a full typed copy (`raw.*`). Profiling and export
  read the Parquet copy. Budget roughly 2–3× the file size.
* **Samples:** text-field and chart statistics come from a sample, and the UI labels them as such. If you change a
  field's type, the stored statistics still describe the inferred type (the UI says so). The export uses your type.
* **Security:** there's no authentication. Keep the ports bound to localhost (the default for `db` and `api`) or put
  the stack behind your SSO proxy.
