# Explore tab: dataset analyses

Purpose, in priority order: understand the dataset before modeling, check review decisions, and report.

## UI

Topbar steps become Review · Bulk edit · Explore · Export & train. Route `/d/:id/explore/:view?` with
view one of `correlations | map | tests | importance` (default `correlations`). All toolbar state lives in the URL.

Shared toolbar:
- **Data**: `raw` (all fields, with a "Kept only" toggle `kept=1`) or `export:<id>` (a finished export's model-ready train split).
- **Target**: one of the dataset's targets (hidden when there are none).
- **Sample**: 1k / 5k (default) / 10k / 20k rows.

Views:
1. **Correlations**: clustered heatmap of association strength, threshold slider, search to add fields, click a cell for a side
   panel with both fields' values, the method, n, and links to Review. Raw mode only; reads `meta.field_correlations`, no job.
   Default cap: the 60 fields most associated with the target (or highest max association with no target).
2. **Map**: 2-D embedding scatter coloured by target. t-SNE default (PCA to 50 first, perplexity control), UMAP optional
   (n_neighbors control). Fixed seed. Hover shows the row's target and top-variance values.
3. **Feature tests**: needs a target. Ranked table: field, test, statistic, effect size, BH-adjusted p, n. Detail panel:
   box plots per class (numeric field) or stacked level bars (categorical field).
   - numeric field × categorical target: one-way ANOVA + Kruskal-Wallis, effect η²
   - categorical field × categorical target: χ², effect Cramér's V
   - numeric target: Pearson + Spearman, effect |ρ|
   - categorical field × numeric target: ANOVA/Kruskal on the target grouped by field level, effect η²
   Target classes with fewer than 5 rows are excluded and the exclusion is listed.
4. **Importance**: needs a target for MI. Mutual information per field (sorted bars) and PCA: explained variance with a
   cumulative line plus a loadings heatmap for the first 5 components (top 20 contributing features).

Each job-backed view shows one of: computing (progress), ready with "as of <time>", out of date with Recompute, failed with Retry,
or a "not enough to run" explanation with a link (e.g. to the target picker).

## Data preparation (raw mode)

Read a seeded row sample from the dataset's Parquet copy, stratified on the target when one is chosen.
Fields used: with kept-only on, fields whose action is keep; with it off, every field except targets. Effective type is the decision's semantic type.
- numeric: median-fill, standardise
- categorical / binary: one-hot of the top 20 levels plus "other"; for MI and tests the integer codes are used directly
- datetime: epoch seconds, then as numeric
- text, id, constant fields: skipped and listed as skipped
Export mode: the export's `model_ready.parquet` rows where split = train, feature columns as-is, target from the export.

## Jobs and caching

Table `meta.analyses`: id, dataset_id (cascade), kind (`map | tests | importance`), params JSONB (normalised: data, kept,
target, sample, method, perplexity / n_neighbors), params_key TEXT (canonical JSON), data_version TEXT, status
(`pending | running | done | failed`), progress, status_message, error, result JSONB, created_at, updated_at, finished_at.

data_version: raw mode = md5 of the ordered (field_id, action, semantic_type) list for the dataset, computed on request, so every
write path (single save, bulk, undo, target change) is covered without touching them. Export mode = `export:<id>`.

API:
- `POST /api/datasets/{id}/analyses` body `{kind, params, recompute?: bool}` → the analysis row. Returns the newest done row with
  the same params_key and current data_version if any (unless recompute); else a running/pending one for the same key and version;
  else creates one and submits a job. Response adds `stale: bool` (done row exists only for an older version — returned with
  `stale: true` and no new job unless recompute) so the UI can show "out of date · Recompute".
- `GET /api/analyses/{id}` → row (for polling).
- `GET /api/datasets/{id}/explore/correlations?kept=0|1&target=&limit=60&add=<field ids>` → `{fields: [{id, name, type}], order:
  [ids], cells: [{a, b, method, value, strength, n}] }` with clustered order.
- Not-enough-data cases return the row with status `failed` and `error` beginning "Not enough" plus `result.reason`; never 500.

One running analysis per dataset: a new job request cancels (marks failed "Superseded") older pending/running rows for that
dataset; jobs check their row status between steps and stop when superseded. Startup marks interrupted running rows failed.
Prune: keep the newest 3 rows per (dataset, kind) that are not current. Dataset delete cascades.

## Dependencies

`umap-learn` added to the API image.

## Testing

Backend unit tests on synthetic data with known answers (planted separating feature tops tests and MI, correlated pair
clusters together, PCA variance sums ≤ 1). API tests for caching, staleness after a decision edit, recompute, no-target and
tiny-sample cases. UI: Playwright over the four views on the sample dataset, light and dark.
