# Explore

*Explore* has four views over one dataset: *Correlations*, *Map*, *Tests* and *Importance*. Terms are
defined in [Concepts](concepts.md).

## The toolbar

The views share a toolbar. Its state lives in the URL, so you can share a link to a view.

| Control | What it sets |
|---|---|
| *Data* | *Raw fields* (with a *Kept only* toggle), or a finished export: its train split, with features as encoded. |
| *Decisions* | For raw fields: whose keep, drop and types the views use (`?decisions=` in the URL). See below. |
| *Target* | One of the targets of the decisions in use. An experiment may have its own targets. Without a target, the views that need one say so and link to the target picker. |
| *Sample* | 1k, 5k (default), 10k or 20k rows. The sample is seeded and stratified on a categorical target. Rows with no target value are left out. |

*Decisions* offers:

- the current baseline as it stands. While you assess it, *Kept only* uses your own choices.
- any finalized baseline
- a team experiment's copy, adjusted or not

*Explore* is shared. Results are cached per decision source, so they never mix people's choices.

## Which fields are used

In raw mode, the views use the kept fields (*Kept only*) or every non-target field. Each field is typed by its
decision.

- Text, identifier and constant fields are skipped and listed. So are fields constant in the sample.
- Categoricals keep their top 20 levels, plus *Other* and *(missing)*.
- Dates become epoch seconds.
- A numeric target with more than 20 distinct values is treated as numeric, unless it is predicted as classes.
  Otherwise it is treated as classes.

On export data, a target predicted as a number is numeric. Every other target is classes.

## The four views

| View | What it computes |
|---|---|
| *Correlations* | A clustered heatmap of the stored associations, for raw data only (no background job). It shows the target and the 59 fields most associated with it, or the 60 fields with the strongest associations overall. They are ordered by average-linkage clustering on 1 − strength. A slider hides weak cells. You can add fields beyond the 60. Click a cell to see the pair, with *Open field* links to *Fields · Focus*. |
| *Map* | A 2-D embedding of the sampled rows, coloured by the target. **t-SNE** is the default: PCA to 50 components first when there are more features, perplexity 5–100, default 30. **UMAP** is the other option: neighbours 5–100, default 15. The seed is fixed. Numerics are median-filled and standardised, categoricals one-hot. Hover shows the 5 fields that spread the rows most (largest weight in the first two principal components). |
| *Tests* | One test per field against the target, with Benjamini–Hochberg adjusted p-values. The UI marks adjusted p < 0.05. Numeric field × class target: one-way ANOVA (effect η²) plus Kruskal–Wallis, with box plots per class. Categorical field × class target: χ² (effect Cramér's V), with level counts per class. Numeric target: Spearman (effect \|ρ\|) plus Pearson, with a scatter of up to 500 points; or ANOVA of the target across the field's levels. Classes with fewer than 5 sampled rows are excluded and listed. |
| *Importance* | Mutual information of each field with the target (needs a target). Also PCA: explained variance for up to 20 components with the cumulative line, plus loadings of the first 5 components for the top 20 features. |

How associations are measured is in [Data preparation](data-preparation.md#associations-and-the-leak-screen).

## Caching and out-of-date results

*Map*, *Tests* and *Importance* run as background jobs. Their results are cached in `meta.analyses`, per set
of parameters.

A result is tied to the decisions it was computed from: every field's keep, drop, target and type. For export data,
it is tied to the export itself.

- After you change a decision, the cached result shows as *Out of date*, with a *Recompute* button. It isn't
  recomputed automatically.
- A new request for the same view supersedes one still running.
- When the sample can't support an analysis (no target, too few rows or fields, a single class), the view explains
  why instead of failing.
