# Data preparation

How Rainforest reads your data, what it suggests for each field, and why. Terms are defined in
[Concepts](concepts.md).

## Type inference

Rainforest proposes a type for every field, with a confidence score and the reasons. The code is in
`api/app/profiling.py`. The types are `numeric_continuous`, `numeric_discrete`, `categorical`, `binary`, `text`,
`datetime`, `identifier` and `constant`.

It detects:

- numbers stored as text
- dates stored as text, in 15 formats, including `YYYYMMDD` as text or as integers
- integer and UUID keys
- free text, as distinct from short categorical strings

### Details

- Parse rates are measured over the whole column. Formats are chosen from values spread across the file. A clean
  head can't hide junk further down.
- Commas are thousands separators only in valid groups (`"1,234.50"`). A column that writes decimals with a comma
  (`"1,5"`, and no dot decimals) is read as decimals. Any other value with a comma doesn't parse.
- Values with a part above 12 decide between `dd/mm` and `mm/dd`. When nothing decides it, Rainforest assumes
  month-first and the reasons say so.
- Values that don't parse as the inferred type, and ±inf, count as *Missing*, because the export reads them as
  missing. They are also reported as `stats.unparseable_count` (with examples) and `stats.inf_count`, and shown as a
  *Read as missing* signal. Unparseable values cap the type confidence at 0.65.
- An integer column retyped to a date is read as `YYYYMMDD` when its values are valid dates. Otherwise it is read as
  epoch timestamps, with the unit (s, ms, µs, ns) taken from their magnitude.
- Whole numbers that all fall within one unit's 2000–2100 epoch range are inferred as dates in the first place. The
  exception is a name that looks like a key (`…_id`, `key`, `code`).
- A target named at upload stays numeric when its values are numbers. Whole numbers that look like keys or dates are
  read as numbers too.
- A non-numeric target with more than 100 distinct values isn't made a target. The dataset gets a warning instead.
- Integers whose name looks like a key (`…_id`, `…_nbr`, `key`, `code`) are read by what they hold. With more than
  1,000 distinct values that also repeat (a patient number seen on several visits), they are identifiers. With 50 or
  fewer, they are categories (`admission_type_id`).
- Text that is mostly numbers but has letter-prefixed codes (`V45`, `E880` among ICD-9 numbers) is categorical, not a
  number with junk.

### Sampled statistics

Counts are always exact. Text-field and chart statistics come from a sample, and the UI labels them as such. If you
change a field's type, the stored statistics still describe the inferred type, and the UI says so. The export uses
your type.

## Shared and per-family decisions

Part of each decision is shared by every model family. The rest is set per family.

| Shared by every family | Per family (`meta.field_settings`) |
|---|---|
| keep / drop / target, *Predict as*, semantic type, notes, reviewed | imputation, missing indicator, transform, scaling, encoding, encoder parameters |

A baseline's family decides which settings you review, export and log. *Fields · Focus*, *Fields · Table* and the
decision panel show that family's settings only. A baseline's snapshot stores only those.

Each family has its own suggestions. A baseline started from a baseline of another family copies the shared
decisions. It keeps each field's settings where they are valid for the new family, and takes the new family's
suggestion for the rest. If you change a field's type, its settings are re-derived for the new type.

Settings a family doesn't use appear as a locked line with the reason. For example: *"Not used for tree models — they
split on value order, so rescaling changes nothing."* The API rejects them too
(`scaling='standard' is not valid … with tree models`).

## Per-family defaults

Tree ensembles split on the order of values. So Rainforest never suggests scaling or log transforms for them, and it
leaves missing values for the model to route.

| | **Tree ensembles** (default) | Linear models | Neural / distance |
|---|---|---|---|
| Models | LightGBM, XGBoost\*, histogram GBM, random forest | logistic regression (ridge for a number) | MLP |
| Missing numerics | left as NaN; the model learns where to send them | median, plus a "was missing" flag from 1% missing | same as linear |
| Skew / outliers | nothing: splits only depend on order | signed log1p when \|skew\| > 1, else winsorize above 3% outliers | same as linear |
| Scaling | **not offered**: it can't change a split | standard | quantile-normal when \|skew\| > 1, else standard |
| Categoricals | ≤ 8 levels one-hot, ≤ 1,000 ordinal codes, else frequency | ≤ 200 levels one-hot (top 30 kept), else frequency | same as linear |
| Binary | 0/1; missing takes the most frequent level under 5% missing, else a "was missing" column | same | same |
| Text | TF-IDF → 32 SVD components | same | same |
| Dates | year, month, day, weekday, hour, day-of-year, epoch days | + median impute + scaling | same |

\* XGBoost trains on the CPU in the default image and on an NVIDIA GPU in the GPU image. See
[GPU](deployment.md#gpu).

## Associations and the leak screen

Rainforest measures how strongly each pair of fields is associated. It uses a seeded 100k-row sample
(`CORR_SAMPLE_ROWS`).

| Pair | Measure |
|---|---|
| Numeric × numeric | Spearman ρ, over the rows where both are present |
| Categorical × categorical | Bias-corrected Cramér's V, with missing as its own level |
| Numeric × categorical | Correlation ratio η |

A field with an association of 0.95 or more to a target is flagged as a **possible leak** and suggested for dropping.

### Details

- Spearman runs as one vectorised pass that ranks each column over its own values. Some pairs have different missing
  rows and a first estimate of at least 0.3. Those pairs are re-ranked within their shared rows, so missing values
  don't understate a monotone relation.
- For Cramér's V, each categorical keeps its 50 most common levels (`CORR_MAX_LEVELS`) and groups the rest.
- **Leak screen.** Categoricals with more levels than that, and text fields (compared by exact value), are also
  measured against each target with every level kept. That value replaces the capped one. So a 500-level code that
  determines the target is flagged.
- The leak screen runs during processing. It runs again in the background whenever you add a target.

## Predicting numbers and bands

Each target has a *Predict as* setting in the decision panel in *Fields*. It is part of the target's decision, like keep
or drop and type.

| Predict as | What it means | Default for |
|---|---|---|
| *Classes* | One of the field's values. Allowed for a target with at most 100 distinct values. | binary and categorical targets, and discrete numbers with at most 100 values |
| *Number* | The value itself (regression). | continuous numbers, and discrete numbers with more than 100 values |
| *Bands* | The number cut into classes. | — |
| *One vs rest* | For a class target: the classes you choose as positive, against everything else. | — |

You define bands in one of two ways:

- *Equal-sized*: 2–20 bands, each with about the same number of rows.
- *Cut points* you type. For example, `10, 25, 80` gives `< 10`, `[10, 25)`, `[25, 80)` and `≥ 80`.

The panel previews how many rows fall in each band, on the whole column. An export cuts equal-sized bands on its
training rows only.

*One vs rest* makes a binary target from a categorical one. For example, `readmitted` becomes `<30` against
`(rest)`.

### Rules

- *Number* and *Bands* are for numeric targets only.
- A continuous target can't be predicted as classes.
- Two assessors who choose a different *Predict as*, or different bands, disagree on the target's core decision. The
  field goes to the [review queue](team-workflow.md#team-review).
- The choice is part of a baseline's snapshot. An experiment can change it like any other adjustment. It shows as
  *Predict as* in the experiment's diff.

How each choice is written to an export is in
[Experiments and training](experiments-and-training.md#targets).

## Derived fields

*Fields · Derived* adds columns computed from the dataset's own fields. There are three kinds:

| Kind | Example |
|---|---|
| A formula over number fields, with `log`, `sqrt`, `min`, `clip` and other functions | `water / (cement + blast_furnace_slag + fly_ash)` |
| A field's values put into named groups, by exact values or ranges | ICD-9 codes into chapters: `390-459, 785` → *Circulatory* |
| How many of the chosen fields hold given values | medications marked `Up` or `Down` |

*Preview* shows the new field's type, missing count and first values.

A derived field is profiled, reviewed, exported and trained on like any other. You can add or remove one while the
baseline is still being set up, or once it is finalized. After a baseline is finalized, start a new version so the
new field gets assessed. You can't add or remove one while a version is assessing.
