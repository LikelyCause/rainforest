# Data preparation

How Rainforest reads your data, what it suggests for each field, and why. Terms are defined in
[Concepts](concepts.md).

## Connected datasets

A dataset can be created from one or more MongoDB collections instead of an uploaded file. Rainforest pulls the
documents into the same Parquet copy and raw table an upload gets, profiles them the same way and remembers where they
came from. Review, baselines, Explore, exports and training treat the dataset like any other. A lead can pull again
with *Refresh*, which carries the team's decisions forward.

### Creating one

Connections are named by the deployment, in `RAINFOREST_MONGO_SOURCES` for Compose or `sources.mongodb` for the
chart (see [Configuration](configuration.md) and [deploy/README.md](../deploy/README.md#mongodb-sources)). Nobody types
a connection string into Rainforest. *Datasets → Connect ▾ → MongoDB collection…* appears when at least one is named.
To try it without a cluster, `make up-mongo` starts a sample MongoDB (database `rf_dev`, collections `curated` and
`massive`) as the connection `dev`; [Deployment](deployment.md#trying-an-option-for-one-run) says what it changes, and
what to add when connections are already set.

Members and leads can create a connected dataset. In the dialog:

1. Choose the **connection**, the **database** and the **collections**. List the most trusted collection first.
2. Optionally give each collection a **filter** and a **projection**, as JSON documents. MongoDB's extended JSON
   (`{"$oid": …}`, `{"$date": …}`) is accepted. They are passed to the server as written, so connect with a read-only
   account.
3. **Preview** samples 200 documents per collection and shows the flattened columns with their types and how often
   each is present, the columns flagged (arrays, mixed types), the columns exact duplicates would be compared on
   ([below](#the-source-column-and-duplicates)), and the estimated rows and cells against the [size budget](#size).
4. Name the dataset and choose whether to add a **source column** and to **leave out exact duplicate rows**
   ([below](#the-source-column-and-duplicates)).

*Connect* stays off until the collections as chosen have been previewed, the dataset has a name and every **Match on**
column is one the preview found; the dialog says what is missing. A label change keeps the preview; another filter,
projection or collection order needs a new one. Choosing another connection or database starts the collections and
options over.

The pull runs as a background job. The dataset is *Pending* while it waits, then *Ingesting* while the documents are
pulled and combined, and then goes through the same profiling, associations and baseline set-up as an upload. The
*Datasets* page marks it as connected and shows when it was last pulled, or that the last pull failed. A pull that
fails (the server is unreachable, the account is refused, the size budget is exceeded, the time limit passes) leaves
the dataset *Failed* with the reason. Error messages name the connection, never its address or credentials.

### How documents become columns

Each document is flattened to one row by these rules, and the column set is the union of the keys over every pulled
document, in the order first seen. A key that first appears in document 400,000 is still a column.

| Value in the document | Becomes |
|---|---|
| nested document `{a: {b: 1}}` | the column `a.b`; dots inside a key are replaced by `_`, and nesting can be any depth |
| array of numbers, text or documents | one text column holding the array as JSON; it profiles as text and can be dropped |
| `ObjectId`, `UUID`, `Binary` | text (binary other than a UUID as base64); `_id` is kept and profiles as an identifier |
| `Int32`, `Int64`, `Double`, `Decimal128` | a number (`Decimal128` as a float) |
| `Date` | a datetime |
| `true`, `false` | a boolean, which profiles as a binary field |
| `null`, or a missing key | missing |
| a key whose values have different kinds across documents (numbers and text, say) | the whole column read as text, and listed in the dataset's warnings |

The names follow what `mongoexport --type=csv` writes, so a baseline built on a CSV export of a collection matches the
connected dataset's fields by name. Field names keep the dots (`a.b`); the raw table that the SQL console reads cleans
every column name to lowercase letters, digits and underscores (`a_b`), as for an upload.

Edge cases:

- **Empty-string keys.** A key `""` gives a column with an empty name, and `{a: {"": 1}}` gives `a.`. Field names keep
  such oddities; in the raw table they become `col_<position>` and `a`.
- **A field that is always `{}`** produces no column: an empty document has no values to flatten, so the field
  vanishes. One that is `{}` in some documents and has keys in others is a column wherever it has keys.
- **Mixed types** are read as text, with numbers and booleans written in their text form, so nothing is lost. The type
  is then inferred from the text as for an upload.
- **Arrays** are JSON text. Elements that are documents keep their keys inside the text, and BSON types inside become
  strings or numbers.
- **Keys that collide** after the dot replacement (`{"a.b": 1, "a_b": 2}` would give `a_b` twice). Within a document,
  keys without a dot keep their names, and a dotted key that would take a name already in use gets a `_2`, `_3`, …
  suffix. Across the pull, the first name given to a key path stays its name in every later document, and a later path
  that would take a name already in use gets the next free suffix, so a key without a dot gets the suffix when its
  dotted twin appeared in an earlier document. The warning lists each rename.

### The source column and duplicates

With several collections, a `source` column names the collection each row came from (or the label typed for it in the
dialog). It is an ordinary categorical field, reviewed and droppable like any other. It is on, and locked on, when more
than one collection is chosen, and optional for one. If the documents have a field named `source`, the column is
`source_2`.

Leaving out exact duplicates removes rows that are equal on the **comparison columns**:

- By default these are the columns every chosen collection has, less `_id`, the `source` column and any column of
  kind identifier: every value an `ObjectId`, `UUID` or `Binary`, or every value text, or every value a number, under
  an identifier-like name (ending in `id`, `uuid`, `key`, `code` and the like, as in `vehicle_id` or `extra.key`),
  because each collection's own keys would make every row distinct. Columns that only some collections have never
  take part, so a row of a collection with extra fields can still match a row without them. The preview lists these
  columns from its sample; the pull applies the same rule to every document it reads, so the two differ only when
  documents beyond the sample bring a column or a kind of value the sample didn't have.
- A **Match on** list in the dialog replaces them, for example the record's natural key. Each column in it must appear
  in the preview's sample (up to 200 documents per collection), or Connect refuses it, so a key that only turns up
  further down a collection can't be named there.
- Duplicates are removed within a collection as well as across them. Of a group of equal rows, the one from the
  collection listed first stays.

The number removed from each collection is recorded on the pull and listed in the dataset's warnings. When the
collections share no comparable columns, the warning says nothing was compared.

### Size

`SOURCE_MAX_CELLS` (600,000,000 by default) caps the whole dataset: documents × projected fields, summed over the
collections and estimated before duplicates are removed. A pull over it is refused before any document is read in full.
The default fits three collections of the vehicle team's size, about 450 million cells.

The pull streams the documents in chunks of 50,000 to Parquet files, but combining the chunks, removing duplicates and
writing the dataset holds all of it in memory at once. Plan on roughly 2–3 times the dataset's in-memory size while a
pull assembles it: about 8 GiB for 1.5 million rows of 300 columns. Profiling, the raw-table load and the rest of
processing are bounded by `PROFILE_SAMPLE_ROWS` and `COPY_BATCH_ROWS`, as for an upload. One pull may take
`SOURCE_PULL_TIMEOUT_S` (an hour) before the job fails.

### Refresh

*Refresh…* in a connected dataset's menu, for leads, pulls every collection again and replaces the dataset's data.
It always re-pulls all of them: the combined rows and the duplicate removal depend on all of them together.

**It keeps** every field that is still in the data, by name, with its id, so its decisions, history, settings and notes
stay. Derived fields are computed again with the same formulas. A target stays a target while profiling still accepts
it on the new data.

**It replaces** the Parquet copy and the raw table, the fields' statistics and inferred types, and the associations.
Every Explore result of the dataset is marked out of date, since it was computed on the old rows. New fields come in
with suggestions, as at upload. A field that is no longer in the data is deleted with its decisions and history, and so
is a derived field whose inputs vanished; each is listed in the dataset's warnings. A target that is no longer in the
data is deleted like any other field, with the warning *Target column no longer in the data*. A target that profiling
now turns down (a non-numeric one with more than 100 distinct values, as at upload) stays a field but stops being a
target: its decision goes back to the suggestion, unreviewed, and the warning *Not used as target(s)* names it. An
experiment's adjustments on a removed field are dropped, and its Adjustments card says so.

**It reopens nothing by itself.** For each model family with a finalized baseline, it compares the new fields with that
family's latest finalized baseline by the rules for [reusing a baseline](team-workflow.md#reusing-another-datasets-baseline):
a field is *carried* when its type is unchanged and its statistics haven't drifted (the missing share within 10 points,
the distinct values within half to double), and *reopened* when it is new, has a new type or has drifted, with the
numbers. Fields of the baseline that are gone are listed. The result is that pull's **drift summary**, shown for each
family above the baselines on *Versions* and under *Latest pull* in the *Team* page's queue, where a lead can start a
new version with the reopened fields selected. The *Team* page opens on it when the pull came after the version was
finalized. A dataset with no finalized baseline gets no drift summary; its decisions that are still at their
suggestion follow the new suggestions. A pull from before a baseline was finalized, such as the first pull, has no
comparison with it and says so; the next refresh compares them.

**While it runs** the dataset is locked. Its pages open read-only with a banner naming who started the refresh, and
experiments take no exports, runs, notes or adjustments until it ends. The API refuses writes to the dataset and new
exports, runs and Explore computations on it with `409`. A refresh is refused with `409` while a baseline version is
being assessed or is in team review, and while an export or a run of the dataset is running. If the pull fails, the
dataset, its fields and its data stay as they were and the failed pull is recorded with its error.

**Exports after a refresh** are built from the new data. Each export of a connected dataset records the pull it was made
from in its manifest and MLflow parameters, and the exports list shows it on the export's row as *pull #<n>*, where n is
the pull's id, so it stays clear which data a run was trained on; an export made earlier keeps its own Parquet file.
The export preview blocks when a field the baseline keeps, or one of its targets, is no longer in the data (a field the
baseline drops may go), and warns when the baseline was finalized before the latest pull.

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

## Other transforms you can choose

Rainforest suggests the defaults above. It also offers options it never suggests, which you can pick in a field's
*Transform* or *Encoding* menu.

| Option | Fields | Families | What it does |
|---|---|---|---|
| Yeo-Johnson (power transform) | numeric | Linear, Neural / distance | Makes skewed numbers closer to normal, and handles zero and negative values, unlike a log. It doesn't standardize: scaling stays its own setting. |

Tree ensembles aren't offered it, because a monotonic transform can't change a split.

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
new field gets assessed. You can't add or remove one while a version is in progress.
