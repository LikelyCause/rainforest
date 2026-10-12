# Missing values and the data check

Goal: a person can say which values in their data mean "missing" (a literal `EMPTY`, a `-999`), choose how each field
handles missing values in plain words (leave them missing, count them as their own category, fill them, or leave
those rows out), and gets a data check right after upload that points out odd columns and asks the few questions
only a person can answer: is this text a number, is this a date and in which format, is this token a blank.

From the owner's feedback (2026-10-12):
- "In the column, it has the literal text EMPTY instead of being blank or being a number. Those should be null."
- "I'd rather it ask me how I want to handle missing, providing the option to push them to NaN's or to drop those
  rows. A missing label doesn't really help me."
- "Warn users that specific fields have weird values and the type can't be auto-detected … ask the user if they want
  to auto-convert date fields to a date type, and perform other validation steps as necessary."

Decisions:
- **Two layers, kept apart.** The data check decides how values are *read*: which values are missing, whether text is
  stored as numbers or dates. These are facts about the dataset, shared by every baseline, Explore and SQL. The
  baseline decides how fields are *used*: type, keep or drop, missing handling, encoding. Those stay assessed blind
  by the team. The data check never sets a type, an action or a setting.
- **Data rules are applied by rewriting the data, not at read time.** One cleaned Parquet copy and one rebuilt raw
  table, so Fields, Explore, the SQL console, exports and derived fields can't disagree. The as-uploaded copy is
  kept so rules can change later.
- **Missing values get one plain-words choice per field.** *Leave missing* is offered for every type a family can
  take NaN for, categoricals and binary included. `__missing__` becomes *Count missing as its own category*, with no
  label to type. *Leave out rows where it's missing* is a choice in the same list.
- **The data check runs in the pipeline and opens with the dataset**, which is usable at once. Starting the first
  baseline asks for the check to be answered, or skipped in one click.
- Before the first baseline starts, anyone who can upload answers the check. After that, only a lead, and only while
  no version is assessing or in team review (the rule derived fields already follow).

Built as four PRs, smallest useful first: (1) missing-value choices, (2) leaving out rows, (3) data rules (missing
tokens), (4) the data check. See [Phases](#phases).

## 1. Missing-value choices (per field, per family)

The setting stays `imputation` in `meta.field_settings`, `assessment_settings` and `experiment_settings`. Two values
are added and the labels are rewritten. The UI row is still called *Missing values*.

| Value | Label | Types |
|---|---|---|
| `none` | Leave missing | numbers, dates, categorical, binary, when the family handles NaN |
| `own_level` (new) | Count missing as its own category | categorical, binary |
| `most_frequent` | Fill with the most frequent value | numeric discrete, categorical, binary |
| `median` / `mean` | Fill with the median / mean | numbers (median also dates) |
| `constant` | Fill with a value… | numbers (a number), categorical and binary (an existing level or any text) |
| `constant` with `""` | Treat as empty text | text (its only fill) |
| `drop_rows` (new) | Leave out rows where it's missing | every type except identifier and constant (see 2) |

`_BASE_ALLOWED` in `semantics.py` becomes:

| Type | Options |
|---|---|
| numeric continuous | none, median, mean, constant, drop_rows |
| numeric discrete | none, median, mean, most_frequent, constant, drop_rows |
| categorical | none, own_level, most_frequent, constant, drop_rows |
| binary | none, own_level, most_frequent, constant, drop_rows |
| text | constant, drop_rows |
| datetime | none, median, drop_rows |

`allowed()` already removes `none` for families without `handles_nan`; that rule extends from numbers and dates to
categorical and binary. The missing flag (`add_missing_indicator`) stays numbers-only.

### How the export reads each choice (`transform.py`)

Today `_categorical_pipeline` always puts a `SimpleImputer` first (constant `__missing__` or most frequent), so a
categorical never reaches the model as NaN. With `none`:

| Encoding | Leave missing | Why it works |
|---|---|---|
| Ordinal | no imputer; `OrdinalEncoder(encoded_missing_value=np.nan)` (sklearn 1.6's default) | a NaN code; LightGBM, XGBoost, histogram GBM and random forest (sklearn ≥ 1.4) all route NaN in a numeric column |
| One-hot | no imputer; a small `MissingOneHot` wrapper fits on present values and writes NaN in every one of the field's columns for a missing row | without it, `OneHotEncoder` makes `nan` its own category, which is *own category* under another name |
| Frequency | `FrequencyEncoder` maps a missing value to NaN instead of the frequency of the text `"nan"` | today `astype(str)` turns missing into a level |
| Binary | `BinaryEncoder(fill=None)` writes NaN and adds no `__isna` column | today a fill not among the levels means 0 plus an indicator |

`own_level` is today's behaviour: `SimpleImputer(strategy="constant", fill_value="missing")` for categoricals (the
feature reads `color_missing`, not `color___missing__`); for binary, 0 plus the `__isna` column. `constant` keeps
`impute_value`. Numbers and dates don't change.

The export's existing NaN guard (`run_export`: "linear models need complete data…") stays as the backstop.

### Defaults

| | Trees | Linear | Neural |
|---|---|---|---|
| Numbers | leave missing (as now) | median, plus the missing flag from 1% missing (as now) | same as linear |
| Categorical | **leave missing** (was its own category) | its own category | same |
| Binary | **leave missing** (was most frequent under 5% missing, else a flag) | most frequent under 5% missing, else its own category | same |
| Text | empty text | same | same |
| Dates | leave missing | median | same |

`drop_rows` is never suggested. Reasons follow the new words: *"Missing values: leave missing; 4.1% are missing and
tree models learn where missing rows go."*

### Existing decisions

Migration `0NN_missing_choices.sql` (the next free number; `021` is taken by work in progress) rewrites, in all
three settings tables and in `meta.fields.suggestion`:
`imputation = 'constant'` with `impute_value = '__missing__'` on a categorical or binary field → `own_level`,
`impute_value = NULL`. Other constants stay *Fill with a value…* with their text. Finalized snapshots are frozen, so
`semantics.normalize_settings()` does the same mapping when a snapshot is read (export, diff, reuse). The new tree
defaults apply to fields profiled after the change (new uploads, and re-profiling by the data check); existing
working decisions keep what they have, so nothing in progress flips under an assessor.

### Team review

The settings are prep settings: adopted by the most common choice, as today. One exception: `drop_rows` against
anything else counts as a **disagreement** and goes to the review queue, because it changes which rows the model sees.
`assessment.disagreements` compares, besides `core_key`, whether each assessor's setting for the baseline's family is
`drop_rows`.

## 2. Leaving out rows where a field is missing

It is a field's choice, because that is where the owner looks ("how do I want to handle missing for this field").
The export carries it out as a row rule, because only the export knows the rows.

- `transform.row_rules` gets a `missing` list: the kept fields (and only kept ones) whose setting is `drop_rows`,
  read from the snapshot. A dropped or non-feature field's settings don't apply.
- Order: duplicates → excluded values → **missing in a field** → one row per key → split. Missing comes before *one row
  per key* so a key's first *complete* row is kept rather than the key being lost.
- The rows leave every split, like excluded values. `assign_splits` already takes `keep`; strata and CV groups are
  computed on what's left. Removed counts land in `split_info.rows_removed` as `missing weight: 1,204`. Rows are
  counted once, under the first rule that removed them, as today.
- `check_target_classes` and the lone-class preview run on the rows that remain, so a class emptied by the rules is
  caught before queueing.
- The fitted preprocessor still has to read a missing value in that field when scoring new data. It handles it as
  *leave missing* for families that take NaN and as *most frequent* / *median* otherwise, and the manifest lists the
  fields under `rows_left_out_when_missing` so scoring code can do the same.
- The export's *Exclude values* rule gains `missing: true` (*"(missing)"* in the value picker), for fields that aren't
  features, such as a group key. This fixes "exclude can't match missing values".

UI: the decision panel shows the cost under the choice, from the profile's exact count: *"Leaves out 1,204 rows
(2.4%) from every split."* The export preview shows the combined, exact effect: *"Missing values leave out 3,880 rows:
weight 1,204, age 2,950 (274 in both)."* A run's detail already shows `rows_removed`.

## 3. Data rules: values that mean missing

### What a rule is

`meta.datasets.data_rules` (JSONB), revisioned by `data_rev`:

```json
{"missing": [{"value": "EMPTY", "fields": ["weight", "height"]},
             {"value": "-999", "fields": ["pressure"]},
             {"value": "N.A.", "fields": "*"}],
 "store": {"order_date": {"as": "date", "format": "%d/%m/%Y"},
           "amount": {"as": "number", "decimal_comma": false}}}
```

- A missing value matches a **whole cell after trimming spaces, ignoring case**. On a number column it matches by
  value (`-999` matches `-999.0`). `"*"` means every field.
- The fixed `NULL_TOKENS` keep working at CSV read, as now. Parquet uploads never had them applied; the data check
  offers them as findings (pre-ticked).
- Rules are declared in the data check (4), from a field's *Read as missing* signal in *Fields · Focus* (*"Treat a value
  as missing…"*, listing the values that don't parse with their counts), or with `missing_values` on the upload API for
  scripts. There is no field for it in the upload dialog: uploads start as soon as a file is dropped.
- `store` comes from the data check only (4).

### Why a dataset fact, not a per-version decision

What a cell *means* doesn't depend on the model or on who assesses it, and Explore, SQL and the profiles are
per-dataset. Per-version rules would need a copy of the data per version and would let Explore show one thing and an
export another. So rules are not assessed and are visible to everyone (no blind phase applies), and they can only
change when no version is assessing or in review, so nobody's assessment shifts under them.

### How a rule is applied

`POST …/data-rules` (or applying the data check) queues `data_rules.apply(dataset_id, rules)`. The dataset reuses the
existing statuses, with new messages: `ingesting` (*"Applying data rules"*) → `profiling` → `correlating` → `ready`.

1. Read the **source**: `datasets/<id>/source.parquet`, the as-uploaded copy (the current `raw.parquet`, renamed by
   the migration; `meta.datasets.source_path`). Derived columns are left out.
2. Apply the rules: matched cells become null; `store` columns are cast (`coerce_numeric` with the field's number
   format; `parse_datetime` with the chosen format). Values that don't convert become null and are counted.
3. Recompute derived fields from their specs (`meta.fields.derived`), in ordinal order, so they see cleaned inputs.
4. Write `datasets/<id>/data_r<rev>.parquet` and point `parquet_path` at it. Rev 0 (no rules) is the source itself,
   so a dataset without rules uses no extra disk.
5. Rebuild `raw.ds_<id>` with `ingest.load_raw_table` from the new file (column types follow: a cleaned `weight` is
   `DOUBLE PRECISION`, a stored date is `DATE`).
6. Re-profile every column **in place**: a new `processing.reprofile` updates `meta.fields` by name, keeping ids,
   instead of `_persist_profiles`' delete-and-insert (which would cascade away decisions). New suggestions are
   written; decisions and settings still at their untouched suggestion (`source = 'suggested' AND NOT reviewed`) follow
   them; anything a person set is kept. Correlations and the leak screen are recomputed as at upload.
7. Bump `data_rev`, record who changed what in `meta.data_checks` (shown as *"Dana treated EMPTY as missing in 2
   fields"*), and delete the previous `data_r*.parquet` once no export or analysis that read it is pending or running
   (`jobs.sweep` gains this).

A failure leaves the dataset `ready` on its previous file with a warning, never `failed`: the swap of `parquet_path`
is the last step.

### Consistency

- **Explore** results are keyed by `data_rev` too, so a result computed before a change shows *stale* and recomputes.
- **SQL console** sees the cleaned `raw.ds_<id>`. The as-uploaded values aren't in Postgres; the data check keeps
  their counts and examples.
- **Field profile** shows rule effects in the reasons: *"2,950 values were EMPTY and read as missing (data rules)."*
- **Exports** read `parquet_path`, as now; the manifest records `data_rev` and the rules.

### Versions

- A version records the `data_rev` it started on (`meta.recipes.data_rev`) and its snapshot stores it with the rules.
- Changing rules after a finalized version: allowed for a lead. The *Baseline* tab then lists the fields whose type or
  missing share changed and offers *Start vN+1, reopening these 3 fields*. It doesn't force it.
- Exporting an older finalized version on newer data is allowed. The export dialog says so: *"The data changed after
  Trees v1 was finalized: EMPTY reads as missing in weight and height."* Past exports keep their own files.
- Reusing another dataset's baseline matches fields after the new dataset's data rules, so drift is measured on clean
  numbers. A dataset uploaded to start from a baseline applies it once its data check is answered (see 4).

## 4. The data check

### When it runs

A `checking` step at the end of `process_dataset`, after correlations, writes the findings to `meta.data_checks`
(`status` gains no new value; the step's message is *"Checking the data"*). The dataset then becomes `ready` as
today, with `data_check_status`:

| Status | Meaning |
|---|---|
| `open` | Findings with a question are waiting. |
| `clear` | Nothing to decide; findings, if any, are notes. |
| `done` | Someone applied it or chose *Keep the data as it is*. |

The dataset opens straight away. Fields, Explore and SQL work. A *"Check the data · 3 questions"* banner shows on the
dataset and its row on *Datasets*. **Starting the first baseline needs the check answered**: the set-up form on the
*Baseline* tab shows it as step 1 above model type, with *Keep the data as it is* as a one-click way through. Why not
a blocking "Review data" status: people want to look at the data (Explore, SQL) to answer the questions, and a dataset
nobody can open can't be checked. Why gate the baseline: rules change types and suggestions, and they must land
before assessors start from them.

A dataset uploaded to start from another baseline (`recipe_id`) waits for the check when it is `open`, then
`_apply_pending_recipe` runs after apply or *Keep as is*. With `clear`, it runs at once, as today.

### What it checks

Each finding: field (or none), kind, counts, up to 5 examples with counts, a proposed answer, and whether it asks a
question. Counts are exact; date-format probing uses the existing spread sample.

| Finding | When | Question and proposed answer |
|---|---|---|
| **Numbers and text mixed** | a text column where 2%–98% of values parse as numbers | *"weight: 7,012 numbers, 2,988 text: EMPTY 2,950, n/a 30, abc 8."* Tick values to treat as missing (pre-ticked: the known blank words below); then *Store as numbers* (the rest, 8 values, become missing) or *Keep as text*. Proposed: the blank words missing, store as numbers when they cover ≥ 95% of the text. |
| **Numbers stored as text** | ≥ 98% parse (inference already reads them as numbers) | *Store as numbers* (proposed), so SQL and Explore see numbers; the values that don't parse are listed. Decimal comma shown when detected. |
| **Dates stored as text** | a text or `YYYYMMDD` column inferred as a date | *Store as dates*, with the detected format and the other day/month order to pick from when nothing decided it. *"order_date: 99.7% read as dd/mm/yyyy; 31 don't: 31/02/2020, n/a."* Proposed: yes. Epoch numbers are a note only. |
| **Blank words** | whole-cell values such as `EMPTY`, `empty`, `NULL`, `null`, `None`, `NA`, `N.A.`, `n.a.`, `-`, `--`, `.`, `#N/A`, `#VALUE!`, `#DIV/0!`, `(blank)`, `nil`, spaces only | One finding per value across fields: *"EMPTY in 4 fields, 12,480 cells."* Pre-ticked for the fields it appears in. `unknown`, `missing`, `none` in a categorical are asked, not pre-ticked: they can be real categories (the diabetes sample's lab `None`). |
| **Sentinel numbers** | `-1`, `-9`, `-99`, `-999`, `-9999`, `99`, `999`, `9999`, `99999` holding ≥ 0.5% of a number column and outside 1.5 × the IQR fence of the other values | *"pressure: -999 in 3.1% of rows; the other values run 40–210."* Not pre-ticked. |
| **Constant or ID-like column** | inferred `constant` or `identifier` | Note: *"Suggested for dropping in the baseline."* |
| **Very high missing** | ≥ 50% missing (≥ 95% is suggested for dropping) | Note. |
| **Duplicate rows** | exact duplicates exist | Note: *"212 rows repeat an earlier row. An export can leave them out (row rules)."* |
| **File reading** | the existing ingest warnings (separator, ragged rows, schema re-read, NUL removed), plus U+FFFD replacement characters per column and a one-column file whose header holds `;` or a tab | Note, with what to fix before uploading again. |

The existing `meta.datasets.warnings` stay; the check shows them as notes so there is one place to read them.

### Who may answer

| | Before the first baseline starts (*Set up*) | A version assessing or in review | Every version finalized |
|---|---|---|---|
| Member | ✓ | — | — |
| Lead | ✓ | — | ✓ |
| Viewer | — | — | — |

The API refuses with `403` (role) or `409` (phase, with the same wording as derived fields: *"Trees v2 is being worked
on; change data rules once it is finalized."*).

### Library samples and generated data

The check runs on them too, so they show the same notes. Library entries can carry `data_rules` in `LIBRARY`, applied
during processing; their check then starts `done` (*"Checked when the sample was prepared"*). Generated samples start
`done` when nothing asks a question, which is the expected case; otherwise `open` like an upload.

### Running it again

*Check the data again* on the *Baseline* tab (and in the dataset menu) re-runs the checks on the current cleaned data,
for example after a lead adds rules from *Fields · Focus*. It is allowed whenever applying is. It creates a new
`meta.data_checks` row; older ones stay as history.

### API

| Endpoint | Does |
|---|---|
| `GET /api/datasets/{id}/data-check` | The latest check: status, findings, proposed answers, current rules and `data_rev`, and `can_apply` with the reason when false. |
| `POST /api/datasets/{id}/data-check/preview` | Body: rules. Per affected field: values that become missing, values that don't convert (with examples), the type it would be inferred as. Synchronous, column reads only. |
| `POST /api/datasets/{id}/data-check/apply` | Body: rules and the `data_rev` they were made on (`409` if it moved). `202`; runs `data_rules.apply`, marks the check `done`. |
| `POST /api/datasets/{id}/data-check/keep` | Marks the check `done` with no change. |
| `POST /api/datasets/{id}/data-check/run` | Re-runs the checks. `202`. |
| `PUT /api/datasets/{id}/data-rules` | Sets rules directly (scripts, the *Treat a value as missing…* action). Same checks as apply. |
| `POST /api/datasets/upload` | Gains optional `missing_values` (comma-separated), applied as a rule for every field before profiling. |
| `GET /api/datasets/{id}` | Gains `data_check_status`, `data_rev`, `data_rules`. |

### Data model (migration `0NN_data_check.sql`)

- `meta.datasets`: `source_path TEXT`, `data_rules JSONB NOT NULL DEFAULT '{}'`, `data_rev INT NOT NULL DEFAULT 0`,
  `data_check_status TEXT NOT NULL DEFAULT 'done' CHECK (… IN ('open', 'clear', 'done'))`. Existing datasets are
  `done`; the migration renames each `raw.parquet` to `source.parquet` (bootstrap, since SQL can't move files) and
  sets both paths.
- `meta.data_checks` (id, dataset_id, data_rev, status `running | open | clear | done | failed`, findings JSONB,
  created_by, created_at, decided_by, decided_at, decision `applied | kept`). Readable by the SQL console role:
  nothing in it is private.
- `meta.recipes.data_rev INT`.
- The `ExcludeRule` schema gains `missing: bool`.

### UI

- **Datasets home.** A processing row's message includes *Checking the data*. A ready row with an `open` check shows a
  chip *Check the data · 3 questions*, linking to it.
- **Data check page** (`/d/:id/check`, reached from the banner, the chip and the *Baseline* tab). Questions first, then
  notes, each a card: the field, the finding in one sentence, the counts as a small bar (numbers / text / missing), the
  examples with counts, and the choices as checkboxes and a segmented *Store as numbers · Keep as text* or *Store as
  dates [dd/mm/yyyy ▾] · Keep as text*. Under the choices, the preview: *"weight: 2,980 values become missing; 8 more
  don't read as numbers and become missing too; will be read as Numeric (continuous)."* Footer: *Apply and re-profile*
  (primary) and *Keep the data as it is*. Read-only for Viewers and for people who can't apply now, with the reason.
- **Baseline tab, set up.** Step 1, *Check the data*, collapsed to *"Done · 3 changes by Dana"* once answered.
- **Fields · Focus.** The *Read as missing N* signal gains *Treat a value as missing…*: the values that don't parse,
  with counts, as checkboxes, then *Apply* (with the preview count). The decision panel's *Missing values* row uses
  the labels in 1, with *Fill with a value…* showing a text box (or a level picker for categoricals) and *Leave out rows
  where it's missing* showing its row count. *Missing label* is gone.
- **Fields · Table.** The bulk action *Set missing values…* lists the choices; fields where a choice isn't valid for
  their type or the family are skipped, and the dry-run plan says why (*"text fields can't be left missing"*).
- **Export dialog.** The missing-rows line from 2, and the data-changed note from 3.

## Docs to update

- `data-preparation.md`: a *Data check* section; *Type inference* notes data rules; the per-family defaults table and
  a new *Missing values* table.
- `concepts.md`: *Data check* and *Data rules* under Data; the code-name table (`data_rules`, `data_rev`,
  `imputation` values).
- `team-workflow.md`: the set-up step, who may answer, rules after a finalized version.
- `experiments-and-training.md`: row-rule order, *(missing)* in excluded values, `rows_left_out_when_missing`.
- `explore.md`, `sql.md`: both read cleaned data.
- `api.md`, `architecture.md` (pipeline step, `data_rules.py`, `reprofile`, migrations table), `user-guide.md`.

## Out of scope

- Re-reading a file with a chosen separator, encoding or header row: the original upload is still deleted after
  ingest; the check says what to fix and to upload again.
- Trimming spaces, merging case variants (`Yes` / `yes `), and other value edits beyond missing and storage type.
- Rules per baseline or per experiment.
- Imputation fitted from other fields (KNN, iterative).
- Detecting units or currencies in text (`"12 kg"`, `"$1,200"`).

## Open questions

1. Should *Leave out rows where it's missing* be shared by every family instead of per family? Per family matches
   the other settings, but a tree baseline and a linear one would then test on different rows. Proposed: per family,
   and the run comparison shows *"different rows"* when two runs' exports removed different rows.
2. Keep `source.parquet` forever, or drop it after some months without rule changes? It doubles disk only for
   datasets with rules.
3. Should the set-up gate also apply to a dataset started from another baseline via the API (`recipe_id`), or should
   the proposed answers be applied automatically there? Proposed: wait, as specified.
4. Sentinel thresholds (0.5% share, IQR fence) need trying on the library datasets before they are fixed.

## Phases

1. **Missing-value choices.** `own_level` and `none` for categoricals and binary; new labels; `MissingOneHot`,
   `FrequencyEncoder` and `BinaryEncoder` changes; tree defaults; the settings migration and `normalize_settings`; decision
   panel and bulk action. Directly answers the owner's second point.
2. **Leaving out rows.** `drop_rows`, the row rule and its order, preview counts, `missing` in excluded values, the
   review-queue rule, manifest field.
3. **Data rules.** `data_rules.apply`, `source.parquet`, `reprofile` in place, raw-table rebuild, Explore keying,
   `PUT …/data-rules`, *Treat a value as missing…* in Focus, upload `missing_values`. Answers the `EMPTY` case
   without the full check.
4. **The data check.** Findings, `meta.data_checks`, the page, the set-up gate, *Store as numbers / dates*, library
   rules, re-run, docs.

## Testing

Unit: `allowed()` per type and family (no `none` for linear categoricals; `drop_rows` never for identifiers);
`normalize_settings` maps old snapshots; each encoder with *leave missing* gives NaN where expected and the same
columns otherwise; `MissingOneHot` names; LightGBM, XGBoost, histogram GBM and random forest each fit on an export with
NaN categoricals; row-rule order (missing before one row per key keeps a key's first complete row); counts in
`rows_removed`; rule matching (trim, case, numbers by value, `"*"`); every finding kind on small hand-made frames,
including a column that is 30% `EMPTY` (today inferred categorical, after the rule numeric) and a `None` category
that is not pre-ticked.

API (DB-backed): apply keeps field ids, decisions a person set, and the raw table's types; `409` while assessing,
`403` for a member after set up; `data_rev` mismatch; the first baseline can't start with an `open` check, and can
after *Keep*; a pending `recipe_id` waits for the check; a failed apply leaves the dataset `ready` on its old file;
Explore results go stale; the migration maps `__missing__` in all three settings tables and suggestions; drop-row
disagreements reach the review queue.

UI (vitest): decision panel labels and row-count line; retype keeps a valid missing choice; data-check card state and
preview. dev-01 (Playwright): upload a CSV with `EMPTY` in a number column and `dd/mm/yyyy` dates, answer the check,
see the field read as numeric with the right missing count in Fields, Explore and SQL, set *Leave out rows where it's
missing*, export, and check the removed rows in the run detail.
