# Export to code

Goal: a person exports from Rainforest a Python script that holds the decisions made on the fields (keep, drop, type,
missing values, transforms, scaling, encodings, targets, row rules, split) and hands it to someone else, who runs it on
the original data and gets exactly the dataset they would have had inside the app, to test with outside it.

## Requirements

From the owner's answers (2026-10-14):

| Question | Decision |
|---|---|
| What is it generated from? | **A finished export.** Its snapshot, row rules, split shares and seed are frozen, so the script's output can be checked against that export's `model_ready.parquet`. |
| What data does the recipient start from? | **The original upload.** The script repeats the app's reading rules (separator, missing-value markers, column names) and recomputes derived fields. |
| What does it hand back? | **Both stages.** `load_clean()`: the decisions applied, nothing fitted. `load_model_ready()`: the same values as the app's `model_ready.parquet`. |
| How is it packaged? | **One readable `.py`.** The decisions as code anyone can read and edit, the app's own code embedded below, pinned versions in a PEP 723 header so `uv run` installs them. No Rainforest import. |
| Sample datasets with no original file? | **"Download it from the same place we seeded from."** Library datasets are rebuilt from their UCI archives the way `sample_library/build.py` built the app's copies (verified identical for all five). Generated samples were never downloaded; the script runs the app's seeded generator again, with settings now recorded (migration 023). Older samples' settings are worked out from their default name and confirmed by making them again. |

Derived from those: anyone who can see the export can download its script (the same rule as the export's files,
Viewers included); the script stops when the data doesn't read as the app read it, unless told not to.

## User stories

1. *As a data scientist on the team,* I export from an experiment and hand the script and the CSV to a colleague
   without Rainforest access, so they can try models in their own notebook on exactly the rows, split and features
   our results came from.
2. *As the colleague,* I run `uv run export12_x.py data.csv` and am told the result is identical to the app's export,
   or exactly which columns of my file differ from what the app read (a re-saved CSV, a changed column).
3. *As a reviewer or auditor,* I read `kept_fields()` and `targets()` to see every preparation decision in one place,
   with the team's notes and the experiment's adjustments beside the field they're about.
4. *As someone trying an idea,* I change one line (`encoding='onehot'` → `'ordinal'`) and get what the app would have
   exported with that decision, and the check tells me the result now differs from the export.
5. *As a person who only has the demo or a library dataset,* I run the script with no file and it makes or downloads
   the same data itself.
6. *As someone deploying a model trained outside the app,* I take `prepare(...).preprocessor` (the fitted
   ColumnTransformer) to transform new rows the way the training rows were transformed.
7. *As a lead,* I can rely on a private (personal) experiment's script being as private as its export.

Out of scope for now: training (the script prepares data; the run's model and settings stay in the app and MLflow),
scripts for an experiment or baseline that hasn't been exported, and the data check / data rules of the 2026-10-12
spec, which isn't built yet (see [Coupling](#coupling-with-planned-work)).

## Design

`GET /api/exports/{id}/files/code` writes the script on request (`export_code/`):

- **`gather.info_for`** reads what the script is written from: the export's manifest (the exact field specs, targets,
  fit-sample size and batch size the export used), its frozen snapshot (notes, left-out fields, target tasks) and
  config (row rules, split, adjustments, and now the derived fields' definitions), where the data came from, and
  per-column summaries of the dataset's Parquet copy and of `model_ready.parquet`, cached per file version.
- **`render.render`** writes the script, top to bottom: the PEP 723 header and a comment block (what it is, how to
  run it); the data (`SOURCE`, `raw_frame()`, `DERIVED`); the decisions (`FAMILY`, `kept_fields()` as `keep.*` calls
  with notes and adjustments as comments, the left-out fields, `targets()` as `predict.*` calls, `ROW_RULES`,
  `SPLIT`, `FIT_SAMPLE_ROWS`, `CHUNK_ROWS`); the driver and the app code it runs; `EXPECTED_INPUT` and
  `EXPECTED_OUTPUT`; and `main()`.
- **`script.py`** is the driver, real app code that is copied into every script: reading (`read_upload`,
  `read_library`), `read_source` (derived fields, then the input check), `prepare` (mirrors `run_export` step by step:
  `prepare_targets`, the seeded fit sample, `build_preprocessor`, batched `transform_into`, the NaN rule), `load_clean`,
  `xy`, the column digests and checks, `--compare`, and the CLI.
- **`bundle.Bundle`** copies app code into the script: starting from the driver's entry points it follows every
  module-level name they reach (same-module definitions, `from .x import y`, `bands.assign` through a module) and
  copies each definition verbatim with its comments, plus the imports they need. `undefined_names` checks the result
  statically. So the script runs `transform.py`, `ingest.py`, `profiling.py`, `derived.py` and `bands.py` as the app
  runs them; a change to any of them reaches new scripts without anyone editing a template.

A field's call is the shortest one that rebuilds exactly its spec (`keep.numeric('age')`), else `keep.field(...)`
with every setting; a target's likewise (`predict.bands(5)`, else the task dict). Tests evaluate every call back to
an equal spec.

**Checks.** Input: rows, then each column's dtype, missing count and a SHA-256 over its values and missing positions
(independent of library versions). A mismatch stops the script (exit 2) with the columns that differ. Output: each
column's digest; for features, also NaN count and sums, so a feature that differs only by floating-point rounding is
told apart from one that's wrong. `--compare` checks value by value against the app's file, with a tolerance.

### What it changes in the app

| Change | Effect |
|---|---|
| Migration 023: `meta.datasets.source_spec` | Set for library datasets (`{kind, key}`) and generated samples (`{kind, settings}`); null for uploads. Readable by the SQL console like the rest of `meta.datasets`. |
| `exports.config.derived` | Each new export freezes the derived fields' definitions. Appears in the manifest's `config` and its MLflow artifact. The UI's re-export and row-rule reuse copy named keys only, and the request schema ignores extra keys. |
| `bands._num` renamed `_fmt` | The bundle is one namespace and `derived._num` exists too. Private; no other users. |
| `GET …/files/code` | A new `kind` on the existing download route, so it inherits `visible_export` (private experiments stay private; the visibility matrix test covers it). Unlike the stored files it works after the files are removed, without the output check. |
| `gather._source` writes `source_spec` on a GET | Only once, for an older generated sample whose settings were just confirmed (the same backfill-on-read pattern as `features_by_type`). |
| UI: *Use from code → Python script (.py)* | Fetched rather than linked, so a refusal (409 with the reason) shows in place; `attachmentName` keeps the downloaded name to a plain file name. |

Nothing changes for existing exports, runs, MLflow logging, the SQL console's privacy rules or the training CLI.

## Validation

- `tests/test_export_code.py` (no database): a deliberately messy CSV (semicolons, a BOM and padded header names, `?`
  and blanks, decimal commas, day-first dates, epoch seconds, ±inf, duplicate rows) goes through the app's own ingest,
  derived fields (formula, groups, count, a formula over a formula) and `run_export`, for a tree export (dedupe,
  excluded values, rows kept together, batches of 400, a 700-row fit sample, class and number targets) and a linear
  export (first row per key, bands and one-vs-rest targets, every scaler and transform). The rendered script, run with
  `python -I`, must write a `model_ready.parquet` equal to the app's, schema included. Also: a changed file stops it
  (rows; values), an edited decision gives what the app exports with that decision, names and notes full of quotes,
  newlines, `"""` and U+2028 change no statement of the script, every `keep.*`/`predict.*` call reads back as the
  same spec, and the bundle's own rules.
- `tests/test_export_code_api.py` (Postgres): the full path through the API (upload, a derived field, baseline,
  experiment adjustments with notes, export with row rules, download, run) is identical; still after the derived field
  is removed; a Viewer can download; a generated sample is made again, from recorded settings and from its name; a
  library dataset reads from its file and is recognised without `source_spec`; refusals (unknown, unfinished, a
  sample that can't be recognised); the input summary is the app's Parquet copy. The visibility matrix in
  `test_experiments.py` now includes the script.
- Full suite: 869 passed, 3 skipped. UI: typecheck and 184 tests.
- On a stack built from the branch: the 50k × 150 demo (450 features) rebuilt from nothing, identical, in 18 s; in a
  clean `python:3.12-slim` with only `uv`, *Concrete strength* (a number target, `fastexcel`) and *Diabetes
  readmission* (first encounter per patient, excluded discharge codes, 69,990 rows, 86 features) downloaded from UCI
  and rebuilt identical. Writing a script takes about 1 s cold for the demo, 0.04 s cached.

## Challenges met building it

1. **Drift between the app and a copy of its code.** A hand-written template would quietly diverge from
   `transform.py`. The bundle copies the app's functions verbatim instead, and the end-to-end tests in CI fail as soon
   as a script's output differs from `run_export`'s.
2. **The bundle is one namespace.** `bands._num` (formats a number) and `derived._num` (parses one) collided; the
   bundle refuses collisions, and one was renamed. Future collisions fail the tests the same way.
3. **Names reached only inside lambdas** (`SCALERS = {"standard": lambda: StandardScaler()}`) were missed at first:
   the script ran until a scaler was needed. Constants are now followed into lambda bodies, and `undefined_names`
   checks every script statically.
4. **App code that reaches the web framework.** `derived._bad` returns an `HTTPException`; the bundle overrides it
   with a plain `ValueError` for the script.
5. **NaN is not null.** `pl.from_pandas` and `pa.Table.from_pandas` turn NaN into null, so a correct rebuild first
   reported rounding differences and wrote a file that didn't equal the app's. Both now keep NaN.
6. **The fit sample depends on server memory settings.** `run_export` caps it by `EXPORT_FIT_MB` for wide exports, so
   the script uses the sample size the manifest recorded, not the one requested.
7. **Derived fields can be removed after an export**, and their definitions weren't stored with it. New exports
   freeze them in `config.derived`; older ones fall back to the current definitions and refuse with the field's name
   when one is gone.
8. **Sample datasets have no original file.** Library datasets rebuild identically from UCI (checked for all five;
   *Concrete strength* needs `fastexcel`, pinned at the version checked). Generated samples didn't record their
   settings; older ones are recognised from their default name and confirmed by regenerating, others are refused with
   the reason.
9. **Bitwise identity across machines isn't guaranteed.** NumPy picks SIMD code paths per CPU for `log1p` and
   friends, and TF-IDF → SVD runs on multithreaded BLAS, so the last digit of a float can differ on another machine.
   The output check says *identical*, *differs only by floating-point rounding* (features named), or what differs;
   rows, split and targets must always be identical.
10. **Library versions.** The script pins the versions the app runs (and caps Python below 3.14, where the pinned
    pandas has no wheels). An export made by an older app with other versions says so in the script's header.
11. **A date field with no recorded format** is worked out per batch in the app, and batches follow the Parquet
    copy's row groups, which change when a derived field rewrites the file. The script mirrors the app's batching and
    warns in its header for such fields; values can differ if the batches do.
12. **Code generation from people's text.** Field names, notes, dataset and experiment names go into the script as
    `repr` literals or comment lines split on every kind of line break (U+2028 included), never into docstrings; a
    test checks hostile names leave the script's statements unchanged.
13. **Checking the input without the data.** Polars' row hash isn't stable across versions, so the digest is a
    SHA-256 over each column's values and missing positions.
14. **CSV quirks.** Rather than re-describe the app's reading (separator guess, schema inference on 50,000 rows then
    the whole file, ragged lines, BOMs), the script runs the app's `ingest._sink`.

## Limitations and open questions

- **Privacy.** The script carries no rows, but it does carry field names, the team's notes, excluded values, derived
  definitions, per-column digests, and per-feature NaN counts and sums (so means). Anyone who can see the export can
  download it. Should notes stay in (they explain the decisions), or be left out by default?
- **Written on request, not stored.** A script reflects the app's code when it's downloaded; after an upgrade that
  changes the pipeline, an old export's script may no longer reproduce it, and its check will say so. Storing the
  script with the export would freeze it but costs storage and wouldn't cover existing exports.
- **Large datasets.** The summaries are computed on the first request per file version (one column at a time); for
  millions of rows and hundreds of columns that first download can take tens of seconds. Computing them during the
  export would remove the wait.
- **Training isn't included.** A follow-up could write a run's model and settings into the script (`train_run()`),
  using the registry's builders the same way.

### Coupling with planned work

The 2026-10-12 *missing values and the data check* spec applies data rules by rewriting the dataset's Parquet copy.
Once it lands, a script must replay the dataset's data rules after reading the original file, or its input check will
fail. The rules should be recorded with each export (as derived fields now are) and applied through an app function
the driver calls, so the bundle carries it. Its new choices (`own_level`, `drop_rows`) live in `transform.py` and
reach scripts automatically; the end-to-end test should gain fields that use them.
