# Deployments

A deployment scores new records with finished runs, inside Rainforest, for testing them on data they haven't seen.
It is not production serving: it runs in the API process, one request at a time per worker thread, with no scaling
or service guarantees. The routes are listed in the [API reference](api.md#deployments-routersdeploymentspy).

A deployment is frozen files: the export's preprocessor and field specs, its frozen derived formulas and input types,
and each run's model. Refreshing the dataset, re-exporting, or changing a decision later doesn't change what it
answers.

## What can be deployed

A deployment is made from runs of **one export**, so they share one preprocessing path and one set of input fields.
It is one of two things:

| Deployment | Runs | Each record gets |
|---|---|---|
| One-vs-rest | [One-vs-rest runs](experiments-and-training.md#one-class-against-the-rest) of one target, at most one per class. A one-vs-rest batch deploys whole; a search batch by choosing its rows. | every class's calibrated probability, the most likely class, and the classes whose model said yes |
| Single run | One run with one target: multiclass, or a number | each class's probability and the most likely class, or the predicted number |

Anyone who can add runs to the experiment can deploy its runs. Viewers can't.

### What is refused, and why

| Refused | Why |
|---|---|
| A run that hasn't finished | There is no model to score with. |
| Runs of two exports | Each export has its own preprocessing; records can only go through one. |
| A chain (a run with several targets) | Its later targets read earlier predictions; a deployment answers for one target. |
| One-vs-rest runs together with a multiclass or number run, or two single runs | Their scores mean different things, so `top` and `matches` couldn't compare them. |
| Two runs of the same class | The class would have two scores. Deploy one of them. |
| One-vs-rest runs of different targets | Their classes aren't alternatives to one another. |
| An export made before exports recorded their input types (*Re-export to deploy*) | Scoring needs each source field's type and the derived formulas as they were at export time. |
| An export whose features read a derived field that can't be computed any more | The field reads a column that was removed from the dataset. |
| An export whose features read a field stored as a time of day or a duration | A record can't carry one yet, so the field would always be missing. |
| A model or preprocessor file that is missing or doesn't unpickle (`410`) | There is nothing to score with. |
| Scores or features that don't reproduce training | See below. |

**The check at creation.** Before a deployment is stored, Rainforest takes the export's test rows (its validation
rows when there are none, else a sample of up to 5,000 training rows), reads their raw values from the dataset, and
scores them through the deployment. Each model must give the scores it gives on the export's own features, within
1e-9, and the features made from the raw values must equal the export's exactly. If not, the deployment is refused
with the first model (or feature) and row that differ. A dataset refreshed after the export can fail this check
because its rows changed; re-export to deploy.

**Model and preprocessor files.** The deployment records the SHA-256 of each model file and of the export's
`preprocessor.joblib`. Every time it loads them it reads each file once, checks those bytes and unpickles the same
bytes; a file that changed is refused (`409`, naming the model or the export) and never loaded. Deploy again to use
the new file. A file that is missing or doesn't unpickle is `410`, and the API logs a warning naming the file and
the error.

## Thresholds

Every one-vs-rest model has a threshold: a record whose score is at or above it is a match. Multiclass and number
runs have none. Thresholds are chosen on each run's validation threshold curve (steps of 0.01), when the deployment
is created, by a rule:

| Rule | Threshold |
|---|---|
| `{"kind": "f1"}` (the default) | the one with the best F1 on validation; the lowest of a tie |
| `{"kind": "precision", "min": 0.9}` | the lowest whose validation precision reaches 0.9; if none does, the F1 choice, with a `threshold_note` saying so |
| `{"kind": "fixed", "value": 0.35}` | 0.35 |

`threshold_rule` sets the rule for every model, and `thresholds` (`{run_id: rule}`) for one. The deployment stores
each model's threshold and the rule that chose it. Scores and thresholds are calibrated probabilities. Choosing a
threshold on the split the model was calibrated on is mildly optimistic; the test split gives the honest number.

## Keys

A script calls a deployment with an API key. The deployment's creator or a lead creates keys and revokes them.

- A key is `rf_` followed by 32 random URL-safe characters. It is shown once, when it is created. Rainforest keeps
  only its SHA-256 and its first 8 characters (to tell keys apart in the list), and never logs it.
- A key opens one thing: the predict call of its own deployment. It isn't a session, it reads nothing else, and
  another deployment refuses it (`401`).
- Revoking a key stops it at once. Retiring the deployment stops every key (`410`) until it is reactivated.
- `last_used_at` shows when a key was last used, to the minute.

A browser on Rainforest's own pages can also call predict with the signed-in session, except a Viewer's. A Viewer can
use a key.

## Predict

```sh
curl -X POST http://localhost:8080/api/deployments/12/predict \
  -H "Authorization: Bearer $RAINFOREST_KEY" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"weight": 1520, "colour": "red", "registered": "2021-03-04"},
                   {"weight": 980, "colour": null, "registered": "2019-11-30"}]}'
```

`GET /api/deployments/{id}` lists the fields a record must carry (`fields`) and two of the dataset's rows of them
(`example`).

```json
{
  "deployment": 12,
  "models": ["V01", "V02", "V03"],
  "results": [
    {"scores": {"V01": 0.91, "V02": 0.04, "V03": 0.12}, "top": {"label": "V01", "score": 0.91},
     "matches": ["V01"], "none_matched": false},
    {"scores": {"V01": 0.08, "V02": 0.21, "V03": 0.17}, "top": {"label": "V02", "score": 0.21},
     "matches": [], "none_matched": true}
  ],
  "warnings": {}
}
```

Results come in record order:

- `scores`: each class's probability (or the predicted number, keyed by the target);
- `top`: the class with the highest score and that score (null for a number);
- `matches`: the one-vs-rest classes at or above their threshold, best first;
- `none_matched`: true when no one-vs-rest model said yes. `matches` and `none_matched` are null for a single run.

### Records

- Every record carries every field in `fields`; send `null` for a missing value. A field missing from a record is
  `422`, naming it and the record. Fields that aren't needed are ignored, so a record can carry its identifier.
- A derived field is computed from its inputs with the formula frozen at export time; a value sent for it is ignored.
- Values are single JSON values. A list or object is `422`, and so are `NaN` and `Infinity`, which aren't JSON.
- Each value is converted to the type the field had in the dataset, whatever type the first record used:
  - numbers as numbers, from JSON numbers or text;
  - a category stored as integers reads `7.0` and `07` as `7`;
  - a date stored as text is read with the format the export pinned; a date stored as a date takes RFC 3339 text
    (`2021-03-04`, `2021-03-04T10:30:00Z`, `T` and `Z` in either case). A time without an offset is UTC, and an
    offset needs a time (`2021-03-04Z` doesn't convert);
  - what the upload reader treats as missing (`""`, `NA`, `null`, `?` and the rest) is missing.
- A value that doesn't convert is scored as missing and counted in `warnings` (`{"weight": 1}`).

### Limits

| Limit | Answer |
|---|---|
| A body over 5 MB | `413` |
| More than `PREDICT_MAX_RECORDS` records (1,000 by default; see [Configuration](configuration.md)) | `413` |
| A retired deployment | `410` |
| No key and no session, a revoked key, or a key of another deployment | `401` |

The models of the last `DEPLOYMENT_CACHE` deployments used (4 by default) stay loaded in the API process. The first
call after that loads them again, which takes longer.

## Testing with a file

A file of records can be scored in the background instead of through predict: `POST /api/deployments/{id}/tests`
with the file as multipart `file`.

- **The file** is read by the same reader as a dataset upload: `.csv`, `.tsv`, `.txt` or `.parquet`, the separator
  guessed, the same missing-value tokens. It needs every field predict needs; extra columns are ignored, and a
  missing field fails the test, naming it.
- **The job** queues with exports and runs, converts the file, and scores it 50,000 rows at a time, exactly as predict
  scores records: the same conversion, the same model checks (a model or preprocessor that changed fails the test
  with the reason). Its `progress` and `status_message` show how far it is.
- **Results** keep every row of the file, in order: its row number (from 1), the dataset's identifier field when the
  file has one (a field decided, else inferred, as an identifier), the file's label when it has one, `top` and
  `top_score`, `matches` and `none_matched` for a one-vs-rest deployment, and a `score_<label>` column per score. A
  column of the file named like a generated one keeps its name, and the generated one gets a leading `_`. The test
  shows the first 500; the download has all of them, as Parquet or CSV (`matches` joined with `; `).
- **Values that don't convert** become missing, as in predict, and are counted per field: `warnings` is
  `{field: {failed, values, flagged}}`, `values` being the values the file gave, and a field is `flagged` when more
  than half of them failed. `notes` holds what the reader noted about the file (its separator, dropped columns).

### Labels

When the file has the deployment's target column, the test reports `metrics` over the rows whose label isn't missing.
Every row is scored; the export's row rules don't apply. A label is read as the dataset's target was, so `7.0` is
class `7` of a target stored as integers.

| Metric | What it is |
|---|---|
| `top1_accuracy` | the share of labelled rows whose `top` is their label. A row whose label has no model (a class skipped, left out or unseen) counts as wrong. |
| `rows_label_without_model` | how many labelled rows have a label no model scores; they have no per-class row |
| `classes` | per class: `support` (labelled rows of the class), `precision`, `recall` and `f1` of its model at its `threshold` (a multiclass run's class counts as predicted when it is `top`; no threshold), and `pr_auc` (average precision of its score on the file). A metric with nothing to divide by is null. |
| `nothing_matched_share` | the share of all rows, labelled or not, where no one-vs-rest model said yes (null for a single run) |
| `confusion` | `labels`: the 20 most frequent labels; `counts[i][j]`: rows labelled `labels[i]` whose `top` is `labels[j]`, the last column counting a `top` outside the list |

A number run reports `mae`, `rmse` and `r2` against the target instead. `labelled_rows` and `rows_without_label`
count the rows either way. The scores are calibrated probabilities and the thresholds were chosen on the validation
split they were calibrated on, so a file the models haven't seen gives the honest numbers.

### Keeping tests

The newest 20 tests of a deployment are kept: starting another deletes the oldest finished one and its files. A test
interrupted by an API restart fails with *Interrupted by an API restart.* Anyone who can see a deployment sees its
tests and downloads their results. Starting one is using the deployment, so it needs write access to the experiment
as for runs, waits for a dataset refresh to finish (`409`), and is refused while the experiment is archived or the
deployment is retired (`409`).

## Retiring and deleting

- *Retire* stops a deployment answering (`410`) and keeps its record, keys and tests. Its creator or a lead can
  reactivate it.
- Retiring and revoking a key stop access, so neither a dataset refresh nor an archived experiment holds them back.
  Creating a deployment, creating a key and reactivating grant it: they wait for a refresh to finish, and are
  refused while the experiment is archived (`409`).
- Deleting a dataset deletes its deployments, their keys and their tests first.
