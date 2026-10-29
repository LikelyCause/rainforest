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

**Model files.** The deployment records each model file's SHA-256. Every time it loads the models it checks them
first; a file that changed is refused (`409`, naming the model) and never loaded. Deploy the run again to use the
new file.

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

## Retiring and deleting

- *Retire* stops a deployment answering (`410`) and keeps its record and keys. Its creator or a lead can reactivate
  it.
- Retiring and revoking a key stop access, so neither a dataset refresh nor an archived experiment holds them back.
  Creating a deployment, creating a key and reactivating grant it: they wait for a refresh to finish, and are
  refused while the experiment is archived (`409`).
- Deleting a dataset deletes its deployments and their keys first.
