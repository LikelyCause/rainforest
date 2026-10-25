# MLflow

Every model run trained from the UI or the API is also logged to an [MLflow](https://mlflow.org) tracking server.
Compose runs it as the `mlflow` service.

A Rainforest experiment and an MLflow experiment are different things. See
[Concepts](concepts.md#a-rainforest-experiment-is-not-an-mlflow-experiment).

## Opening MLflow

Open `http://localhost:8080/mlflow/`, or *MLflow ↗* in your profile menu.

- MLflow sits behind the Rainforest sign-in. Before every request, nginx asks the API whether the session cookie is
  valid. If it isn't, you go to the sign-in page and back.
- A Viewer can browse MLflow but not change it.
- The server has no published port of its own.

The runs table has an *MLflow ↗* link per logged run, and the run's detail links to it too. An experiment's page
links to the dataset's MLflow experiment, filtered to its runs. `meta.model_runs` stores `mlflow_experiment_id` and
`mlflow_run_id`, and the API returns `mlflow_url`, built on `MLFLOW_BROWSE_URL` (the address browsers open MLflow at;
empty means the bundled `/mlflow/`).

## Layout

- One MLflow experiment per dataset and prefix, named `<prefix>/<dataset id>-<dataset name>`, where the prefix is
  `MLFLOW_EXPERIMENT_PREFIX` (`rainforest` by default). Each experiment is tagged with its dataset id and prefix, and
  Rainforest finds it again by both, so renaming a dataset keeps its experiment, and releases sharing one MLflow server
  each keep their own. Changing the prefix starts a new experiment for each dataset; the old experiments stay in MLflow
  under their old names. Under the prefix `rainforest`, an experiment from before the prefix tag (tagged only with
  the dataset id, and named `rainforest/<id>-<name>` after whatever the dataset was called then) is adopted and given
  the tag, even when the dataset has been renamed since. The prefix may hold only letters, digits and `.`, `_`, `/`
  and `-`; the API refuses to start with any other. The Helm chart sets it to the release's namespace unless
  `mlflow.experimentPrefix` says otherwise ([Running Rainforest on Kubernetes](../deploy/README.md#mlflow)).
- One MLflow run per model run, named `run <id> · <model label>`, such as `run 12 · LightGBM` (or `run 13 · Ridge
  regression` for the linear model on number targets).

## What each run records

### Params

| Param | Contents |
|---|---|
| `model`, `order`, `cv`, `cv_folds`, `seed`, `device`, `chain_method`, `export_family`, `n_train`, `n_features` | the run's set-up |
| `estimator.<name>` | every effective estimator parameter: defaults, your overrides and the device settings |
| `prep.adjusted_fields` | how many fields the experiment adjusted from its baseline |
| `target.<name>.task` | each target's *Predict as*: `classes`, `regression` or `bins` |
| `prep.valid_size`, `prep.test_size`, `prep.split_seed` | the export's split |
| `rainforest.pull_id` | for a connected dataset, the pull the export's data came from |

### Metrics

- Each scalar metric per split and target, as `<split>.<target>.<metric>`. Examples: `test.column_a.roc_auc`,
  `valid.column_c.r2`.
- Cross-validation means as `cv.<target>.<metric>`, and spreads as `cv_std.<target>.<metric>`.
- `fit_seconds`.

Characters MLflow doesn't allow in a key become `_`.

### Artifacts

| Artifact | Contents |
|---|---|
| `report.json` | the run's report, the same file the API serves |
| `model/classifier_chain.joblib` | the trained chain, the same file the API serves; `model/target_chain.joblib` when any target is a number |
| `adjustments.json` | the baseline and the experiment's adjustments, field by field |
| `preparation/manifest.json` | the export's manifest: every field's type and settings, the targets and the split |

So each run records the full preparation behind it. The server stores artifacts on the `mlartifacts` volume and
proxies uploads and downloads, so clients only talk HTTP to it.

### Tags

| Tag | Contents |
|---|---|
| `rainforest.dataset_id`, `rainforest.dataset_name` | the dataset |
| `rainforest.export_id` | the export |
| `rainforest.recipe_id`, `rainforest.recipe_version` | the baseline version the export was built from |
| `rainforest.baseline` | its label, such as `Trees baseline v2` |
| `rainforest.baseline_family` | its model family |
| `rainforest.adjusted` | the number of adjusted fields |
| `rainforest.targets_changed` | `true` when the experiment's targets differ from the baseline's |
| `rainforest.experiment_id`, `rainforest.experiment_name` | the Rainforest experiment |
| `rainforest.run_id` | the Rainforest model run |
| `rainforest.model` | the model |
| `rainforest.model_family` | the family the export was prepared for |
| `rainforest.user` | who started the run |
| `rainforest.git_sha`, `mlflow.source.git.commit` | the commit, when the image was built with `GIT_SHA` (`make up` passes the checkout's commit) |

## Private experiments wait

MLflow has no access control of its own. Anyone who can sign in to Rainforest can browse everything in it.

So a run in a [private experiment](experiments-and-training.md#privacy) is not logged while its baseline version is
assessing. The runs table shows *MLflow after review*. When review opens, those runs are logged. Runs that finish
after that are logged at once. The API also logs any still waiting when it starts.

## When MLflow is down

The run still finishes. Logging gives up after a short timeout: 15 seconds per request, with two retries. The run
records the reason as `mlflow_error` in its report. The runs table shows *Not in MLflow*, with the reason on hover.

Nothing is lost. `make mlflow-backfill` logs every finished run that has no MLflow run, such as one trained while
MLflow was down or turned off. It skips runs of an experiment that is still private. It runs only when you start it.

## Turning it off

Set `MLFLOW_TRACKING_URI=` (empty) in `.env`. Runs are then not logged and show no link. The `mlflow` service still
starts. Stop it with `docker compose stop mlflow` if you don't want it.

## Storage

The tracking data lives in a database named `mlflow` in the same Postgres. The `mlflow` service creates it on start
if it is missing. So it works on an existing `pgdata` volume as well as a new one.

## Not logged

Runs from the command line (`make train`) aren't model runs in `meta.model_runs`, so they aren't logged to MLflow.
See [Training from the command line](experiments-and-training.md#training-from-the-command-line).
