# 🌲 Rainforest

Rainforest helps a team agree, field by field, on how to prepare a wide, mixed-type dataset for tree-based models,
then exports it model-ready and trains models on it.

<!-- TODO: add a screenshot of the Review screen at docs/images/review.png -->
![The Review screen](docs/images/review.png)

## What it does

- **Blind team review.** Each assessor reviews every field on their own. Disagreements go to a queue where a lead
  settles them. The result is a frozen, versioned **baseline** that always exports the same way.
- **Tree-aware defaults.** Tree ensembles split on the order of values, so Rainforest never suggests scaling or log
  transforms for them, and leaves missing values for the model to route. *Linear* and *Neural* baselines get their own
  defaults.
- **Explained suggestions.** Every field gets an inferred type and suggested settings, with the evidence. Possible
  target leaks are flagged.
- **Multi-target model chains.** Predict several targets at once, each as classes, a number or bands. Each model in
  the chain also learns from the predictions for the targets before it.
- **Experiments without side effects.** Try other choices on a copy of the baseline, export with a leakage-safe
  split, and train LightGBM, XGBoost (CPU or GPU) and other models.
- **MLflow logging.** Every run is logged with its settings, metrics, model and full preparation.

It is for data science teams who need to agree on, and record, how a dataset is prepared. It works solo too.

## How it works

```
Upload → Assess → Submit → Team review → Finalize baseline → Experiment → Export → Train → MLflow
```

1. **Upload.** You add a CSV or Parquet file, and Rainforest profiles and suggests a decision for every field.
2. **Assess.** Each assessor keeps, drops or retypes every field and checks its settings, on their own.
3. **Submit.** Each assessor submits their assessment once every field is reviewed.
4. **Team review.** Fields everyone agreed on are adopted, and a lead settles the rest.
5. **Finalize baseline.** A lead freezes the decisions as a numbered baseline version.
6. **Experiment.** You start an experiment from the baseline and adjust its copy if you want to.
7. **Export.** Rainforest splits the rows, fits the preprocessing on training rows only, and writes Parquet.
8. **Train.** You train a chain of models on the export and compare the runs.
9. **MLflow.** Every run is logged with the preparation behind it.

The services:

```
┌────────────┐  /api/*   ┌──────────────────────────┐        ┌────────────────────────────────┐
│  ui        │ ────────► │  api (FastAPI)           │ ─────► │  db (Postgres 16)              │
│  React +   │  nginx    │  ingest · profile ·      │  SQL   │  meta.*  fields, stats,        │
│  ECharts   │  proxy    │  correlate · explore ·   │        │          decisions, settings,  │
│  :8080     │           │  export · train chains   │        │          exports, model runs,  │
└────────────┘           └─────────┬────────────────┘        │          analyses              │
                                   │ /data volume             │  raw.ds_<id>  your data        │
                                   │ Parquet copies, exports, └────────────────────────────────┘
                                   │ trained models
```

## Quick start

### Prerequisites

- Docker with Compose v2.
- At least 6 GB of RAM free. The stack uses about 2.3 GB at rest, and large exports peak at about 2 GB more.
- Disk space of roughly 2–3× the size of the files you upload, plus room for the images.

See [Sizing](docs/deployment.md#sizing) for where these figures come from.

### Commands

> **Warning**
> Set `POSTGRES_PASSWORD` and `READONLY_PASSWORD` in `.env` before the first `make up`. Postgres reads them only when
> it creates its data volume, so changing them later has no effect.

```bash
cp .env.example .env    # then set POSTGRES_PASSWORD and READONLY_PASSWORD
make up                 # builds and starts everything; the first build takes a few minutes
```

### What you'll see

Open `http://localhost:8080`. From another machine, use the Docker host's address. `UI_PORT` sets the port.

The sign-in page shows one profile, *Admin*, a lead. Pick it. The empty *Datasets* page offers *Open the 50k × 150
demo*.

> **Warning**
> The default sign-in mode doesn't authenticate. Anyone who reaches the page can pick any profile, including a lead.
> Use it for a single person or a trusted network only. Teams use OIDC sign-in: see
> [Deployment](docs/deployment.md#sign-in).

## Your first dataset

This walkthrough uses the *Concrete strength* sample: predict a concrete's strength from its mix. You are the lead
and the only assessor.

1. On *Datasets*, choose *Add a sample ▾ → Real datasets* and add *Concrete strength*. Its target, `strength_mpa`, is
   already set.
2. Open the dataset when it is ready. A new dataset opens on its *Baseline* tab, at *Set up the first baseline*.
   Choose the model type *Trees*, keep *the suggestions* as the starting point, keep the target ticked, and leave
   yourself as the only assessor. Choose *Start Trees baseline v1*.
3. You land in *Fields*, on the first field. Walk through the fields and their *Why these suggestions*. Press `A` to
   use a suggestion and confirm it. The target's *Predict as* says it is predicted as a number. You can change it to
   bands.
4. When every field is reviewed, the button at the right of the top bar's second row reads *Submit assessment*.
   Choose it, and confirm. With one assessor there is nothing to settle, so every field is adopted.
5. The same button now reads *Finalize Trees v1*. Choose it, and confirm. This freezes the baseline as *Trees v1* and
   takes you to *Experiments*.
6. Choose *New experiment from v1*, give it a name, and create it.
7. In the experiment, choose *Export for trees models* on *Exports*. Then, on *Train*, train a model. LightGBM is the
   default.
8. Compare the runs on *Runs*, and open *MLflow ↗* on a run to see it in MLflow.

To change the baseline later, or prepare the data for another model type, use the *Baseline* tab.

Next, read the [user guide](docs/user-guide.md) and the [team workflow](docs/team-workflow.md).

## Deployment options

| Setup | Use when | How |
|---|---|---|
| Solo / demo | One person, or a trusted network | `make up` with the defaults. See [Sign-in](docs/deployment.md#sign-in). |
| Team with OIDC | Several people, real sign-in | `RAINFOREST_AUTH_MODE=oidc` and your provider's settings in `.env`. See the [checklist](docs/deployment.md#team-deployment-checklist). |
| Team with the reference Keycloak | Trying OIDC (development only) | `make up-keycloak`, or `COMPOSE_PROFILES=keycloak` in `.env`. See [Reference Keycloak](docs/deployment.md#reference-keycloak-development-only). |
| GPU | XGBoost on an NVIDIA GPU | `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` in `.env`, or `make up-gpu`. See [GPU](docs/deployment.md#gpu). |

## Common commands

| Command | Purpose |
|---|---|
| `make up` | Build and start everything as `.env` describes |
| `make up-gpu` | `make up` plus the CUDA XGBoost image, for this run only |
| `make up-keycloak` | `make up` plus the development Keycloak, for this run only |
| `make gpu-check` | Show what the running API sees of the GPU |
| `make down` | Stop and remove the containers; data volumes are kept |
| `make logs` | Follow the API's logs |
| `make ps` | List the containers |
| `make train EXPORT=<id>` | Train on an export from the command line (`MODEL=`, `DEVICE=` optional) |
| `make models` | List registered models and whether they're installed |
| `make mlflow-backfill` | Log every finished run that has no MLflow run |
| `make psql` | Open psql as the database owner |
| `make sample-csv` | Write a 200k-row sample CSV to `./sample.csv` |
| `make test` | Run the API tests locally; database-backed tests skip |
| `make test-db` | Run the full API suite against a throwaway Postgres |
| `make clean` | Stop everything and **delete all data volumes** |

## Documentation

**Using Rainforest**

- [Concepts](docs/concepts.md): the terms every other page uses.
- [User guide](docs/user-guide.md): each screen, the top bar and keyboard shortcuts.
- [Team workflow](docs/team-workflow.md): roles, assessing, team review, finalizing and new versions.
- [Data preparation](docs/data-preparation.md): type inference, defaults, leak screen, numbers and bands.
- [Explore](docs/explore.md): correlations, the row map, feature tests and importance.
- [Experiments and training](docs/experiments-and-training.md): adjustments, exports, chains and metrics.
- [MLflow](docs/mlflow.md): what each run logs, and where.
- [SQL](docs/sql.md): the SQL console, example queries and the schema.

**Running it**

- [Deployment](docs/deployment.md): team checklist, sign-in, GPU, sizing, backups, troubleshooting.
- [Configuration](docs/configuration.md): every `.env` setting.
- [Security](docs/security.md): network exposure, sessions, OIDC and privacy.

**Developing it**

- [Development](docs/development.md): local setup, tests and CI.
- [Architecture](docs/architecture.md): how the pieces fit, and where to add things.
- [API](docs/api.md): every HTTP endpoint.

## Data attribution

*Add a sample ▾ → Real datasets* adds one of five datasets from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/), used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

| Dataset | Rows × columns | Targets | What it exercises | Source |
|---|---|---|---|---|
| Diabetes readmission | 101,766 × 50 | `readmitted` (3 classes) | heavy missingness, ID columns, high-cardinality diagnosis codes | Clore, Cios, DeShazo & Strack (2014), [doi:10.24432/C5230J](https://doi.org/10.24432/C5230J) |
| Student grades | 649 × 33 | `G1`, `G2`, `G3` (grades 0–20) | a chain where each grade helps predict the next | Cortez (2008), [doi:10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T) |
| Student dropout | 4,424 × 37 | `Target` (dropout, enrolled, graduate) | an imbalanced target; categoricals stored as numeric codes | Realinho, Vieira Martins, Machado & Baptista (2021), [doi:10.24432/C5MC89](https://doi.org/10.24432/C5MC89) |
| Steel plate faults | 1,941 × 34 | seven yes/no fault types | long chains | Buscema, Terzi & Tastle (2010), [doi:10.24432/C5J88N](https://doi.org/10.24432/C5J88N) |
| Concrete strength | 1,030 × 9 | `strength_mpa` (a number) | a regression target, or the same cut into bands | Yeh (1998), [doi:10.24432/C5PK67](https://doi.org/10.24432/C5PK67) |
