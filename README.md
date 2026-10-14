# 🌲 Rainforest

Rainforest is a self-hosted web app for preparing wide, mixed-type tabular data for machine learning. A team reviews
every field of a dataset, agrees on how each one is prepared, and freezes that as a versioned **baseline**. Rainforest
then exports model-ready Parquet from it, trains models, and logs every run to MLflow.

It is for data science teams who need to agree on, and record, how a dataset is prepared. It works solo too.

![Reviewing a field: its profile, distribution and suggested settings, with the reasons](docs/images/review.png)

## How it works

```
Upload → Assess → Team review → Finalize baseline → Experiment → Export → Train
```

1. **Upload** a CSV or Parquet file. Rainforest profiles it and suggests a decision for every field.
2. **Assess.** Each assessor keeps, drops or retypes every field and checks its settings, on their own, then submits.
3. **Team review.** Fields everyone agreed on are adopted. A lead settles the rest.
4. **Finalize.** A lead freezes the decisions as a numbered baseline version that always exports the same way.
5. **Experiment.** Start an experiment from the baseline, and adjust its copy if you want to.
6. **Export.** Rainforest splits the rows, fits the preprocessing on training rows only, and writes Parquet.
7. **Train** a chain of models on the export, compare the runs, and open each one in MLflow.

Along the way:

- **Suggestions come with reasons.** Every field gets an inferred type and settings, with the evidence. Possible
  target leaks are flagged.
- **Defaults depend on the model type.** *Trees* baselines never scale or log-transform, and leave missing values for
  the model to route. *Linear* and *Neural* baselines get their own defaults.
- **Several targets at once.** Each target is predicted as classes, a number or bands, and each model in the chain
  also learns from the predictions for the targets before it.
- **Models:** LightGBM, XGBoost (CPU or NVIDIA GPU), scikit-learn's histogram GBM and random forest, logistic and ridge
  regression, and an MLP, with cross-validation and parameter sweeps.
- **Explore** correlations, a row map, feature tests and importance, or query the data in a read-only SQL console.

## Quick start

You need Docker with Compose v2, `make`, at least 6 GB of free RAM, and disk space of about 2–3× the files you upload.
See [Sizing](docs/deployment.md#sizing).

```bash
git clone https://github.com/LikelyCause/rainforest.git && cd rainforest
cp .env.example .env    # then set POSTGRES_PASSWORD and READONLY_PASSWORD
make up                 # builds and starts everything; the first build takes a few minutes
```

> [!IMPORTANT]
> Set `POSTGRES_PASSWORD` before the first `make up`. Postgres reads it only when it creates its data volume, so
> changing it later has no effect.

Open `http://localhost:8080` (or the Docker host's address; `UI_PORT` sets the port) and pick the *Admin* profile.
MLflow is at `/mlflow/`.

> [!WARNING]
> The default sign-in mode doesn't authenticate: anyone who reaches the page can pick any profile, a lead's included.
> Use it alone or on a trusted network. Teams should use OIDC; see [Sign-in](docs/deployment.md#sign-in).

## Try it

This uses the *Concrete strength* sample, which predicts a concrete's strength from its mix, with you as the only
assessor.

1. On *Datasets*, choose *Add a sample ▾ → Real datasets* and add *Concrete strength*. Open it when it's ready.
2. At *Set up the first baseline*, keep the defaults (*Trees*, the suggestions, the target `strength_mpa`, and you as
   the only assessor) and choose *Start Trees baseline v1*.
3. In *Fields*, step through the fields and read *Why these suggestions*. Press `A` to accept a suggestion and move
   on.
4. Choose *Submit assessment* at the top right, then *Finalize Trees v1*. With one assessor there is nothing to settle.
5. Choose *New experiment from v1*, name it, and choose *Export for trees models*.
6. On *Train*, choose *Train* (LightGBM is the default). The run appears on *Runs*, with a link to it in MLflow.

The empty *Datasets* page also offers a generated *50k × 150 demo*, and you can upload your own CSV or Parquet file.

## Deploying

| Setup | Use when | How |
|---|---|---|
| Solo / demo | One person, or a trusted network | `make up` with the defaults |
| Team with OIDC | Several people, real sign-in | `RAINFOREST_AUTH_MODE=oidc` and your provider's settings in `.env`; see the [checklist](docs/deployment.md#team-deployment-checklist) |
| Reference Keycloak | Trying OIDC (development only) | `make up-keycloak`; see [Reference Keycloak](docs/deployment.md#reference-keycloak-development-only) |
| GPU | XGBoost on an NVIDIA GPU | `make up-gpu`; see [GPU](docs/deployment.md#gpu) |

Everything is configured in `.env`; [Configuration](docs/configuration.md) lists every setting. The stack runs four
containers:

| Service | What it does |
|---|---|
| `ui` | nginx serving the React app; proxies `/api/` to the API and `/mlflow/` to MLflow |
| `api` | FastAPI: profiling, exports and training jobs; keeps files on the `appdata` volume |
| `db` | Postgres 16: app metadata (`meta.*`), each dataset's rows (`raw.ds_<id>`), and MLflow's runs |
| `mlflow` | MLflow tracking server, with model artifacts on the `mlartifacts` volume |

Useful commands:

| Command | Purpose |
|---|---|
| `make up` / `make down` | Start everything / stop it, keeping the data |
| `make logs` | Follow the API's logs |
| `make train EXPORT=<id>` | Train on an export from the command line (`MODEL=`, `DEVICE=` optional) |
| `make psql` | Open psql as the database owner |
| `make test-db` | Run the API tests against a throwaway Postgres |
| `make clean` | Stop everything and **delete all data volumes** |

The [Makefile](Makefile) has the rest.

## Documentation

- **Using it:** [Concepts](docs/concepts.md) · [User guide](docs/user-guide.md) ·
  [Team workflow](docs/team-workflow.md) · [Data preparation](docs/data-preparation.md) · [Explore](docs/explore.md) ·
  [Experiments and training](docs/experiments-and-training.md) · [MLflow](docs/mlflow.md) · [SQL](docs/sql.md)
- **Running it:** [Deployment](docs/deployment.md) · [Configuration](docs/configuration.md) ·
  [Security](docs/security.md)
- **Developing it:** [Development](docs/development.md) · [Architecture](docs/architecture.md) · [API](docs/api.md)

## License

Rainforest is released under the [MIT License](LICENSE).

The five sample datasets under *Add a sample ▾ → Real datasets* come from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/) and are used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

| Dataset | Rows × columns | Targets | Source |
|---|---|---|---|
| Diabetes readmission | 101,766 × 50 | `readmitted` (3 classes) | Clore, Cios, DeShazo & Strack (2014), [doi:10.24432/C5230J](https://doi.org/10.24432/C5230J) |
| Student grades | 649 × 33 | `G1`, `G2`, `G3` (grades 0–20) | Cortez (2008), [doi:10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T) |
| Student dropout | 4,424 × 37 | `Target` (dropout, enrolled, graduate) | Realinho, Vieira Martins, Machado & Baptista (2021), [doi:10.24432/C5MC89](https://doi.org/10.24432/C5MC89) |
| Steel plate faults | 1,941 × 34 | seven yes/no fault types | Buscema, Terzi & Tastle (2010), [doi:10.24432/C5J88N](https://doi.org/10.24432/C5J88N) |
| Concrete strength | 1,030 × 9 | `strength_mpa` (a number) | Yeh (1998), [doi:10.24432/C5PK67](https://doi.org/10.24432/C5PK67) |
