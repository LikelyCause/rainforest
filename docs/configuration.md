# Configuration

`.env` is the deployment's whole configuration. Compose reads it on every `docker compose` or `make` command, so
`make up`, `make down`, `make logs` and the rest always apply it. Start from `.env.example`.

Write a literal `$` in a value as `$$`.

The defaults below are the ones `docker-compose.yml` applies when `.env` leaves a variable unset.

## Compose settings

| Variable | Default | Purpose |
|---|---|---|
| `COMPOSE_FILE` | `docker-compose.yml` | `docker-compose.yml:docker-compose.gpu.yml` for the GPU image (see [GPU](deployment.md#gpu)) |
| `COMPOSE_PROFILES` | – | `keycloak` to run the reference Keycloak (see [Reference Keycloak](deployment.md#reference-keycloak-development-only)) |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | rainforest | database name and owner; Postgres applies them when it creates the `pgdata` volume, so set them before the first start |
| `READONLY_PASSWORD` | rainforest_ro | password for the SQL console's `rainforest_ro` role |
| `UI_PORT`, `API_PORT`, `DB_PORT` | 8080, 8000, 5432 | published ports (`API_PORT` and `DB_PORT` bind to 127.0.0.1) |
| `API_WORKERS` / `JOB_WORKERS` | 2 / 2 | uvicorn workers / background job threads per worker |
| `JOB_SLOTS` | 1 | background jobs running at once across all workers; each uses `cpu_count // JOB_SLOTS` threads, so the default gives one job every core |
| `LOG_LEVEL` | info | API log level |
| `RAINFOREST_AUTH_MODE` | `pick` | `oidc` for team deployments; `pick` lists users to choose from and doesn't authenticate (see [Sign-in](deployment.md#sign-in)) |
| `RAINFOREST_USERS` | – | pick mode's accounts, `key[#id]:Name:role;…` (roles `lead`, `member`, `viewer`); when set, the whole list, synced on every start (see [Accounts from .env](deployment.md#accounts-from-env)) |
| `RAINFOREST_OIDC_ISSUER` | – | the provider's issuer URL, exactly as it appears in its tokens (e.g. `https://sso.example.com/realms/rainforest`). Required in `oidc` mode |
| `RAINFOREST_OIDC_CLIENT_ID`, `RAINFOREST_OIDC_CLIENT_SECRET` | – | the confidential client registered for Rainforest. Required in `oidc` mode |
| `RAINFOREST_OIDC_REDIRECT_URL` | from the request | `<Rainforest URL>/api/auth/oidc/callback`; set it when Rainforest sits behind a proxy |
| `RAINFOREST_OIDC_DISCOVERY_URL` | issuer + `/.well-known/openid-configuration` | where the API reads discovery, when it can't reach the issuer URL itself (the Keycloak profile: `http://keycloak:8080/realms/rainforest/.well-known/openid-configuration`) |
| `RAINFOREST_OIDC_SCOPES` | `openid profile email` | requested scopes |
| `RAINFOREST_OIDC_GROUPS_CLAIM` | `groups` | ID token claim with the user's groups (Keycloak's full group paths like `/rainforest-lead` also work) |
| `RAINFOREST_OIDC_PROVIDER_NAME` | `SSO` | the name on the *Sign in with …* button |
| `RAINFOREST_LEAD_GROUP`, `RAINFOREST_MEMBER_GROUP` | `rainforest-lead`, `rainforest-member` | groups that make a lead or a member |
| `KEYCLOAK_URL`, `RAINFOREST_URL` | `http://localhost:8081`, `http://localhost:8080` | Keycloak profile only: the URLs browsers use for Keycloak (its issuer) and Rainforest (written into the realm's redirect URIs on first import) |
| `KEYCLOAK_PORT`, `KEYCLOAK_ADMIN_PASSWORD` | 8081, admin | Keycloak profile only: published port and admin console password |
| `RAINFOREST_SESSION_SECRET` | generated | key that signs session cookies; if empty, one is generated on first start and kept in the data volume (`/data/session_secret`) |
| `RAINFOREST_ALLOWED_HOSTS` | `*` | comma-separated hostnames / IPs the API answers to (`*.example.com` allowed). Set this to the names and addresses you use, e.g. `rainforest.lan,192.168.1.20`; `localhost`, `127.0.0.1` and `::1` are always accepted |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000/mlflow` | where the API logs model runs; empty turns MLflow logging off (see [MLflow](mlflow.md#turning-it-off)) |
| `GIT_SHA` | – | commit tagged on MLflow runs, baked into the API image; `make up` sets it from the checkout |
| `API_EXTRA_PIP` | – | extra packages baked into the API image (e.g. `catboost`) |
| `XGBOOST_GPU_PKG` | `xgboost==3.4.1` | the CUDA XGBoost package for the GPU image (`xgboost-cu12==3.4.1` for 570–579 drivers) |

## API tuning

These are read by the API (`api/app/config.py`), also from `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `PROFILE_SAMPLE_ROWS` | 500000 | rows for quantiles, histograms and text stats (counts are always exact) |
| `CORR_SAMPLE_ROWS` | 100000 | rows for associations |
| `CORR_MAX_LEVELS` | 50 | categorical levels kept for associations |
| `COPY_BATCH_ROWS` | 100000 | rows per batch when loading `raw.ds_<id>` |
| `EXPORT_CHUNK_MB`, `EXPORT_FIT_MB` | 256, 512 | export memory budgets (dense output per transform chunk, and for the fit sample) |
| `SAMPLE_MAX_CELLS` | 60000000 | cap on rows × fields for generated samples (about 2 GB peak) |
| `DB_POOL_MAX` | 10 | connections per API worker |
| `SQL_TIMEOUT_MS`, `SQL_MAX_ROWS` | 15000, 2000 | SQL console guard rails |
| `SQL_POOL_MAX` | 4 | read-only console connections per API worker (also the number of console queries it runs at once) |
