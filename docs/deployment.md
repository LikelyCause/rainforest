# Deployment

How to run Rainforest for a team, and how to keep it running. Every setting named here is in
[Configuration](configuration.md). Security details are in [Security](security.md).

## Team deployment checklist

`.env` is the whole configuration. Compose reads it on every command, so every `make up` applies it. The commented
*team deployment* block in `.env.example` has every setting below.

1. **Set the passwords** before the first start: `POSTGRES_PASSWORD` and `READONLY_PASSWORD`. Postgres reads them only
   when it creates its data volume.
2. **Set up OIDC sign-in.** Set `RAINFOREST_AUTH_MODE=oidc` and the `RAINFOREST_OIDC_*` settings for your provider.
   See [Sign-in](#sign-in) and [Registering with an OIDC provider](#registering-with-an-oidc-provider).
3. **Set the allowed hosts.** Set `RAINFOREST_ALLOWED_HOSTS` to the names and addresses people use, such as
   `rainforest.lan,192.168.1.20`.
4. **Put a TLS proxy in front** of the UI port, anywhere beyond a trusted LAN. The proxy must set
   `X-Forwarded-Proto`. See [Security](security.md#network).
5. **Optional: GPU.** Add `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml`. It builds the API with CUDA XGBoost
   and gives it the host's NVIDIA GPUs. See [GPU](#gpu).
6. **Optional: the reference Keycloak**, for trying OIDC. Add `COMPOSE_PROFILES=keycloak`, set `KEYCLOAK_URL` and
   `RAINFOREST_URL` to the URLs browsers use, and point the OIDC settings at it. See
   [Reference Keycloak](#reference-keycloak-development-only).
7. Run `make up`.

Any combination works, Keycloak with the GPU image included. Every later `make up` keeps it.

### Trying an option for one run

`make up-keycloak` and `make up-gpu` try an option without editing `.env`. Each starts the stack as `.env` describes,
plus one option, for that run only. Each prints what to add to `.env` to keep it. The next `make up` starts the stack
as `.env` describes.

| Command | Adds |
|---|---|
| `make up-gpu` | the GPU image |
| `make up-keycloak` | the Keycloak profile, plus the reference Keycloak's sign-in settings for each one `.env` leaves empty (none when `.env` names its own issuer) |

`make up-keycloak` stops with an error when `.env` sets `RAINFOREST_AUTH_MODE` to something other than `oidc`.

## Sign-in

`RAINFOREST_AUTH_MODE` chooses how people sign in.

| Mode | Use for | How it works |
|---|---|---|
| `pick` (default) | development, demos, a single person, or a trusted single-team network | The sign-in page is a centred grid of the active users' profiles (picture or initials, name and role). Picking one signs you in as them. The page says plainly that this doesn't authenticate. |
| `oidc` | team deployments | Standard OpenID Connect against your identity provider. |

### Pick mode

*Switch profile* in the profile menu goes back to the sign-in page.

Without `RAINFOREST_USERS`, leads manage the user list on the *Team* page. The first start creates one lead, *Admin*.

### Accounts from .env

In pick mode, `RAINFOREST_USERS` defines the accounts as `key[#id]:Name:role;…`. For example:

```
tanner:Tanner:lead;test-member:Test Member:member;test-viewer:Test Viewer:viewer
```

- `key` is a stable slug: lowercase letters, digits and dashes.
- The list needs at least one lead.
- The list is synced on every start. A listed key creates its account, or updates its name and role. A key that
  returns reactivates its account. Every account not listed is deactivated, never deleted, so its work stays attached.
- `#id` gives an existing account without a key this key, once, so it keeps its history.
- With the list set, accounts can't be added or edited in the app. The *Team* page lists them read-only.
- A list with a problem stops the API from starting, with a message naming it, before anything changes. Problems are a
  malformed entry, an unknown role, a repeated key, or no lead.

### OIDC mode

The API runs the authorization-code flow with PKCE. The sign-in page shows *Sign in with <provider>*, named by
`RAINFOREST_OIDC_PROVIDER_NAME`.

- A user is created the first time they sign in.
- Their name, email and role are refreshed from the ID token at every sign-in.
- The role comes from the token's groups claim. `RAINFOREST_LEAD_GROUP` (default `rainforest-lead`) makes a lead.
  `RAINFOREST_MEMBER_GROUP` (default `rainforest-member`) makes a member. There is no Viewer group.
- Someone in neither group, or deactivated, gets a *No access* page.
- *Sign out* also ends the provider's session.

### Registering with an OIDC provider

1. Register Rainforest as a **confidential client**.
2. Set the redirect URI to `<Rainforest URL>/api/auth/oidc/callback`.
3. Set `<Rainforest URL>/login` as a post-logout redirect URI.
4. Configure a groups claim in the ID token, holding the lead and member groups.
5. Put the issuer, client id and client secret in `.env` as `RAINFOREST_OIDC_ISSUER`, `RAINFOREST_OIDC_CLIENT_ID` and
   `RAINFOREST_OIDC_CLIENT_SECRET`.

Set `RAINFOREST_OIDC_REDIRECT_URL` when Rainforest sits behind a proxy. Set `RAINFOREST_OIDC_DISCOVERY_URL` when the
API can't reach the issuer URL itself.

## Reference Keycloak (development only)

The `keycloak` profile runs [Keycloak](https://www.keycloak.org/) on port 8081 (`KEYCLOAK_PORT`) as a reference
identity provider. With a fresh `.env`, `make up-keycloak` runs it at `http://localhost:8081` and signs Rainforest in
against it.

> **Warning**
> This is a development setup, not a production one. Keycloak runs in `start-dev` mode over plain HTTP. The client
> secret (`rainforest-dev-secret`) and the passwords below are public. The realm allows HTTP from anywhere. For real
> use, run Keycloak (or any OIDC provider) in production mode behind TLS, and set the `RAINFOREST_OIDC_*` settings for
> it.

Keycloak keeps its own `keycloak` database in the same Postgres. On first start it imports the `rainforest` realm
from `keycloak/realm-rainforest.json`. The realm has:

- the confidential client `rainforest`
- the groups `rainforest-lead` and `rainforest-member`, mapped into the ID token's `groups` claim
- these users:

| User | Password | Group |
|---|---|---|
| `lead1` | `lead1-dev-password` | rainforest-lead |
| `ds1` | `ds1-dev-password` | rainforest-member |
| `ds2` | `ds2-dev-password` | rainforest-member |
| `guest1` | `guest1-dev-password` | none (shows the *No access* page) |

Keycloak's admin console is at `<KEYCLOAK_URL>/admin`. Sign in as `admin` with `KEYCLOAK_ADMIN_PASSWORD` (default
`admin`).

Keycloak takes a minute or so to start. Signing in fails until it is ready.

### Using it from other machines

Set both URLs in `.env` as the other machines see them. For example:

```
KEYCLOAK_URL=http://rainforest.lan:8081
RAINFOREST_URL=http://rainforest.lan:8080
```

Set the issuer and redirect URL to match. `.env.example` shows the set. `make up-keycloak` derives the issuer and
redirect URL from the two URLs when `.env` leaves them empty. It also takes the URLs on its command line:
`make up-keycloak KEYCLOAK_URL=… RAINFOREST_URL=…`.

### How the issuer works

Tokens name their issuer by the URL browsers use. So Keycloak runs with `KC_HOSTNAME` set to `KEYCLOAK_URL`, and the
API checks tokens against the same issuer.

The API container can't reach that URL when it names `localhost`. So the API reads discovery from
`http://keycloak:8080` (`RAINFOREST_OIDC_DISCOVERY_URL`). Keycloak's dynamic backchannel
(`KC_HOSTNAME_BACKCHANNEL_DYNAMIC`) answers it with token and key URLs on that internal name. The issuer and the
browser-facing URLs stay external.

### Changing RAINFOREST_URL later

The redirect URIs are written into the realm on its first import only. After changing `RAINFOREST_URL`, do one of
these:

- Edit the `rainforest` client in Keycloak's admin console.
- Reimport the realm:
  1. Stop Keycloak: `docker compose stop keycloak`.
  2. Drop its database: `docker compose exec db dropdb -U <POSTGRES_USER> keycloak`.
  3. Start it again: `make up` with the profile in `.env`, or `make up-keycloak`.

## GPU

XGBoost can train on an NVIDIA GPU. How to choose the GPU for a run is in
[Experiments and training](experiments-and-training.md#gpu-training).

The default image ships the 5 MB CPU build of XGBoost (`xgboost-cpu`). The GPU image swaps in the CUDA build
(`xgboost`, CUDA 13), which adds about 800 MB, mostly its NCCL dependency.

### Host prerequisites

- An NVIDIA driver **580 or newer** for the default CUDA 13 build.
  - Blackwell cards (RTX 50xx) need the open kernel modules (`nvidia-open`).
  - With a 570–579 driver, build with `XGBOOST_GPU_PKG=xgboost-cu12==3.4.1` (CUDA 12.9).
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  registered with Docker:

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

- Containers must see the GPU. Check with:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
  ```

The host needs no CUDA toolkit. The XGBoost wheel bundles its CUDA runtime, and the Container Toolkit mounts the
driver.

### Starting with the GPU

1. Add `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` to `.env`. For a 570–579 driver, also add
   `XGBOOST_GPU_PKG`.
2. Run `make up`. Every later `make up` keeps the GPU image while `.env` names the GPU file.
3. Run `make gpu-check`. It shows what the API sees: the GPU's name, memory and driver, and whether XGBoost can use
   it.

`make up-gpu` builds and starts the GPU image for one run, without editing `.env`. With `COMPOSE_FILE` unset, or
without the GPU file, `make up` builds the CPU image.

### Details

- `docker-compose.gpu.yml` builds the API with `GPU=1` and reserves the host's GPUs for it.
- `GET /api/system` reports `{"gpu": null}` when no GPU is visible. When XGBoost can't use a visible GPU (CPU build,
  driver too old), it reports the GPU with `available: false` and a `reason`.
- Each API worker detects the GPU once, in the background as it starts, with a one-tree test fit. This is needed
  because XGBoost otherwise falls back to the CPU silently.
- A GPU run that falls back to the CPU anyway fails, rather than being recorded as a GPU run.

## Sizing

These figures come from the code's defaults and limits. Measure your own data before you commit to hardware.

| Resource | What uses it | Figure |
|---|---|---|
| Disk | each uploaded file: the upload on the `/data` volume, a Parquet copy, and a full typed copy in Postgres (`raw.*`) | roughly 2–3× the file size |
| Disk | the GPU image | about 800 MB more than the CPU image |
| RAM | the stack at rest (measured on a new install, with the defaults) | about 2.3 GB: MLflow about 1.9 GB, the API about 340 MB, Postgres about 90 MB, the UI under 10 MB |
| RAM | Postgres, as data grows | up to `shared_buffers=512MB` (set in `docker-compose.yml`) plus working memory |
| RAM | an export | peaks at a few times `EXPORT_CHUNK_MB` (256 MB) and `EXPORT_FIT_MB` (512 MB). A 300k-row export peaks at about 1.3–2 GB. |
| RAM | a generated sample at the `SAMPLE_MAX_CELLS` cap (60M cells) | about 2 GB peak |
| CPU | a background job | `cpu_count // JOB_SLOTS` threads. With the default `JOB_SLOTS=1`, one job uses every core. |

So plan for at least 6 GB of RAM free for the stack: about 2.3 GB at rest, plus one job peaking at about 2 GB, plus
headroom. Add more for wider or longer data, for `JOB_SLOTS` above 1, or for the Keycloak profile. A small dataset
adds little: training on *Concrete strength* (1,030 rows) raised the stack's total by under 100 MB.

### How background jobs run

Ingest, target associations, export, training and *Explore* analyses run as background jobs.

- They run in a thread pool inside the API process. Their state lives in Postgres.
- At most `JOB_SLOTS` run at once across all workers (one by default, using every core).
- The rest wait as *Queued* and start oldest first. So a quick job submitted behind a long training run waits for it.
- Each job row records the process that runs it. When a worker starts, it fails the queued and running jobs of
  processes that are gone, and leaves its siblings' jobs alone.
- An interrupted target-associations job leaves its dataset ready, with a warning to choose the targets again.
- For multi-node deployments, swap `app/jobs.py` for a queue (RQ, Celery, Arq).

## What to back up

| What | Where | Contents |
|---|---|---|
| `.env` | the repository root | the whole configuration, including passwords and `RAINFOREST_SESSION_SECRET` if you set it |
| `pgdata` volume | Postgres data directory | the `rainforest` database (`meta.*` and `raw.*`), plus the `mlflow` and `keycloak` databases |
| `appdata` volume | mounted at `/data` in the API | uploads, Parquet copies, exports, trained models, profile pictures (`avatars`), and the generated `session_secret` |
| `mlartifacts` volume | mounted in the MLflow server | MLflow's run artifacts |

Compose names the volumes with the project name, so on the host they are `rainforest_pgdata`, `rainforest_appdata`
and `rainforest_mlartifacts`.

## Upgrading

1. Update your checkout.
2. Run `make up`. It rebuilds the images and restarts the containers.

The API applies new database migrations once on start, under an advisory lock, and records them in
`meta.schema_migrations`. It recreates the views on every start. See [SQL](sql.md#migrations-and-views).

## Stopping and resetting

| Command | Effect |
|---|---|
| `make down` | stops and removes the containers, the Keycloak profile's included. Data volumes are kept. |
| `docker compose stop mlflow` | stops only the MLflow server, for example after turning MLflow off |
| `make clean` | stops everything and **deletes all data volumes** |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| A changed `POSTGRES_PASSWORD` or `POSTGRES_USER` has no effect, or the API can't connect | Postgres applies these only when it creates the `pgdata` volume. Set them back to the values of the first start. Or, to start over with no data, run `make clean` and then `make up`. |
| Signing in through the reference Keycloak fails right after start | Keycloak takes a minute or so to start. Wait and try again. |
| Keycloak rejects the redirect after you changed `RAINFOREST_URL` | The realm took the redirect URIs on its first import. See [Changing RAINFOREST_URL later](#changing-rainforest_url-later). |
| `make up-keycloak` stops with an error about `RAINFOREST_AUTH_MODE` | `.env` sets another mode. Set it to `oidc`, or remove it. |
| The API doesn't start after you edited `RAINFOREST_USERS` | The list has a problem. `make logs` shows the message naming it. |
| No CPU / GPU switch in the *Train a model* panel | Run `make gpu-check`. It says whether the API sees a GPU and why XGBoost can't use it. Check the [host prerequisites](#host-prerequisites). |
| The runs table shows *not logged* | MLflow was down or turned off. The reason is in the run's report as `mlflow_error`. Run `make mlflow-backfill` once MLflow is up. See [MLflow](mlflow.md#when-mlflow-is-down). |
