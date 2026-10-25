# rainforest

The Helm chart for Rainforest: the API, the UI, and optionally MLflow and Postgres. This page lists every value in
[`values.yaml`](values.yaml). How to install, upgrade and back up a release is in [`deploy/README.md`](../../README.md).

The defaults are a dev install (pick-mode sign-in, which does not authenticate; Postgres and MLflow inside the release;
no TLS). [`ci/dev-values.yaml`](ci/dev-values.yaml) is a working dev file and [`ci/team-values.yaml`](ci/team-values.yaml)
a team install with OIDC, an external Postgres and MLflow server, and a GPU. `values.schema.json` rejects an empty
`hostname`, so lint and template with `-f ci/dev-values.yaml`:

```bash
helm lint deploy/chart/rainforest -f deploy/chart/rainforest/ci/dev-values.yaml
```

## Top level

| Key | Default | Meaning |
|---|---|---|
| `hostname` | `""` | **Required.** The name people open Rainforest at. Feeds the Ingress host and the API's `RAINFOREST_ALLOWED_HOSTS`. |
| `imagePullSecrets` | `[]` | Registry credentials for the images, as in any pod spec: `[{name: regcred}]`. |
| `clusterDomain` | `cluster.local` | Used to build the in-cluster addresses nginx proxies to; its resolver ignores search domains. |
| `nameOverride` | `""` | Replaces the chart name in resource names. |
| `fullnameOverride` | `""` | Replaces the whole resource name prefix. |

## auth

| Key | Default | Meaning |
|---|---|---|
| `auth.mode` | `pick` | `pick`: the sign-in page lists users to choose from and does **not** authenticate. `oidc`: single sign-on. |
| `auth.users` | `""` | Pick mode's `RAINFOREST_USERS`, `key[#id]:Name:role;…` with roles `lead`, `member` or `viewer`. Ignored under `oidc`. |
| `auth.oidc.issuer` | `""` | The provider's issuer URL. |
| `auth.oidc.clientId` | `rainforest` | The client id registered with the provider. The client secret is the `oidcClientSecret` Secret key. |
| `auth.oidc.providerName` | `SSO` | The name on the *Sign in with …* button. |
| `auth.oidc.scopes` | `openid profile email` | Scopes requested. |
| `auth.oidc.groupsClaim` | `groups` | The ID token claim that holds the user's groups. |
| `auth.oidc.leadGroup` | `rainforest-lead` | Members of this group are leads. |
| `auth.oidc.memberGroup` | `rainforest-member` | Members of this group are members. Someone in neither gets a *No access* page. |
| `auth.oidc.discoveryUrl` | `""` | Only when the pod can't reach the issuer URL itself. |

The OIDC redirect URL is derived: `https://<hostname>/api/auth/oidc/callback` when `ingress.tls.secretName` is set,
`http://` otherwise.

## postgresql

| Key | Default | Meaning |
|---|---|---|
| `postgresql.enabled` | `true` | `true`: a Postgres StatefulSet in the release. `false`: the server under `postgresql.external`. The image reads `POSTGRES_PASSWORD` only when it initialises an empty volume, so rotating the password also means `ALTER ROLE` in the database. |
| `postgresql.database` | `rainforest` | The database the in-chart server is created with. |
| `postgresql.user` | `rainforest` | The user the in-chart server is created with. |
| `postgresql.image.repository` | `postgres` | Image of the in-chart server. |
| `postgresql.image.tag` | `16-alpine` | |
| `postgresql.image.pullPolicy` | `IfNotPresent` | |
| `postgresql.persistence.size` | `20Gi` | Size of the server's volume. |
| `postgresql.persistence.storageClass` | `""` | Empty: the cluster's default class. |
| `postgresql.resources` | `{}` | Pod resource requests and limits. |
| `postgresql.tuning.shared_buffers` | `512MB` | Server settings, passed as `postgres -c key=value`; the defaults are the Compose file's. |
| `postgresql.tuning.work_mem` | `32MB` | |
| `postgresql.tuning.maintenance_work_mem` | `256MB` | |
| `postgresql.tuning.max_wal_size` | `4GB` | |
| `postgresql.readonlyRole` | `""` | The SQL console's `SELECT`-only role. Empty: `<database>_ro`, so releases sharing a server don't share a role. |

### postgresql.external

Used when `postgresql.enabled` is `false`.

| Key | Default | Meaning |
|---|---|---|
| `postgresql.external.host` | `""` | The server's host. Required when `postgresql.enabled` is `false`. |
| `postgresql.external.port` | `5432` | |
| `postgresql.external.database` | `rainforest` | The release's database. |
| `postgresql.external.user` | `rainforest` | The user that owns it. Its password is the `postgresPassword` Secret key. |
| `postgresql.external.managedRoles` | `true` | `true`: the API creates the console role on start (needs `CREATEROLE`), and in-chart MLflow creates its database (needs `CREATEDB`). `false`: the operator created the console role and, with in-chart MLflow, the MLflow database; the statements are in [deploy/README.md](../../README.md#external-postgres). The in-chart server is always managed. |

## mlflow

| Key | Default | Meaning |
|---|---|---|
| `mlflow.mode` | `in-chart` | `in-chart`: an MLflow server in the release, reached by people at the UI's `/mlflow/` behind the Rainforest sign-in. `external`: an existing server; the API logs to `trackingUri` (or `externalUrl`) and *MLflow ↗* opens `externalUrl`. `off`: no experiment logging and no MLflow link. |
| `mlflow.externalUrl` | `""` | Where browsers open an external server. Required when `mlflow.mode` is `external`. |
| `mlflow.trackingUri` | `""` | Where the API logs to when that differs from `externalUrl`, often an internal address without the single sign-on in front. |
| `mlflow.experimentPrefix` | `rainforest` | Experiments are named `<prefix>/<dataset id>-<dataset name>` and found by dataset id and prefix. Releases sharing one server each set their own. Changing it on an install with runs starts new experiments; the old ones stay. |

Credentials for an external server are the optional Secret keys `mlflowTrackingUsername`, `mlflowTrackingPassword` and
`mlflowTrackingToken`.

### In-chart mode only

| Key | Default | Meaning |
|---|---|---|
| `mlflow.database` | `""` | The server's backend database on the release's Postgres server. Empty: `<postgresql database>_mlflow`. Created on start when the chart manages the server's roles; otherwise the operator creates it first. |
| `mlflow.image.repository` | `registry.example.com/rainforest/mlflow` | The repository's `mlflow/` image; set it to your registry. |
| `mlflow.image.tag` | `""` | Empty: the chart's `appVersion` (Rainforest's version, not MLflow's). |
| `mlflow.image.pullPolicy` | `IfNotPresent` | |
| `mlflow.persistence.size` | `20Gi` | Artifacts (`/mlartifacts`); one ReadWriteOnce volume. |
| `mlflow.persistence.storageClass` | `""` | Empty: the cluster's default class. |
| `mlflow.resources` | `{}` | Pod resource requests and limits. |
| `mlflow.nodeSelector` | `{}` | |
| `mlflow.tolerations` | `[]` | |

## api

One replica, by design: background jobs run in the API process and the data volume is ReadWriteOnce.

| Key | Default | Meaning |
|---|---|---|
| `api.image.repository` | `registry.example.com/rainforest/api` | Set it to your registry. |
| `api.image.tag` | `""` | Empty: the chart's `appVersion`. |
| `api.image.pullPolicy` | `IfNotPresent` | |
| `api.resources` | `{}` | Pod resource requests and limits. |
| `api.nodeSelector` | `{}` | |
| `api.tolerations` | `[]` | |
| `api.workers` | `2` | Worker processes per API pod (`WEB_CONCURRENCY`). |
| `api.logLevel` | `info` | |
| `api.jobs.workers` | `2` | Threads in the background job pool (`JOB_WORKERS`). |
| `api.jobs.slots` | `1` | Background jobs running at once across all workers (`JOB_SLOTS`); each gets `cpu_count // slots` threads. |

### api.gpu

GPU training on the CUDA build of the API. Which driver each tag needs is in
[deploy/README.md](../../README.md#gpu). Needs the NVIDIA device plugin on the cluster.

| Key | Default | Meaning |
|---|---|---|
| `api.gpu.enabled` | `false` | Enabled, the pod runs `api.gpu.imageTag` instead of `api.image.tag`, limits `nvidia.com/gpu` to `count` (added to `api.resources.limits`) and sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. |
| `api.gpu.imageTag` | `""` | Required when enabled: the full tag of the CUDA build of the same version, such as `0.1.0-cuda13` or `0.1.0-cu12`. |
| `api.gpu.count` | `1` | GPUs the pod claims. |
| `api.gpu.nodeSelector` | `{}` | Added to `api.nodeSelector` (this wins on a shared key), only when enabled. |
| `api.gpu.tolerations` | `[]` | Added after `api.tolerations`, only when enabled. |

### api.tuning

Settings by their `.env.example` names; the defaults match `api/app/config.py`. Each key becomes an environment
variable of the same name, and any other `.env.example` setting can be added here.

| Key | Default |
|---|---|
| `api.tuning.PROFILE_SAMPLE_ROWS` | `500000` |
| `api.tuning.CORR_SAMPLE_ROWS` | `100000` |
| `api.tuning.CORR_MAX_LEVELS` | `50` |
| `api.tuning.COPY_BATCH_ROWS` | `100000` |
| `api.tuning.EXPORT_CHUNK_MB` | `256` |
| `api.tuning.EXPORT_FIT_MB` | `512` |
| `api.tuning.SAMPLE_MAX_CELLS` | `60000000` |
| `api.tuning.DB_POOL_MAX` | `10` |
| `api.tuning.SQL_TIMEOUT_MS` | `15000` |
| `api.tuning.SQL_MAX_ROWS` | `2000` |
| `api.tuning.SQL_POOL_MAX` | `4` |

[Configuration](../../../docs/configuration.md) says what each one does.

## ui

| Key | Default | Meaning |
|---|---|---|
| `ui.replicas` | `1` | |
| `ui.image.repository` | `registry.example.com/rainforest/ui` | Set it to your registry. |
| `ui.image.tag` | `""` | Empty: the chart's `appVersion`. |
| `ui.image.pullPolicy` | `IfNotPresent` | |
| `ui.resources` | `{}` | Pod resource requests and limits. |
| `ui.nodeSelector` | `{}` | |
| `ui.tolerations` | `[]` | |

## networkPolicy

| Key | Default | Meaning |
|---|---|---|
| `networkPolicy.enabled` | `true` | NetworkPolicies that admit only the `api` and `ui` pods to the in-chart MLflow (port 5000) and only the `api` and `mlflow` pods to the in-chart Postgres (port 5432). Each renders only when its component is in the release, and takes effect only on a CNI that enforces NetworkPolicy. |

## persistence

| Key | Default | Meaning |
|---|---|---|
| `persistence.keep` | `true` | Annotates the data and MLflow claims with `helm.sh/resource-policy: keep` and `argocd.argoproj.io/sync-options: Delete=false,Prune=false`, so `helm uninstall` and an Argo CD delete or prune leave them in place. `false`: they go with the release. The in-chart Postgres volume is kept either way. |
| `persistence.data.size` | `50Gi` | Datasets, exports and models; one ReadWriteOnce volume owned by the API pod, mounted at `/data`. |
| `persistence.data.storageClass` | `""` | Empty: the cluster's default class. |

## secrets

| Key | Default | Meaning |
|---|---|---|
| `secrets.existingSecret` | `""` | The name of a Secret you manage, in the release's namespace. Real installs use this, so no credential sits in a values file. Empty: the chart creates one from `secrets.values`, which is for dev and CI only. |
| `secrets.values.sessionSecret` | `""` | Used only when `existingSecret` is empty. Empty: the API generates a secret into the data volume. |
| `secrets.values.oidcClientSecret` | `""` | Required under `auth.mode: oidc` (when the chart creates the Secret). |
| `secrets.values.readonlyPassword` | `""` | Required (when the chart creates the Secret). The SQL console's role password. |
| `secrets.values.postgresPassword` | `""` | Required (when the chart creates the Secret). The database user's password. |
| `secrets.values.mlflowTrackingUsername` | `""` | Optional, external MLflow. |
| `secrets.values.mlflowTrackingPassword` | `""` | Optional, external MLflow. |
| `secrets.values.mlflowTrackingToken` | `""` | Optional, external MLflow. |

The keys of an `existingSecret` are the names under `secrets.values`: `sessionSecret`, `oidcClientSecret`,
`readonlyPassword`, `postgresPassword`, `mlflowTrackingUsername`, `mlflowTrackingPassword` and `mlflowTrackingToken`.
Nothing is generated, so every render of the chart is identical. The Postgres password reaches the API and MLflow as
`PGPASSWORD`, never inside a URL, so it may contain any character.

## ingress

| Key | Default | Meaning |
|---|---|---|
| `ingress.className` | `""` | Your controller's class. `nginx` also sets the ingress-nginx annotations large uploads need: no request-body limit, no request buffering, and 600-second read and send timeouts. For any other controller, set the equivalents with `ingress.annotations` or the controller's own settings, or uploads over its default size fail with `413`. |
| `ingress.annotations` | `{}` | Annotations on the Ingress, such as `cert-manager.io/cluster-issuer: letsencrypt`. One with the same name as an annotation the chart sets wins. |
| `ingress.tls.secretName` | `""` | The name of the TLS Secret. Empty: plain HTTP. Set, it also makes the OIDC redirect URL `https`. The chart doesn't create this Secret. |
