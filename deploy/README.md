# Running Rainforest on Kubernetes

How to install, configure and keep up a Rainforest release for one team, using the Helm chart in
[`chart/rainforest`](chart/rainforest). Every value is listed in the chart's [values reference](chart/rainforest/README.md).
The Compose setup for one machine is in [Deployment](../docs/deployment.md).

One release serves one team. Rainforest keeps no tenancy model inside an install: isolation between teams comes from a
namespace per release, and each release has its own database, its own console role and its own MLflow experiments.

What a release runs:

| Component | Kind | Notes |
|---|---|---|
| `api` | Deployment, one replica | Owns the `/data` volume (ReadWriteOnce). Background jobs run inside it, so it is never scaled out. An upgrade stops the old pod before it starts the new one. |
| `ui` | Deployment | nginx serving the app; proxies `/api/` to the API and, with in-chart MLflow, `/mlflow/` to MLflow. |
| `mlflow` | Deployment, one replica | Only with `mlflow.mode: in-chart`. Reached through the UI, behind the Rainforest sign-in. |
| `postgresql` | StatefulSet | Only with `postgresql.enabled: true`. |
| Ingress | Ingress | One host, `/` to the UI. |

Resource names start with the release name, plus `-rainforest` when the release name doesn't already contain
`rainforest`. The examples below use a release called `rainforest`, so the names read `rainforest-api`,
`rainforest-postgresql` and so on.

## Prerequisites

- Kubernetes 1.28 or newer, and Helm 3.
- An ingress controller. The chart writes a plain `networking.k8s.io/v1` Ingress and depends on no particular
  controller.
- A storage class that provides ReadWriteOnce volumes, as the cluster default or named in the `storageClass` values.
- For GPU training: nodes with NVIDIA GPUs and the
  [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) installed on the cluster.
- Container images for `api`, `ui` and `mlflow` in a registry the cluster can pull from. The chart's repositories
  default to the placeholders `registry.example.com/rainforest/{api,ui,mlflow}`; set them (see below). When the
  registry needs credentials, create a pull secret and list it in `imagePullSecrets`.

Neither Keycloak nor any secret tooling comes with the chart. Sign-in uses your own OIDC provider.

## First install

This installs Rainforest with the Postgres and MLflow that run inside the release. It is the quickest route to a
working install and suits a team that has no managed database. Use [an external Postgres](#external-postgres) for
anything the team can't afford to lose.

1. **Make the namespace and the Secret.** The chart reads credentials from a Secret you create, so no password sits
   in a values file:

   ```bash
   kubectl create namespace rainforest-vehicles

   kubectl -n rainforest-vehicles create secret generic rainforest-secrets \
     --from-literal=postgresPassword="$(openssl rand -hex 24)" \
     --from-literal=readonlyPassword="$(openssl rand -hex 24)" \
     --from-literal=oidcClientSecret='<the client secret from your identity provider>' \
     --from-literal=sessionSecret="$(openssl rand -hex 32)"
   ```

   | Key | Holds | Needed |
   |---|---|---|
   | `postgresPassword` | the database user's password | always |
   | `readonlyPassword` | the SQL console's read-only role's password | always |
   | `oidcClientSecret` | the OIDC client secret | with `auth.mode: oidc` |
   | `sessionSecret` | the key that signs session cookies | optional; the API generates one into the data volume when it is absent |
   | `mlflowTrackingUsername`, `mlflowTrackingPassword` | basic-auth credentials for an external MLflow server | optional, external MLflow only |
   | `mlflowTrackingToken` | a bearer token for an external MLflow server | optional, external MLflow only |

   The values file below names a TLS Secret, `vehicles-rainforest-tls`, which the chart does not create. Create it
   in the same namespace (`kubectl -n rainforest-vehicles create secret tls vehicles-rainforest-tls --cert=… --key=…`),
   or leave it to cert-manager by putting its annotation under `ingress.annotations` (for example
   `cert-manager.io/cluster-issuer: letsencrypt`); until the Secret exists the controller serves its own default
   certificate.

   The database password reaches the API and MLflow as `PGPASSWORD`, never inside a URL, so it may contain any
   character.

2. **Write a values file** for the team, `vehicles.yaml`:

   ```yaml
   hostname: vehicles.rainforest.example.com

   auth:
     mode: oidc
     oidc:
       issuer: https://sso.example.com/realms/rainforest
       leadGroup: rainforest-vehicles-lead
       memberGroup: rainforest-vehicles-member

   mlflow:
     experimentPrefix: vehicles
     image: {repository: registry.example.com/team/rainforest-mlflow}

   api:
     image: {repository: registry.example.com/team/rainforest-api}
   ui:
     image: {repository: registry.example.com/team/rainforest-ui}

   secrets:
     existingSecret: rainforest-secrets

   ingress:
     className: nginx
     tls:
       secretName: vehicles-rainforest-tls
   ```

   A complete example, with an external database and MLflow server and a GPU, is [`ci/team-values.yaml`](chart/rainforest/ci/team-values.yaml).

3. **Install:**

   ```bash
   helm upgrade --install rainforest deploy/chart/rainforest \
     --namespace rainforest-vehicles -f vehicles.yaml --wait --timeout 10m
   ```

   From a checkout that is the chart as it stands. To install a release instead, use the packaged chart from the
   project's Helm registry (see [Delivery](#delivery)): replace `deploy/chart/rainforest` with `rainforest/rainforest
   --version 1.2.3`.

   The images default to the chart's `appVersion`, so a chart version and the images it runs move together. Set
   `api.image.tag`, `ui.image.tag` and `mlflow.image.tag` to override that. The MLflow image's tag is Rainforest's
   version, not MLflow's.

   The notes printed after the install give the URL and, under OIDC, the redirect URL to register with your provider
   (`<URL>/api/auth/oidc/callback`; also set `<URL>/login` as a post-logout redirect URI). See
   [Registering with an OIDC provider](../docs/deployment.md#registering-with-an-oidc-provider) for what the client
   needs.

4. **Point DNS at the ingress controller** and open the URL. Under OIDC, a lead is someone in `auth.oidc.leadGroup`
   and a member someone in `auth.oidc.memberGroup`.

`hostname` is required: the chart refuses to render without it. It feeds the Ingress host and the API's
`RAINFOREST_ALLOWED_HOSTS`. The ingress controller answers a request for any other name itself, usually with a
`404`. The API answers `400` when the UI is reached under a host that isn't `hostname`, such as through a
`kubectl port-forward`: the page loads and its `/api/` calls fail.

### Trying it without a team

The defaults are a dev install: pick-mode sign-in, in-chart Postgres and MLflow, no TLS. Pick mode lists users to
choose from and does **not** authenticate; anyone who reaches the page can sign in as a lead. The install notes
say so. Use it on a throwaway cluster only. [`ci/dev-values.yaml`](chart/rainforest/ci/dev-values.yaml) is a working
file, with the passwords in `secrets.values` (the chart creates the Secret from them only when
`secrets.existingSecret` is empty, and never generates a password).

## Values that change per team

| Value | Set it to |
|---|---|
| `hostname` | the name people open Rainforest at |
| `api.image.repository`, `ui.image.repository`, `mlflow.image.repository` | your registry's paths; `imagePullSecrets` when it needs credentials |
| `postgresql.database`, `postgresql.user` (in-chart) or `postgresql.external.*` | the team's database and user; one database per release on a shared server |
| `auth.oidc.issuer`, `clientId`, `leadGroup`, `memberGroup` | your provider, the client registered there for this release (default `rainforest`; teams sharing one provider usually register one each), and this team's groups |
| `mlflow.experimentPrefix` | a name unique among the releases that share an MLflow server |
| `secrets.existingSecret` | the Secret you created for this release |
| `ingress.className`, `ingress.tls.secretName` | your controller's class and the TLS Secret (see [Uploads and the ingress controller](#uploads-and-the-ingress-controller)) |

`ingress.tls.secretName` names a Secret the chart does not create: make it yourself or have cert-manager issue it
(put its annotation under `ingress.annotations`). Setting it also makes the OIDC redirect URL `https`.

The sizes of the three volumes (`persistence.data.size`, `mlflow.persistence.size`, `postgresql.persistence.size`),
`api.resources` and `api.gpu` depend on the team's data. [Sizing](../docs/deployment.md#sizing) gives the figures: a
file takes roughly 2–3 times its size on disk, and one background job peaks at about 2 GB of RAM on top of a stack that
idles at about 2.3 GB. Volumes can't shrink, so start with room.

## MLflow

`mlflow.mode` is one of three.

| | `in-chart` (default) | `external` | `off` |
|---|---|---|---|
| Runs | an MLflow server in the release, with a volume for artifacts | nothing; you have a server | nothing |
| The API logs to | the server in the release | `mlflow.trackingUri`, or `mlflow.externalUrl` when that is empty | nowhere |
| *MLflow ↗* | opens `/mlflow/` on the Rainforest host, behind the Rainforest sign-in; a Viewer can browse and not change | opens `mlflow.externalUrl` in a new tab, with that server's own sign-in | hidden |
| Run links | `/mlflow/#/experiments/…` | `<externalUrl>/#/experiments/…` | none |

With an external server, Rainforest stores a URL and proxies nothing. Who can read and change that server's data is
its own matter; the Rainforest roles don't reach it.

```yaml
mlflow:
  mode: external
  externalUrl: https://mlflow.example.com               # what browsers open; required
  trackingUri: http://mlflow.infra.svc.cluster.local:5000   # what the API logs to; defaults to externalUrl
  experimentPrefix: vehicles
```

`trackingUri` is for when the API reaches the server over a different address than browsers do, often an internal
one without the single sign-on in front. When the server wants a credential, add `mlflowTrackingUsername` and
`mlflowTrackingPassword`, or `mlflowTrackingToken`, to the Secret. The MLflow client reads them natively; Rainforest
adds no authentication of its own.

**Experiment prefix.** Rainforest names an MLflow experiment `<prefix>/<dataset id>-<dataset name>` and finds it again
by dataset id **and** prefix, so releases that share a server and happen to share dataset ids never share an
experiment. Give each release its own `mlflow.experimentPrefix` (default `rainforest`). Changing the prefix on an
install that has runs starts a new experiment per dataset; the old experiments stay in MLflow under their old names.

With `in-chart`, MLflow keeps its runs in a database on the same Postgres server as Rainforest, named
`<postgresql database>_mlflow` unless `mlflow.database` says otherwise, and its artifacts on the `mlflow` volume.

## External Postgres

Set `postgresql.enabled: false` and describe the server:

```yaml
postgresql:
  enabled: false
  external:
    host: pg.example.com
    port: 5432
    database: rainforest_vehicles
    user: rainforest_vehicles
    managedRoles: false
```

Rainforest is tested against Postgres 16. The release needs its own database and a user that owns it, and the
password of that user goes in the Secret as `postgresPassword`. Each release on a shared server also needs its own
console role: the SQL console's `SELECT`-only role. It is named `<database>_ro` unless `postgresql.readonlyRole` says
otherwise, so releases never share a role.

**Roles the API creates itself (`managedRoles: true`).** The API creates the console role on every start, sets its
password from `readonlyPassword`, and grants it its read access. That needs `CREATEROLE` on the user. With in-chart
MLflow, the MLflow server also creates its database on start, which needs `CREATEDB`.

**Roles the operator creates (`managedRoles: false`).** On a shared server, where a DBA won't give a tenant
`CREATEROLE`, set `managedRoles: false` and have the server's administrator run this once per release (names from the
example above):

```sql
-- the release's user and database, if they don't exist yet
CREATE ROLE rainforest_vehicles LOGIN PASSWORD '…';           -- postgresPassword
CREATE DATABASE rainforest_vehicles OWNER rainforest_vehicles;

-- the SQL console's role: LOGIN and a password, nothing more
CREATE ROLE rainforest_vehicles_ro LOGIN PASSWORD '…';        -- readonlyPassword

-- only with in-chart MLflow
CREATE DATABASE rainforest_vehicles_mlflow OWNER rainforest_vehicles;
```

The API grants and revokes the console role's privileges itself on every start. It does not alter the role, and it
refuses to start while the role is missing, with a message naming it. [SQL](../docs/sql.md#when-the-api-cant-create-roles)
describes the rule. With in-chart MLflow, the MLflow pod likewise fails with a message while its database is
missing. The install notes repeat the database name.

With `managedRoles: false`, rotating `readonlyPassword` means running `ALTER ROLE … PASSWORD` yourself as well as
changing the Secret.

**The in-chart server.** `postgresql.enabled: true` runs `postgres:16-alpine` as a StatefulSet with the Compose
file's tuning (`postgresql.tuning`). The image reads `POSTGRES_PASSWORD` only when it first initialises an empty
volume. Changing `postgresPassword` later does not change it in the database: run `ALTER ROLE` there too, or the API can no
longer connect. When the chart created the Secret, changing it restarts the API and MLflow; with `existingSecret`,
restart them yourself (see [Upgrading](#upgrading)).

## GPU

XGBoost can train on an NVIDIA GPU. The GPU build of the API is a separate image per CUDA version, which CI publishes
next to the CPU image:

| Image tag | CUDA | Node driver |
|---|---|---|
| `<version>-cuda13` | 13 | 580 or newer (Blackwell cards, RTX 50xx, need the open kernel modules, `nvidia-open`) |
| `<version>-cu12` | 12.9 | 570 to 579 |

```yaml
api:
  gpu:
    enabled: true
    imageTag: 0.1.0-cuda13        # required when enabled; the same version as the other images
    count: 1                      # GPUs the pod claims
    nodeSelector: {nvidia.com/gpu.present: "true"}
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
```

With `api.gpu.enabled`, the pod runs `api.gpu.imageTag` instead of `api.image.tag`, limits `nvidia.com/gpu` to `count`
(added to any limits in `api.resources`), and sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. The node selector
and tolerations apply only then. The cluster needs the NVIDIA device plugin; nothing else in the release changes.

To check it, sign in and look for the CPU / GPU switch in the *Train a model* panel. `GET /api/system` reports the
GPU the API sees, or why XGBoost can't use it. How to choose the device for a run is in
[Experiments and training](../docs/experiments-and-training.md#gpu-training). The Compose side of the same choice
is in [GPU](../docs/deployment.md#gpu).

## Uploads and the ingress controller

Uploads stream through the controller to the API and can reach 20 GB, which is far above most controllers' default
request-body limit. With `ingress.className: nginx` the chart sets the ingress-nginx annotation
`nginx.ingress.kubernetes.io/proxy-body-size: "0"` for you. For any other class it sets nothing: raise the body limit
yourself, with the annotation or the controller-level setting your controller uses, and put the annotation under
`ingress.annotations`. Until you do, an upload larger than the controller's limit fails with `413`.

The controller must also set `X-Forwarded-Proto`, which the API reads to mark its session cookie `Secure`; every
common controller does.

## Upgrading

1. **Back up** (see [Backups](#backups)). The API applies new database migrations on start and has no way back, so a
   rollback of the chart doesn't undo them.
2. Change the chart version, or the image tags, and run the same `helm upgrade --install` command as at install.
3. The API pod is replaced (the old one stops first, because the data volume is ReadWriteOnce), so Rainforest is
   unavailable for a minute or so while the new pod runs its migrations. The API's startup probe allows five
   minutes for a large database.

Changing a value in the Secret rolls the API and MLflow, because the chart checksums it into the pod template, when
the chart created the Secret. When you manage the Secret yourself, restart the Deployments after changing it:
`kubectl -n rainforest-vehicles rollout restart deploy/rainforest-api deploy/rainforest-mlflow`.

## Backups

Three things hold a release's state.

| What | Where | Back up with |
|---|---|---|
| The databases: Rainforest's (`meta.*` and `raw.*`), and MLflow's with in-chart MLflow | Postgres | `pg_dump` of each |
| Uploads, Parquet copies, exports, trained models, profile pictures and a generated session secret | the `rainforest-data` volume, mounted at `/data` in the API pod | a volume snapshot, or a copy of `/data` |
| MLflow's run artifacts | the `rainforest-mlflow` volume, with in-chart MLflow | a volume snapshot, or a copy of `/mlartifacts` |

The database and `/data` must come from the same moment, or a dataset's rows and its files disagree. Take both while
nobody is uploading or training.

For the in-chart Postgres:

```bash
ns=rainforest-vehicles
kubectl -n $ns exec sts/rainforest-postgresql -- pg_dump -U rainforest -Fc rainforest > rainforest.dump
kubectl -n $ns exec sts/rainforest-postgresql -- pg_dump -U rainforest -Fc rainforest_mlflow > rainforest_mlflow.dump
```

An external server is backed up the way the rest of that server is. The `/data` volume can be copied through the API
pod:

```bash
kubectl -n $ns exec deploy/rainforest-api -- tar -C /data -cf - . > rainforest-data.tar
```

Prefer your storage's volume snapshots for large volumes. The configuration is the values file plus the Secret; keep
both where you keep other secrets.

## Uninstalling

`helm uninstall` removes what the chart installed, and what happens to the volumes depends on the volume:

| Volume | After `helm uninstall` |
|---|---|
| `rainforest-data` and `rainforest-mlflow` | **Deleted** with the release. The chart sets no `helm.sh/resource-policy: keep`. Whether the data behind them survives then follows the storage class's reclaim policy: `Delete` (the usual default for dynamically provisioned volumes) removes it, `Retain` keeps the volume for you to reclaim. |
| The in-chart Postgres volume (`data-rainforest-postgresql-0`) | **Kept.** StatefulSet volumes are not removed with the StatefulSet. A later install under the same release name and namespace reuses it, with the password it was first created with. |

An Argo CD Application that is deleted with its resources behaves the same way. So before removing a release you want
the data from, take the [backups](#backups), or check the storage class's reclaim policy and change a volume's with
`kubectl patch pv <name> -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'`. The Secret you created, the TLS
Secret and an external database are not touched.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The page loads but its calls fail with `400` | The UI was reached under a host that isn't `hostname` (a port-forward, say). Open the site at the name you set, or change `hostname`. |
| The ingress controller answers `404` | The request's host isn't `hostname`, or the Ingress's `className` doesn't match a controller. |
| An upload fails with `413` | The ingress controller's body limit. See [Uploads and the ingress controller](#uploads-and-the-ingress-controller). |
| The API pod restarts and its log names a missing role | `managedRoles: false` and the console role doesn't exist. Create it as above. |
| The MLflow pod restarts and its log names a missing database | `managedRoles: false` with in-chart MLflow and no `<database>_mlflow`. Create it as above. |
| A pod stays `Pending` | Its volume has no storage class or no capacity, or (GPU) no node offers `nvidia.com/gpu` or tolerates the taints. `kubectl describe pod` says which. |
| `helm lint` fails on `hostname` | The chart requires it. Lint with `helm lint deploy/chart/rainforest -f deploy/chart/rainforest/ci/dev-values.yaml`. |
| The API can't connect after changing `postgresPassword` | The in-chart Postgres keeps the password it was created with. Run `ALTER ROLE` in the database, and restart the API and MLflow if you manage the Secret yourself. |
| Sign-in reaches the provider and then fails | The redirect URL registered there differs from `<URL>/api/auth/oidc/callback`, in scheme too. |

Pick mode works in Kubernetes but doesn't authenticate; the install notes say so.

## Constraints

- One API replica. Background jobs run in the API process and share the `/data` volume.
- One host per release. Rainforest can't run under a URL path prefix such as `/rainforest/`.
- The chart ships no identity provider. The Compose reference Keycloak (`make up-keycloak`) is a development
  convenience; see [Reference Keycloak](../docs/deployment.md#reference-keycloak-development-only).

The UI image renders its nginx configuration at container start from `API_UPSTREAM`, `MLFLOW_UPSTREAM` and
`MLFLOW_ENABLED`, with the chart passing full service addresses (`<service>.<namespace>.svc.<clusterDomain>`).
Set `clusterDomain` if your cluster doesn't use `cluster.local`. The nginx resolver is the cluster DNS from the
container's `resolv.conf`.

## Smoke test

`deploy/smoke.sh` installs the chart on a throwaway [kind](https://kind.sigs.k8s.io/) cluster and drives a real
workflow through the UI service: it generates the 50k × 150 demo, sets up and finalizes a baseline, exports, trains
one LightGBM run and checks that the in-chart MLflow logged it. It prints `smoke: ok` and exits 0 only when all of
that worked. It needs `docker`, `kind`, `helm`, `kubectl`, `jq` and `curl`.

```bash
bash deploy/smoke.sh                                   # builds the three images, then runs
REGISTRY=registry.example.com/rainforest IMAGE_TAG=1.2.3 BUILD=0 bash deploy/smoke.sh   # uses images that exist
KEEP=1 bash deploy/smoke.sh                            # leaves the cluster up and says how to reach the UI
```

| Variable | Default | Meaning |
|---|---|---|
| `REGISTRY` | `rainforest` | image repository prefix: `<REGISTRY>/{api,ui,mlflow}:<IMAGE_TAG>` |
| `IMAGE_TAG` | `smoke` | tag of the three images |
| `BUILD` | `1` | `1` builds the images from `api/`, `ui/` and `mlflow/`; `0` uses images already present locally, pulling any that are missing |
| `KEEP` | `0` | `1` leaves the cluster running |
| `CLUSTER` | `rf-smoke` | kind cluster name |
| `LOCAL_PORT` | `18080` | local port the script forwards to the UI |
| `HELM_TIMEOUT`, `DATASET_TIMEOUT`, `EXPORT_TIMEOUT`, `RUN_TIMEOUT`, `FORWARD_TIMEOUT` | `10m`, 600 s, 600 s, 600 s, 60 s | the upper bound of each wait; a timeout prints the pods and their logs and fails |

The chart's own unit tests run with `bash deploy/chart/test.sh` (they need Docker).

## Delivery

`.gitlab-ci.yml` at the repository root has four stages. `.github/workflows/ci.yml` runs the same test commands, so
the two stay in step.

| Stage | Runs | When |
|---|---|---|
| `test` | `api-tests` (Postgres and MLflow as services; the API tests, then `mlflow/`), `ui-tests` (typecheck, Vitest, build, `ui/nginx/test.sh`), `lint-api` (ruff), `lint-ui` (ESLint, Prettier), `chart-lint` (`helm lint` and `helm template` with both `ci/` values files, `bash -n deploy/smoke.sh`), `chart-unittest` (helm-unittest from the same image as `deploy/chart/test.sh`) | merge requests, the default branch and tags |
| `build` | five images, built in Docker-in-Docker and pushed to the project's container registry: `api-cpu`, `api-cuda13` (`GPU=1`), `api-cu12` (`GPU=1`, `xgboost-cu12==3.4.1`), `ui`, `mlflow` | the default branch and release tags |
| `smoke` | `deploy/smoke.sh` on a `kind` cluster, against the pushed images | release tags |
| `chart` | `helm package` and an upload to the project's Helm registry, after `smoke` passes | release tags |

**What a push to the default branch produces:** images at `$CI_REGISTRY_IMAGE/{api,ui,mlflow}:<commit SHA>`, and the
two GPU builds as `api:<commit SHA>-cuda13` and `-cu12`.

**What a release tag produces:** the same images again, tagged with the release tag (`1.2.3`, or `v1.2.3`, with the
`-cuda13` and `-cu12` suffixes on the GPU builds); the smoke test; then the chart `rainforest-1.2.3.tgz`, whose `version` is the tag
without a leading `v` and whose `appVersion` is the tag as written. The chart's default image tag is its `appVersion`,
so the chart installs exactly the images that tag built. The tag must be a semantic version, or `helm package`
fails, so only semantic-version tags (`1.2.3`, `v1.2.3`, `1.2.3-rc1`) start a release; any other tag runs only the tests.
The smoke test is the release gate: the chart is uploaded only after it passes, so a failed release leaves its images
in the container registry but no chart in the Helm registry; fix forward with a new tag.

**Installing from the registry.** The chart is in the project's Helm package registry, in the `stable` channel. Create
a [deploy token](https://docs.gitlab.com/user/project/deploy_tokens/) with the `read_package_registry` scope, and a
second one with `read_registry` for the cluster to pull the images (a pull Secret listed in `imagePullSecrets`), then:

```bash
helm repo add rainforest https://gitlab.example.com/api/v4/projects/<project id>/packages/helm/stable \
  --username <deploy token user> --password <deploy token>
helm repo update
helm search repo rainforest --versions
helm upgrade --install rainforest rainforest/rainforest --version 1.2.3 \
  --namespace rainforest-vehicles -f vehicles.yaml --wait --timeout 10m
```

Set `api.image.repository`, `ui.image.repository` and `mlflow.image.repository` to
`registry.gitlab.example.com/<group>/<project>/{api,ui,mlflow}`. With `api.gpu.enabled`, `api.gpu.imageTag` is the
release tag plus `-cuda13` or `-cu12`, as in [GPU](#gpu).

**The smoke job.** It needs a runner with the Docker executor in privileged mode, for the `docker:27-dind` service.
`deploy/smoke.sh` runs with `BUILD=0`: it pulls the three images with the job's `docker login`, loads them into the
kind cluster, and installs the chart with `pullPolicy: Never`, so the cluster never talks to the private registry and
needs no `imagePullSecrets`. Because kind runs inside the Docker-in-Docker service, the job creates the cluster first,
with the API server listening on all interfaces and `docker` as a certificate name, and points the kubeconfig at
`docker`; `smoke.sh` reuses a cluster that exists. It installs the chart from the checked-out source, because it runs before the package exists.

To check the pipeline file without a GitLab project, [`gitlab-ci-local`](https://github.com/firecow/gitlab-ci-local)
runs the test jobs on a laptop (`gitlab-ci-local api-tests`, `chart-lint`, and so on). Everything that pushes
(`build`, `chart`) or needs the Docker-in-Docker service (`smoke`) only runs on a real runner.

`deploy/argocd/` holds the Argo CD layout: one Argo Application per team, generated from the `teams/<name>/`
directory that holds that team's chart version and values file. Adding a team is adding a directory, and each team
pins its own chart version. The files themselves are the reference for the pipeline and for the layout.
