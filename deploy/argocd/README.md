# Rainforest deploy repository

This directory is the seed of a separate repository, `rainforest-deploy`: the Git repository Argo CD reads to run one
Rainforest release per team. It lives in the Rainforest repository so the layout and the chart change together; copy
the directory into a new repository of its own and that repository is the deploy repository.

```
applicationset.yaml        # one Argo Application per teams/<name>/, applied once to the Argo CD cluster
teams/
  vehicles/
    config.yaml            # the chart registry and the chart version this team runs
    values.yaml            # the team's release: the chart's values that differ from its defaults
  <next-team>/
```

The ApplicationSet uses the Git file generator over `teams/*/config.yaml`. Each team becomes an Application named
`rainforest-<name>` that installs the chart into the namespace `rainforest-<name>`, with the chart taken from the
project's Helm registry at the version in that team's `config.yaml` and the values from that team's `values.yaml`.
Adding a team is adding a directory; each team pins its own chart version.

Every hostname, registry URL, group and project number in these files is a placeholder (`example.com`).

## Seeding the repository

1. Copy this directory's contents into the root of a new Git repository and push it to the Git server. The
   repository's root must hold `applicationset.yaml` and `teams/`.
2. In `applicationset.yaml`, set both `repoURL` values (the generator's and the `ref: values` source's) to the new
   repository's URL, and `revision`/`targetRevision` if its branch is not `main`.
3. In each team's `config.yaml`, set `chart.repo` to the Helm registry of the Rainforest GitLab project:
   `https://<gitlab>/api/v4/projects/<project id>/packages/helm/stable`. This is an HTTP Helm repository, not an OCI
   one. The release pipeline uploads the chart there on every release tag, as chart version `<tag without v>`.
4. Give Argo CD access to both (next section), then apply the ApplicationSet to the cluster that runs Argo CD:

   ```bash
   kubectl apply -n argocd -f applicationset.yaml
   ```

The Applications use multi-source syntax (the chart from the registry, the values file from this repository through
`$values`), so the Argo CD version must support it (2.6 or later).

## How Argo CD authenticates

Argo CD needs read access to the Git repository and to the Helm registry. Both are registered as repository
credentials, which Argo stores as Secrets in its own namespace. In GitLab, a deploy token is enough for each:
`read_repository` on this repository, `read_package_registry` on the Rainforest project.

```bash
# the deploy repository
argocd repo add https://gitlab.example.com/platform/rainforest-deploy.git \
  --username <deploy-token-username> --password <deploy-token>

# the Rainforest project's Helm registry
argocd repo add https://gitlab.example.com/api/v4/projects/42/packages/helm/stable \
  --type helm --name rainforest \
  --username <deploy-token-username> --password <deploy-token>
```

The URL given to `--type helm` must be exactly the `chart.repo` in the `config.yaml` files.

The cluster pulling the images is a separate concern: when the image registry needs credentials, create a pull
Secret in each team's namespace and list it under `imagePullSecrets` in that team's `values.yaml`.

## Adding a team

1. Copy a team directory, `cp -r teams/vehicles teams/<name>`, and edit both files. In `config.yaml`, set the chart
   version. In `values.yaml`, set `hostname`, the identity provider and groups, the database, MLflow, resources and
   `secrets.existingSecret` (the convention is `rainforest-<name>`). When `api.gpu.enabled` is true, `api.gpu.imageTag`
   is the release tag as pushed plus `-cuda13` (or `-cu12`): `v0.1.0-cuda13` for the tag `v0.1.0`. It includes the `v`
   when the tag has one, whereas the chart version in `config.yaml` never does.
2. Create the namespace and the Secret named in `secrets.existingSecret`, before the first sync. The keys are the ones
   `NOTES.txt` lists for the team's settings; for the OIDC and external-MLflow team above:

   ```bash
   kubectl create namespace rainforest-<name>
   kubectl -n rainforest-<name> create secret generic rainforest-<name> \
     --from-literal=postgresPassword='...' \
     --from-literal=readonlyPassword='...' \
     --from-literal=oidcClientSecret='...'
   ```

   `sessionSecret` (cookie signing) and `mlflowTrackingUsername`, `mlflowTrackingPassword` and
   `mlflowTrackingToken` (external MLflow credentials) are optional keys. The Secret is created by hand or by the
   platform's secret tooling and never committed here; the values files hold no credentials.
3. Commit and push. The ApplicationSet creates the Application `rainforest-<name>` within a few minutes. Sync it in
   the Argo CD UI or with `argocd app sync rainforest-<name>`.

## Rolling a team forward

Change `chart.version` in that team's `config.yaml` (and `api.gpu.imageTag` in its `values.yaml`, which carries the
release tag as pushed, `v` included), commit, push and sync the team's Application. The chart's default image tags are its `appVersion`,
the release tag, so the images move with the chart. The other teams stay where they are, so the first team can run a
release before the rest.

To roll every team forward, make the same change in each team's `config.yaml` and `values.yaml` in one commit.
Review the diff Argo shows before syncing: a chart upgrade can include template changes as well as new images.

## Removing a team

Deleting `teams/<name>/` deletes the Application but not the workload: the template sets no resources finalizer, so
the release keeps running in its namespace. To remove the workload as well, delete the directory and push, then
delete the namespace (`kubectl delete namespace rainforest-<name>`) or the leftover objects by hand. The data and
MLflow volumes go with them (see below), so take the backups first. Deleting the Application with
`argocd app delete rainforest-<name> --cascade` also removes the resources, but the ApplicationSet recreates the
Application while the directory exists, so remove the directory first. The Secret created by hand and an external
database are not touched.

## Automated sync and pruning

The Applications sync by hand. `applicationset.yaml` carries the `automated` policy (`prune: true`, `selfHeal: true`)
commented out, and a team that wants commits applied without a click enables it there. Enable `prune` knowingly:
deleting an Application with its resources, or pruning the release's objects, deletes the data and MLflow volumes with
them, and the data behind a volume follows the storage class's reclaim policy. The Rainforest repository's
`deploy/README.md`, section "Uninstalling", lists which volumes survive. Take backups or set `Retain` on the volumes
before enabling it for a team that holds data.
