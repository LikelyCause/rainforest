{{/* Names and labels, as `helm create` writes them. */}}
{{- define "rainforest.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "rainforest.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "rainforest.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "rainforest.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rainforest.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/* Usage: include "rainforest.labels" (dict "root" . "component" "api") */}}
{{- define "rainforest.labels" -}}
helm.sh/chart: {{ include "rainforest.chart" .root }}
{{ include "rainforest.selectorLabels" . }}
app.kubernetes.io/version: {{ .root.Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .root.Release.Service }}
{{- end }}

{{/* Usage: include "rainforest.image" (dict "image" .Values.api.image "root" .) */}}
{{- define "rainforest.image" -}}
{{- printf "%s:%s" .image.repository (default .root.Chart.AppVersion .image.tag) }}
{{- end }}

{{/* Values.yaml numbers arrive as float64, which Go prints as 1e+06; render whole numbers as integers, others as is. */}}
{{- define "rainforest.num" -}}
{{- if and (kindIs "float64" .) (eq (. | int64 | float64) .) }}{{ . | int64 }}{{ else }}{{ . }}{{ end }}
{{- end }}

{{/* Services are addressed by FQDN: the UI's nginx resolver ignores search domains. */}}
{{- define "rainforest.serviceFqdn" -}}
{{- printf "%s-%s.%s.svc.%s" (include "rainforest.fullname" .root) .component .root.Release.Namespace .root.Values.clusterDomain }}
{{- end }}

{{- define "rainforest.apiUpstream" -}}
{{- printf "http://%s:8000" (include "rainforest.serviceFqdn" (dict "root" . "component" "api")) }}
{{- end }}

{{- define "rainforest.mlflowUpstream" -}}
{{- printf "http://%s:5000" (include "rainforest.serviceFqdn" (dict "root" . "component" "mlflow")) }}
{{- end }}

{{/* The Secret holding credentials: the operator's, or the one this chart creates. */}}
{{- define "rainforest.secretName" -}}
{{- default (include "rainforest.fullname" .) .Values.secrets.existingSecret }}
{{- end }}

{{/* Database coordinates, shared by the API and the in-chart MLflow server. */}}
{{- define "rainforest.databaseHost" -}}
{{- if .Values.postgresql.enabled }}{{ include "rainforest.fullname" . }}-postgresql{{ else }}{{ required "postgresql.external.host is required when postgresql.enabled is false" .Values.postgresql.external.host }}{{ end }}
{{- end }}

{{- define "rainforest.databasePort" -}}
{{- if .Values.postgresql.enabled }}5432{{ else }}{{ include "rainforest.num" .Values.postgresql.external.port }}{{ end }}
{{- end }}

{{- define "rainforest.databaseName" -}}
{{- if .Values.postgresql.enabled }}{{ .Values.postgresql.database }}{{ else }}{{ .Values.postgresql.external.database }}{{ end }}
{{- end }}

{{- define "rainforest.databaseUser" -}}
{{- if .Values.postgresql.enabled }}{{ .Values.postgresql.user }}{{ else }}{{ .Values.postgresql.external.user }}{{ end }}
{{- end }}

{{/*
Connection URI without the password, which reaches libpq clients as PGPASSWORD instead.
Usage: include "rainforest.postgresUri" (dict "root" . "database" (include "rainforest.databaseName" .))
*/}}
{{- define "rainforest.postgresUri" -}}
{{- printf "postgresql://%s@%s:%s/%s" (include "rainforest.databaseUser" .root) (include "rainforest.databaseHost" .root) (include "rainforest.databasePort" .root) .database }}
{{- end }}

{{/* The SQL console's SELECT-only role; each release on a shared server needs its own. */}}
{{- define "rainforest.readonlyRole" -}}
{{- default (printf "%s_ro" (include "rainforest.databaseName" .)) .Values.postgresql.readonlyRole }}
{{- end }}

{{/* The in-chart server is always managed; an external one follows managedRoles. */}}
{{- define "rainforest.managedRoles" -}}
{{- if .Values.postgresql.enabled }}true{{ else }}{{ toString .Values.postgresql.external.managedRoles }}{{ end }}
{{- end }}

{{/* The in-chart MLflow server's backend database. */}}
{{- define "rainforest.mlflowDatabase" -}}
{{- default (printf "%s_mlflow" (include "rainforest.databaseName" .)) .Values.mlflow.database }}
{{- end }}

{{/*
Host headers MLflow accepts (it compares them exactly): "mlflow" is what the UI's nginx sends, the FQDN with the port
is what the API's client sends, and the short Service name covers anything else in the namespace.
*/}}
{{- define "rainforest.mlflowAllowedHosts" -}}
{{- $short := printf "%s-mlflow" (include "rainforest.fullname" .) }}
{{- $fqdn := include "rainforest.serviceFqdn" (dict "root" . "component" "mlflow") }}
{{- join "," (list "mlflow" $short (printf "%s:5000" $short) $fqdn (printf "%s:5000" $fqdn)) }}
{{- end }}

{{/* Where the API logs experiments. Empty turns logging off. */}}
{{- define "rainforest.mlflowTrackingUri" -}}
{{- if eq .Values.mlflow.mode "in-chart" }}{{ include "rainforest.mlflowUpstream" . }}/mlflow
{{- else if eq .Values.mlflow.mode "external" }}{{ default (include "rainforest.mlflowExternalUrl" .) .Values.mlflow.trackingUri }}
{{- end }}
{{- end }}

{{- define "rainforest.mlflowExternalUrl" -}}
{{- required "mlflow.externalUrl is required when mlflow.mode is external" .Values.mlflow.externalUrl }}
{{- end }}

{{/* Where browsers open MLflow. Empty is the same-origin /mlflow/ the UI proxies. */}}
{{- define "rainforest.mlflowBrowseUrl" -}}
{{- if eq .Values.mlflow.mode "external" }}{{ include "rainforest.mlflowExternalUrl" . }}{{ end }}
{{- end }}

{{/* The scheme people open Rainforest with: publicScheme, else https exactly when the Ingress terminates TLS. */}}
{{- define "rainforest.scheme" -}}
{{- if .Values.publicScheme }}{{ .Values.publicScheme }}{{ else if .Values.ingress.tls.secretName }}https{{ else }}http{{ end }}
{{- end }}

{{/* Each release on a shared MLflow server needs its own prefix, and the namespace already differs per release. */}}
{{- define "rainforest.experimentPrefix" -}}
{{- default .Release.Namespace .Values.mlflow.experimentPrefix }}
{{- end }}

{{/*
The ConfigMap's keys the chart sets from other values. api.tuning may not repeat one, nor a name in
rainforest.podEnvNames (pod env wins over envFrom, so a tuning key there would be ignored without a word).
*/}}
{{- define "rainforest.configData" -}}
RAINFOREST_AUTH_MODE: {{ .Values.auth.mode | quote }}
RAINFOREST_USERS: {{ .Values.auth.users | quote }}
RAINFOREST_OIDC_ISSUER: {{ .Values.auth.oidc.issuer | quote }}
RAINFOREST_OIDC_DISCOVERY_URL: {{ .Values.auth.oidc.discoveryUrl | quote }}
RAINFOREST_OIDC_CLIENT_ID: {{ .Values.auth.oidc.clientId | quote }}
RAINFOREST_OIDC_REDIRECT_URL: {{ printf "%s://%s/api/auth/oidc/callback" (include "rainforest.scheme" .) .Values.hostname | quote }}
RAINFOREST_OIDC_SCOPES: {{ .Values.auth.oidc.scopes | quote }}
RAINFOREST_OIDC_GROUPS_CLAIM: {{ .Values.auth.oidc.groupsClaim | quote }}
RAINFOREST_OIDC_PROVIDER_NAME: {{ .Values.auth.oidc.providerName | quote }}
RAINFOREST_LEAD_GROUP: {{ .Values.auth.oidc.leadGroup | quote }}
RAINFOREST_MEMBER_GROUP: {{ .Values.auth.oidc.memberGroup | quote }}
RAINFOREST_ALLOWED_HOSTS: {{ .Values.hostname | quote }}
RAINFOREST_MANAGED_ROLES: {{ include "rainforest.managedRoles" . | quote }}
READONLY_ROLE: {{ include "rainforest.readonlyRole" . | quote }}
WEB_CONCURRENCY: {{ include "rainforest.num" .Values.api.workers | quote }}
JOB_WORKERS: {{ include "rainforest.num" .Values.api.jobs.workers | quote }}
JOB_SLOTS: {{ include "rainforest.num" .Values.api.jobs.slots | quote }}
LOG_LEVEL: {{ .Values.api.logLevel | quote }}
MLFLOW_TRACKING_URI: {{ include "rainforest.mlflowTrackingUri" . | quote }}
MLFLOW_BROWSE_URL: {{ include "rainforest.mlflowBrowseUrl" . | quote }}
MLFLOW_EXPERIMENT_PREFIX: {{ include "rainforest.experimentPrefix" . | quote }}
{{- end }}

{{/*
The variables api-deployment.yaml sets under env:, space-separated: the fixed ones, RAINFOREST_MONGO_SOURCES and one
MONGO_URI_<NAME> per source.
*/}}
{{- define "rainforest.podEnvNames" -}}
PGPASSWORD DATABASE_URL READONLY_PASSWORD DATA_DIR RAINFOREST_SESSION_SECRET NVIDIA_DRIVER_CAPABILITIES RAINFOREST_OIDC_CLIENT_SECRET MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_TOKEN RAINFOREST_MONGO_SOURCES
{{- range .Values.sources.mongodb }} {{ include "rainforest.mongoUriVar" . }}{{ end }}
{{- end }}

{{/* The environment variable that holds a source's URI; fails the render for a name the API would not accept. */}}
{{- define "rainforest.mongoUriVar" -}}
{{- $name := toString (default "" .name) }}
{{- if not (regexMatch "^[a-z0-9-]+$" $name) }}
{{- fail (printf "sources.mongodb: the name %q is not a slug; use lowercase letters, digits and dashes" $name) }}
{{- end }}
{{- printf "MONGO_URI_%s" (upper (replace "-" "_" $name)) }}
{{- end }}

{{/*
The api container's env entries for the MongoDB sources. Kubernetes expands $(VAR) in a later entry's value from the
earlier ones, so RAINFOREST_MONGO_SOURCES composes the URIs without them landing in the ConfigMap or the manifest.
*/}}
{{- define "rainforest.mongoEnv" -}}
{{- $seen := dict }}
{{- $parts := list }}
{{- range .Values.sources.mongodb }}
{{- $var := include "rainforest.mongoUriVar" . }}
{{- if hasKey $seen .name }}
{{- fail (printf "sources.mongodb: the name %q appears twice" .name) }}
{{- end }}
{{- $_ := set $seen .name true }}
{{- if not .existingSecret }}
{{- fail (printf "sources.mongodb: %q needs existingSecret, the Secret holding its URI" .name) }}
{{- end }}
{{- $parts = append $parts (printf "%s=$(%s)" .name $var) }}
- name: {{ $var }}
  valueFrom:
    secretKeyRef:
      name: {{ .existingSecret | quote }}
      key: {{ default "uri" .key | quote }}
{{- end }}
{{- if $parts }}
- name: RAINFOREST_MONGO_SOURCES
  value: {{ join ";" $parts | quote }}
{{- end }}
{{- end }}

{{/*
The API image: under api.gpu.enabled, api.gpu.imageTag as given, else <api.image.tag or appVersion>-<api.gpu.cuda>.
*/}}
{{- define "rainforest.apiImageTag" -}}
{{- $base := default .Chart.AppVersion .Values.api.image.tag }}
{{- if not .Values.api.gpu.enabled }}{{ $base }}
{{- else }}
{{- $cuda := .Values.api.gpu.cuda }}
{{- if and $cuda (not (has $cuda (list "cuda13" "cu12"))) }}
{{- fail (printf "api.gpu.cuda is %q: use cuda13 or cu12" $cuda) }}
{{- end }}
{{- if .Values.api.gpu.imageTag }}{{ .Values.api.gpu.imageTag }}
{{- else if $cuda }}{{ printf "%s-%s" $base $cuda }}
{{- else }}{{ fail "api.gpu.cuda is required when api.gpu.enabled is true: cuda13 or cu12 (deploy/README.md, GPU, says which driver each needs)" }}
{{- end }}
{{- end }}
{{- end }}

{{/* On the data and MLflow claims: helm uninstall and an Argo CD delete or prune leave them in place. */}}
{{- define "rainforest.keepAnnotations" -}}
helm.sh/resource-policy: keep
argocd.argoproj.io/sync-options: Delete=false,Prune=false
{{- end }}
