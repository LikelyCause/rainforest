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

{{/* Values.yaml integers arrive as float64, which Go prints as 1e+06; render whole numbers as integers. */}}
{{- define "rainforest.num" -}}
{{- if kindIs "float64" . }}{{ . | int64 }}{{ else }}{{ . }}{{ end }}
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

{{/* Database coordinates, shared by the API and (later) the in-chart MLflow server. */}}
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

{{/* The SQL console's SELECT-only role; each release on a shared server needs its own. */}}
{{- define "rainforest.readonlyRole" -}}
{{- default (printf "%s_ro" (include "rainforest.databaseName" .)) .Values.postgresql.readonlyRole }}
{{- end }}

{{/* The in-chart server is always managed; an external one follows managedRoles. */}}
{{- define "rainforest.managedRoles" -}}
{{- if .Values.postgresql.enabled }}true{{ else }}{{ toString .Values.postgresql.external.managedRoles }}{{ end }}
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

{{- define "rainforest.scheme" -}}
{{- if .Values.ingress.tls.secretName }}https{{ else }}http{{ end }}
{{- end }}
