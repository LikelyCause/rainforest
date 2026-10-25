#!/usr/bin/env bash
# Installs the Helm chart on a throwaway kind cluster and drives a real workflow through the UI service:
# generate the 50k x 150 demo, set up and finalize a baseline, export, train one LightGBM run, and check that the
# in-chart MLflow logged it. Exits 0 and prints "smoke: ok" only when all of that worked.
#
# Needs: docker, kind, helm, kubectl, jq, curl.
#
#   bash deploy/smoke.sh
#
# Environment:
#   REGISTRY      image repository prefix: <REGISTRY>/{api,ui,mlflow}:<IMAGE_TAG>   (default: rainforest)
#   IMAGE_TAG     tag of the three images                                           (default: smoke)
#   BUILD         1: build the images from api/, ui/ and mlflow/ (default). 0: use images already present
#                 locally, pulling any that are missing (CI passes the images it just pushed).
#   KEEP          1: leave the cluster running and print how to reach the UI.
#   CLUSTER       kind cluster name                                                 (default: rf-smoke)
#   LOCAL_PORT    port-forward port on localhost                                    (default: 18080)
#   HELM_TIMEOUT (10m), DATASET_TIMEOUT, EXPORT_TIMEOUT, RUN_TIMEOUT (600 s each), FORWARD_TIMEOUT (60 s)
#                 upper bounds for each wait; on a timeout the script prints the pods and their logs and fails.
#
# The Postgres passwords contain URI-reserved characters on purpose: the chart hands the password to libpq as
# PGPASSWORD, so neither the API nor MLflow may ever need it URL-encoded.
set -Eeuo pipefail

REGISTRY=${REGISTRY:-rainforest}
IMAGE_TAG=${IMAGE_TAG:-smoke}
BUILD=${BUILD:-1}
KEEP=${KEEP:-0}
CLUSTER=${CLUSTER:-rf-smoke}
LOCAL_PORT=${LOCAL_PORT:-18080}

RELEASE=rainforest
NAMESPACE=default
# "localhost" is always in the API's allowed hosts, so the port-forward needs no Host override.
HOSTNAME_VALUE=localhost
PG_PASSWORD='smoke/pass#1?x%y'
RO_PASSWORD='ro/pass#2?z%w'

HELM_TIMEOUT=${HELM_TIMEOUT:-10m}
DATASET_TIMEOUT=${DATASET_TIMEOUT:-600}
EXPORT_TIMEOUT=${EXPORT_TIMEOUT:-600}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
FORWARD_TIMEOUT=${FORWARD_TIMEOUT:-60}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHART=$ROOT/deploy/chart/rainforest
CTX=kind-$CLUSTER
WORK=$(mktemp -d)
COOKIES=$WORK/cookies
BASE=http://localhost:$LOCAL_PORT
FORWARD_PID=
CREATED=0   # 1 once this run has created the kind cluster; only then does cleanup delete it

log() { printf 'smoke: [%4ss] %s\n' "$SECONDS" "$*"; }

k() { kubectl --context "$CTX" -n "$NAMESPACE" "$@"; }

# ---------------------------------------------------------------- diagnostics and cleanup

# What the pods are doing and why, for a failed phase.
diagnose() {
  log "--- diagnostics ---"
  k get pods -o wide 2>&1 || true
  local pod
  for pod in $(k get pods -o name 2>/dev/null); do
    log "--- describe $pod (events) ---"
    k describe "$pod" 2>&1 | sed -n '/^Events:/,$p' | tail -15 || true
    log "--- logs $pod ---"
    k logs "$pod" --all-containers --tail=80 2>&1 || true
  done
}

# Diagnostics print once, even when fail runs in a subshell and the parent then trips the ERR trap too.
fail() {
  log "FAILED: $*" >&2
  if [ ! -e "$WORK/diagnosed" ]; then
    : >"$WORK/diagnosed"
    diagnose >&2
  fi
  exit 1
}

cleanup() {
  local code=$?
  if [ -n "$FORWARD_PID" ]; then
    kill "$FORWARD_PID" 2>/dev/null || true
    wait "$FORWARD_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
  if [ "$CREATED" != 1 ]; then
    log "cluster $CLUSTER was not created by this run; it is left as it is"
  elif [ "$KEEP" = 1 ]; then
    log "cluster $CLUSTER kept. Reach the UI with:"
    echo "  kubectl --context $CTX port-forward svc/$RELEASE-ui $LOCAL_PORT:80   # then open $BASE"
    echo "  kind delete cluster --name $CLUSTER   # when done"
  else
    kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true
  fi
  if [ "$code" -eq 0 ]; then echo "smoke: ok"; else log "failed (exit $code)" >&2; fi
}

# An unexpected error inside any phase lands here with the failing line.
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR
trap cleanup EXIT

# ---------------------------------------------------------------- helpers

# api METHOD PATH [JSON-BODY [EXPECTED-STATUS]]: prints the response body. Fails on any status of 400 or more and,
# when EXPECTED-STATUS is given (the one docs/api.md names), on any other status; the message shows status and body.
api() {
  local method=$1 path=$2 body=${3:-} expected=${4:-} out status
  local args=(-sS -o "$WORK/body" -w '%{http_code}' -X "$method" -b "$COOKIES" -c "$COOKIES" "$BASE$path")
  [ -n "$body" ] && args+=(-H 'Content-Type: application/json' --data "$body")
  status=$(curl --max-time 60 "${args[@]}") || fail "curl $method $path did not complete"
  out=$(cat "$WORK/body")
  if [ "$status" -ge 400 ] || { [ -n "$expected" ] && [ "$status" != "$expected" ]; }; then
    fail "$method $path -> HTTP $status${expected:+ (expected $expected)}: $out"
  fi
  printf '%s' "$out"
}

# poll LABEL TIMEOUT-SECONDS PATH JQ-STATUS-EXPR: polls a row until its status is done/ready; failed ends it at once.
poll() {
  local label=$1 timeout=$2 path=$3 want=$4 deadline=$((SECONDS + $2)) row status
  while :; do
    row=$(api GET "$path")
    status=$(jq -r '.status' <<<"$row")
    [ "$status" = "$want" ] && return 0
    [ "$status" = failed ] && fail "$label failed: $(jq -c '{status, status_message, error}' <<<"$row")"
    [ "$SECONDS" -ge "$deadline" ] && fail "$label still '$status' after ${timeout}s: $(jq -c '{progress, status_message}' <<<"$row")"
    sleep 5
  done
}

# ---------------------------------------------------------------- phases

check_tools() {
  local t
  for t in docker kind helm kubectl jq curl git; do
    command -v "$t" >/dev/null || fail "$t is required but not installed"
  done
}

create_cluster() {
  if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
    log "cluster $CLUSTER already exists: reusing it and keeping it when the run ends"
    # A leftover release would carry over old volumes (and an old Postgres password); start clean.
    helm --kube-context "$CTX" -n "$NAMESPACE" uninstall "$RELEASE" --wait >/dev/null 2>&1 || true
    k delete pvc --all --wait >/dev/null 2>&1 || true
  else
    log "creating kind cluster $CLUSTER"
    kind create cluster --name "$CLUSTER" --wait 120s
    CREATED=1
  fi
}

# kind load docker-image imports every platform in the image's manifest list, which fails (content digest not found)
# when Docker's containerd image store holds only the host's platform of a multi-arch image such as postgres. Saving
# just that platform and loading the archive works.
load_image() {
  local ref=$1 platform tar
  kind load docker-image "$ref" --name "$CLUSTER" && return 0
  log "kind load docker-image failed for $ref, retrying with a single-platform archive"
  platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$ref")
  tar=$WORK/image.tar
  docker save --platform "$platform" -o "$tar" "$ref"
  kind load image-archive "$tar" --name "$CLUSTER"
  rm -f "$tar"
}

prepare_images() {
  local c ref
  for c in api ui mlflow; do
    ref=$REGISTRY/$c:$IMAGE_TAG
    if [ "$BUILD" = 1 ]; then
      log "building $ref"
      if [ "$c" = api ]; then
        docker build --build-arg "GIT_SHA=$(git -C "$ROOT" rev-parse HEAD)" -t "$ref" "$ROOT/$c"
      else
        docker build -t "$ref" "$ROOT/$c"
      fi
    elif ! docker image inspect "$ref" >/dev/null 2>&1; then
      log "pulling $ref"
      docker pull "$ref"
    fi
    load_image "$ref"
  done
  # The chart's Postgres image; loaded so the cluster never needs the internet for it.
  if docker image inspect postgres:16-alpine >/dev/null 2>&1; then
    load_image postgres:16-alpine
  fi
}

install_chart() {
  log "helm install (timeout $HELM_TIMEOUT)"
  local set=(
    --set "hostname=$HOSTNAME_VALUE"
    --set-string "secrets.values.postgresPassword=$PG_PASSWORD"
    --set-string "secrets.values.readonlyPassword=$RO_PASSWORD"
  )
  local c
  for c in api ui mlflow; do
    set+=(--set-string "$c.image.repository=$REGISTRY/$c" --set-string "$c.image.tag=$IMAGE_TAG" --set "$c.image.pullPolicy=Never")
  done
  # No ingress controller in kind; everything goes through the ui Service.
  helm --kube-context "$CTX" -n "$NAMESPACE" upgrade --install "$RELEASE" "$CHART" \
    -f "$CHART/ci/dev-values.yaml" "${set[@]}" --wait --timeout "$HELM_TIMEOUT" \
    || fail "helm install did not become ready"
  k get pods
}

port_forward() {
  local svc
  svc=$(k get svc -o name | grep -- '-ui$') || fail "no ui Service found"
  log "port-forward $svc -> localhost:$LOCAL_PORT"
  k port-forward "$svc" "$LOCAL_PORT:80" >"$WORK/forward.log" 2>&1 &
  FORWARD_PID=$!
  local deadline=$((SECONDS + FORWARD_TIMEOUT))
  # The UI's nginx serves its own pages at / and proxies /api/ only, so /api/auth/config (open, no session) is what
  # proves the path ui -> api works.
  until curl -fsS --max-time 5 "$BASE/api/auth/config" >"$WORK/config" 2>/dev/null; do
    kill -0 "$FORWARD_PID" 2>/dev/null || { cat "$WORK/forward.log" >&2; fail "port-forward exited"; }
    [ "$SECONDS" -ge "$deadline" ] && { cat "$WORK/forward.log" >&2; fail "the API is not reachable through the port-forward after ${FORWARD_TIMEOUT}s"; }
    sleep 1
  done
  jq -e '.mode == "pick"' "$WORK/config" >/dev/null || fail "unexpected /api/auth/config: $(cat "$WORK/config")"
  log "ui -> api: $(jq -c . "$WORK/config")"
}

# The API's own /health (it runs SELECT 1), reached through the Kubernetes API server's Service proxy, and the
# Postgres password as the pod sees it: byte for byte what was passed in, reserved characters and all.
check_api_health() {
  local api_svc pod_pw
  api_svc=$(k get svc -o name | grep -- '-api$' | cut -d/ -f2)
  k get --raw "/api/v1/namespaces/$NAMESPACE/services/$api_svc:8000/proxy/health" | jq -e . >/dev/null \
    || fail "the API's /health did not answer with JSON"
  pod_pw=$(k exec "deploy/$api_svc" -- printenv PGPASSWORD)
  [ "$pod_pw" = "$PG_PASSWORD" ] || fail "PGPASSWORD in the API pod differs from the password given to Helm"
  log "api /health ok; PGPASSWORD reached the pod verbatim"
}

sign_in() {
  local id
  id=$(api GET /api/auth/users | jq -r '.[] | select(.name == "Admin") | .id')
  [ -n "$id" ] || fail "no user named Admin in the sign-in list"
  ADMIN_ID=$id
  api POST /api/auth/pick "{\"user_id\": $ADMIN_ID}" | jq -e '.role == "lead"' >/dev/null || fail "Admin is not a lead"
  log "signed in as Admin (id $ADMIN_ID)"
}

# The SQL console connects as the read-only role, so this proves READONLY_PASSWORD reached the API intact.
check_readonly_console() {
  api POST /api/sql '{"sql": "SELECT 1 AS ok"}' | jq -e '.rows[0][0] == 1 or .rows[0].ok == 1' >/dev/null \
    || fail "the SQL console did not answer"
  log "SQL console (read-only role) answers"
}

generate_dataset() {
  log "generating the 50k x 150 demo (timeout ${DATASET_TIMEOUT}s)"
  DATASET_ID=$(api POST /api/datasets/sample '{"rows": 50000, "fields": 150, "name": "smoke demo"}' 202 | jq -r '.id')
  [ -n "$DATASET_ID" ] && [ "$DATASET_ID" != null ] || fail "sample generation returned no dataset id"
  poll "dataset $DATASET_ID" "$DATASET_TIMEOUT" "/api/datasets/$DATASET_ID" ready
  log "dataset $DATASET_ID is ready"
}

finalize_baseline() {
  local targets fields
  targets=$(api GET "/api/datasets/$DATASET_ID" | jq -c '.target_columns // []')
  [ "$targets" != "[]" ] || fail "dataset $DATASET_ID has no target_columns; the generated sample should set them"
  log "setting up the Trees baseline, targets $targets, Admin as the only assessor"
  api POST "/api/datasets/$DATASET_ID/baselines" \
    "{\"family\": \"tree\", \"start\": \"scratch\", \"targets\": $targets, \"assessors\": [$ADMIN_ID]}" >/dev/null

  fields=$(api GET "/api/datasets/$DATASET_ID/fields" | jq -c '[.[].id]')
  api POST "/api/datasets/$DATASET_ID/decisions/bulk" \
    "{\"field_ids\": $fields, \"accept_suggestion\": true, \"reviewed\": true}" \
    | jq -e '.matched > 0' >/dev/null || fail "bulk accept matched no fields"

  api POST "/api/datasets/$DATASET_ID/assessment/submit" >/dev/null
  api POST "/api/datasets/$DATASET_ID/finalize" '{"accept_open": true}' >/dev/null
  RECIPE_ID=$(api GET "/api/datasets/$DATASET_ID/team" | jq -r 'select(.recipe.phase == "finalized") | .recipe.id')
  [ -n "$RECIPE_ID" ] || fail "the baseline is not finalized"
  log "baseline $RECIPE_ID finalized"
}

export_and_train() {
  EXPERIMENT_ID=$(api POST "/api/datasets/$DATASET_ID/experiments" \
    "{\"name\": \"smoke\", \"source\": \"recipe:$RECIPE_ID\"}" 201 | jq -r '.id')
  log "experiment $EXPERIMENT_ID; exporting (timeout ${EXPORT_TIMEOUT}s)"
  EXPORT_ID=$(api POST "/api/datasets/$DATASET_ID/exports" "{\"experiment_id\": $EXPERIMENT_ID}" 202 | jq -r '.id')
  poll "export $EXPORT_ID" "$EXPORT_TIMEOUT" "/api/exports/$EXPORT_ID" done

  log "export $EXPORT_ID done; training lightgbm (timeout ${RUN_TIMEOUT}s)"
  RUN_ID=$(api POST "/api/exports/$EXPORT_ID/runs" '{"model": "lightgbm", "cv_folds": 0}' 202 | jq -r '.id')
  poll "run $RUN_ID" "$RUN_TIMEOUT" "/api/runs/$RUN_ID" done
  log "run $RUN_ID done"
}

check_mlflow() {
  api GET /api/system | jq -e '.mlflow.enabled == true' >/dev/null || fail "/api/system reports MLflow disabled"
  local url
  url=$(api GET "/api/runs/$RUN_ID" | jq -r '.mlflow_url // empty')
  [ -n "$url" ] || fail "run $RUN_ID has no mlflow_url (report: $(api GET "/api/runs/$RUN_ID" | jq -c '.report.mlflow_error'))"
  log "run logged to MLflow: $url"

  # /mlflow/ is proxied by the UI behind the API's sign-in check: no session, no MLflow.
  local code location
  read -r code location < <(curl -sS --max-time 10 -o /dev/null -w '%{http_code} %{redirect_url}\n' "$BASE/mlflow/")
  case $code in
    301 | 302 | 303 | 307 | 308) ;;
    *) fail "GET /mlflow/ without a session returned HTTP $code, expected a redirect to /login" ;;
  esac
  case $location in
    */login*) ;;
    *) fail "GET /mlflow/ redirected to '$location', expected /login" ;;
  esac
  log "/mlflow/ without a session redirects to $location"
}

# ---------------------------------------------------------------- main

main() {
  check_tools
  create_cluster
  prepare_images
  install_chart
  port_forward
  check_api_health
  sign_in
  check_readonly_console
  generate_dataset
  finalize_baseline
  export_and_train
  check_mlflow
}

main
