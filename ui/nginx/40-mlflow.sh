#!/bin/sh
# Renders the nginx config from the environment; the nginx image runs this from /docker-entrypoint.d/ before starting.
# The env overrides below exist for ui/nginx/test.sh.
set -eu

template_dir=${NGINX_TEMPLATE_DIR:-/etc/nginx/rainforest}
conf_dir=${NGINX_CONF_DIR:-/etc/nginx/conf.d}
resolv_conf=${RESOLV_CONF:-/etc/resolv.conf}

API_UPSTREAM=${API_UPSTREAM:-http://api:8000}
MLFLOW_UPSTREAM=${MLFLOW_UPSTREAM:-http://mlflow:5000}

# Docker's embedded DNS (127.0.0.11) and the cluster DNS both appear as the first nameserver, so nothing needs setting.
if [ -z "${DNS_RESOLVER:-}" ]; then
  DNS_RESOLVER=$(awk '$1 == "nameserver" { print $2; exit }' "$resolv_conf" 2>/dev/null || true)
  if [ -z "$DNS_RESOLVER" ]; then
    DNS_RESOLVER=127.0.0.11
    echo "40-mlflow.sh: no nameserver in $resolv_conf, using $DNS_RESOLVER" >&2
  fi
  # nginx wants IPv6 addresses in brackets
  case $DNS_RESOLVER in *:*) DNS_RESOLVER="[$DNS_RESOLVER]" ;; esac
fi
export API_UPSTREAM MLFLOW_UPSTREAM DNS_RESOLVER

# Explicit variable lists keep nginx's own $variables intact.
envsubst '${API_UPSTREAM} ${DNS_RESOLVER}' < "$template_dir/default.conf.template" > "$conf_dir/default.conf"

if [ "${MLFLOW_ENABLED:-}" = 1 ]; then
  envsubst '${API_UPSTREAM} ${MLFLOW_UPSTREAM}' < "$template_dir/mlflow.inc.template" > "$conf_dir/mlflow.inc"
else
  : > "$conf_dir/mlflow.inc"
fi
