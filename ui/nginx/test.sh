#!/usr/bin/env bash
# Runs the image's entrypoint script (40-mlflow.sh) against temp directories and checks the rendered nginx config.
# Needs envsubst (gettext).
set -u

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }

# render <name> [VAR=value ...]: runs the script into $work/<name>/conf.d with a resolv.conf holding $RESOLV_CONTENT.
render() {
  local name=$1; shift
  local dir=$work/$name
  mkdir -p "$dir/conf.d"
  printf '%s' "${RESOLV_CONTENT-nameserver 10.96.0.10
}" > "$dir/resolv.conf"
  env -u API_UPSTREAM -u MLFLOW_UPSTREAM -u MLFLOW_ENABLED -u DNS_RESOLVER \
    NGINX_CONF_DIR="$dir/conf.d" NGINX_TEMPLATE_DIR="$here" RESOLV_CONF="$dir/resolv.conf" "$@" \
    sh "$here/40-mlflow.sh" 2> "$dir/stderr" || fail "$name: script exited non-zero"
}

has() { grep -qF -- "$2" "$work/$1/conf.d/$3"; }
expect() { has "$1" "$2" "$3" || fail "$1: $3 lacks '$2'"; }
refuse() { ! has "$1" "$2" "$3" || fail "$1: $3 contains '$2'"; }

# MLflow on, defaults for upstreams, resolver from resolv.conf
render enabled MLFLOW_ENABLED=1
expect enabled 'location /mlflow/' mlflow.inc
expect enabled 'proxy_pass $mlflow;' mlflow.inc
expect enabled 'location = /_auth' mlflow.inc
expect enabled 'location @signin' mlflow.inc
expect enabled 'set $mlflow http://mlflow:5000;' mlflow.inc
expect enabled 'set $api http://api:8000;' mlflow.inc
expect enabled 'set $api http://api:8000;' default.conf
expect enabled 'resolver 10.96.0.10 valid=10s ipv6=off;' default.conf
expect enabled 'include /etc/nginx/conf.d/mlflow.inc;' default.conf
# nginx's own variables survive the substitution
expect enabled 'proxy_set_header Host $http_host;' default.conf
expect enabled 'try_files $uri /index.html;' default.conf

# Custom upstreams
render custom MLFLOW_ENABLED=1 API_UPSTREAM=http://rainforest-api:8000 MLFLOW_UPSTREAM=http://rainforest-mlflow:5000
expect custom 'set $api http://rainforest-api:8000;' default.conf
expect custom 'set $api http://rainforest-api:8000;' mlflow.inc
expect custom 'set $mlflow http://rainforest-mlflow:5000;' mlflow.inc

# MLflow off: unset, 0 and any other value leave an empty include and no mlflow in the config
for state in unset 0 true; do
  if [ "$state" = unset ]; then render "off-$state"; else render "off-$state" MLFLOW_ENABLED="$state"; fi
  [ -f "$work/off-$state/conf.d/mlflow.inc" ] || fail "off-$state: mlflow.inc missing"
  [ ! -s "$work/off-$state/conf.d/mlflow.inc" ] || fail "off-$state: mlflow.inc not empty"
  # the include line itself names the file; nothing else may mention MLflow
  ! grep -v 'include /etc/nginx/conf.d/mlflow.inc;' "$work/off-$state/conf.d/default.conf" | grep -qi mlflow ||
    fail "off-$state: default.conf mentions mlflow"
  expect "off-$state" 'set $api http://api:8000;' default.conf
done

# An explicit DNS_RESOLVER wins over resolv.conf
render explicit MLFLOW_ENABLED=1 DNS_RESOLVER=10.0.0.2
expect explicit 'resolver 10.0.0.2 valid=10s ipv6=off;' default.conf

# First nameserver line is used; comments and other directives are skipped
RESOLV_CONTENT='# generated
search svc.cluster.local
nameserver 127.0.0.11
nameserver 8.8.8.8
' render first
expect first 'resolver 127.0.0.11 valid=10s ipv6=off;' default.conf
[ ! -s "$work/first/stderr" ] || fail "first: unexpected warning: $(cat "$work/first/stderr")"

# An IPv6 nameserver is bracketed for nginx
RESOLV_CONTENT='nameserver fd00::10
' render ipv6
expect ipv6 'resolver [fd00::10] valid=10s ipv6=off;' default.conf

# No nameserver line: Docker's resolver and a warning
RESOLV_CONTENT='search example.com
' render fallback
expect fallback 'resolver 127.0.0.11 valid=10s ipv6=off;' default.conf
grep -q 'nameserver' "$work/fallback/stderr" || fail "fallback: no warning on stderr"

if [ "$failures" -ne 0 ]; then
  echo "$failures check(s) failed" >&2
  exit 1
fi
echo "ui/nginx: all checks passed"
