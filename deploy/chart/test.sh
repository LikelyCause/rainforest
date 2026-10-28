#!/usr/bin/env bash
# Runs the chart's helm-unittest suites from the pinned Docker image (the local helm refuses the unsigned plugin).
# Extra arguments go to helm-unittest, e.g. `./test.sh -f 'tests/api_test.yaml'`.
# The image runs as uid 1000 and helm-unittest creates tests/__snapshot__, so the chart is mounted read-only and
# tested from a copy: a checkout owned by another user (a CI runner's) would otherwise refuse the write.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec docker run --rm -v "$here:/src:ro" --entrypoint sh helmunittest/helm-unittest:3.17.3-0.8.2 \
  -c 'cp -R /src /tmp/chart && cd /tmp/chart && exec helm unittest "$@" rainforest' sh "$@"
