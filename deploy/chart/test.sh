#!/usr/bin/env bash
# Runs the chart's helm-unittest suites from the pinned Docker image (the local helm refuses the unsigned plugin).
# Extra arguments go to helm-unittest, e.g. `./test.sh -f 'tests/api_test.yaml'`.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec docker run --rm -v "$here:/apps" helmunittest/helm-unittest:3.17.3-0.8.2 "$@" rainforest
