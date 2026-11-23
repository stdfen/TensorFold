#!/bin/sh
set -eu

if test "${TF_TP_RANK:-}" = 1; then
    kill -0 1
    exit $?
fi

curl --fail --silent --show-error \
    "http://127.0.0.1:${TF_HTTP_PORT:-8080}/health" \
    >/dev/null

