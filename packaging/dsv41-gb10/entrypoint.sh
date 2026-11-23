#!/bin/sh
set -eu

die() {
    printf 'tensorfold-entrypoint: %s\n' "$*" >&2
    exit 64
}

case "${TF_TP_WORLD:-}" in
    2) ;;
    *) die "TF_TP_WORLD must be 2 for this image" ;;
esac

case "${TF_TP_RANK:-}" in
    0|1) ;;
    *) die "TF_TP_RANK must be 0 or 1" ;;
esac

case "${TF_TP_MASTER:-}" in
    ""|localhost|127.0.0.1) die "TF_TP_MASTER must be rank 0's fabric address, reachable from rank 1" ;;
esac

test -d "${TF_MODEL_DIR}" || die "model directory does not exist: ${TF_MODEL_DIR}"
test -f "${TF_MODEL_DIR}/config.json" || die "model config is missing: ${TF_MODEL_DIR}/config.json"
test -f "${TF_MODEL_DIR}/tokenizer.json" || die "tokenizer is missing: ${TF_MODEL_DIR}/tokenizer.json"
test -d "${TF_DSV41_ASSETS}" || die "asset directory does not exist: ${TF_DSV41_ASSETS}"
test -d "${TF_DSV41_ENGRAM_DIR}" || die "Engram directory does not exist: ${TF_DSV41_ENGRAM_DIR}"
test -f "${TF_DSV41_ENGRAM_DIR}/engram-l1-r${TF_TP_RANK}of${TF_TP_WORLD}.bin" || die "rank Engram shard is missing: layer 1"
test -f "${TF_DSV41_ENGRAM_DIR}/engram-l14-r${TF_TP_RANK}of${TF_TP_WORLD}.bin" || die "rank Engram shard is missing: layer 14"
test -f "${TF_DSV41_ASSETS}/rope.bin" || die "asset is missing: rope.bin"
test -f "${TF_DSV41_ASSETS}/engram-token-map.bin" || die "asset is missing: engram-token-map.bin"
test -d "${TF_DSV41_ASSETS}/aot" || die "asset directory is missing: aot/"

if test "${TF_REQUIRE_ASSET_HASHES:-1}" = 1; then
    test -f "${TF_DSV41_ASSETS}/SHA256SUMS" || die "asset SHA256SUMS is required"
    (cd "${TF_DSV41_ASSETS}" && sha256sum --check --strict SHA256SUMS) \
        || die "asset checksum verification failed"
    test -f "${TF_DSV41_ENGRAM_DIR}/SHA256SUMS" || die "Engram SHA256SUMS is required"
    (cd "${TF_DSV41_ENGRAM_DIR}" && sha256sum --check --strict SHA256SUMS) \
        || die "Engram checksum verification failed"
fi

if test "$#" -gt 0; then
    exec "$@"
fi

if test "${TF_MODE:-serve}" = preflight; then
    exec tensorfold-preflight
fi

if test "${TF_TP_RANK}" = 1; then
    exec tf-dsv41 follow "${TF_MODEL_DIR}" "${TF_DSV41_ASSETS}"
fi

set -- serve "${TF_MODEL_DIR}" \
    --host "${TF_HTTP_HOST:-0.0.0.0}" \
    --port "${TF_HTTP_PORT:-8080}" \
    --name "${TF_SERVED_NAME:-deepseek-v4.1-flash-1017}" \
    --context "${TF_CONTEXT:-32768}" \
    --parallel "${TF_PARALLEL:-1}" \
    --max-tokens "${TF_MAX_TOKENS:-32768}" \
    --temperature "${TF_TEMPERATURE:-1.0}" \
    --top-p "${TF_TOP_P:-0.95}" \
    --top-k "${TF_TOP_K:-20}" \
    --no-update-check \
    --metrics-open

if test "${TF_THINKING:-1}" = 1; then
    set -- "$@" --thinking
else
    set -- "$@" --no-thinking
fi

if test -n "${TF_REASONING_EFFORT:-}"; then
    set -- "$@" --reasoning-effort "${TF_REASONING_EFFORT}"
fi

if test -n "${TF_THINKING_BUDGET:-}"; then
    set -- "$@" --thinking-budget "${TF_THINKING_BUDGET}"
fi

if test "${TF_DSV41_DRAFTS:-0}" = 0; then
    set -- "$@" --no-drafts
fi

if test -n "${TF_CHAT_TEMPLATE:-}"; then
    set -- "$@" --chat-template "${TF_CHAT_TEMPLATE}"
fi

exec tensorfold-dsv41 "$@"

