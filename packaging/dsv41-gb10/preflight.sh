#!/bin/sh
set -eu

fail() {
    printf 'tensorfold-preflight: FAIL: %s\n' "$*" >&2
    exit 1
}

printf '%s\n' 'tensorfold-preflight: build'
cat /opt/tensorfold-build-info
tensorfold-dsv41 --version

test "$(uname -m)" = aarch64 || fail "runtime architecture must be aarch64"
test -r "${TF_MODEL_DIR}/config.json" || fail "config.json is not readable"
test -r "${TF_MODEL_DIR}/tokenizer.json" || fail "tokenizer.json is not readable"
test -r "${TF_DSV41_ASSETS}/rope.bin" || fail "rope.bin is not readable"
test -r "${TF_DSV41_ASSETS}/engram-token-map.bin" || fail "engram-token-map.bin is not readable"
test -d "${TF_DSV41_ASSETS}/aot" || fail "aot is not a directory"

check_engram() {
    layer=$1
    file="${TF_DSV41_ENGRAM_DIR}/engram-l${layer}-r${TF_TP_RANK}of${TF_TP_WORLD}.bin"
    test -r "$file" || fail "Engram shard is not readable: $file"
    set -- $(od -An -v -t u8 -N 48 "$file")
    test "$#" -eq 6 || fail "Engram header is truncated: $file"
    test "$1" = 3545544865734873924 || fail "Engram magic is invalid: $file"
    test "$2" = "$layer" || fail "Engram layer mismatch: $file"
    lo=$3
    hi=$4
    total=$5
    test "$6" = 264 || fail "Engram row width is invalid: $file"
    test "$TF_TP_WORLD" -gt 0 && test "$TF_TP_WORLD" -le 24 || fail "Engram TP world must be in 1..24"
    test "$TF_TP_RANK" -ge 0 && test "$TF_TP_RANK" -lt "$TF_TP_WORLD" || fail "Engram TP rank is outside world"
    if test "$TF_TP_WORLD" -eq 1; then
        expected_lo=0
        expected_hi=$total
    else
        # DSENGR01 stores the 24 canonical column offsets at byte 232 for
        # layer 1 and byte 464 for layer 14. Runtime ownership is the same
        # ceil-divided column range as Host.headShard; total-row division is
        # wrong because the 24 prime buckets have different lengths.
        case "$layer" in
            1) offsets_at=232 ;;
            14) offsets_at=464 ;;
            *) fail "unsupported Engram layer: $layer" ;;
        esac
        part=$(((24 + TF_TP_WORLD - 1) / TF_TP_WORLD))
        col_lo=$((TF_TP_RANK * part))
        col_hi=$((col_lo + part))
        test "$col_lo" -lt 24 || fail "Engram rank owns no hash columns"
        test "$col_hi" -le 24 || col_hi=24
        expected_lo=
        expected_hi=
        column=0
        for offset in $(od -An -v -t u4 -j "$offsets_at" -N 96 "${TF_DSV41_ASSETS}/engram-token-map.bin"); do
            test "$column" -ne "$col_lo" || expected_lo=$offset
            test "$column" -ne "$col_hi" || expected_hi=$offset
            column=$((column + 1))
        done
        test "$column" -eq 24 || fail "Engram token-map offsets are truncated"
        test -n "$expected_lo" || fail "Engram lower column offset is missing"
        test "$col_hi" -ne 24 || expected_hi=$total
        test -n "$expected_hi" || fail "Engram upper column offset is missing"
    fi
    test "$lo" -eq "$expected_lo" || fail "Engram lower row bound mismatch: $file"
    test "$hi" -eq "$expected_hi" || fail "Engram upper row bound mismatch: $file"
    expected_size=$((4096 + (hi - lo) * 264))
    actual_size=$(stat -c %s "$file")
    test "$actual_size" -eq "$expected_size" || fail "Engram shard size mismatch: $file"
}

check_engram 1
check_engram 14

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
else
    fail "nvidia-smi is unavailable; run the container with --gpus all"
fi

ldconfig -p | grep -q 'libcuda.so.1' || fail "libcuda.so.1 is unavailable"
ldconfig -p | grep -q 'libnccl.so.2' || fail "libnccl.so.2 is unavailable"

if test "${TF_COMM_BACKEND:-nccl}" = roce; then
    test "${TF_TP_ROCE_FALLBACK:-nccl}" = error \
        || fail "RoCE validation requires TF_TP_ROCE_FALLBACK=error"
    test -r "${TF_TP_KERNEL_IMAGE:-}" \
        || fail "RoCE mailbox kernel is not readable: ${TF_TP_KERNEL_IMAGE:-unset}"
    test -d /dev/infiniband \
        || fail "RoCE requires /dev/infiniband in the container"
    test -n "${TF_TP_ROCE_HCA:-}" \
        || fail "RoCE requires an explicit validated TF_TP_ROCE_HCA"
fi

printf '%s\n' 'tensorfold-preflight: PASS'

