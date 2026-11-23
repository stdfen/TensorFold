#!/bin/sh
set -eu

if test "$#" -lt 5; then
    echo "usage: $0 IMAGE ENV_FILE MODEL_DIR ASSET_DIR ENGRAM_DIR [STATE_DIR]" >&2
    exit 64
fi

image=$1
env_file=$2
model_dir=$3
asset_dir=$4
engram_dir=$5
state_dir=${6:-/var/lib/tensorfold-dsv41}

test -f "$env_file" || { echo "missing env file: $env_file" >&2; exit 64; }
test -d "$model_dir" || { echo "missing model directory: $model_dir" >&2; exit 64; }
test -d "$asset_dir" || { echo "missing asset directory: $asset_dir" >&2; exit 64; }
test -d "$engram_dir" || { echo "missing Engram directory: $engram_dir" >&2; exit 64; }
mkdir -p "$state_dir"

set -- docker run --rm \
    --name tensorfold-dsv41 \
    --network host \
    --ipc host \
    --gpus all \
    --ulimit memlock=-1:-1 \
    --cap-add IPC_LOCK \
    --env-file "$env_file" \
    --mount "type=bind,src=$model_dir,dst=/models/model,readonly" \
    --mount "type=bind,src=$asset_dir,dst=/assets,readonly" \
    --mount "type=bind,src=$engram_dir,dst=/engram,readonly" \
    --mount "type=bind,src=$state_dir,dst=/var/lib/tensorfold" \
    "$image"

if test -d /dev/infiniband; then
    set -- docker run --rm \
        --name tensorfold-dsv41 \
        --network host \
        --ipc host \
        --gpus all \
        --ulimit memlock=-1:-1 \
        --cap-add IPC_LOCK \
        --device /dev/infiniband \
        --env-file "$env_file" \
        --mount "type=bind,src=$model_dir,dst=/models/model,readonly" \
        --mount "type=bind,src=$asset_dir,dst=/assets,readonly" \
        --mount "type=bind,src=$engram_dir,dst=/engram,readonly" \
        --mount "type=bind,src=$state_dir,dst=/var/lib/tensorfold" \
        "$image"
fi

exec "$@"

