# DeepSeek V4.1 Flash 1017 on two GB10 nodes

This package builds the experimental TensorFold v1.2.1 DeepSeek CUDA engine for native ARM64 and `sm_121`.
The image does not contain model weights or generated assets. Both are mounted read-only so their revisions and
checksums remain independent of the engine image.

## Fixed build inputs

- TensorFold base: `release/v1.2.1` at `1db3edbe7e692e83c7f5056754dc4e69e79a6c22`.
- CUDA: `13.0.1`, Ubuntu `24.04`, ARM64.
- Zig: `0.17.0`; the archive comes from `pkg.hexops.org`, listed by Zig as a community mirror, and is verified with the signature from
  `ziglang.org` and the official Zig minisign key.
- GPU code: `sm_121` only.
- Checkpoint target: `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` at the separately recorded revision.

## Build

Run on an ARM64 GB10 host for the qualification build:

```bash
docker buildx build --platform linux/arm64 \
  --file packaging/dsv41-gb10/Dockerfile \
  --target runtime \
  --build-arg BUILD_JOBS=2 \
  --tag tensorfold-dsv41:v1.2.1-sm121 \
  --load .
```

Windows can validate the build graph through `build.ps1`. QEMU compilation is not a GB10 qualification receipt:

```powershell
.\packaging\dsv41-gb10\build.ps1 -Target toolchain
.\packaging\dsv41-gb10\build.ps1 -Target runtime -Jobs 2
```

The build runs `test-dsv41-assets-host` and `test-dsv41-token-map` before compiling both serving binaries with real
CUDA and family images.

## Required mounts

```text
/models/model/
  config.json
  tokenizer.json
  tokenizer_config.json
  model*.safetensors
  model.safetensors.index.json

/assets/
  rope.bin
  engram-token-map.bin
  aot/
  SHA256SUMS

/engram/
  engram-l1-r<R>of2.bin
  engram-l14-r<R>of2.bin
  engram-manifest.json
  SHA256SUMS
```

Create `SHA256SUMS` only after the v1.2.1 asset pipeline has finished:

```bash
cd /path/to/assets
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum --check --strict SHA256SUMS
```

The same model snapshot, asset directory and image digest must be present on both nodes.

## Preflight

Copy `env.example` to a private file on each node. Set the same master address and set rank to `0` or `1`.

```bash
docker run --rm --gpus all --network host --ipc host \
  --env-file ./rank0.env \
  -e TF_MODE=preflight \
  --mount type=bind,src=/path/to/model,dst=/models/model,readonly \
  --mount type=bind,src=/path/to/assets,dst=/assets,readonly \
  --mount type=bind,src=/path/to/rank-engram,dst=/engram,readonly \
  tensorfold-dsv41:v1.2.1-sm121
```

Preflight checks the ARM64 architecture, exact binary version, GPU visibility, CUDA/NCCL libraries, model metadata
files, required assets and asset hashes. It does not prove that the checkpoint loader or AOT set can execute the model.

## Start the TP2 pair

Start rank 1 first and then rank 0:

```bash
packaging/dsv41-gb10/run-node.sh \
  tensorfold-dsv41:v1.2.1-sm121 ./rank1.env /path/to/model /path/to/assets \
  /path/to/rank1-engram

packaging/dsv41-gb10/run-node.sh \
  tensorfold-dsv41:v1.2.1-sm121 ./rank0.env /path/to/model /path/to/assets \
  /path/to/rank0-engram
```

Rank 0 runs `tensorfold-dsv41 serve`; rank 1 runs `tf-dsv41 follow`. TP settings come from `TF_TP_*`, not from the
ordinary server's `--tp` flags. The first gate deliberately uses NCCL, 32K context, one slot, drafts off and no optional
graph/prefill/session tuning. Enable RoCE and optimizations only after the fixed-token NCCL baseline passes.

For RoCE, mount `/dev/infiniband`, set `TF_COMM_BACKEND=roce`, provide the qualified mailbox kernel through
`TF_TP_KERNEL_IMAGE`, and use `TF_TP_ROCE_FALLBACK=error` for the final receipt. A successful service start with NCCL
fallback is not a RoCE pass.

## Image boundary

This image is build-verified when its ARM64 image builds and its no-GPU checks pass. It becomes integration-verified
only after both GB10 nodes load the pinned checkpoint and pass token parity. It becomes deployable for the requested
profile only after NCCL, strict RoCE, API, long-context, concurrency and rank-failure gates have receipts.

