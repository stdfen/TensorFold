# DeepSeek-V4.1-Flash-1017 native TP4 roadmap

Status: active local experiment  
Last updated: 2026-10-28  
Target checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017`  
Pinned revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`

## Objective

Run the full official DeepSeek-V4.1-Flash-1017 checkpoint with TensorFold's native CUDA engine, text-only and
tensor-parallel world 4. The runtime must consume the official safetensors as its authority and preserve:

- MXFP8 E4M3 dense weights with UE8M0 scales and checkpoint-defined dynamic activation quantization.
- MXFP4 E2M1 packed routed experts with K-group 32 and UE8M0 scales.
- BF16 embedding, head, norms and other checkpoint-defined BF16 tensors.
- Official Engram FP8 rows and UE8M0 scales.
- Exact TensorFold engine contracts for plain, drafted, resumed and concurrent execution as each feature is enabled.

Correctness is established first on a large-memory GPU environment. Four-machine GB10 NVMe residency begins only
after the all-resident whole-model gate passes.

This document is the implementation source of truth. Update its status, decisions, gates and change log whenever the
implementation changes direction.

## Scope and constraints

The first whole-model path is deliberately narrow:

- Included: text-only, TP4, greedy decoding, one request, short context, official checkpoint arithmetic.
- Initially disabled: vision, aligner, MTP/DSpark, prompt cache, sessions, concurrency and long-context tuning.
- Lossless runtime slicing, padding, permutation and prepared caches are allowed.
- Requantizing official tensors is not allowed.
- The official safetensors remain authoritative; any prepared store is derived, discardable and revision-keyed.
- The existing EXL3 path remains separate and unchanged.

## P0 evidence baseline

The P0 implementation is `tools/zig/dsv41_native_checkpoint.py`, with tests in
`tools/zig/test_dsv41_native_checkpoint.py`.

The pinned official revision passed index/header and mixed-format validation:

| Item | Result |
| --- | ---: |
| Shards | 48 |
| Tensors | 96,085 |
| Payload bytes | 510,286,023,000 |
| Payload size | 475.241 GiB |
| Text scope | 466.949 GiB |
| Excluded multimodal scope | 0.904 GiB |
| Optional draft scope | 7.388 GiB |
| TP4 text-only raw slice per rank | 117.492 GiB |
| Resident slice per rank | 3.047 GiB |
| Routed-expert slice per rank | 67.236 GiB |
| Engram slice per rank | 47.208 GiB |
| Row-split read spans per rank | 157,709,580 |

The row-span count makes direct runtime streaming of current TP4 `w2` slices from the official row-major files
impractical. The GB10 phase must use a lossless prepared layout, change the runtime layout, or adopt expert-ID
partitioning.

## Milestone status

| Milestone | State | Exit criterion |
| --- | --- | --- |
| M0: checkpoint inventory and TP4 plan | Complete | Official 48-shard header plan passes |
| M1: native official-safetensors loader | Complete | Native plan equals P0 and covers every tensor |
| M2: mixed projection host references | Complete | MXFP8/MXFP4/BF16 references pass real slices |
| M3: CUDA kernels and all-resident bring-up | In progress | Minimal TP4 model produces qualified output |
| M4: large-memory whole-model correctness | Not started | All required engine equalities pass |
| M5: lossless rank-local prepared store | Not started | Prepared bytes reproduce official tensors |
| M6: four-GB10 NVMe residency | Not started | Correct and stable four-machine deployment |

## M0: checkpoint inventory and TP4 plan

State: complete.

Delivered:

- Local-directory and Hugging Face Range-header input.
- Config, index, dtype, shape, byte-range and shard placement validation.
- MXFP8, MXFP4, UE8M0 and BF16 classification.
- Text, optional-draft and multimodal scope classification.
- Resident, routed-expert and Engram residency accounting.
- Current TensorFold TP split planning and row-major fragmentation counts.

Receipt complete: the local weight host reproduced the 48-shard, 96,085-tensor P0 plan. See
[`docs/receipts/deepseek-v41-1017-native-m1-20261028.md`](../receipts/deepseek-v41-1017-native-m1-20261028.md).

Command:

```bash
python tools/zig/dsv41_native_checkpoint.py \
  plan-native /data/models/DeepSeek-V4.1-Flash-1017 \
  --world 4 \
  --json /tmp/deepseek-v41-1017-tp4-plan.json
```

## M1: native official-safetensors loader

State: complete.

Delivered so far:

- Added a Zig header-only admission path for the official mixed checkpoint without changing the EXL3 loader.
- Added config, dtype, MXFP4/MXFP8 scale-geometry, scope, residency, TP split and row-span validation.
- Added `inspect-native` and `plan-native --world N` command routing to `tf-dsv41`.
- Added streaming `plan-native --world N --json PATH` output for the complete sorted per-tensor plan.
- Added a P0 `--compare-native PATH` gate covering every tensor plan field without relying on map order.
- Added host fixtures for the official mixed layout, invalid expert-scale geometry, unclassified I8,
  unindexed and misplaced tensors, index total-size mismatch and malformed shard headers.

Receipt: the native and P0 planners matched all 96,085 tensors across all 12 plan fields on the AArch64
weight host. The native host suite passed 9/9. See the linked M1 receipt above.

Goal: let the native DeepSeek family recognize and plan the official mixed checkpoint without requiring an EXL3 pack.

Work:

1. Add an `official_mixed` checkpoint format independent of `exl3`.
2. Validate the official model type and quantization contract.
3. Reuse `core/safetensors.zig` for header parsing and byte ranges.
4. Build tensor-name to shard/offset/dtype/shape metadata without reading tensor payloads.
5. Map dense MXFP8 weights/scales, routed MXFP4 weights/scales, BF16 tensors and Engram tables.
6. Represent `wo_a` explicitly; do not treat it as an ordinary dense projection.
7. Port P0 TP4 slicing and capacity accounting to Zig.
8. Refuse vision and unsupported draft paths clearly.
9. Keep all loader changes separate from the current EXL3 loader.

Commands to deliver:

```text
tf-dsv41 inspect-native MODEL
tf-dsv41 plan-native MODEL --world 4
tf-dsv41 plan-native MODEL --world 4 --json native-plan.json
python tools/zig/dsv41_native_checkpoint.py plan-native MODEL --world 4 --compare-native native-plan.json
```

Gate:

- Native output matches the P0 JSON for all 96,085 tensors.
- Zero missing, duplicate, misplaced or unclassified tensors.
- All dtype, shape and scale geometry failures are explicit.
- Planning completes without reading the 510 GB payload.
- Host tests include malformed index, header, geometry and unsupported-scope cases.

## M2: mixed projection host references

State: complete.

Delivered:

- Checkpoint-independent `bf16`, `mxfp8`, `mxfp4` and special `wo_a` projection classification.
- Independent scalar FP64 and FP32 prompt/decode references.
- E4M3, E2M1 packed, UE8M0, BF16 and dynamic-activation arithmetic contracts.
- Deterministic digest-keyed slices from four real official tensors.
- Local and target Linux AArch64 reference and projection suites.

Receipt: [`docs/receipts/deepseek-v41-1017-native-m2-20261028.md`](../receipts/deepseek-v41-1017-native-m2-20261028.md).

Goal: establish independent arithmetic references before adding GPU kernels.

Work:

- Decouple projection dispatch from checkpoint format.
- Add BF16, MXFP8, MXFP4 and special-projection kinds.
- Implement MXFP8 E4M3 plus UE8M0 decode.
- Implement MXFP4 E2M1 packed plus UE8M0 decode, K-group 32.
- Implement checkpoint-defined dynamic activation quantization.
- Implement the official `wo_a` behavior.
- Extract small, digest-keyed fixtures from real official tensors.
- Add FP32 and FP64 references for prompt and decode rows.

Gate:

- Real tensor slices decode element-for-element as specified.
- Each projection class has an independent FP64 reference.
- Reports include maximum absolute error, relative error and worst column.
- No existing EXL3 behavior changes.

## M3: CUDA kernels and all-resident bring-up

State: in progress.

Delivered so far:

- Correctness-first CUDA projection operators for BF16, official MXFP8 dynamic/plain, official MXFP4 group-32,
  and block-diagonal `wo_a`, with FP32 and BF16 output variants.
- Real official-safetensors slices for all four projection classes pass on an SM121 GB10; synthetic dynamic
  activation and BF16-output cases pass on the same device.
- A correctness-first routed MoE pair covers MXFP4 routed experts and the MXFP8 shared expert without EXL3
  rotations or requantization. Its gate/up, bounded SwiGLU, down and route combine device fixture passes on GB10.
- Zig launch bindings, extension replay validation, family fatbin registration and an isolated host contract build.
- Lossless rank-local payload reads and one contiguous all-resident device image keyed by official tensor names.
- Rank-local projection geometry validation and 16-byte-aligned per-layer device pointer tables for routed plus
  shared experts.
- Decode and prefill emitters select the official mixed extensions without EXL3/x3 calls.
- Dual-backend runner lookup, official-name runtime aliases, required FP32 forms, Engram q/k forms, and a TP4
  native-resident owner that loads and binds all backbone expert tables.
- Production model/HTTP/follower opt-in for direct official loading, with minimal-mode admission and an AArch64
  Linux cross-compile gate. Required mHC `fn16` persistent forms are explicitly built and externally bound.
- TP4 teacher-forced capture is wired through the production `Forward`: every rank participates in GPU local top-K
  plus all-gather, and rank 0 deterministically merges by logit then token id. `tf-dsv41 native-capture` writes the
  frozen prompt/decode rows in the strict comparison schema, including checkpoint digests and cross-engine margins.
- Runtime weight views now resolve both legacy and official backends for embedding and auxiliary declarations;
  the host native gate and AArch64 Linux production gate pass after this production boot audit.

Current remaining gate:

- Run a complete TP4 forward in an environment with enough memory for four approximately 117.5 GiB rank images
  plus runtime state, then compare its `native-capture` output against the frozen reference capture. A single
  121 GiB unified-memory GB10 is intentionally not treated as that environment.

Evidence: [`docs/receipts/deepseek-v41-1017-native-m3-20261028.md`](../receipts/deepseek-v41-1017-native-m3-20261028.md).

Goal: run the minimal text-only official checkpoint on TP4 in a sufficiently large-memory GPU environment.

### M3.1 MXFP8 dense path

- Add checkpoint-math activation quantization.
- Do not assume the current weight-only MXFP8 path is numerically equivalent.
- Cover attention, shared experts, routing and other dense projections.
- Gate `wo_a` independently.

### M3.2 MXFP4 routed-MoE path

- Add native E2M1 packed, group-32, UE8M0 CUDA operators.
- Do not reuse NVFP4, whose scale geometry and semantics differ.
- Cover routed `w1`, `w3` and `w2` for prompt and decode.
- Preserve current TP4 collectives for the first implementation.

### M3.3 All-resident loading

- Read tensor payloads from official shards.
- Allow lossless rank-local staging and packing.
- Prohibit requantization.
- Keep all target weights resident for this milestone.

### M3.4 Minimal served configuration

- Text only.
- TP4.
- Greedy, draft off.
- One request and short context.
- Prompt cache, sessions and concurrency off.

Gate:

- Each GPU projection meets its FP64 contract on served shapes.
- Prompt and decode are checked separately.
- Teacher-forced logits record top-token differences and margins.
- TP4 produces the reference tokens for the frozen prompt corpus.
- No silent format conversion or CPU fallback exists.

## M4: large-memory whole-model correctness

Goal: qualify the official checkpoint path before any GB10 streaming optimization.

Enable in this order:

1. Longer prompts.
2. Engram.
3. Concurrent requests.
4. Sessions and resume.
5. Prompt cache.
6. MTP/DSpark drafting.
7. Long-context windows.

Gate at the same checkpoint, backend, runtime, world and settings:

- Drafted output equals plain output token-for-token.
- Concurrent replies equal their solo replies.
- Resumed execution equals fresh full-prompt execution.
- Cancellation, reset and partial acceptance leave the next request valid.
- Prompt/decode seams preserve all required state.
- Teacher-forced comparisons retain every top-token difference and margin.
- Hardware, CUDA, libraries, configuration and token arrays are recorded.

Do not start M5 until this gate passes.

## M5: lossless rank-local prepared store

Goal: remove the official row-major `w2` fragmentation while retaining official weights as the authority.

Initial decision: prototype a TP4 prepared store before considering EP4.

Requirements:

- One revision-keyed manifest per rank.
- Source tensor, shard, offset, shape, dtype and digest provenance.
- Lossless rank-local contiguous layouts for `w1`, `w3` and especially `w2`.
- Expert-slot granularity suitable for sequential NVMe reads.
- Atomic publication, interruption recovery and complete validation.
- No numerical conversion or requantization.
- Prepared data is discardable and reproducible from official shards.

Decision gate:

| Candidate | Benefit | Cost or risk |
| --- | --- | --- |
| TP4 prepared store | Reuses current execution graph and collectives | Every rank still addresses every expert |
| EP4 expert-ID partition | Experts are naturally contiguous per rank | Requires token all-to-all and a larger graph rewrite |

Switch to EP4 only if measured TP4 capacity, IO or communication cannot meet the deployment gate.

## M6: four-GB10 NVMe residency

Goal: deploy the full official checkpoint on four 128 GB unified-memory GB10 machines.

Per-rank tiers:

- Resident: dense/shared weights, embedding, head, norms, router, runtime workspace, KV and communication buffers.
- Cached: routed-expert slots and Engram rows.
- NVMe: a complete local official checkpoint plus the rank-local prepared store.

Work:

- Implement a Linux/CUDA FSR physical owner.
- Compare `io_uring`, O_DIRECT and positional reads.
- Use aligned host buffers and explicit CUDA event ownership.
- Add expert forecasting, prefetch, leases, publication and eviction.
- Reuse and adapt the existing Engram row cache and prefetch path.
- Keep normal weight misses local to each machine's NVMe.
- Make peer, IO and collective failures fail fast across all ranks.

Primary tuning variables:

- Expert-pool GiB.
- Expert prefetch depth.
- Engram row-cache GiB.
- IO queue depth and request size.
- Slot granularity.
- KV budget.
- Prompt chunk size.
- Compute, IO and collective overlap.

Gate:

- Cold boot succeeds on all four machines.
- Memory remains above the declared floor with no OOM or unified-memory thrashing.
- Greedy output matches the M4 all-resident baseline.
- Streaming preserves drafted/plain, concurrent/solo and resumed/fresh equalities.
- Node or link failure terminates all ranks rather than hanging a collective.
- Receipts include TTFT, decode rate, NVMe bandwidth, cache hit rate, evictions and tail latency.

## Implementation order

Work proceeds strictly in this order unless the decision log records a change:

1. Finish the local-weight-host M0 receipt. Complete.
2. Implement M1 and pass its header-only native gate. Complete.
3. Implement M2 references and freeze arithmetic fixtures. Complete.
4. Implement M3 kernels and minimal all-resident TP4. In progress; production boot and hardware gate remain.
5. Complete M4 whole-model correctness.
6. Prototype and choose the M5 storage layout.
7. Implement and qualify M6 on four GB10 machines.

## Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-28 | Correctness precedes GB10 streaming | Avoid debugging arithmetic and residency simultaneously |
| 2026-10-28 | Keep the official safetensors authoritative | Prevent untracked or lossy checkpoint conversion |
| 2026-10-28 | First bring-up is text-only and drafts-off | Minimize the first whole-model correctness surface |
| 2026-10-28 | Preserve current TP4 expert semantics through M4 | Avoid an EP4 all-to-all rewrite before arithmetic is proven |
| 2026-10-28 | Prototype a lossless TP4 prepared store before EP4 | P0 proves direct row-major `w2` streaming is impractical |

## Change log

### 2026-10-28

- Created the active roadmap.
- Recorded the completed P0 official-header receipt.
- Completed M1 on the AArch64 weight host: 9/9 host tests and all 96,085 tensor plans matched P0.
- Corrected excluded optional-draft span accounting discovered by the first real-checkpoint comparison.
- Completed M2 independent BF16/MXFP8/MXFP4/`wo_a` host references and froze deterministic real-tensor fixtures.
- Passed M2 arithmetic and projection suites on both local x86_64 and target Linux AArch64 hosts.
