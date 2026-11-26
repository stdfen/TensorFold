# DeepSeek-V4.1-Flash-1017 native M3 receipt (in progress)

Date: 2026-10-28  
Scope: local experiment, no PR, official checkpoint arithmetic, text-only TP4 target

## Passed gates

- Isolated Zig official-mixed contract build passes through `build-dsv41-native.zig`, including the real CUDA
  runtime types, native launch bindings, all-resident metadata, projection dispatch and replay bindings.
- The SM121 device fixture passes BF16, BF16 output, dynamic MXFP8, group-32 MXFP4 and `wo_a`.
- Four digest-keyed slices read directly from the pinned official checkpoint pass on device:
  `embed.weight`, `layers.0.attn.wq_a`, `layers.0.ffn.experts.0.w1`, and `layers.0.attn.wo_a`.
- The correctness-first routed/shared MoE device fixture passes. It combines an MXFP4 routed expert with the
  MXFP8 shared expert, including separate activation semantics and route weights.
- Production fatbins compile for SM121: mixed projections and native MoE.
- Decode and prefill emission contract tests select native dense/MoE extensions and reject EXL3/x3 calls.
- The runtime can resolve either legacy packed weights or the official all-resident image. Official native tensors
  are bound to existing runtime roles; BF16/FP16 norms, router bias and indexer forms are widened explicitly,
  while matrix payloads remain byte-identical. Engram q/k runtime forms are reproduced explicitly.
- A TP4 native-resident owner now opens the official checkpoint, loads one contiguous rank image, constructs all
  40 layers' aligned expert pointer tables, and binds them to the runner.
- The production CUDA model, HTTP engine and follower CLI now expose the opt-in
  `TF_DSV41_NATIVE_OFFICIAL=1` path. It refuses non-TP4, drafts, partial backbones and multiple slots, selects
  native emission, and skips legacy EXL3 forms, stacks and L2-prefetch construction.
- Official FP32 mHC function tensors are RNE-converted into the required BF16 `fn16` runtime forms and bound as
  persistent externals; this closes the zero-initialized-form gap found during production boot role auditing.
- A dedicated AArch64 Linux cross-compile gate covers the production model and serving entry point in addition to
  the host-native arithmetic/emission gate. The full 40-layer decode emitter produces 80 native MoE calls and no
  EXL3/x3 calls.
- The production `Forward` exposes an explicit teacher-forced top-K diagnostic. Each TP rank selects local
  candidates on GPU, all four ranks exchange candidates, and the host merge is deterministic for equal logits.
  The plan-link gives followers the same capture operation, avoiding a leader-only collective deadlock.
- `tf-dsv41 native-capture MODEL ASSETS REQUEST OUT` opens the official TP4 backend, captures prompt and decode
  rows, and writes `tensorfold.deepseek_v41.teacher_forced.v1` with checkpoint digests. The strict comparator rejects
  setting/phase drift, requires both engines' winning tokens to be retained, and reports every mismatch and margin.
- A dual-backend `Runner.weightView` closes remaining direct legacy-weight access in embedding, batch and DSpark
  declarations. The isolated host native gate and AArch64 Linux production cross-compile gate pass after the change;
  the Python checkpoint, fixture and comparator suites pass 9/9.
- The lossless rank-local payload reader passed on host and Linux AArch64. Exact official payload samples were
  read for replicated MXFP8, column-split MXFP4 and fragmented row-split MXFP4 tensors.

## Arithmetic decisions now encoded

- Dense MXFP8 uses E4M3 weights, block-32 UE8M0 scales and checkpoint dynamic activation quantization.
- Routed experts use low-nibble-first E2M1 MXFP4 and one UE8M0 scale per logical group of 32 K values.
- Shared experts remain MXFP8 and therefore dynamically quantize their activations independently of routed
  MXFP4 branches.
- `wo_a` decodes each weight and rounds it to BF16 before block-diagonal accumulation.
- No official tensor is requantized. Rank slicing changes layout only by lossless byte selection.

## Capacity result

The official text-only TP4 plan needs up to 126,155,597,640 payload bytes per rank (about 117.492 GiB), before
CUDA context, KV state, activations, collectives and temporary buffers. The available single GB10 has about
121 GiB unified memory, so it is a valid kernel fixture host but not a valid all-resident whole-model host.

## Remaining M3 exit work

The production served-model constructor is wired. M3 is not complete until a four-rank all-resident run produces
qualified prompt/decode output in a sufficiently large-memory GPU environment. The capture and comparison commands
are now implemented; the missing evidence is the real all-resident run, not an absent test harness.

The only currently authorized GPU host is an SM121 GB10 with 121 GiB system/unified memory. At inspection time it
had about 60 GiB available. It cannot hold the 117.492 GiB rank payload plus CUDA/runtime state, so it is retained
as the real-tensor kernel fixture host and is not reported as an all-resident pass.
