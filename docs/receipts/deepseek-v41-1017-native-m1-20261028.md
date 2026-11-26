# DeepSeek-V4.1-Flash-1017 native M1 receipt

Date: 2026-10-28  
Verification: target weight host, Linux AArch64, Zig 0.17.0, Python 3.12.3  
Scope: header-only official-safetensors admission and TP4 planning; no tensor payload was read

## Checkpoint identity

| Item | Value |
| --- | --- |
| Model type | `deepseek_v41` |
| Shards | 48 |
| Tensors | 96,085 |
| Indexed payload | 510,286,023,000 bytes |
| `config.json` SHA-256 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| index SHA-256 | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |

## TP4 text-only plan

| Item | Bytes / count |
| --- | ---: |
| Text payload | 501,382,611,408 bytes |
| Optional draft, excluded | 7,932,874,632 bytes |
| Multimodal, excluded | 970,536,960 bytes |
| Per-rank resident | 3,271,631,808 bytes |
| Per-rank routed experts | 72,194,457,600 bytes |
| Per-rank Engram | 50,689,508,232 bytes |
| Per-rank total | 126,155,597,640 bytes (117.492 GiB) |
| Replicated spans | 1,099 |
| Row-table spans | 6 |
| Column-split spans | 63,320 |
| Row-split spans | 157,709,580 |

## Gates

- The native Zig planner completed directly against all 48 official shard headers: `PASS`.
- The independent Python P0 planner completed against the same checkpoint: `PASS`.
- All 96,085 sorted tensor records matched for name, shard, dtype, shape, scope, residency,
  encoding, split, axis, rank bytes, read spans and contiguity: `PASS`.
- Saved P0 plan: 38,517,636 bytes, SHA-256
  `2796b6c2f66f2dd22ae015d8616464aa29d6cfb97b32ff743206601d5a4c8211`.
- Saved native plan: 38,516,787 bytes, SHA-256
  `aac4557d824ea72c2661d48d7886136e80db7fc9baf8c54680c420ccc9d9ec08`.
- Native host suite on the target architecture: 9/9 tests passed.

The first comparison exposed an optional-draft accounting bug: excluded `mtp` tensors were assigned zero
bytes but retained row-fragment spans. The native planner was corrected to assign one non-scheduled span,
a regression test was added, and the complete real-checkpoint comparison then passed.

## Commands

```bash
tf-dsv41-native plan-native "$MODEL" --world 4 --json native-plan-tp4.json
python3 tools/zig/dsv41_native_checkpoint.py plan-native "$MODEL" \
  --world 4 --json p0-plan-tp4.json --compare-native native-plan-tp4.json
```

This receipt qualifies M1 metadata admission and planning only. It does not qualify MXFP8/MXFP4 arithmetic,
GPU kernels, whole-model logits, NVMe streaming or four-node serving.
