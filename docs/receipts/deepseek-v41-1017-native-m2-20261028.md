# DeepSeek-V4.1-Flash-1017 native M2 receipt

Date: 2026-10-28  
Verification: Windows x86_64 and target Linux AArch64, Zig 0.17.0; Python fixture extraction on the official checkpoint  
Scope: independent host arithmetic references and real-tensor fixtures; no CUDA kernel or whole-model inference qualification

## Checkpoint identity

| Item | Value |
| --- | --- |
| Model type | `deepseek_v41` |
| Official revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| `config.json` SHA-256 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| index SHA-256 | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| Fixture SHA-256 | `d3b90699274c24067f49d1cb239658126bd730c8adbb5c0c3d10b512b255695b` |

## Arithmetic contracts

- BF16 matrix values and BF16 input/output boundaries.
- MXFP8 E4M3 dense weights with official 32-by-32 UE8M0 scale geometry.
- MXFP4 routed-expert weights with low-nibble-first E2M1 packing and one UE8M0 scale per 32 K values.
- Checkpoint-defined dynamic MXFP8 activation quantization with power-of-two UE8M0 scale.
- Special `wo_a`: block-diagonal attention groups, with decoded MXFP8 weights rounded through BF16 before multiplication.
- Independent FP64 and FP32 prompt/decode matrix references.

Projection classification is independent of the EXL3 checkpoint path and validates dtype and scale geometry before
dispatching `bf16`, `mxfp8`, `mxfp4` or `wo_a`.

## Real official-tensor fixtures

| Projection class | Tensor slice | Payload SHA-256 |
| --- | --- | --- |
| BF16 | `embed.weight` | `269fdc105163ccdbbfa75b99266a3538a64849dcb011157ce4c041224301a273` |
| MXFP8 | `layers.0.attn.wq_a` | `4309702c2b347db33fcce7f7ce9e83c87c56b693b155248a7ee6c193f14a329d` |
| MXFP4 | `layers.0.ffn.experts.0.w1` | `b1a1aa5bb95ac0b9b3828b5b00001c58e43c85db65acaab7bc1088ec9129c5cf` |
| Special `wo_a` | `layers.0.attn.wo_a` | `b430aae38cb7383e415621e3af28b7a7470dcb5ea1fb4a0366a11cb2fdb83ed1` |

The fixture extractor was run twice against the official checkpoint. Both complete JSON outputs had the same
SHA-256 digest, qualifying deterministic extraction. The Python teacher implements BF16 round-to-nearest-even,
E4M3, E2M1 and UE8M0 decoding independently of the Zig implementation.

## Gates

- Local native arithmetic reference suite: 6/6 tests passed.
- Target Linux AArch64 native arithmetic reference suite: 6/6 tests passed.
- Local projection classification and geometry suite: 3/3 tests passed.
- Target Linux AArch64 projection classification and geometry suite: 3/3 tests passed.
- Python fixture teacher and extraction tests: 2/2 tests passed.
- Real BF16, MXFP8, MXFP4 and `wo_a` fixture cases passed element and matvec checks.
- Existing EXL3 loader behavior was not changed.

This receipt closes M2 host-reference qualification only. It does not qualify CUDA numerical error, GPU served
shapes, TP4 collectives, logits, full-model inference, NVMe streaming or four-node deployment. Those gates begin
at M3 and M4.
