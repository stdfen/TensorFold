# TensorFold v1.2.1 双 GB10 原生启动回执

## 结论

2026-11-03 已在 `intellif-74`（rank 0）与 `intellif-15`（rank 1）完成 TensorFold v1.2.1 原生 TP2 启动。
服务使用 EXL3 2.9 bpw 量化 PACK 在线推理，并从同模型的全量权重离线生成 v1.2.1 Engram TP2 分片。

本回执证明 NCCL、32K context、单 slot、draft 关闭基线达到 `integration_verified`。它不等价于 RoCE、长上下文、并发或生产验收。

## 固定输入

| 输入 | 值 | 原因 |
|---|---|---|
| TensorFold | v1.2.1，提交 `1db3edbe7` | 固定原生框架 ABI 与行为基线 |
| 镜像 | `tensorfold-dsv41:v1.2.1-sm121-native` | ARM64、CUDA 13.0.1、SM121 |
| 在线 PACK | `/data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` | 适配 GB10 内存容量的作者量化权重 |
| Engram 来源 | `/data/models/DeepSeek-V4.1-Flash-1017` | 量化索引不含 Engram embed tensors，必须从全量权重离线提取 |
| 拓扑 | TP=2，rank 0/1，NCCL | 先建立框架原生正确性基线，再评估 RoCE |
| profile | context 32768、slots 1、drafts 0 | 降低首次启动变量并与 AOT needs 完全绑定 |

## 原生资产生成结果

- `tf-dsv41-token-map` 从量化模型 tokenizer/config 生成 `DSENGR01` token map。
- v1.2.1 `aot-needs` 在两个 rank 上生成相同 SHA-256 的 92 个 specialization 需求。
- 外部参考工具使用 Triton 3.7.1 与 ptxas 13.3.73 为 SM121 编译 92/92 specialization，失败 0。
- `rope.bin` 按 `32768 + 2048 = 34816` 行生成。
- rank-aware `tf-dsv41-engram-pack` 为 layer 1/14 分别生成 `r0of2` 与 `r1of2`。
- 每个 rank 回读比较 200,004 行，差异 0；资产与 Engram 均有 `SHA256SUMS`。

旧 v1.1.6 AOT 仅精确覆盖当前需求的一部分，因此没有直接复用；旧资产只用于提供 kernel source、签名和编译选项来源。

## 启动参数

两个 rank 共享：

```text
TF_EXPERIMENTAL=deepseek
TF_TP_WORLD=2
TF_TP_DEVICE=0
TF_TP_FAILFAST=1
TF_COMM_BACKEND=nccl
TF_CONTEXT=32768
TF_DSV41_CONTEXT=32768
TF_PARALLEL=1
TF_DSV41_SLOTS=1
TF_DSV41_DRAFTS=0
TF_MAX_TOKENS=32768
TF_TEMPERATURE=1.0
TF_TOP_P=0.95
TF_TOP_K=20
TF_THINKING=1
TF_REQUIRE_ASSET_HASHES=1
```

每节点只改变 `TF_TP_RANK` 和 rank 对应的 Engram/asset 目录。rank 1 先启动 `tf-dsv41 follow`，rank 0 后启动 `tensorfold-dsv41 serve`。

## 运行证据

- 两节点 preflight 均返回 `PASS`，识别 GPU 为 NVIDIA GB10、compute capability 12.1、驱动 580.178.04。
- 两 rank 均报告 TP2/NCCL/fail-fast 已启用，每 rank 加载 94.8 GiB，未发生容器重启。
- rank 1 日志为 `loaded, following rank 0`。
- rank 0 在 8080 提供 OpenAI API，加载完成耗时约 278.5 秒。
- `/health` 返回 `status=ok`、`version=1.2.1`、`backend=cuda`、`warming=false`。
- `/v1/models` 返回模型 `deepseek-v4.1-flash-1017`，context length 32768。
- 关闭 thinking 的短请求精确返回 `READY`；TTFT 约 0.287 秒，3-token decode 约 20.26 tok/s。
- 首次真实请求后两 rank 日志未出现 `MissingTritonVariant`、`no captured Triton`、panic 或 error。

## 已知限制与下一步

1. 当前仅验证 NCCL over Ethernet；RoCE 尚未验收。
2. 当前只有单 slot、短请求证据；尚未完成 32K 边界、并发、vision、故障注入与长稳测试。
3. `libtfgrammar.so` 未包含，结构化输出和 required tool calls 会返回 HTTP 400；普通 chat/completions 不受影响。
4. 启动时启用完整资产哈希，约 101 GiB Engram 会增加重启前校验时间，这是有意的 fail-fast 行为。
5. GB10 可用内存余量约 14 GiB；扩大 context、slot 或启用 draft 前必须重新做内存门禁。
