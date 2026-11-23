# TensorFold v1.2.1 双 GB10 Engram 分片修复与复测记录

## 1. 结论

2026-11-03 已修复 `EngramRowOutsideShard`。修复后的 TP2 服务在 `intellif-74`（rank 0）和 `intellif-15`（rank 1）完成重启，两个容器的 restart count 均为 0。作者 `tools/bench.py` 的 2K、8K、30K prefill 和 256-token decode 共 8 个 run 全部完成，`errors=0`，未再出现 Engram 越界、TP fail-fast 或 HTTP 503。

作者 `tools/smoke.py` 不再在 `stop string` 用例处击穿引擎，后续 stream、tool call、tool result、greedy repeat 和 thinking 用例均继续执行并通过。但是 `stop string` 本身仍为 FAIL：模型生成 64 个重复 BOS 文本并以 `length` 结束，而不是生成 `1 2 3 4` 后由字符串 `5` 截断。因此当前结论是：**Engram 数据布局异常已修复并通过长提示验证；作者完整 smoke 正确性门禁仍剩 1 个独立输出语义问题。**

## 2. 根因与代码修复

运行时 `engram_host.zig` 按 24 个哈希列做 ceil-divide，把完整列分给各 rank。每列占用不同长度的 prime bucket，因此 TP2 的 rank 边界必须落在第 12 列的 canonical offset 上。原 packer 按总 rows 等分，导致 rank 1 缺失合法行：

| 层 | 原 rank 1 起点 | 正确起点 | 缺失合法行 |
| --- | ---: | ---: | ---: |
| layer 1 | 192003084 | 192001740 | 1344 |
| layer 14 | 192008341 | 192007016 | 1325 |

本次调整：

1. `engram_pack.zig` 的 `Shard.range` 改为接收 `layer`，从 `engram_hash.layers` 读取 canonical offsets，并复制 `Host.headShard` 的列分配公式。原因是离线 payload、header、manifest 必须与运行时 hash-to-rank 所有权使用同一事实源。
2. `world > 1` 时强制 layer 和总行数匹配模型元数据，拒绝用错误模型或错误表规模生成“内部自洽但运行时缺行”的分片。`world = 1` 保持完整 `[0, rows)`，兼容通用单 rank 与小型合成测试。
3. pack、read-back verify、manifest build 和 manifest rebuild 全部复用同一 `Shard.range`，避免不同工具再次产生边界漂移。
4. 新测试覆盖 layer 1/14 的 TP2 golden boundary、连续无缝覆盖、每列首尾都落入所属 shard、真实哈希结果不越界、非 canonical TP 分片拒绝。
5. `tensorfold-preflight` 不再按总 rows 均分；它从 `engram-token-map.bin` 的 DSENGR01 元数据读取 24 个 offsets，按运行时列所有权计算期望 header。原因是旧 preflight 会拒绝正确分片、接受错误分片。

## 3. 构建与资产验证

定向 ARM64 测试：

```text
zig build test-dsv41-engram-pack dsv41-engram-pack -j2
14/14 PASS
```

修复后实际 header：

| 节点 | 文件 | `[lo, hi)` | payload 行数 |
| --- | --- | --- | ---: |
| rank 0 | layer 1 | `[0, 192001740)` | 192001740 |
| rank 1 | layer 1 | `[192001740, 384006168)` | 192004428 |
| rank 0 | layer 14 | `[0, 192007016)` | 192007016 |
| rank 1 | layer 14 | `[192007016, 384016682)` | 192009666 |

两 rank 均完成：

- `200004 rows read back through engram_rows, 0 differ`；
- 三个文件的 `SHA256SUMS` 全部通过；
- 更新后的布局感知 `tensorfold-preflight` 在两台 GB10 上均返回 PASS；
- 模型约 269.3 秒完成加载，rank 1 进入 `loaded, following rank 0`；
- `/health` 返回 HTTP 200、`warming=false`、版本 `1.2.1`、backend `cuda`。

最终可传输镜像：

```text
dist/tensorfold-dsv41-v1.2.1-sm121-arm64-engram-fix-final.tar
size: 1816691712 bytes
sha256: b59593ae9a6ed4386b90f74e582218035cb534576e3c635b625d331205171a7a
```

## 4. 作者 smoke 复测

| 项目 | 修复前 | 修复后 |
| --- | --- | --- |
| health | PASS | PASS |
| plain reply | PASS | PASS |
| no special-token leak | PASS | PASS |
| max_tokens | PASS | PASS |
| stop string | 引发 Engram 越界，双 rank 重启 | 不再崩溃；仍因重复 BOS 输出而 FAIL |
| stream | BLOCKED | PASS |
| tool call | BLOCKED | PASS |
| tool result round trip | BLOCKED | PASS |
| greedy repeat | BLOCKED | PASS |
| thinking on/off | BLOCKED | INFO，行为正确 |

修复后 smoke 结束时两个 rank 仍为 running、restart count 为 0。剩余 stop string 失败不应再归因于 Engram shard coverage；后续应单独审计该 prompt 的 logits/token 解码、BOS 抑制及 stop matcher 输入序列。

## 5. 作者 bench 复测与对比

当前部署和作者主页测试都满足“Engram 位于两台机器各自的本地磁盘”，因此本节以作者 README 的本地 Engram/sparkDash 结果作为主要部署性能基线。当前测试为固定 salt、thinking off、drafts off、context 32768、单并发；作者为 context 524288、drafts on、GPU 2200 MHz、RoCE、真实 prose/code prompt、最多 4 streams。

| 指标 | v1.2.1 修复后 | 作者双机、本地 Engram | 当前相对作者 | 解释 |
| --- | ---: | ---: | ---: | --- |
| 2K prefill | 126.59 tok/s，TTFT 15.65s | README 未公布 2K | 不计算 | 当前档位只用于稳定性门禁 |
| 8K prefill | 127.58 tok/s，TTFT 61.67s | 2,152.0 tok/s，TTFT 3.99s | throughput -94.1%，TTFT +1445.5% | 最接近的同规模本地磁盘结果 |
| 30K/32K prefill | 127.62 tok/s，TTFT 225.26s（实际约 28.7K tokens） | 2,153.6 tok/s，TTFT 15.40s（实际 32,788 tokens） | throughput -94.1%，TTFT +1362.7% | token 数不完全相同，作为近似规模对比 |
| 256-token decode / prose | 14.42 tok/s，TTFT 0.65s | 80.7 tok/s，TTFT 0.133s | throughput -82.1%，TTFT +385.9% | 作者为真实 prose、drafts on；当前为作者 bench 随机词、drafts off |
| 256-token decode / code | 14.42 tok/s，TTFT 0.65s | 118.1 tok/s，TTFT 0.213s | throughput -87.8%，TTFT +203.4% | 仅作为作者 code 上限参考 |
| errors | 0 | 作者主页结果成功 | — | 正确性路径已恢复 |

这轮结果证明修复解决了稳定性和覆盖问题，不证明已达到作者性能。性能差距应在 smoke 剩余问题修复之后，按 prompt chunk/prefill planning、graphs、draft、RoCE 和作者性能补丁逐项 A/B，不能通过放宽数据校验或隐藏异常换取吞吐。

### 5.1 作者保存 JSON 的脚本原生对比

另外直接执行了作者脚本自带的比较入口：

```bash
python -I tools/bench.py --compare \
  runs/base-ctx131072-rr16.json \
  dist/v121-author-bench-engram-fix.json
```

作者保存的 `base-ctx131072-rr16.json` 与本轮使用的是同一份 `bench.py`，但模型名、salt、context 和测试 token 目标不同；该保存结果也对应作者说明中的约 690–890 tok/s 慢速存储路径区间。当前部署明确为两节点本地 Engram，因此它只用于验证统计脚本口径，不能作为主要部署性能基线：

| 近似对应指标 | 作者保存结果 | v1.2.1 修复后 | 差异 |
| --- | ---: | ---: | ---: |
| 2K prefill throughput | 689.44 tok/s（2000 target） | 126.59 tok/s（2048 target） | -81.6% |
| 2K TTFT | 2.81s | 15.65s | +456.8% |
| 8K prefill throughput | 890.61 tok/s（8000 target） | 127.58 tok/s（8192 target） | -85.7% |
| 8K TTFT | 8.67s | 61.67s | +611.4% |
| decode throughput | 54.23 tok/s | 14.42 tok/s | -73.4% |
| decode TTFT | 0.17s | 0.65s | +271.2% |
| errors | 0 | 0 | 一致 |

作者脚本同时明确提示 model 和 salt 不同；本轮也没有执行作者基线中的 multi 1/2/4。因此这些百分比只用于量化当前性能差距，不能证明某一单独优化补丁的收益。

脚本身份：

- `tools/smoke.py`: `3dfc3c32b8fac5323df9f7436932a3aaec3d2dca2c6e06c1f103d51707b2edac`
- `tools/bench.py`: `c692a88f5e771ca6c6c4580737d4332cb294ab353035c5356da01c5c1a391689`

## 6. 证据

- `dist/v121-author-smoke-engram-fix.json`
- `dist/v121-author-bench-engram-fix.json`
- `dist/v121-author-bench-engram-fix.log`
- `docs/recipes/deepseek-v41-flash-1017-v121-engram-failure-audit.md`
- `zig/src/families/deepseek_v41/engram_pack.zig`
- `zig/src/families/deepseek_v41/engram_pack_test.zig`
- `packaging/dsv41-gb10/preflight.sh`

## 7. 下一步处理顺序

1. 先复现并审计 stop string 的重复 BOS：记录该请求逐 token id、top logits 和 detokenize 结果，确认是模型 logits、采样/BOS 屏蔽还是 HTTP stop matcher 之前的文本问题。
2. stop smoke 全通过后，再调整 `TF_DSV41_PREFILL_PIECES`/prompt chunk、graphs、draft 和 RoCE；每次只启用一个变量，重复固定 salt bench。
3. 性能配置稳定后补做 32K 边界、并发、故障注入和长稳测试，再决定是否清理保留的旧容器和旧 Engram 目录。
