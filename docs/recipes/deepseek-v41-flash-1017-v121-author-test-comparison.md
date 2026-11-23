# TensorFold v1.2.1 双 GB10：作者测试脚本执行与结果对比

## 1. 结论

本次直接使用参考仓库 `MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks` 的原始 `tools/smoke.py` 和 `tools/bench.py`，对当前 TensorFold v1.2.1 双机服务进行测试。脚本未做修改，远端文件 SHA-256 与本地作者仓库一致。

当前结果不能判定为可交付部署：基础健康检查、普通回复、特殊 token 防泄漏和 `max_tokens` 通过，但 `stop string` 用例触发 rank1 的 `EngramRowOutsideShard`，随后 Tensor Parallel fail-fast 使两端容器成对重启。服务恢复后，`bench.py` 的首个 2K prefill 又复现同一故障。因此没有产生合法性能样本，性能对比在正确性门禁处被阻断。

这也修正了此前“短提示可生成”的结论边界：它证明模型能加载并完成部分请求，但不能证明任意提示下的 Engram 访问完整或服务稳定。

## 2. 测试对象与可复现性

| 项目 | 当前验证环境 | 作者参考环境 |
| --- | --- | --- |
| TensorFold | v1.2.1 基线 `1db3edbe7` 加当前适配代码 | 作者仓库固化的 TensorFold 补丁集合 |
| 模型 | `deepseek-v4.1-flash-1017` | `DeepSeek-V4.1-Flash-1017` |
| 节点 | `intellif-74` rank0、`intellif-15` rank1 | 2 台 DGX Spark / GB10 |
| 并行 | TP=2，API lane=1 | TP=2，最多 4 streams（公开性能表） |
| 上下文 | 32,768 | 524,288（公开主性能表） |
| Draft | 关闭 | 开启（公开主性能表） |
| 通信 | NCCL Ethernet，RoCE fast path 关闭 | 作者调优网络路径 |
| 镜像 | `tensorfold-dsv41:v1.2.1-sm121-native` | 作者 recipe 镜像 |

作者脚本来源版本：参考仓库 `main`，commit `f424dfe`。

| 文件 | SHA-256 |
| --- | --- |
| `tools/smoke.py` | `3dfc3c32b8fac5323df9f7436932a3aaec3d2dca2c6e06c1f103d51707b2edac` |
| `tools/bench.py` | `c692a88f5e771ca6c6c4580737d4332cb294ab353035c5356da01c5c1a391689` |

## 3. 作者 smoke 脚本结果

执行命令：

```bash
python3 -I tools/smoke.py \
  --url http://127.0.0.1:8080 \
  --model deepseek-v4.1-flash-1017 \
  --json v121-author-smoke.json
```

| 检查项 | 当前结果 | 作者项目记录 | 分析 |
| --- | --- | --- | --- |
| health | PASS | PASS | v1.2.1 API 健康，模型已加载，lane=1 |
| plain reply | PASS | PASS | 返回 `Hello!`，3 completion tokens，`finish_reason=stop` |
| no special-token leak | PASS | PASS | 普通响应未泄漏内部特殊 token |
| max_tokens | PASS | PASS | 5 tokens，`finish_reason=length` |
| stop string | **FAIL** | PASS | HTTP 500 `Aborted`；这是本次首个独立失败 |
| stream | BLOCKED | PASS | 引擎已进入不可用状态，HTTP 503；不能归因于 stream 功能本身 |
| tool call / round trip | BLOCKED | PASS | 同一连锁 503；本轮未进入 grammar/tool 能力判定 |
| greedy repeat | BLOCKED | PASS | 同一连锁 503 |
| thinking on | BLOCKED | PASS | 同一连锁 503 |
| thinking off | BLOCKED | PASS | 同一连锁 503 |

作者项目文档记录 `tools/smoke.py` 全部通过。当前脚本打印出的 6 个 FAIL 中，只有 `stop string` 是在健康引擎上发生的首发故障；后五项是在 fail-fast 已经触发后收到 503，必须标为 BLOCKED，不能误报为五个新的兼容性缺陷。

## 4. 作者 bench 脚本结果

考虑当前最大上下文 32K、lane=1，本次选择与作者 `tools/bench.py` 公布结果可直接对齐的 2K、8K、30K prefill 和单流 256-token decode；未执行超出当前配置能力的 100K、4-stream 和 needle 测试。

```bash
python3 -I tools/bench.py \
  --url http://127.0.0.1:8080 \
  --model deepseek-v4.1-flash-1017 \
  --suite prefill,decode \
  --prefill-sizes 2048,8192,30000 \
  --decode-tokens 256 \
  --reps 2 \
  --salt v121-native-author-compare-20261103 \
  --health --metrics \
  --out v121-author-bench.json
```

| 指标 | 当前结果 | 作者 `tools/bench.py` 记录 | 是否可比较 |
| --- | --- | --- | --- |
| Prefill 2K | 首次请求失败；无 token/usage | 约 1,500–1,600 tok/s，TTFT 1.2s | 否 |
| Prefill 8K | 连锁 HTTP 503 | 约 1,960 tok/s，TTFT 3.9s | 否 |
| Prefill 30K | 连锁 HTTP 503 | 约 2,000 tok/s，TTFT 14.4s | 否 |
| Decode，1 stream | 连锁 HTTP 503 | 55–61 tok/s | 否 |
| Decode，4 streams | 未执行，当前 lane=1 | 113–117 tok/s aggregate | 配置不等价 |
| Prefill 100K / needle | 未执行，超过 32K context | 约 1,950 tok/s / 作者另有长上下文记录 | 配置不支持 |

首个 2K run 的服务指标显示请求已接收 1,987 个 prompt tokens，约 6.90 秒后在首 token 前失败；随后双 rank 第二次成对重启。由于 `bench.py` summary 只有 `errors=8`，不存在可以用于百分比对比的 TTFT、prefill tok/s 或 decode tok/s。用此前 3-token 短请求的 20 tok/s 代替作者 256-token decode 结果也不成立，因为测量长度、脚本和稳态区间均不同。

作者 README 的 sparkDash 主表（例如 8K prefill 2,152 tok/s、单流 prose decode 80.7 tok/s）使用 524,288 context、drafts on、固定 GPU 频率和真实文本，与本轮 `bench.py` 随机词、drafts off 配置不同，只能作为旁证，不能作为直接分母。

## 5. 故障链与代码含义

实际故障链如下：

```text
作者请求触发未覆盖的 Engram row
        ↓
rank1: EngramRowOutsideShard（layer 1 / layer 14）
        ↓
rank1 follower 退出
        ↓
rank0 fate channel 关闭，TP fail-fast 返回 Aborted
        ↓
当前请求 HTTP 500 或流无 token，后续请求 HTTP 503
        ↓
Docker restart policy 使两个 rank 成对重启并重新加载约 95 GiB/rank 权重
```

该结果直接关联当前 v1.2.1 适配中的“按 TP rank 生成/加载 Engram shard”改动。静态 manifest、文件校验和短请求均可通过，但运行时 hash 得到的 row 并不保证属于执行该层读取的 rank 本地 shard；或者当前 packer 与 runtime 对 row 的 shard 映射定义不一致。因而下一步应优先审计：

1. `zig/tools/dsv41_engram_pack.zig` 的 row 分片规则；
2. `zig/src/families/deepseek_v41/engram_pack.zig` 的 shard 起止区间和 manifest 写入；
3. runtime Engram row 计算、TP rank 选择与 `EngramRowOutsideShard` 边界检查；
4. 作者补丁中 Engram 表在两机的实际放置策略，确认是否本来要求每个 rank 持有完整表，而不是按 rank 切行。

在此问题修复前，不应继续性能调优或扩大 context/parallel，因为正确性和可用性门禁尚未通过。

## 6. 容器影响与证据

- smoke 前：双 rank running，restart count 0。
- smoke 后：双 rank restart count 1；日志明确包含 `EngramRowOutsideShard` 与 TP fail-fast。
- bench 前：服务完成自动恢复，`/health` 为 200。
- bench 后：双 rank restart count 2；约 4 分钟后自动重新加载完成，rank0 `/health` 恢复 HTTP 200，rank1 恢复 `loaded, following rank 0`。未人工停止、删除或替换容器。
- 原始结果保存在本地 `dist/v121-author-smoke.json` 与 `dist/v121-author-bench.json`，其中不包含凭据。

## 7. 验收判定

| 门禁 | 判定 | 原因 |
| --- | --- | --- |
| 模型加载 | PASS | 双 rank 可加载并健康响应 |
| 最小文本生成 | PASS | 普通短请求与 `max_tokens` 通过 |
| 作者完整 smoke | FAIL | `stop string` 触发 Engram 越界并使集群重启 |
| 长提示稳定性 | FAIL | 作者 2K prefill 首次 run 即复现 |
| 性能可比性 | BLOCKED | 无合法 benchmark 样本 |
| 可部署验收 | FAIL | 任意合法提示可能触发缺失 Engram row，服务会整体重启 |

最终结论：当前镜像属于“可加载、部分短请求可运行”的开发验证状态，不是作者方案同等的稳定可部署状态。应先修复 Engram 数据布局/运行时寻址一致性，然后从作者 smoke 全通过开始，重新执行固定 salt 的 prefill/decode 对比。
