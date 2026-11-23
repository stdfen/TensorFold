# 作者方案参考内容与 v1.2.1 新部署方案影响分析

## 1. 结论

当前 `tensorfold-dsv41:v1.2.1-sm121` 镜像没有直接 cherry-pick、复制或编译作者的 34 个 engine patch、2 个 inputs patch、自定义 `tf-dsv41-lanes`、rank-cache 或 kit。镜像中的推理二进制来自 TensorFold v1.2.1 提交 `1db3edbe7e692e83c7f5056754dc4e69e79a6c22`。

作者方案被参考的部分主要在部署层：双节点角色、服务参数、GB10/SM121 目标、容器运行方式和性能特性的启用顺序。此前“作者补丁与 v1.2.1 的追踪矩阵”是设计决策和验证清单，不代表这些补丁已经进入镜像。

新方案采用：

```text
EXL3 2.9bpw 量化目录 -> v1.2.1 在线 PACK
全量 FP8/FP4 目录     -> 离线生成 v1.2.1 TP2 Engram rank shards
作者优化              -> 在 v1.2.1 原生能力上逐项验证后启用
```

它修正了之前对 Engram 数据来源的判断，但不改变 v1.2.1 作为唯一运行时 ABI 的原则。

## 2. 当前镜像实际参考了作者方案的哪些内容

### 2.1 已进入镜像或启动封装的参考项

| 作者方案内容 | 当前镜像中的体现 | 代码位置 | 参考性质 |
|---|---|---|---|
| 双节点、leader/follower 角色 | rank 0 执行 `tensorfold-dsv41 serve`，rank 1 执行 `tf-dsv41 follow` | `packaging/dsv41-gb10/entrypoint.sh` | 参考拓扑；命令和协议使用 v1.2.1 原生实现 |
| GB10、SM121 | CUDA 13.0.1、`-Dsm=121`、linux/arm64 | `packaging/dsv41-gb10/Dockerfile` | 参考目标硬件 |
| host network、host IPC、GPU、memlock | `--network host --ipc host --gpus all --ulimit memlock=-1` | `packaging/dsv41-gb10/run-node.sh` | 参考容器部署形态 |
| 服务名和 HTTP 参数 | `TF_SERVED_NAME`、host、port | `entrypoint.sh` | 参数映射，不是作者代码 |
| 生成长度和采样参数 | max-tokens=32768、temperature=1.0、top-p=0.95、top-k=20 | `entrypoint.sh`、`env.example` | 采用作者默认值作为可配置初值 |
| thinking 参数 | thinking、reasoning-effort、thinking-budget | `entrypoint.sh` | 采用作者启动接口语义，调用 v1.2.1 CLI |
| 双 rank 启动前检查 | world/rank/master、模型、资产、CUDA/NCCL 检查 | `entrypoint.sh`、`preflight.sh` | 根据作者双机部署风险重新设计，代码为本项目新增 |
| 模型和资产只读挂载 | `/models/model`、`/assets` | Dockerfile、run-node.sh | 部署原则参考，路径和校验为本项目设计 |

### 2.2 没有进入当前镜像的作者内容

| 未采用内容 | 原因 |
|---|---|
| `tf-dsv41-lanes` follower 命令 | 与 v1.2.1 `tf-dsv41 follow PACK ASSETS` ABI 不同 |
| rank-cache、kit、`/rank-cache` | v1.2.1 直接读取 EXL3 PACK，并有自己的 prepared/AOT 体系 |
| 作者 `engram-host.bin` token-map | v1.2.1 强制使用 `DSENGR01` `engram-token-map.bin` |
| 作者旧 AOT/captured kit | kernel signature、planner 和运行 profile 不可证明与 v1.2.1 一致 |
| 作者 RDMA runtime 实现 | v1.2.1 使用 `fabric/tp` 的 NCCL/RoCE/fate/planlink |
| 34 个 engine patch | 均未直接合并；按“已有等价、架构替代、按需适配”处理 |
| 2 个 inputs mixed-pack patch | 服务 EAQ v1.1 mixed BF16，不属于当前 EXL3 2.9bpw 首启路径 |
| 作者 start.sh/config.sh/nodes.sh | 只提取参数和启用顺序，没有复制进镜像 |

因此，“参考作者方案”不能表述为“镜像包含作者 patched TensorFold”。准确说法是：“镜像使用 v1.2.1 推理实现，并参考作者的部署经验设计容器参数和后续优化矩阵”。

## 3. 新部署方案对之前详细设计的影响

### 3.1 不受影响的设计

以下设计继续成立：

- 使用 v1.2.1 独立 `tensorfold-dsv41` CUDA engine，不解除通用 release binary 的 DeepSeek gate。
- TP world 固定为 2，rank 0 提供 HTTP，rank 1 只 follow。
- 第一阶段使用 NCCL、32K context、单 slot、关闭 drafts 建立 token 正确性基线。
- v1.2.1 原生 KV split/session/fate/graph/API 架构优先于作者旧 patch。
- 模型、Engram、AOT 等大文件不写入镜像，使用只读挂载和 manifest。
- RoCE、drafts、graph、side streams 和 1M context 在正确性基线后逐项开启。

### 3.2 被修正的设计

| 之前判断 | 新判断 | 影响 |
|---|---|---|
| EXL3 索引缺少 Engram embed，因此必须重建在线 PACK | EXL3 在线 PACK 可以保持不变；embed 只需存在于全量权重，供离线 Engram packer 使用 | 删除“重建量化 PACK”步骤，降低 loader 改动范围 |
| 作者独立 `/engram` 整体视为旧 ABI | 现存大文件的 header 是 v1.2.1 可识别的 `DSV41EN1`，但它们是 world=4 分片 | 可作为格式和数值参考，不可直接作为 TP2 文件使用 |
| 模型启动主要阻断是主 PACK 不完整 | 主要阻断变为 TP2 rank-aware Engram 生成、token-map、RoPE 和 profile-matched AOT | 开发焦点从 checkpoint loader 转移到资产工具链 |
| Engram 需要合入 `/assets` | Engram 应独立挂载到 `TF_DSV41_ENGRAM_DIR=/engram` | Dockerfile、run-node、entrypoint、preflight 都要调整 |

### 3.3 新增的必要代码调整

| 调整 | 修改位置 | 原因 | 是否来自作者 patch |
|---|---|---|---|
| `tf-dsv41-engram-pack --rank/--world` | `zig/tools/dsv41_engram_pack.zig`、`engram_pack.zig`、manifest/tests | 当前 v1.2.1 只生成 `r0of1`，TP2 reader 打开 `r0of2/r1of2` | 否；由新数据布局直接产生 |
| 将 asset/token-map/engram-pack 工具放入 preparation image | `packaging/dsv41-gb10/Dockerfile` | 两节点需从 `/data/models` 生成原生资产 | 否 |
| `/engram` 只读挂载 | Dockerfile、run-node.sh | 在线 reader 通过 `TF_DSV41_ENGRAM_DIR` 读取 NVMe 行分片 | 参考作者独立 Engram 挂载思想，但使用 v1.2.1 格式 |
| Engram header/range/hash preflight | entrypoint/preflight | 防止 rank/world 或行区间错误导致 collective 挂起 | 与作者 0035 的故障目标一致，代码需按 v1.2.1 实现 |
| 资产来源 manifest | 新增准备脚本/manifest | 证明 EXL3 主干、全量 Engram、tokenizer、AOT 属于同一模型/profile | 否；属于可追溯性增强 |

## 4. 新方案对 34+2 补丁决策的影响

### 4.1 优先级提升为首启门禁

| 作者补丁 | 原决策 | 新影响 | 新决策 |
|---|---|---|---|
| 0012 base fixes | v1.2.1 架构替代，故障验证 | 新生成的 Engram 会实际走 read-ahead、cancel 和 recovery | 不移植；首启必须做 cancel/read failure 验证 |
| 0017 kit row split | AOT/profile 决定 | 现在必须为 v1.2.1 首启 profile 重新生成 AOT | 不移植旧 split；AOT needs 完整性成为启动门禁 |
| 0020 pair broken | 复用+验证 | Engram 或 AOT 一端失败时必须让另一端退出 | 不移植；rank kill/文件损坏 fail-fast 成为首启验收 |
| 0035 Engram failure paths | 架构替代、待验证 | TP2 rank shard 是新生成资产，EOF/range/短文件风险成为现实 | 不移植；短表、坏 header、I/O error 注入成为强制测试 |
| 0037 early rendezvous | 验证/适配 | 两节点读取约 197 GiB PACK 和各自 Engram，加载时序可能不对称 | 首启必须验证 follower 先启动和 bootstrap 超时 |
| 0038 aligned index views | profile 决定 | 新 AOT 必须覆盖首启实际 alignment/row shape | AOT needs/calls check 成为首启门禁 |

### 4.2 仍放在正确性基线之后

| 作者补丁/思路 | 影响结论 |
|---|---|
| 0001 indexer split | 32K 首启不需要；131K/1M context profile 后再决定 |
| 0005 prefill interleave | 首启保持 v1.2.1 默认/pieces；混合负载性能数据触发 |
| 0008 pool/window | 使用 v1.2.1 KV pool/slots；不因 Engram 来源变化而修改 |
| 0010 RDMA idle nap | NCCL 基线无影响；切 RoCE 后测 CPU/唤醒延迟 |
| 0011 parked pages budget | 仍需在 GB10 实测；不会阻塞资产生成 |
| 0013–0019 spill/KV/session | v1.2.1 架构替代结论不变，首启不打开 session disk |
| 0022 KV shard ring | NCCL baseline 后比较 v1.2.1 KV split × RoCE |
| 0023 prepare memory | 1M/并发长 prompt 阶段再验证 |
| 0033 graph cap | v1.2.1 已有等价/超集；首启可关闭 graph 降低变量 |

### 4.3 完全不受新方案影响

- 0004、0027、0028、0032、0036、0046：API、文本、keepalive、metrics 和 HTTP 连接行为。
- 0029–0031：Anthropic 客户端兼容差异；不阻塞 OpenAI API 模型启动。
- 0024–0026 和 inputs 0001–0002：只面向 EAQ v1.1 mixed BF16，当前 EXL3 2.9bpw 不采用。

## 5. 对当前镜像状态的影响

当前镜像仍可作为“v1.2.1 SM121 二进制构建通过”的 bootstrap image，但不是新方案的最终部署镜像，因为：

1. 没有复制 `tf-dsv41-engram-pack`、`tf-dsv41-assets` 和 `tf-dsv41-token-map` 到运行/准备阶段。
2. 没有声明 `TF_DSV41_ENGRAM_DIR` 或 `/engram` volume。
3. `run-node.sh` 不接受 Engram 目录参数。
4. entrypoint/preflight 不验证 TP2 Engram 文件。
5. 已导出的 tar 对应上述旧入口，完成这些修改后必须重新构建、导出和计算 SHA-256。

镜像内的 v1.2.1 CUDA server/follower 二进制不因新方案失效；需要重做的是工具和部署层。

## 6. 修订后的实施顺序

1. 为 Engram packer 实现 `--rank/--world` 和 header/range/manifest tests。
2. 修改 Dockerfile，产出 preparation target 和最终 runtime target。
3. 更新 entrypoint、preflight、run-node 和 env.example，加入 `/engram`。
4. 用全量权重分别生成 TP2 rank 0/1 Engram，并执行 sampled verify。
5. 用 v1.2.1 生成 DSENGR01 token-map、RoPE 和首启 profile AOT。
6. 重建镜像并在两个节点执行 preflight。
7. 先 rank 1、后 rank 0，以 NCCL/32K/slot1/no-drafts 启动。
8. 完成 token golden、Engram 故障注入和 rank fail-fast。
9. 再参考作者方案逐项开启 Engram AIO/prefetch、drafts、graphs、side streams 和 RoCE。

## 7. 证据边界

本分析基于：

- 当前 Dockerfile、entrypoint、preflight、run-node 和最终镜像构建记录；
- 作者 34+2 补丁逐代码路径审计；
- 两个节点 `/data/models` 的 config、index、safetensors header 和 tokenizer 哈希只读检查；
- v1.2.1 Engram packer、row reader、token-map、AOT 和 TP 源码。

当前验证等级仍为 `build_verified`。在新 Engram 工具、资产和双节点启动完成前，不能提升为 `integration_verified`。
