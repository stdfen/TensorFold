# DeepSeek V4.1 Flash 1017 作者 34+2 补丁代码路径审计

## 1. 审计结论

本审计逐个读取作者仓库 34 个 engine patch 和 2 个 inputs/rank-cache patch 的 diff 文件、修改文件和 hunk 上下文，并在 TensorFold `release/v1.2.1` 中定位对应源码、数据结构、环境变量和测试。

审计基线：

- 作者 engine：`bertholomus/TensorFold@fbbc97132cd004237d9887b8d3185896efaedc5c` 加顺序 patch。
- 作者 inputs：`bertholomus/TensorFold@59402086381a42c74d53053443224eec2790b4c5` 加 2 个 patch。
- 当前目标：`stdfen/TensorFold release/v1.2.1@1db3edbe7e692e83c7f5056754dc4e69e79a6c22`。

代码级结论分为：

- **等价/超集**：v1.2.1 具有相同根因处理，通常还包含更完整的测试或通用抽象。
- **架构替代**：目标相同，但数据结构、协议或模块已经重写，不能移植旧 patch。
- **部分覆盖**：v1.2.1 有相关能力，但缺少作者 patch 的某个策略或需要硬件确认。
- **明确差异**：v1.2.1 源码仍保留作者 patch 要消除的行为。
- **不适用 EAQ v1**：只服务 EAQ v1.1 mixed pack，不进入首轮。

注意：代码级静态等价不等于 GB10 集成通过；所有结论仍需对应 receipt。

## 2. 逐补丁代码审计：计算、调度、内存与缓存

| 补丁 | 作者代码路径和实际修改 | 改动原因 | v1.2.1 对应代码 | 代码级结论 | 实施与验证 |
| --- | --- | --- | --- | --- | --- |
| 0001 indexer split | `dsv41/model.zig` 读取 `TF_DS_IDX_SPLIT`；`prompt.zig::indexer` 将 row blocks 按 rank 分半，各自执行 selection，再 all-gather；以 compressed key 数和 scratch 容量决定是否 split | 1M prefill 的 CSA2 indexer 对大量压缩 keys 做 score/top-k；两个 rank 重复计算会在约 131K 后成为瓶颈 | `deepseek_v41/block_prefill.zig`、`block_index.zig`、`serve_knobs.zig` 已有 stream top-k、`INDEX_BUDGET_MIB`、`INDEX_BOUND`，但没有同样的 TP2 row-half/all-gather 策略 | **部分覆盖**。内存分块已重写，跨 rank 计算拆分未证明存在 | 不直接移植 `prompt.zig` patch；先 profile 128K/256K/512K。若 index kernel 主导，基于新 block planner 增加 rank-local row range、selection gather 和 parity test |
| 0005 prefill interleave | `core/lane_host.zig` 增加 `between()` hook、其他 stream decode 和 cancel；`lanes/stream.zig` 保存 Between；`dsv41/lanes.zig` 在 prompt chunks 间调用 | 长 prompt 的整段 prefill 会让在线 decode 数分钟无进展 | `deepseek_v41/sched_host.zig`、`core/lane_rounds.zig`、`lane_host_fill.zig` 以 `TF_DSV41_PREFILL_PIECES=1` 在 decode rounds 间调度 prompt pieces | **架构替代/超集目标**。v1.2.1 把 family hook 提升为 round planner，不应叠加旧 Between | 启用 pieces 并做“长 prompt + 活跃 chat + cancel”测试。只有 pieces 仍不满足延迟目标才在新 scheduler 内调优，不恢复旧 hook |
| 0008 pool/window | 新增 `dsv41/pool.zig`，分离共享 extent pool 和单请求 window；`model.zig` 按 window 设置 RoPE rows；`native.zig` 解析 `TF_DS_POOL` | 多个长会话共享容量需要大于单请求 1M window，同时 RoPE 不应按总 pool 扩张 | `deepseek_v41/cuda_kv.zig` 的 `TF_DSV41_POOL_TOKENS`、paged pool；`sessions_gpu.zig` 管 resident/disk；`kv/layout.zig` 管 page ownership | **架构替代**。旧连续 extent pool 已被 paged KV/session 模型替代 | 用 pool pages × page size 验证总容量和单请求 1M 上限；不要增加旧 `TF_DS_WINDOW` 双语义变量 |
| 0010 RDMA idle nap | `dsv41/rdma.zig::Ring.proxy` 增加 monotonic idle 计时、`nanosleep`、Linux timer slack；解析 `TF_RDMA_IDLE_S/NAP_US` | 旧 RDMA proxy 空闲时 busy-spin，长期占满每 rank 一个 CPU core | v1.2.1 使用 `fabric/tp/roce*.zig` 的新 proxy/verbs/multi-HCA 路径；未发现等价 idle-nap 配置 | **明确缺少该策略，但非正确性缺口** | 在 v1.2.1 RoCE 上测 idle CPU、首轮唤醒延迟和 shutdown。若确有问题，在 `roce_host`/proxy 循环实现通用 backoff，不移植旧 Ring 代码 |
| 0011 parked pages budget | `native/cuda.zig::pool/openWith` 读取 `/proc/zoneinfo`；`native/cuda_memory.zig::parkedBytes/counting` 将超过 resting level 的 per-CPU pages 加回 MemAvailable | 镜像 load/save 后 Linux per-CPU page lists 暂存数 GiB，单看 MemAvailable 会错误拒绝 GB10 启动 | v1.2.1 `deepseek_v41/boot_mem.zig`、`kv/budget.zig` 有 boot growth、serving dip 和 hard floor，但未找到 `parkedBytes` 等价解析 | **部分覆盖**。预算模型更完整，但 parked-pages 修正未见 | 先在 GB10 镜像构建后对比 `/proc/meminfo`、`zoneinfo`、启动判定。只有误拒绝可复现时，将 parked 统计作为 Linux-specific meter 输入并加 parser tests |
| 0012 base fixes | `model.zig` 在新 prompt pass 前 drop 旧 Engram read-ahead，并收敛 cleanup；`rdma.zig` 将 `prctl` 限定 Linux | 被取消/中断 prompt 的异步 Engram rows 可能污染下一 prompt；macOS host tests 不应链接 Linux `prctl` | v1.2.1 拆为 `engram_rows_*`、`engram_recovery_test.zig`、`engram_gate` 和 portable platform files | **架构替代，静态上覆盖目标** | 运行 cancel/resume/read-failure tests，并在 Linux/非 Linux host tests 验证资源回收；不移植旧 model/rdma hunk |
| 0013 spill store | 新增 `core/spill/{format,fsio,slots,store}.zig` 及大量恢复/损坏/配额测试，形成 2048-token durable block store | 长会话和 forks 的 KV 状态需要落盘、复用、恢复和容量治理 | v1.2.1 使用 `deepseek_v41/kv/sess.zig`、`sessions_gpu.zig`、core KV pool/session store | **架构替代**。持久化协议不同 | 以 v1.2.1 session manifest、CRC/identity、容量和恢复测试为准；禁止混用旧 spill 文件 |
| 0014 cache tier | 扩展 `engine_api.Open/Status`、`LaneHost`、`prompt_cache`，加入 durable probe/restore/background jobs；server 暴露 spill flags/metrics | prompt cache 只在内存中无法跨重启，也无法把恢复 I/O 与 decode 并行 | v1.2.1 的 `sessions_gpu.zig`、`kv/sched.zig`、`kv/sess.zig`、LaneHost round scheduler 承担这些职责 | **架构替代** | 验证 fresh=resume、I/O 与 decode 并行、失败回滚和 metrics；不复制旧 prompt_cache Tier API |
| 0015 KV shard attention | 新增 `parts.zig`、`shard.zig`、`shard_ops.zig` 和 CUDA gather/fetch/union kernels；修改 prompt/round attention，使 compressed latents 按 pool part 只存一份 | TP2 每 rank 复制全部 compressed KV 浪费统一内存，限制长会话并发 | v1.2.1 原生 `kv/layout.zig`、`kv/split.zig`、`cuda_kv.zig`，支持 dense/compact/dedup/history/early | **架构替代且 v1.2.1 功能更细** | 以 replicated 为 golden，对 split 的 prefill/decode、各 ratio/source layer 做 byte/token parity；不移植 3000+ 行旧 parts/shard |
| 0016 spill blocks | 新增 `fingerprint`、`rank_identity/wire`、`spill_state/tier`；LaneHost 和 follower 同步 keep/restore；模型加载时协商 layout 和特征 | 双 rank durable state 必须以相同模型、布局、代际和容量恢复，否则会读错 KV | v1.2.1 `kv/sess.zig`、`sessions_gpu.zig`、`kv/session_slots_test.zig`、TP config digest | **架构替代** | 验证两 rank manifest、缺 rank 数据、错误 fingerprint、重启恢复；失败必须 fail closed |
| 0017 kit row split | 工具端生成/校验 kit variants；`tri_norm.rowmmGatePieces` 在 kit 无 >16-row variant 时拆为 16-row；新增 `dsv41_port.cu` plane staging | 作者固定 kit 只有至多 16 rows 的 MoE gate variant，32-row round 会 `MissingTritonVariant` | v1.2.1 用 `aot_needs.zig` 枚举真实 shape，`cuda/aot.zig` 严格选择 specialization，资产为 `sm_121` 重建 | **架构替代；是否缺 variant 取决于新资产** | 用最终 slots/draft/context 生成 needs，检查每个 MoE row shape。若缺失，补 reference compiler/needs，不在 runtime 静默拆旧 kit |
| 0018 KV shard lanes | 新增 `core/lanes/kv_parts.zig` 和 `dsv41/kv_lanes.zig`；LaneHost 参与 part placement、carry、borrow、twins、digest；扩展多 stream follower 协议 | 0015 只解决单 stream attention，生产并发还需要 ownership、迁移和请求调度 | v1.2.1 `kv/sched.zig`、`kv/session_slots*`、`kv/split*` 已把 split 纳入多 slot scheduler | **架构替代** | 1/2/4 slots 下验证 owner、carry、borrow、cancel 和 token parity；不要移植旧 lanes protocol |
| 0019 spill + KV shard | 修改 spill manifest owner layout、rank wire epoch、Kv/Follower/Leader restore，将 durable blocks 恢复到原 owner part | spill 和 sharding 独立可用不代表组合安全；恢复到错误 rank 会静默破坏 attention | v1.2.1 session + split layout、session slot split golden tests、union plan tests | **架构替代** | 组合测试 disk park + split + restart；删除 rank1 manifest/改变 layout 必须拒绝恢复 |
| 0021 spill pictures | 扩展 request/cache/spill manifest，记录图片 span 和完整 SHA-256；chat/prompt 生成 media identity | 图片 token 由内容和位置决定，只有文本 token 相同不足以复用缓存 | v1.2.1 vision/session 路径和 media identity；需以 image resume receipt 确认 | **架构替代，需集成验证** | 同图同位置命中；换图或换位置不得命中；图片会话重启恢复等于 fresh |
| 0022 KV shard ring | 新增 `shard_ring.zig`、ring CUDA kernels 和 RDMA peer ops；decode 只交换实际跨 owner rows，并可 overlap | NCCL 上 sharded decode 交换开销抵消内存收益；小而稀疏的 rows 更适合 RoCE ring | v1.2.1 `kv/split.zig` 的 compact/dedup/history/early 与 `fabric/tp/roce`，另有 `kv/split_roce_test.zig` | **架构替代/超集目标** | 比较 dense/compact/history × NCCL/RoCE；确认严格 RoCE、bytes 计数和 token parity |
| 0023 prepare memory | template/tokenizer 改为可释放 scratch；新增 `server/prepare_gate.zig` 和 `TENSORFOLD_PREPARE_FLOOR_GIB`，长请求 prepare 在低内存时等待 | 多个约 1M resume 同时 render/tokenize 会让 rank0 MemAvailable 归零，统一内存机器可能冻结 | v1.2.1 有 LaneHost memory admission、DeepSeek `kv/budget` hard floor，但未发现完全相同的 Linux prepare floor；template/tokenizer 实现已重构 | **部分覆盖** | 在 8 个长 resume 压测中分别记录 prepare scratch 和 admission。若 API prepare 仍越过 floor，在 server prepare 前增加通用 Linux memory gate |
| 0033 graph cap | 新增 family `graph_cache.zig` LRU；`round.Graphs` 超 cap 释放最久未 replay graph；rank1 CLI 同步 cap | rows/context bucket/stretch 组合长期增长，每个 captured graph 占 device+host memory，最终触发 memguard | v1.2.1 `cuda/graph_cache.zig` 已有 `Settings.max=48`、LRU `evictOldest`、memory-floor shedding、rank agreement、metrics | **等价且 v1.2.1 为通用超集** | 不移植。验证 DeepSeek family 传入 Settings、两 rank digest、长期 held≤max、floor eviction 和 token parity |
| 0034 rebase followups | `lane_host.zig` 在 piece 运行时标记 passing；`server/prompt.zig` 拒绝准备后 0-token 图片 | 恢复状态若误判为普通 admission 会破坏生命周期；0-token 图片导致 `tokens-1` 下溢 | v1.2.1 LaneHost 已拆分 fill/admission 状态；vision admission 独立 | **需针对性验证，不宜移植旧行号 patch** | cancellation/piece restore ownership test；0-token image 必须返回请求错误而非 panic |
| 0037 early rendezvous | `link.zig` 拆 `listen/accept`，`model.zig` 在加载权重前 open port；`prompt2d` 调整 split | follower 加载较快时，rank0 还未监听，300s 重试后错误退出 | v1.2.1 `fabric/tp/bootstrap.zig`、`planlink.zig`、`Config.link_port/connect_timeout`；bootstrap 是独立 TP 层 | **架构替代，需启动时序验证** | 人为制造 rank0 慢加载、rank1 快加载，确认 early listener/backlog；若仍失败，修 bootstrap 生命周期而非只加超时 |
| 0038 aligned index views | `prompt.zig::indexer` 将 blocks 对齐 16 rows；`replayCut` 保证位置指针 16-byte alignment；AIO gather 检查 | captured Triton variant带 div16 假设，奇数 block 或 replay offset 会 `MissingTritonVariant` | v1.2.1 `cuda/aot.zig` 会按 pointer/count alignment 选 variant；DeepSeek block/index planner 已重写 | **部分覆盖，需 shape 审计** | 用 `aot-needs` 枚举 odd cut/long prefill，检查 aligned/fallback variants；缺失时修 planner 或生成 variant，不复用旧 image |

## 3. 逐补丁代码审计：可靠性、API 与可观测性

| 补丁 | 作者代码路径和实际修改 | 改动原因 | v1.2.1 对应代码 | 代码级结论 | 实施与验证 |
| --- | --- | --- | --- | --- | --- |
| 0004 call gate | 新增 `core/lanes/call_gate.zig::Gate` 状态机；扩展 `engine_api.CallLex`、Stream Spec；server 提供 token text/encode；engine 在 think block 后强制 call opener 和 offered name | `tool_choice=required/named` 不能只靠模板提示，否则模型可能回答普通文本或调用错误函数 | v1.2.1 `core/engine_api.CallGate`、server `tool_specs/tool_grammar`、DeepSeek template/DSML fixtures | **等价/已上游化** | 不移植。运行 required、named、auto、none，覆盖 open think block、stream 和 malformed call |
| 0020 pair broken | 扩展 comm/link/rdma/model/native/LaneHost fatal；监控进度和 timeout；rank break 后 health 503、drain、exit 75 | 一 rank 死亡或挂起后，另一 rank 继续等待 collective 会形成半活服务和永久请求 | v1.2.1 `fabric/tp/fate.zig`、fail-fast config、DeepSeek `model_supervisor`、`round_progress`/stall exit | **架构替代/超集** | kill、SIGSTOP、断链和 kernel stall；所有请求终止、health 失败、双 rank 一致退出并可重启 |
| 0027 tool-call prose | `server/reply_text.zig` 调整 prose cut；`openai.zig` 用正确 content；补 prose+call test | DeepSeek spaced DSML 前的正文不应因工具调用变成 `null`，stream 与 whole 也应一致，否则下一轮 prompt 不匹配 | v1.2.1 `reply_text.zig` 识别 spaced DSML；`openai.zig::finish` 调 `parsedContent`；`stream_dsml_test` 有 DeepSeek V4.1 prose/call whole+stream | **等价且测试更完整** | 不移植。运行现有 `stream_dsml_test` 和真实 checkpoint round-trip |
| 0028 plain request text | tokenizer 增加 plain marking/encode；messages 遍历用户和工具字符串，阻止 added/special token match；raw completions保留特殊 token | 用户上传文本可能含 role、DSML、`</think>`，若按控制 token 编码会注入模板结构 | v1.2.1 `model_text.VTable.encode_plain/encode_masked`、tokenizer plain path、server request range handling | **架构替代/等价目标** | 对消息、工具描述、网页和代码中的控制串做 token fixture；chat 必须 plain，raw completion 保持既有语义 |
| 0029 Anthropic budget default | `anthropic_translate.zig` 允许 enabled thinking 缺 `budget_tokens`，交给 server 默认 thinking budget | OpenAI 路由能使用服务默认，Anthropic 路由却直接 400，两个前端语义不一致 | v1.2.1 当前仍要求 `budget != null && budget < max_tokens` | **明确差异** | 若目标要求兼容作者行为，做独立 API patch：缺省时不写 `thinking_budget`；显式非法值继续拒绝；增加 translation tests |
| 0030 Anthropic context lenient | 放宽 `context_management.edits`：接受任意 `clear_thinking_*` 与 keep，而其他 edit 仍拒绝 | 客户端会发送不同日期/`keep:none`，云 API 接受但旧本地服务 400 | v1.2.1 当前只接受精确 `clear_thinking_20251015 + keep:all` | **明确差异** | 先确认产品兼容范围；若启用，按 type prefix + allowlisted keep 验证，未知 edit 继续 fail closed |
| 0031 Anthropic system anywhere | 删除 system message 必须处于 user 后、assistant 前的邻接限制，保留 `clear_at/output_config` 拒绝 | 客户端会在最新 user 前插入 system reminder；旧校验错误拒绝整个中途 turn | v1.2.1 `anthropic_translate.messages` 仍执行相邻 role 检查 | **明确差异** | 作为独立 API patch；证明任意 system 顺序渲染与 chat route 一致，同时保留 turn-scoped unsupported 检查 |
| 0032 stream keepalive | 为 sink 增加 keepalive callback；generation 等待期间周期调用；Anthropic 发送 `event: ping`，OpenAI 发送 SSE comment | 250K prefill 约数分钟无 token，代理会因 idle timeout 断开 | v1.2.1 `stream_live.zig`、`TF_SSE_KEEPALIVE_S`、OpenAI/Responses keepalive、Anthropic ping、`keepalive_test.zig` | **等价且 v1.2.1 更通用** | 不移植。验证 response head 先发、首 token 前多次 keepalive、客户端断开触发 cancel |
| 0035 Engram failure paths | `engram/aio/io/lead/prompt` 将 EOF/OOM/row-range/read fail 升级为 rank fault；follower `roundRefused` 传播；join readers；open failure 逐层释放 | 一 rank Engram 失败而另一 rank进入 collective，会永远等待；短表或异步分配失败还可能留下悬挂 batch | v1.2.1 `engram_gate/rows/io/recovery`、`engram_recovery_test`、model supervisor/fate | **架构替代，功能范围看似覆盖** | 故障注入 short table、EOF、read fail、OOM、cancel；要求两 rank fail closed 且无线程/分配泄漏 |
| 0036 cached prompt metrics | `engine_api.Info` 增 max context；chat 记录 cached prompt tokens；metrics/health/models 输出 totals 和 vLLM aliases | 只在单次响应显示 cached tokens 无法做长期 cache 命中率和容量分析 | v1.2.1 已有 `prompt_tokens_cached_total`、prefix aliases、health totals、`max_model_len` | **等价/已上游化** | 字段和累计语义 contract test；API key 模式下确认 health 降级不泄露详情 |
| 0046 GET/DELETE body discard | `server/routes.zig` 在所有 GET/DELETE route 前 `discardBody` | keep-alive 连接若不消费 body，剩余字节会被下一请求误解析 | v1.2.1 `routes.zig` 多处分支调用 `discardBody` | **等价/已上游化** | 运行 bodyless、带 body、同连接下一请求 claim test |

## 4. EAQ v1.1 mixed-pack 补丁与 inputs 工具

| 补丁 | 作者代码路径和实际修改 | 改动原因 | v1.2.1 对应代码 | 代码级结论 | 实施与验证 |
| --- | --- | --- | --- | --- | --- |
| 0024 mixed pack | `dsv41/weights.zig` 识别 rank-cache plain entries；prompt/round 对 27 个 indexer/compressor linear 用 cuBLAS BF16，而其他仍走 EXL3 | EAQ v1.1 有 27 个未量化 BF16 linear，旧 EXL3-only loader 无法表达混合 storage | v1.2.1 有 EXL3/dsq storage 和 BF16 sibling能力，但未证明能在同一 Flash checkpoint按这 27 个名字混合执行 | **不适用 EAQ v1；v1.1 明确待审** | 首轮不改。v1.1 阶段先做 tensor inventory，再为每个 linear 显式 storage type 和 target/prefill parity |
| 0025 widen BF16 | `dsv41/checkpoint.zig::widenHalf` 增 BF16→f32：源 16 bits 放高半、低半清零 | `weights_proj.weight` 在 pack 为 BF16，而 plan 要 f32；旧 reader只支持 FP16 widen | v1.2.1 core checkpoint conversion需以真实 EAQ v1/v1.1 dry-open验证 | **EAQ v1 条件验证；主要服务 v1.1** | 若 v1 不触发不改；若触发，加入 bit-exact BF16 widen test，拒绝尺寸/类型不匹配 |
| 0026 plain rows | model option `TF_DS_PLAIN_ROWS`；prompt 的 BF16 mm 至少按固定 N rows 执行 | cuBLAS 对不同 M shape 选择不同 kernel/split-K，舍入变化会使 draft on/off 或并发输出不同 | v1.2.1 EAQ v1 EXL3 kernels按 row 固定；mixed BF16 路径尚未纳入首轮 | **不适用 EAQ v1** | v1.1 时以 draft off/on、1/8并发 token parity决定固定 row策略，不提前污染 EXL3路径 |
| inputs 0001 mixed rank-cache | `plan.zig::Builder` 为指定 27 tensors 生成 `.plain` part；`rank_cache_main` 写 plain bytes并报告数量 | engine 只有在 cache manifest声明 plain entries时才能选择 BF16 path | v1.2.1 优先直接读 pack并可用 `prepared.zig` 缓存内部 images；旧 rank-cache ABI 不同 | **不移植；v1.1 重新设计** | 若需要 prepared cache，把 dtype/storage 纳入 v1.2.1 manifest和digest，不读取旧 cache |
| inputs 0002 widen BF16 | inputs `checkpoint.zig::widenHalf` 同步增加 BF16→f32 | rank-cache writer和runtime必须使用同一转换，否则 cache与直接加载数值不同 | v1.2.1 prepared path必须共享同一 loader/conversion代码 | **设计原则保留，代码不移植** | direct load 与 prepared load逐字节/逐token一致；转换只实现一次并由两路径复用 |

## 5. 审计后对原追踪矩阵的修订

### 5.1 已确认无需开发

- 0004 call gate。
- 0027 tool-call prose。
- 0032 stream keepalive。
- 0033 graph cap。
- 0036 cached prompt metrics。
- 0046 body discard。

原因：v1.2.1 不仅存在同目标代码，还存在对应的 DSML、keepalive、graph LRU/floor、metrics 或 route tests。

### 5.2 已确认是新架构替代，禁止旧 patch 直接移植

- 0005 prefill interleave。
- 0008 pool/window。
- 0013–0016 spill/session/KV 基础。
- 0018–0019 多 lane 与持久化组合。
- 0020 pair failure。
- 0021 picture cache。
- 0022 KV/RoCE exchange。
- 0035 Engram failure。
- 0037 rendezvous。

原因：v1.2.1 已将旧 fork 的 family-local逻辑重构为 round scheduler、paged KV/session、通用 TP fabric/fate 等模块；旧 patch 的数据结构和 wire protocol不再兼容。

### 5.3 明确仍存在行为差异

- 0029 Anthropic thinking budget 缺省。
- 0030 Anthropic context management 宽容度。
- 0031 Anthropic system message 位置。

原因：已直接读取 v1.2.1 `anthropic_translate.zig`，当前仍保留作者 patch 前的严格检查。这些差异不阻塞模型推理，但可能阻塞目标客户端。

### 5.4 必须依赖 GB10/profile 决定

- 0001 indexer TP split。
- 0010 RDMA idle nap。
- 0011 parked pages budget。
- 0017 row split/AOT variant。
- 0023 prepare memory floor。
- 0038 aligned index views。

原因：这些问题依赖长上下文、Linux unified memory、实际 AOT shape或 RoCE proxy行为；仅靠源码不能证明目标机器会触发。

### 5.5 首轮延期

- 0024–0026。
- inputs 0001–0002。

原因：它们服务 EAQ v1.1 mixed BF16 pack；用户目标的作者推荐基线是 EAQ v1 EXL3 2.9 bpw。

## 6. 开发优先级

1. EAQ v1 checkpoint dry-open 和 asset manifest。
2. TP2/NCCL token golden。
3. 严格 RoCE parity。
4. v1.2.1 现有 session/KV split/Engram/graph/fate 集成验证。
5. 根据产品需求决定 0029–0031 API compatibility。
6. 根据 GB10 profile决定 0001/0010/0011/0017/0023/0038。
7. EAQ v1.1 单独分支处理 0024–0026 和 inputs patches。

顺序原因：loader和资产决定能否运行；TP/NCCL决定数值基线；RoCE和性能策略必须建立在基线之上；API兼容和 v1.1 不应干扰 EAQ v1 首次部署。

## 7. 静态审计证据范围

已读取：

- 34 个作者 engine `.patch` 的 diff 文件、修改路径和 hunk。
- 2 个 inputs `.patch` 的 diff 文件、修改路径和 hunk。
- 作者 patch index、版本 pin、配置和部署说明。
- v1.2.1 DeepSeek family、TP fabric、server/API、graph cache、KV/session、Engram、AOT 和相关 tests。

尚未完成：

- GB10 build。
- EAQ v1 实包 metadata/load。
- 两节点 NCCL/RoCE。
- GPU token parity、长上下文、故障注入和性能 receipt。

因此本文验证等级为 `static_verified`，不是 `build_verified` 或 `integration_verified`。

