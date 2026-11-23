# TensorFold v1.2.1 双机 GB10 部署 DeepSeek V4.1 Flash 1017 实现方案

配套设计文档：

- [详细架构设计](deepseek-v41-flash-1017-v121-architecture-design.md)
- [作者改动与 v1.2.1 追踪矩阵](deepseek-v41-flash-1017-v121-change-traceability.md)
- [作者 34+2 补丁代码路径审计](deepseek-v41-flash-1017-v121-code-path-audit.md)

## 1. 目标与范围

本文说明如何基于 TensorFold `v1.2.1`，使用两台 NVIDIA GB10 和作者推荐的 EAQ v1 2.9 bpw 量化权重，实现 DeepSeek V4.1 Flash 1017 的双机推理。

目标模型基线：

```text
Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw
revision: 45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b
```

TensorFold 基线：

```text
repository: https://github.com/stdfen/TensorFold.git
base branch: release/v1.2.1
commit: 1db3edbe7e692e83c7f5056754dc4e69e79a6c22
```

本文只在 `stdfen/TensorFold` fork 中规划代码和文档变更。作者部署仓库只作为行为、权重和性能参考，不在该仓库中开发 TensorFold v1.2.1，也不把作者的整套补丁或运行缓存直接复制进来。

## 2. 总体结论

TensorFold v1.2.1 已经具备实现该目标的大部分基础能力：

- DeepSeek V4.1 CUDA family。
- 专用 `dsv41-serve` 构建入口。
- rank 0 的 `tensorfold-dsv41` 和 follower 的 `tf-dsv41`。
- TP=2。
- NCCL、mailbox 和 RoCE 通信后端。
- GB10 的 `sm_121` 编译目标。
- EXL3/dsq-1 量化加载基础。
- DSpark draft。
- 磁盘会话、KV split、fail-fast 和 stall watchdog。
- CUDA graph cache 和 prefill pieces。

但当前不能直接得出“作者权重可以无修改丝滑部署”的结论。首要未知项是：

1. EAQ v1 checkpoint metadata、tensor names、EXL3 trellis/codebook 是否完全符合 v1.2.1 loader。
2. v1.2.1 需要的 Engram token map、RoPE 和 AOT kernel 资产与作者旧 kernel kit 的格式不同。
3. v1.2.1 专用引擎的双 rank 启动方式与作者 fork 的旧命令和环境变量不同。
4. 两台 GB10 上的加载峰值、RoCE 设备选择、长上下文和 vision 路径尚无运行证据。

因此实施顺序必须是：

```text
冻结输入
  → 权重静态兼容
  → 生成 v1.2.1 原生资产
  → 构建 sm_121 专用引擎
  → TP2/NCCL 正确性
  → TP2/RoCE
  → 功能完整性
  → 长上下文与可靠性
  → 性能差距分析
  → 按证据移植优化
```

## 3. 为什么首轮使用专用引擎

v1.2.1 的标准发布服务仍对 `deepseek_v41` 保留明确拒绝门禁；DeepSeek V4.1 通过独立的实验构建和二进制提供。

首轮应使用：

- `tensorfold-dsv41`：rank 0 和 HTTP API。
- `tf-dsv41`：rank 1 follower。

不应在第一阶段删除 `zig/src/native/cuda_refusal.zig` 中的门禁。

原因：

- 专用引擎已经限定模型类型，失败面更小。
- 标准二进制门禁代表发布资格尚未完成，不只是一个命令行开关。
- 同时修改 loader、资产、CLI 和发布门禁，会使故障无法归因。
- 专用引擎跑通后，再决定是否需要标准 CLI 集成，不影响双机部署目标。

通过条件：专用二进制可以加载固定 checkpoint，并在两台 GB10 上完成 TP2 推理。

## 4. 实施阶段与步骤

## 阶段 0：建立独立开发基线

### 步骤 0.1：确认本地仓库和远端

执行：

```bash
git remote -v
git fetch origin --tags
git rev-parse release/v1.2.1
git status --short
```

原因：

- 后续代码必须提交到 `stdfen/TensorFold` fork。
- 必须避免误从作者 fork 或 upstream 的其他分支开始开发。
- 未识别的本地修改可能与适配代码混合，影响审查和回滚。

通过条件：`release/v1.2.1` 指向批准的基线提交，工作区没有未解释的修改。

### 步骤 0.2：创建功能分支

建议：

```bash
git switch release/v1.2.1
git switch -c feature/dsv41-flash1017-tp2-gb10
```

原因：

- `main` 和 `release/v1.2.1` 必须保持可比较、可回退。
- 权重加载、资产工具、通信和部署编排应通过小提交逐步引入。
- 后续可以分别 review loader 兼容、资产生成、运行脚本和性能优化。

通过条件：功能分支的 merge-base 是 v1.2.1 固定提交。

### 步骤 0.3：建立输入清单

记录但不提交敏感地址或凭据：

- TensorFold commit。
- 模型仓库和完整 revision。
- Engram 基础模型仓库和 revision。
- tokenizer/config 文件 SHA256。
- CUDA、NVCC、Zig、Triton、Python 和驱动版本。
- 两台 GB10 的架构、内存、ConnectX-7 和 RoCE 信息。
- 生成资产使用的工具 revision 和参数。

原因：

- AOT kernel 和 token map 与模型几何、上下文和工具版本有关。
- 只记录模型名字不足以复现一次成功运行。
- 发生结果差异时，必须能够判断是代码、资产、权重还是环境变化。

通过条件：形成可机器读取的 manifest，且不包含 token、私钥和完整敏感地址。

## 阶段 1：先证明权重格式可读

### 步骤 1.1：检查 checkpoint 配置和文件完整性

检查模型目录中的：

- `config.json`。
- tokenizer 配置和 chat template。
- safetensors index。
- 所有量化 shard。
- EXL3/dsq-1 metadata。
- DSpark 和 vision 相关 tensor 是否存在。

原因：

- 下载不完整和 loader 不兼容会产生相似的“缺 tensor”错误。
- 必须先排除文件缺失，才能决定是否修改代码。
- 作者 EAQ v1 权重包含量化主模型、vision 和 DSpark，不能只验证文本层的一小部分。

通过条件：文件数量、大小、index 引用和固定 revision 一致；index 不引用不存在的 shard。

### 步骤 1.2：建立 checkpoint metadata dry-open 工具或测试

优先复用 v1.2.1 已有 checkpoint reader；如果现有入口会立即分配大块 GPU 内存，则增加只读取 metadata、tensor names、dtype 和 shape 的 host-side 检查。

建议输出：

```text
model_type
quantization format/version
tensor count
missing expected tensors
unexpected tensors
per-tensor dtype/shape
EXL3 trellis/codebook/scale metadata
vision tensors
DSpark tensors
Engram external dependencies
```

原因：

- metadata 检查成本远低于每次完整加载约百 GiB 权重。
- 可以把“格式不支持”和“GPU/内存/通信失败”分离。
- loader 修改必须由具体的缺失字段或 tensor 映射证据驱动。

通过条件：dry-open 可以完整遍历 checkpoint；所有必须 tensor 被识别，dtype 和 shape 符合 family 规划。

### 步骤 1.3：仅在 dry-open 失败时扩展 loader

可能涉及：

- DeepSeek V4.1 checkpoint metadata 解析。
- `cuda_format.zig` 的量化表示。
- family `config.zig` 的模型几何映射。
- plan/named tensor/load 路径。
- EXL3 trellis、codebook、scale 或 output scale 读取。

修改原则：

1. 先为实际失败的 metadata/tensor 写最小 fixture 或测试。
2. 只增加 EAQ v1 所需格式，不顺带支持 EAQ v1.1。
3. 对未知 dtype、shape 和 metadata 继续 fail closed。
4. 错误信息必须包含 tensor 名和期望格式，但不能输出敏感路径。

原因：

- 放宽所有检查可能让错误权重进入 CUDA kernel，产生更难定位的错误结果或越界访问。
- EAQ v1.1 的 27 个 BF16 indexer/compressor linear 会引入另一套混合格式路径，应独立评审。
- 最小修改便于与 v1.2.1 原始行为比较。

通过条件：新增测试通过，错误格式仍被拒绝，EAQ v1 dry-open 全部通过。

## 阶段 2：生成 v1.2.1 原生资产

### 步骤 2.1：构建资产工具并运行 host tests

执行：

```bash
zig build dsv41-assets
zig build test-dsv41-assets-host test-dsv41-token-map
```

原因：

- v1.2.1 明确区分 runtime 和资产生成工具。
- host tests 可以在投入 GPU 编译前发现模型几何、token map 或文件格式问题。
- 旧作者仓库的资产格式不能假定与 v1.2.1 ABI 相同。

通过条件：两个 host test 均成功，生成工具来自当前功能分支的同一提交。

### 步骤 2.2：生成 Engram token map

执行：

```bash
zig-out/bin/tf-dsv41-token-map \
  --tokenizer MODEL/tokenizer.json \
  --config MODEL/config.json \
  --output ASSETS/engram-token-map.bin
```

原因：

- v1.2.1 serving asset 使用 `DSENGR01` 格式的 `engram-token-map.bin`。
- 作者旧方案中的 JSON 或 legacy `engram-host.bin` 不能作为 serving asset。
- token map 与 tokenizer 和模型配置绑定，必须保存来源 SHA256。

通过条件：二进制和 provenance sidecar 生成成功；sidecar 中的 tokenizer/config SHA256 与固定输入一致。

### 步骤 2.3：按最终运行形状生成 AOT needs

先设置与最终 serving 相同的关键参数，再执行：

```bash
zig-out/bin/tf-dsv41-assets aot-needs ASSETS/sigs.json ASSETS/needs.json
```

关键输入至少包括：

- `TF_TP_WORLD=2`。
- rank/world 对应形状。
- context/window。
- slot/parallel。
- draft 开关。
- sampling/prefill 设置。
- image/vision 是否启用。

原因：

- `aot-needs` 运行的是 family 的真实符号 planner。
- kernel specialization 取决于实际运行形状；只为单 rank 或短上下文生成的集合可能在双机或长上下文时缺 kernel。
- 运行中缺 AOT kernel 不应通过临时 fallback 掩盖。

通过条件：`sigs.json` 和 `needs.json` 覆盖计划中的 TP2、上下文、并发、draft 和 vision 组合。

### 步骤 2.4：为 GB10 编译 AOT kernel

使用与 v1.2.1 recipe 配套的外部 Triton reference tools，根据 `needs.json` 编译实际目标：

```text
target SM: 121
output: ASSETS/aot/
```

保存：

- reference tools revision。
- Triton/CUDA/Python 版本。
- 编译参数。
- `needs.json`、产物 manifest 和 SHA256。

原因：

- GB10 使用 `sm_121`，不能复用其他 SM 的 kernel image。
- 作者旧 kernel kit 对应旧 fork 和旧资产契约，不能直接视为 v1.2.1 兼容。
- 缺少生成 provenance 会导致成功构建无法复现。

通过条件：`needs.json` 中每个 specialization 都有对应产物，无遗漏或错误 SM。

### 步骤 2.5：生成 RoPE 资产

使用 v1.2.1 recipe 配套的 reference RoPE generator 生成：

```text
ASSETS/rope.bin
```

RoPE rows 必须覆盖：

```text
目标 context + 2048
```

原因：

- 少 2048 行可能在上下文边界附近失败。
- RoPE 数据与上下文上限相关，不能只在短 prompt 下验证。

通过条件：资产行数和 checksum 记录在 manifest，覆盖最大计划上下文。

### 步骤 2.6：验证最终资产目录

目标结构：

```text
ASSETS/
├── rope.bin
├── engram-token-map.bin
├── engram-token-map.bin.json
├── sigs.json
├── needs.json
├── manifest.json
└── aot/
```

运行时只通过以下变量引用：

```bash
export TF_DSV41_ASSETS=/absolute/path/to/ASSETS
```

原因：

- 统一入口避免 rank 0/rank 1 使用不同资产。
- 绝对路径可以减少容器、SSH 和工作目录变化造成的歧义。

通过条件：两节点读取到的 manifest 和所有资产 SHA256 完全一致。

## 阶段 3：构建 GB10 专用二进制

### 步骤 3.1：在 ARM64 GB10 环境构建

执行：

```bash
zig build dsv41-serve \
  -Doptimize=ReleaseSafe \
  -Dnvcc=/usr/local/cuda/bin/nvcc \
  -Ddsv41-nvcc=/usr/local/cuda/bin/nvcc \
  -Dsm=121
```

原因：

- recipe 要求在原生 ARM64 host 上构建真实 runtime 和 family CUDA images。
- `ReleaseSafe` 在首轮保留安全检查，比直接追求极限性能更适合作为正确性基线。
- `sm=121` 是 GB10 目标，错误架构可能构建成功但运行失败或降级。

通过条件：生成 `tensorfold-dsv41` 和 `tf-dsv41`，二进制可执行且依赖库完整。

### 步骤 3.2：记录构建 provenance

记录：

```bash
git rev-parse HEAD
zig version
/usr/local/cuda/bin/nvcc --version
sha256sum zig-out/bin/tensorfold-dsv41 zig-out/bin/tf-dsv41
ldd zig-out/bin/tensorfold-dsv41
ldd zig-out/bin/tf-dsv41
```

原因：

- 双节点必须运行同一提交和同一二进制。
- 构建缓存或工具链变化可能生成不同 image。
- 运行问题需要能够关联到精确二进制，而不是仅关联分支名。

通过条件：节点 A/B 的 binary SHA256 一致，动态库依赖均可解析。

## 阶段 4：建立 TP2/NCCL 最小正确性基线

### 步骤 4.1：关闭非必要优化

首轮配置建议：

```text
world=2
backend=nccl
小 context
parallel/slots=1
draft=off
graph cache=off
prefill pieces=off
KV split=off 或默认
disk session=off
vision=off
```

具体 CLI 参数以当前构建的 `--help` 为准，先保存：

```bash
zig-out/bin/tensorfold-dsv41 --help
zig-out/bin/tf-dsv41 --help
```

原因：

- 最小配置可以证明 checkpoint、kernel、模型规划和基本 TP 通信正确。
- draft、graphs、session 和 KV 策略同时开启后，输出错误无法快速归因。
- NCCL 是 TP 默认后端，应先作为 RoCE 优化前的对照组。

通过条件：配置清单明确记录，两个 rank 使用相同模型和资产 fingerprint。

### 步骤 4.2：配置 rank 0

基础通信环境：

```bash
export TF_EXPERIMENTAL=deepseek
export TF_DSV41_ASSETS=/absolute/path/to/ASSETS
export TF_TP_WORLD=2
export TF_TP_RANK=0
export TF_TP_MASTER='<NODE_A_FABRIC_ADDRESS>'
export TF_TP_PORT=29500
export TF_TP_DEVICE=0
export TF_COMM_BACKEND=nccl
export TF_TP_FAILFAST=1
```

原因：

- `TF_EXPERIMENTAL=deepseek` 是实验 family 的显式准入，不应通过删除检查绕过。
- rank 0 同时负责模型主进程和 API。
- master 必须使用节点 B 可达的 fabric 地址，不能使用 `localhost`。

通过条件：rank 0 启动后等待 rank 1，日志中模型类型、world、rank、资产路径符合预期。

### 步骤 4.3：配置 rank 1

```bash
export TF_EXPERIMENTAL=deepseek
export TF_DSV41_ASSETS=/absolute/path/to/ASSETS
export TF_TP_WORLD=2
export TF_TP_RANK=1
export TF_TP_MASTER='<NODE_A_FABRIC_ADDRESS>'
export TF_TP_PORT=29500
export TF_TP_DEVICE=0
export TF_COMM_BACKEND=nccl
export TF_TP_FAILFAST=1
```

使用 `tf-dsv41` 的 follower 子命令启动；精确参数以该提交的 `--help` 为准。

原因：

- 两 rank 除 `rank` 和必要的本地 device 外，服务形状和通信配置必须一致。
- v1.2.1 会比较配置 digest；显式设置可以减少不同 shell 默认值导致的不一致。

通过条件：两 rank rendezvous 成功，NCCL 初始化成功，配置 digest 一致。

### 步骤 4.4：执行最小输出测试

使用固定 token prompt、greedy sampling、短输出，记录：

- prompt token ids。
- output token ids。
- rank 0/rank 1 启动日志。
- health 和 metrics。
- GPU/统一内存峰值。

原因：

- token ids 比自然语言文本更适合判断微小输出差异。
- greedy、短输出减少 sampling 和长上下文噪声。
- 该结果是后续开启 draft、RoCE 和 graph 的 golden baseline。

通过条件：重复运行输出 token 完全一致，无 NaN、CUDA error、missing kernel 或 rank timeout。

## 阶段 5：适配和验证权重分片及内存

### 步骤 5.1：记录每 rank 的权重规划

输出或增加诊断信息，记录每个 rank：

- tensor 数量和总字节数。
- quantized/BF16/FP16 分布。
- vision、DSpark、Engram 归属。
- host、device 和 mmap/文件映射占用。
- 加载峰值和稳态占用。

原因：

- 作者方案约每 rank 98–106 GiB，GB10 的统一内存余量有限。
- 只看最终稳态不足以发现加载期间的瞬时双份内存。
- rank 分配错误可能仍能启动，但在推理阶段读错 tensor。

通过条件：两个 rank 的规划互补且符合 TP2 设计，加载峰值不触发系统内存压力或冻结。

### 步骤 5.2：确定权重存储策略

优先级：

1. 保留一个不可变的模型 snapshot。
2. 每节点将高频随机读取的 Engram 数据放到本地磁盘。
3. 如果 v1.2.1 直接读 checkpoint 能满足启动时间和内存要求，不引入旧 rank-cache。
4. 只有实测证明直接读取不可接受时，才为 v1.2.1 设计带版本、checksum 和 rank fingerprint 的 prepared cache。

原因：

- 作者旧 rank-cache 绑定旧 loader 和 tensor layout，不能直接作为 v1.2.1 的事实标准。
- 先复用 v1.2.1 原生路径可以减少额外格式和缓存失效逻辑。
- Engram 随机小读对远程文件系统非常敏感，本地放置有明确性能理由。

通过条件：冷启动、热启动、内存峰值和磁盘占用均有实测记录；缓存不会在模型 revision 变化后被误用。

## 阶段 6：切换 TP2/RoCE

### 步骤 6.1：先验证网络和 verbs

在模型进程之外验证：

- ConnectX-7 link active。
- RoCE v2 GID。
- MTU 和 traffic class。
- 双向带宽、时延和错误计数。
- 容器或服务进程有 verbs 设备权限。

原因：

- 网络配置错误不应该通过修改 TensorFold 代码解决。
- 在大模型启动后诊断 GID、MTU 或设备权限成本很高。

通过条件：两节点 verbs 测试稳定，无持续丢包、重传或端口错误增长。

### 步骤 6.2：以允许 fallback 的模式首次启动

```bash
export TF_COMM_BACKEND=roce
export TF_TP_ROCE_FALLBACK=nccl
export TF_TP_ROCE_HCA='<HCA[:GID]>'
```

可按环境设置：

```text
TF_TP_ROCE_HCAS
TF_TP_ROCE_TC
TF_TP_ROCE_FAST
TF_TP_ROCE_POLLERS
TF_TP_ROCE_STAGGER_NS
TF_TP_ROCE_BLOCKS
TF_TP_ROCE_CPU
TF_TP_ROCE_CHUNKED_KB
```

原因：

- 首次切换需要保留 NCCL 对照，避免因 RoCE setup 失败完全阻断功能验证。
- 日志必须明确实际使用 RoCE 还是发生 fallback，不能只看服务是否可用。

通过条件：日志证明 RoCE setup 和 load-time probe 成功；如果发生 fallback，本阶段不得标记通过。

### 步骤 6.3：切换严格 RoCE 模式

```bash
export TF_TP_ROCE_FALLBACK=error
```

原因：

- 性能验收时必须保证测量的是 RoCE，而不是静默 NCCL fallback。
- 严格模式可以在启动时暴露两 rank 配置或 HCA/GID 错误。

通过条件：严格模式双 rank 正常启动，固定 token 输出与 NCCL baseline 一致。

### 步骤 6.4：比较 NCCL 与 RoCE

对相同 prompt 和配置记录：

- 首 token 延迟。
- prefill tokens/s。
- decode tokens/s。
- 小消息/大消息路径计数。
- CPU 核占用。
- 网络吞吐和错误计数。
- 输出 token hash。

原因：

- RoCE 是优化，不应改变数值结果。
- 如果性能没有改善，需要先判断消息尺寸、HCA 选择和 CPU proxy，而不是立即移植作者补丁。

通过条件：输出一致；RoCE 无可靠性回退，性能和 CPU 开销有可解释结果。

## 阶段 7：逐项开启 v1.2.1 已有能力

每次只开启一类能力，并与上一阶段 token baseline 比较。

### 步骤 7.1：开启 DSpark draft

原因：

- 作者 checkpoint 自带 DSpark block，但 draft 是独立执行路径。
- 必须证明 draft 只改变执行效率，不改变最终接受 token。

验证：draft off/on 使用相同 prompt、seed 和 sampling；比较最终 token ids、drafted/accepted 统计和速度。

通过条件：确定性配置下输出一致，draft 统计合理，无拒绝率异常或性能倒退。

### 步骤 7.2：开启 CUDA graph cache

原因：

- graph 可以降低重复 shape 的 launch 开销，但会增加显存/统一内存占用。
- 作者 fork 有 graph cap 补丁，v1.2.1 虽有 graph cache 基础设施，仍需证明长期 shape 增长不会无界占用。

验证：覆盖多种 rows、context buckets 和并发 shape，记录 graph 数量和内存曲线。

通过条件：输出一致，捕获/复用正常，长期内存有界；否则先补充 cap/eviction 测试，再决定开发。

### 步骤 7.3：开启 prefill pieces

相关设置：

```bash
export TF_DSV41_PREFILL_PIECES=1
```

并满足运行日志要求的 served shape，例如 own prefill 和 prompt tail 设置。

原因：

- 分片 prefill 的目标是避免长 prompt 长时间阻塞 decode。
- 该能力影响调度公平性，不应在基础正确性前开启。

通过条件：长 prompt 与短 decode 并存时延迟改善，输出和 prompt token 计数不变。

### 步骤 7.4：开启 disk session 和 KV split

原因：

- v1.2.1 已提供这些能力，应优先配置和验证，而不是移植作者旧补丁。
- 它们会改变缓存所有权、恢复和磁盘生命周期，需要独立故障测试。

验证：fresh/resume 一致、双 rank manifest 一致、缺失 rank 数据时拒绝错误恢复、重启后无临时文件残留。

通过条件：恢复结果等于 fresh baseline，rank 数据不一致时 fail closed。

### 步骤 7.5：开启 vision

原因：

- 文本成功不能代表完整 Flash 1017 checkpoint 成功。
- vision 会引入额外 tensor、image preprocessing 和 AOT shape。

验证：固定图片和 prompt，比较单次、并发和 resume；确认 `aot-needs` 已包含 image shape。

通过条件：图片请求稳定、可重复，不缺 kernel，恢复结果一致。

## 阶段 8：API 和功能验收

至少验证：

- OpenAI chat/completions。
- streaming 和非 streaming。
- thinking。
- tool choice：auto、none、required、named。
- structured output/grammar。
- 图片输入。
- prompt cache。
- session resume。
- 1、2、4 路并发。

原因：

- 作者方案不只是裸 token generator，还依赖服务 API 行为。
- v1.2.1 已包含若干 API 修复，不能因为作者补丁列表中出现同名问题就重复移植。

通过条件：所有必测接口有 JSON receipt；skip 项明确记录原因，不能算作通过。

## 阶段 9：长上下文、内存和可靠性

### 步骤 9.1：上下文阶梯测试

顺序：

```text
32K → 128K → 256K → 512K → 1,048,576
```

每档记录 prefill、decode、needle/recall、内存峰值、cache、graph 和两个 rank 日志。

原因：

- 直接测试 1M 失败时无法判断阈值。
- 作者的 indexer split 优化以长上下文为主要触发条件，必须先得到 v1.2.1 的分段曲线。

通过条件：每档正确召回、无 OOM/stall/transport timeout，测试结束后内存不持续增长。

### 步骤 9.2：并发阶梯测试

顺序：1、2、4 路；必要时再扩大。

原因：

- 两台 GB10 的 pool、context 和统一内存共享策略会随并发变化。
- 单流性能不能代表生产调度行为。

通过条件：所有请求结束，无会话串扰；吞吐、首 token 延迟和每流速度均有记录。

### 步骤 9.3：rank 故障和恢复

测试：

- rank 1 kill。
- rank 1 stop/hang。
- 通信断开。
- rank 0 退出。
- 双 rank clean restart。
- session/KV 数据不一致。

原因：

- v1.2.1 默认 fail-fast，但必须在目标网络和模型负载下验证。
- 半活服务比明确退出危险，会导致请求长期挂起或错误结果。

通过条件：故障在限定时间内被双方发现，API 不继续报告健康；重启后 baseline 恢复。

## 阶段 10：性能差距分析和按需开发

只有前面所有正确性门禁通过后，才比较作者方案和 v1.2.1。

### 步骤 10.1：建立同条件基准

保持一致：

- 模型 revision。
- prompt/token 集。
- context、parallel、draft。
- GPU 时钟和温度。
- RoCE/NCCL 配置。
- graph 和 cache 状态。
- 冷/热启动定义。

原因：

- 不同模型、prompt 或热缓存下的数字不能证明代码差距。
- 性能补丁必须有可重复的 baseline 和目标。

通过条件：至少三轮稳定结果，原始 JSON 和环境 manifest 可复查。

### 步骤 10.2：评估 indexer TP split

触发条件：超过约 131K 上下文后，indexer 明显成为单 rank 瓶颈，且 profile 与作者补丁描述一致。

原因：

- 这是长上下文优化，不应为短上下文基线增加复杂度。
- v1.2.1 的内部计划可能已变化，旧 patch 不能直接套用。

开发要求：重新基于 v1.2.1 planner 实现，补充 128K 边界前后、TP2 数值一致性和性能测试。

### 步骤 10.3：评估 RDMA idle nap

触发条件：空闲时 RoCE proxy 长期占满一个 CPU core，影响系统运行或功耗。

原因：

- idle nap 是资源治理，不是正确性必需项。
- nap 会引入唤醒延迟和 shutdown 延迟，必须量化。

开发要求：增加空闲检测、可配置 sleep、唤醒上界和关闭测试；比较 decode latency 与空闲 CPU。

### 步骤 10.4：评估 graph cap/eviction

触发条件：多日运行、多 shape 或会话增长时 graph cache 内存持续上升。

原因：

- 短测无法证明 graph cache 有界。
- 无证据引入固定 cap 可能降低高频 shape 命中率。

开发要求：使用 v1.2.1 graph cache API 实现 LRU/上限，加入活跃 graph 不被错误释放的并发测试。

### 步骤 10.5：评估额外 prefill interleave

触发条件：`TF_DSV41_PREFILL_PIECES=1` 仍不能满足长 prompt 与在线 decode 的延迟目标。

原因：

- v1.2.1 已有 pieces 能力，应先验证现有机制。
- 再增加调度策略会影响公平性、吞吐和可重复性。

开发要求：用长 prompt + 持续 decode 混合负载证明改动收益，并验证单请求吞吐没有不可接受退化。

## 5. 作者补丁的处理分类

### A. 优先使用 v1.2.1 现有实现

- TP2。
- NCCL/RoCE backend。
- fail-fast 和 stall exit。
- disk session。
- KV split/compact/dedup。
- tool choice。
- prefix-cache metrics。
- HTTP body discard。
- CUDA graph cache 基础能力。
- prefill pieces。

原因：重复移植会制造两套状态机，并可能回退 v1.2.1 已修复的行为。

### B. 必须通过真实数据验证或适配

- EAQ v1 checkpoint loader。
- Engram token map 和外部 Engram 权重路径。
- `sm_121` AOT kernel 资产。
- TP2 权重规划和加载峰值。
- vision tensor 与 AOT shape。
- v1.2.1 专用二进制启动编排。

原因：这些项目直接决定模型能否启动和正确执行，不能靠源码中存在同名能力推断通过。

### C. 仅在 profile 证明需要后开发

- indexer TP split。
- RDMA idle nap。
- graph cap/eviction。
- 额外 prefill interleave。
- 特定 pool/window 调度。

原因：这些是性能或长期运行优化，未经 profile 移植会扩大维护面且无法证明收益。

### D. 首轮明确不做

- EAQ v1.1 混合 BF16 支持。
- abliterated overlay。
- 复制作者旧 rank-cache 格式。
- 直接复用作者旧 kernel kit。
- 整体应用作者 34 个 engine patches。
- 删除标准二进制 release refusal。

原因：它们不是完成 EAQ v1 双机部署的最小必要条件。

## 6. 建议代码与文件组织

在不破坏现有 family 结构的前提下：

```text
docs/recipes/
  deepseek-v41-flash-1017-two-gb10-v121.md

zig/src/families/deepseek_v41/
  仅放 checkpoint、planner、runtime 的必要适配

zig/src/fabric/tp/
  仅放经独立测试证明需要的通信修改

tools/ 或 scripts/
  放可复用的资产检查、checkpoint dry-open、双机启动和 receipt 工具

docs/receipts/<run-id>/
  只放脱敏、小体积、可审查的验收证据
```

原因：

- family 兼容、通信和部署编排是不同责任域，应保持独立提交。
- 大模型、AOT 二进制、编译缓存和完整运行日志不得进入 Git。
- receipt 应能证明结果，但不应成为私有环境转储。

## 7. 建议提交拆分

1. `docs: add v1.2.1 two-GB10 DeepSeek V4.1 plan`
2. `test(dsv41): add EAQ v1 checkpoint metadata fixture`
3. `feat(dsv41): admit EAQ v1 checkpoint metadata`，仅在需要时。
4. `tools(dsv41): validate asset manifest and hashes`
5. `tools(dsv41): add TP2 launch/receipt workflow`
6. `test(dsv41): add NCCL/RoCE token parity receipt`
7. 单独的性能补丁，每个补丁对应独立 profile 和 receipt。

原因：每个提交应回答一个问题，便于 bisect、review 和回滚。不要把 loader、RoCE 和调度优化塞进同一提交。

## 8. 验收门禁

| 门禁 | 必须满足的条件 | 失败时行动 |
| --- | --- | --- |
| G0 基线 | 分支、commit、模型和工具链固定 | 停止开发，修正来源 |
| G1 权重 | EAQ v1 metadata/tensor dry-open 全通过 | 最小扩展 loader 并加测试 |
| G2 资产 | token map、RoPE、AOT 完整且有 SHA256 | 重新生成，不允许运行时猜测 |
| G3 构建 | ARM64 `sm_121` 两个二进制成功 | 定位构建/依赖，不进入模型加载 |
| G4 NCCL | TP2 固定 token 输出稳定 | 定位 loader/planner/TP，不启用 RoCE |
| G5 RoCE | 严格模式启动，输出等于 NCCL | 定位网络/HCA/transport |
| G6 功能 | 文本、thinking、tools、vision、resume 通过 | 修复具体功能，不做性能优化 |
| G7 压力 | 长上下文、并发、故障恢复通过 | 定位内存/调度/可靠性 |
| G8 性能 | 同条件可重复基准完成 | profile 后决定是否移植优化 |

## 9. 最终完成标准

只有同时满足以下条件，才能称为“TensorFold v1.2.1 双机 GB10 部署实现完成”：

1. 使用 `release/v1.2.1` 派生分支和固定 EAQ v1 revision。
2. 资产全部由 v1.2.1 对应工具链生成，目标为 `sm_121`。
3. 两台 GB10 使用相同 binary 和资产 fingerprint。
4. TP2/NCCL 和 TP2/RoCE 都通过固定 token 正确性测试。
5. 严格 RoCE 模式下没有静默 fallback。
6. text、thinking、tools、structured output、vision 和 resume 通过。
7. 1/2/4 路并发通过。
8. 128K、512K 和目标最大上下文通过；如果 1M 不作为正式支持范围，必须明确降级并记录原因。
9. 任一 rank 故障时服务 fail closed，重启后恢复基线。
10. 内存峰值、稳态占用、性能数据和环境 manifest 有脱敏 receipt。

## 10. 回滚策略

每个阶段都应可回到上一门禁：

- 代码：回到 v1.2.1 基线或上一通过提交。
- 模型：保持固定 immutable snapshot，不原地修改。
- 资产：按 manifest 建版本目录，不覆盖上一通过版本。
- 二进制：按 commit/SHA256 保存，不使用模糊的 `latest`。
- 通信：RoCE 失败时回到已验证 NCCL 基线，但不得把 fallback 结果报告为 RoCE 通过。
- 性能补丁：一次只回滚一个独立提交，并重复 token parity。

原因：大模型双机故障通常同时涉及代码、资产和环境；没有分层回滚就无法确定哪一层导致退化。

## 11. 已知风险

| 风险 | 影响 | 缓解方式 |
| --- | --- | --- |
| EAQ v1 metadata 不兼容 | 无法加载模型 | dry-open、最小 fixture、严格 loader |
| AOT shape 缺失 | 特定请求运行失败 | 按最终 TP/context/vision 生成 needs |
| 旧 kernel kit ABI 不同 | 错误或缺 kernel | 全部使用 v1.2.1 资产链重建 |
| 加载峰值过高 | GB10 冻结或 OOM | 分 rank 记账、避免内存双份、保留安全余量 |
| Engram 远程随机读 | decode 严重变慢 | 两节点本地放置并校验 fingerprint |
| RoCE 静默 fallback | 性能结论无效 | 最终验收使用 fallback=error |
| graph cache 增长 | 长期内存泄漏式增长 | 长跑监控，证据触发 cap/eviction |
| 直接移植旧补丁 | 重复实现和回归 | 按 A/B/C/D 分类逐项决策 |

## 12. 相关 v1.2.1 源码和文档

- `docs/recipes/deepseek-v41-cuda.md`
- `zig/src/families/deepseek_v41/serve_engine.zig`
- `zig/src/fabric/tp/config.zig`
- `zig/src/native/cuda_refusal.zig`
- `docs/quantization.md`
- `docs/kernel-contract.md`
- `docs/roce-gather.md`
- `docs/multi-rank-receipts.md`

本文是实现和验证计划，不代表已经完成 GB10 构建、实包加载或双机集成验收。运行状态必须以实际 build、GPU、双 rank 和 API receipt 为准。

