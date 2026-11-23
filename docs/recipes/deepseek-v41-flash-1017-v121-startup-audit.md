# DeepSeek-V4.1-Flash-1017：TensorFold v1.2.1 镜像启动审计

> 更新说明：在确认 `/data/models` 同时存在 EXL3 量化权重和完整 FP8/FP4 权重后，启动方案已调整为“EXL3 在线 PACK + 全量权重离线生成 `TF_DSV41_ENGRAM_DIR`”。量化索引无需包含 Engram embed 张量。后续实施以 [deepseek-v41-flash-1017-v121-native-startup-plan.md](deepseek-v41-flash-1017-v121-native-startup-plan.md) 为准。

## 结论

当前 `tensorfold-dsv41:v1.2.1-sm121` 镜像不能直接使用作者仓库的启动参数和现有模型目录完成启动。参数层可以映射，但运行时 ABI 有三项确定性阻断：

1. 作者 follower 使用 `tf-dsv41-lanes MODEL /rank-cache ... KIT`，v1.2.1 使用 `tf-dsv41 follow PACK ASSETS`；两者不是同一命令或资产协议。
2. 作者资产包含旧格式 `engram-host.bin`；v1.2.1 强制读取 `engram-token-map.bin`，其文件头必须为 `DSENGR01`。把旧文件改名不能兼容。
3. 作者部署目录没有提供可直接证明适用于 v1.2.1 TP2 的 Engram 资产。v1.2.1 在线 PACK 可以继续使用 EXL3；`layers.*.engram.embed.weight/scale` 从全量权重离线读取，转换成 `TF_DSV41_ENGRAM_DIR` 的 TP2 行分片。

因此没有创建必然失败的远端容器。只有重新生成 v1.2.1 PACK 和与精确运行 profile 匹配的资产后，才应执行双机启动。

## 已核实环境

| 项目 | 结果 | 影响 |
|---|---|---|
| 两台节点 | aarch64、GB10、SM 12.1、驱动 580.178.04 | 满足镜像架构和 CUDA 目标 |
| Docker | linux/arm64，GPU runtime 可用 | 可运行最终镜像 |
| 内存 | 每节点约 121 GiB，总可用约 117–118 GiB | 需要先以 32K/单槽建立基线 |
| 本地镜像 | `tensorfold-dsv41:v1.2.1-sm121`，arm64 | 构建通过，但尚未传到节点 |
| 作者权重 | 主目录约 197 GiB，另有每节点 Engram 约 48 GiB | 旧 ABI，不能原样交给 v1.2.1 |
| 作者资产 | 有 `rope.bin`、`aot/`、`engram-host.bin` | 缺 v1.2.1 token-map；AOT provenance/profile 也不匹配 |

## 作者参数与 v1.2.1 参数映射

| 作者设置 | 作者值 | v1.2.1 映射 | 结论/原因 |
|---|---:|---|---|
| master port | `29620` | `TF_TP_PORT=29620` | 可直接映射；plan link 还会使用 `port + 7` |
| HTTP host/port | `0.0.0.0:8888` | `TF_HTTP_HOST` / `TF_HTTP_PORT` | 可直接映射，仅 rank 0 启 HTTP |
| served name | `DeepSeek-V4.1-Flash-1017` | `TF_SERVED_NAME` | 可直接映射 |
| context | `1048576` | `TF_CONTEXT` + `TF_DSV41_CONTEXT` | 不能只传 CLI；v1.2.1 模型/KV/RoPE profile 也必须一致。首启用 32768 |
| parallel | `4` | `TF_PARALLEL=4` + `TF_DSV41_SLOTS=4` | CLI 并发和 CUDA slot 数须一致；首启用 1 降低内存和 AOT 组合数 |
| max tokens | `32768` | `TF_MAX_TOKENS=32768` | 可直接映射 |
| sampling | `1.0 / 0.95 / 20` | `TF_TEMPERATURE` / `TF_TOP_P` / `TF_TOP_K` | 可直接映射 |
| thinking | `1` | `TF_THINKING=1` | 可直接映射 |
| reasoning effort/budget | 空 | 对应同名 `TF_` 环境变量 | 空值时不传参数 |
| drafts | `1` | `TF_DSV41_DRAFTS=1` | 参数可映射，但需要 DSpark 对应 AOT；首启关闭 |
| graphs/side/prefetch/AIO | 开 | v1.2.1 的 `TF_DSV41_*` knob 与 AOT/profile | 不能按同名猜测照搬，必须按 v1.2.1 捕获资产启用 |
| RDMA | 开 | 先 `TF_COMM_BACKEND=nccl`，验证后再切 `roce` | v1.2.1 transport 协议和作者旧 runtime 不同 |
| rank 1 | `tf-dsv41-lanes ...` | `tf-dsv41 follow PACK ASSETS` | 必须改命令 |

## 推荐的首启 profile

首启不是作者的性能 profile，而是最小正确性 profile：

```text
TF_TP_WORLD=2
TF_TP_PORT=29620
TF_COMM_BACKEND=nccl
TF_CONTEXT=32768
TF_DSV41_CONTEXT=32768
TF_PARALLEL=1
TF_DSV41_SLOTS=1
TF_MAX_TOKENS=32768
TF_DSV41_DRAFTS=0
TF_THINKING=1
TF_TEMPERATURE=1.0
TF_TOP_P=0.95
TF_TOP_K=20
```

原因：world、context、slots、drafts 和 prefill shape 会改变 KV 规划、buffer plan 或 AOT 需要集合。先缩小变量，拿到双 rank 一致性和 HTTP token 基线，再逐项恢复作者的 drafts、4 slots、图优化、RoCE 和 1M context。

## v1.2.1 最终启动命令

下面命令只在 PACK 和 ASSETS 重新生成并通过哈希校验后执行。`<rank0-fabric-address>` 必须是 rank 1 可达的 rank 0 fabric 地址。

### rank 1：先启动 follower

```bash
docker run -d --name tensorfold-dsv41-r1 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/v121-pack:/models/model:ro \
  -v /path/to/v121-assets-r1:/assets:ro \
  -e TF_MODEL_DIR=/models/model \
  -e TF_DSV41_ASSETS=/assets \
  -e TF_EXPERIMENTAL=deepseek \
  -e TF_TP_WORLD=2 -e TF_TP_RANK=1 \
  -e TF_TP_MASTER=<rank0-fabric-address> -e TF_TP_PORT=29620 \
  -e TF_TP_DEVICE=0 -e TF_COMM_BACKEND=nccl \
  -e TF_CONTEXT=32768 -e TF_DSV41_CONTEXT=32768 \
  -e TF_PARALLEL=1 -e TF_DSV41_SLOTS=1 \
  -e TF_DSV41_DRAFTS=0 \
  tensorfold-dsv41:v1.2.1-sm121
```

### rank 0：follower 就绪后启动 API

```bash
docker run -d --name tensorfold-dsv41-r0 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/v121-pack:/models/model:ro \
  -v /path/to/v121-assets-r0:/assets:ro \
  -e TF_MODEL_DIR=/models/model \
  -e TF_DSV41_ASSETS=/assets \
  -e TF_EXPERIMENTAL=deepseek \
  -e TF_TP_WORLD=2 -e TF_TP_RANK=0 \
  -e TF_TP_MASTER=<rank0-fabric-address> -e TF_TP_PORT=29620 \
  -e TF_TP_DEVICE=0 -e TF_COMM_BACKEND=nccl \
  -e TF_HTTP_HOST=0.0.0.0 -e TF_HTTP_PORT=8888 \
  -e TF_SERVED_NAME=DeepSeek-V4.1-Flash-1017 \
  -e TF_CONTEXT=32768 -e TF_DSV41_CONTEXT=32768 \
  -e TF_PARALLEL=1 -e TF_DSV41_SLOTS=1 \
  -e TF_MAX_TOKENS=32768 -e TF_THINKING=1 \
  -e TF_TEMPERATURE=1.0 -e TF_TOP_P=0.95 -e TF_TOP_K=20 \
  -e TF_DSV41_DRAFTS=0 \
  -p 8888:8888 \
  tensorfold-dsv41:v1.2.1-sm121
```

`--network host` 下 `-p` 不产生端口映射效果，可以省略；保留仅用于说明服务端口。生产命令应二选一，推荐 host network 时删除 `-p`。

## 启动前必须完成的资产工作

1. 直接使用 `/data/models` 的 EXL3 2.9 bpw 目录作为在线 PACK；使用全量权重中的两个 Engram `embed.weight/scale` 离线生成 TP2 rank 分片。原因：主干加载与 Engram NVMe 行读取在 v1.2.1 中是两条独立路径。
2. 使用 v1.2.1 的 `tf-dsv41-token-map` 从相同 tokenizer/config 生成 `engram-token-map.bin`。原因：hash bucket、multiplier、lookup id 是 loader ABI 的组成部分。
3. 对 world=2、context=32768、slots=1、drafts=0 的精确 profile 生成每 rank AOT，并记录生成 commit、SM、profile 和 SHA-256。原因：AOT 选择依赖实际 launch constexpr/shape。
4. 生成覆盖 `rope.bin`、token-map、`aot.json` 和所有 cubin 的 `SHA256SUMS`。原因：避免两个 rank 使用不同资产而在 collective 阶段挂死。
5. 两端先运行 `TF_MODE=preflight`，再按 rank 1、rank 0 顺序启动。原因：follower 必须先进入 plan link 等待，rank 0 才能完成集体初始化。

## 升级到作者性能参数的顺序

1. NCCL、32K、单槽、无 drafts：验证启动和固定 token 输出。
2. 开 drafts：重新生成相应 AOT，验证 acceptance 和输出一致性。
3. parallel/slots 从 1 升到 4：重新核算 KV/统一内存并生成 profile 资产。
4. 开 v1.2.1 已审计的 graph、side-stream、prefetch knob：每次只启一组并保留回退基线。
5. NCCL 切 RoCE：验证 HCA/GID、mailbox 上限和 failfast。
6. context 从 32K 分级提升，最后才尝试 1M：每一级重新生成 RoPE/AOT并执行长上下文内存测试。

这一路径把“能启动”“结果正确”和“达到作者性能”拆成三个可诊断门槛，避免把 ABI、显存、通信和 kernel profile 问题混在一次失败中。
