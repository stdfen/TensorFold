# TensorFold v1.2.1 双 GB10 Engram 异常审计与处理方案

## 1. 审计结论

两次 `EngramRowOutsideShard` 的直接原因已经由源码、远端 shard header 和作者方案三条证据链交叉确认：当前新增的 TP shard packer 使用 `rows * rank / world` 按总行数等分 Engram 表，但 v1.2.1 运行时使用的是“先按哈希列分 rank，再在各列自己的素数桶中取 row”的布局。24 个哈希列的桶长度并不完全相等，因此哈希列边界不等于总行数的二等分边界。

对于 TP2，rank1 应从第 12 个哈希列的 offset 开始；当前文件却从 `floor(total_rows / 2)` 开始。实际缺口为：

| 层 | rank1 正确起点 | 当前 shard 起点 | 缺失合法 rows |
| --- | ---: | ---: | ---: |
| layer 1 | 192,001,740 | 192,003,084 | 1,344 |
| layer 14 | 192,007,016 | 192,008,341 | 1,325 |

因此 rank1 首个哈希列可能产生位于上述缺口中的合法全局 row。短提示偶然没有命中缺口，所以能返回 `READY`；作者 `stop string` 和 2K 随机词 prompt 命中后，reader 正确地抛出 `EngramRowOutsideShard`，rank1 退出，rank0 通过 TP fail-fast 成对重启。

这不是 stop-string 实现缺陷，也不是 NCCL、RoCE、AOT 或模型权重加载故障。它是 Engram 离线数据切分规则与运行时哈希寻址规则不一致。

## 2. v1.2.1 原生代码与本次适配回顾

### 2.1 v1.2.1 已有实现

v1.2.1 已包含完整的 DeepSeek V4.1 Engram 运行时：

- `engram_hash.zig` 固化模型的 2 个 Engram 层、24 个哈希列、每列的 prime bucket 长度及累计 offset；
- `engram_host.zig` 用 `headShard(rank, world)` 按哈希列切分 rank，并由 `shardIndex` 只选取本 rank 的列；
- `engram_rows.zig` 与 Linux reader 按 shard header 的 `[lo, hi)` 校验全局 row，并以 `row - lo` 转换成文件内偏移；
- `forward_engram.zig` 按实际 TP rank/world 打开 `engram-l<L>-r<R>of<W>.bin`；
- 原生 `tf-dsv41-engram-pack` 只生成完整的 `r0of1` 表，没有提供 TP2 分片生成能力。

这里的关键语义是：文件名中的 rank/world 决定打开哪个文件，但文件的 row 范围必须和该 rank 获得的哈希列覆盖范围一致；不能由文件名推导为“总行数均分”。

### 2.2 当前 v1.2.1 适配做了什么

为了从全量 FP8/FP4 权重生成 TP2 可加载资产，当前分支增加了：

1. `Shard{rank, world}` 与 `--rank/--world` CLI；
2. `range(rows)`，用 `rows * rank / world` 计算 `[lo, hi)`；
3. packer 只写本地范围并使用 `row - lo` 作为 payload 偏移；
4. manifest 写入 shard 的 lo/hi、大小和 SHA-256；
5. verify 只从本 shard 范围抽样，并与原始权重逐行比较；
6. 一个 TP2 合成表测试，证明两个等分文件互斥、payload 和 header 自洽。

这些修改保证了“文件内部一致”，但没有验证“运行时哈希产生的本 rank rows 全部属于文件范围”。因此 200,004 行抽样零差异、manifest 校验和 preflight 都能通过，同时真实 prompt 仍会失败。

## 3. 作者方案如何实现

作者方案没有把 Engram 表按总行数二等分。其 `prepare.sh` 从基础模型直接下载 Engram 所在的 safetensors shards，并把同一组完整文件复制到 worker 本地；文档明确记录每个节点都需要约 203 GB Engram tables。rank0 读取本机/HF snapshot，rank1 读取 `WORKER_ENGRAM_DIR`，两端拥有相同完整表数据。

作者 TensorFold 分支再由运行时根据 rank 选择哈希列和对应 rows。因为底层数据源覆盖全局表，列边界不精确等分也不会产生缺行。作者的 `0035-engram-lead-failure-paths.patch` 解决的是“某 rank 读取失败后如何让整对节点快速失败并释放资源”，不是生成或修正 shard 数据；当前 v1.2.1 的 fail-fast 正是在正确暴露数据问题。

对比关系如下：

| 层面 | 作者方案 | 当前 v1.2.1 适配 | 影响 |
| --- | --- | --- | --- |
| Engram 源 | 每节点完整 safetensors tables | 从全量权重转换为 packed rank shard | 当前需要精确计算 rank row 范围 |
| rank 划分 | 运行时按哈希列选 rows | pack 时按总行数均分，运行时仍按哈希列选 rows | 定义不一致 |
| 数据覆盖 | 每 rank 可访问全部 rows | 每 rank 只能访问 `[lo, hi)` | rank1 边界缺口会直接失败 |
| 故障处理 | 0035 使 pair 快速失败 | v1.2.1 TP fail-fast 成对重启 | 行为正确，不应绕过 |
| 验证 | 作者 smoke/bench 使用真实 prompt | pack verify 仅抽本地范围 | 当前缺少 hash-to-shard 覆盖测试 |

## 4. 异常的完整代码路径

```text
token IDs
  → engram_hash.hashValues
      每个哈希列：row = rolling % prime[column] + offset[column]
  → engram_host.headShard(rank=1, world=2)
      rank1 取得 columns 12..23
  → engram_host.shardIndex
      layer1 最小合法 row = offsets[12] = 192001740
  → engram_rows_linux.unique
      当前 rank1 header.lo = 192003084
  → row < header.lo
  → EngramRowOutsideShard
  → rank1 follower exits
  → rank0 fate channel closes
  → Aborted / HTTP 500，随后 HTTP 503
  → Docker restart policy 重启两个 rank
```

远端文件 header 与当前 packer 计算完全一致：

- layer1 rank0 `[0, 192003084)`，rank1 `[192003084, 384006168)`；
- layer14 rank0 `[0, 192008341)`，rank1 `[192008341, 384016682)`。

运行时真实边界来自 `engram_hash.layers[*].offsets[12]`，分别为 `192001740` 和 `192007016`。这证明异常不是文件损坏或下载不完整，而是 pack 时选择了错误的逻辑边界。

## 5. 为什么已有验证没有发现

| 现有检查 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| header/文件长度 | payload 长度匹配声明的 `[lo, hi)` | lo/hi 是否符合模型哈希布局 |
| manifest SHA-256 | 文件没有在生成后变化 | 文件是否包含所有本 rank 可达 rows |
| 100K 随机 row verify | shard 内数据与全量权重相同 | shard 外是否存在本 rank 合法 row |
| 合成 TP2 单测 | 等分切片读写自洽 | DeepSeek 24 个 prime buckets 的真实边界 |
| engram-probe | shard 内随机 I/O 和 reader 性能 | tokenizer/hash 生成的 row 是否全被覆盖 |
| `READY` 短请求 | 某一小组 hash rows 可读取 | 全部合法 prompt 的覆盖与稳定性 |

测试设计的核心遗漏是：验证输入由“文件范围随机 row”生成，而不是由“真实 token → hash → 本 rank row”生成。前者天然不会走出错误范围。

## 6. 处理方案

### 6.1 推荐方案：按哈希列边界生成 packed shard

保留 packed shard 设计，但把 `Shard.range(rows)` 改为模型布局感知的 `Shard.range(layer, rows)`：

1. 从 `engram_hash.layers` 查找 layer metadata；
2. 验证 release tensor rows 与 metadata.rows 完全一致；
3. 使用与 `Host.headShard` 相同的列切分公式得到 `[column_lo, column_hi)`；
4. `lo = offsets[column_lo]`；
5. `hi = column_hi == columns ? rows : offsets[column_hi]`；
6. manifest、pack、verify 共用同一函数，禁止各自重复实现边界计算。

TP2 修复后的 header 应为：

| 层 | rank0 | rank1 |
| --- | --- | --- |
| layer 1 | `[0, 192001740)` | `[192001740, 384006168)` |
| layer 14 | `[0, 192007016)` | `[192007016, 384016682)` |

该方案不增加总磁盘占用，只需重新生成四个 Engram 文件和 manifest。它也保持 reader、AIO、缓存和 runtime API 不变，改动范围最小。

### 6.2 安全基线方案：每个 rank 使用完整 packed table

为两个 rank 分别生成 header `[0, total)` 的完整文件，只改变文件名为 r0of2/r1of2。这个方案最接近作者“每节点完整表”的数据覆盖方式，适合作为诊断基线，但每节点 Engram 占用会从约 101.5 GB 增至约 203 GB；当前节点余量不足以无风险并存生成，因此不作为首选部署方案。

如果推荐方案修复后仍出现 row 越界，可在释放旧 shard 空间并确认磁盘余量后，用完整表方案做 A/B：完整表通过则说明边界计算仍有遗漏；完整表也失败则应转向 token map/hash metadata 或表版本一致性。

### 6.3 不应采用的方案

- 不应捕获 `EngramRowOutsideShard` 后返回零值：会静默改变模型输出；
- 不应扩大 rank1 `lo` 而不重写 payload：header 与文件内 offset 会错位；
- 不应关闭 fail-fast：只会把快速失败变成 rank hang/NCCL hang；
- 不应先调 RoCE、draft、graph 或并发：它们不影响 row 覆盖关系；
- 不应把问题归因于 stop 参数并跳过 smoke 项：2K prefill 已独立复现。

## 7. 代码调整清单

| 优先级 | 文件 | 调整 | 原因 |
| --- | --- | --- | --- |
| P0 | `engram_pack.zig` | range 接收 layer，并按 hash offsets 划分 | 修复根因 |
| P0 | `engram_manifest.zig` | build/manifestFor 使用同一布局函数 | 防止 manifest 接受错误边界 |
| P0 | `engram_pack_test.zig` | 删除“总行数等分即正确”的假设 | 当前测试固化了错误设计 |
| P0 | 新增/扩展 host-pack 交叉测试 | 对真实 layer metadata 验证每 rank 所有列的最小/最大 row 均在 shard 内 | 建立真正正确性门禁 |
| P1 | `engram_probe.zig` | 增加 `--hash-coverage` 或启动前轻量 coverage 检查 | 让目标机在启动模型前发现布局错误 |
| P1 | preflight | 核验 header lo/hi 等于模型 metadata 的预期边界 | 阻止错误资产进入容器 |
| P1 | manifest schema/内容 | 记录布局来源或 hash metadata digest | 避免同 rows、不同布局被误用 |
| P2 | 文档 | 明确“TP rank 按 hash columns，不按 rows 均分” | 防止后续再次引入同类错误 |

## 8. 必须新增的测试

### 8.1 静态边界测试

- layer1/14，world=1、2、4 的精确 golden lo/hi；
- 所有 rank 范围首尾连续、无 gap、无 overlap，最终 hi 等于 rows；
- 每个 rank 的每个所属 column：`offset[col] >= lo` 且 `offset[col] + prime[col] <= hi`；
- 非 canonical layer 或 rows 不匹配必须拒绝，不能回退到均分。

### 8.2 hash-to-shard 覆盖测试

使用真实 `engram_hash` metadata 和确定性 token 序列生成 rows，对 world=2 的两个 rank 验证每个 row 都属于其预期范围。必须包含每列边界构造或直接遍历 metadata 边界，避免只靠低概率随机命中 1.3K-row 缺口。

### 8.3 pack/verify 测试

verify 的抽样集合除 shard 内随机 rows 外，必须加入本 rank 各 hash column 的：

- first row；
- last row；
- column 交界两侧 rows；
- 从固定 token fixture 得到的真实 rows。

### 8.4 双机验收顺序

1. 两节点 preflight 必须显示修正后的四个 header；
2. 启动后执行一个已知会落入旧缺口的定向 prompt/hash fixture；
3. 作者 `smoke.py` 全部通过，restart count 保持 0；
4. 作者 `bench.py` 先跑 2K 单次，确认不重启；
5. 再跑 2K/8K/30K、2 reps 和 256-token decode；
6. 最后才恢复 RoCE、draft、graph、并发和更长 context 的性能阶段。

## 9. 实施判定

根因置信度为高：源码定义、实际文件 header、两次运行时错误和作者完整表方案相互吻合。推荐修复只涉及离线 Engram range 计算、manifest/preflight 和测试，不需要修改模型 forward、reader、TP fail-fast 或 HTTP server。

在修复并重新生成资产前，当前镜像代码本身仍可保留用于开发，但现有四个 TP2 Engram shard 必须标记为无效，不能继续用于功能或性能验收。
