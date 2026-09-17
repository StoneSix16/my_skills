# nsys + NVTX 的采集与读取

前置：区间由被测进程自己发射（NVTX），nsys 只负责采。

## 两个开关，缺一不可

| 变量 | 作用 |
| :--- | :--- |
| 让进程调 `cudaProfilerStart()` / `cudaProfilerStop()` | 决定 nsys 的采集区间能否开启 |
| 让进程在阶段进出时 push/pop NVTX 区间 | 决定报告里有没有可归因的区间 |

只开前者能采到 kernel，但没有区间可归因；只开后者区间根本不会开启，nsys 采不到任何东西。两者互相独立，都必须设。

**用相位记录文件判断 nsys 是否生效是错的**——相位打点与这些开关无关，只要进程被包装过就会产出。它总是有内容。

## 启动

```
nsys profile -o <out> --force-overwrite=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
    -t cuda,nvtx \
    <launch command>
```

- `--force-overwrite=true` **不能省**。目标文件已存在时，nsys 只打印一行 `Failed to create ...: File exists.`，然后**继续运行目标进程**，最后不落盘。这行提示混在常规输出里极易漏看，表现是「一切正常，只是报告没更新」。
- `--capture-range=cudaProfilerApi` 配合进程内的 `cudaProfilerStart/Stop` 限定采集窗口。没有它，整个长驻进程的报告会大到无法处理。
- `--capture-range-end=stop-shutdown`：区间一闭合，nsys 自己写报告并退出，同时带走目标进程。省掉了「什么时候按 Ctrl-C」这个时机问题。
- nsys **会跟进子进程**（实测 `bash → torchrun → worker` 三层派生都能采到），不需要 `--trace-fork-before-exec`。

## 区间必须闭合在进程内

nsys **只为已闭合的区间写 kernel 数据**。把区间闭合在被测的那个请求/迭代内部，之后的一切都在采集之外。

收尾方式的实测结果：

| 方式 | 结果 |
| :--- | :--- |
| 区间闭合后 Ctrl-C | 安全。报告在 `Stop()` 时已落盘 |
| 区间**中途** Ctrl-C | 报告文件存在，但**没有 kernel 数据** |
| `stop-shutdown` | 推荐，不需要手动收尾 |

## 成功与失败的判据

按顺序排查，每步都有可观测量：

| 现象 | 含义 |
| :--- | :--- |
| 启动时进程打印出自定义提示 | 开关已生效 |
| 运行中 nsys 打印 `Capture range started / ended in the application` | 区间开过也关过 |
| 报告文件被新建或更新 | 采集成功 |
| **打印 `Generated:` 后无文件路径，且目录里找不到报告** | 区间**从未开启**，nsys 不落盘 |
| 报告存在，但 `nsys stats` 报 `does not contain CUDA kernel data` | 区间开了，但进程在区间内被杀 |
| 打印 `Failed to create ...: File exists.` | 漏了 `--force-overwrite` |

「目录里找不到报告」与「报告是空的」是两种不同的失败，指向不同原因。

## 读报告

```
nsys stats --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum:nvtx-name,cuda_kern_exec_sum <rep>
```

| 报告 | 给什么 | 能否相加 |
| :--- | :--- | :--- |
| `nvtx_gpu_proj_sum` | 区间 CPU 墙钟（Range Time）、GPU 跨度（Proj Time）、子区间数 | **不可跨嵌套区间求和** |
| `cuda_gpu_kern_sum:nvtx-name` | 每个区间内 kernel 时长之和 | 可相加，合计等于 kernel 总时长 |
| `cuda_kern_exec_sum` | 启动开销与执行时长的占比 | —— |

**同一个区间有三个不同的数，别混**：CPU 括号（`Range Time`，实例时长之和）、投影跨度（`Proj Time`，首末 GPU op 之间）、投影 op 的时长之和（`cuda_gpu_kern_sum`）。只有最后一个是可加的。

`Range Time` 与括号并集在单线程下相等——NVTX 是栈式的，同一线程上同名区间的实例不可能同时打开（实测三个区间，和与并集逐位相同）。需要并集的是**跨线程**统计，或把嵌套的多层混在一起算的时候。

`cuda_gpu_kern_sum:nvtx-name` **只统计 kernel**；`nvtx_gpu_proj_sum` 的 `Total GPU Ops` 统计的是 kernel + memcpy + memset，所以两个报告的 op 数对不上是正常的（实测同一个区间 8640 对 8010，差的是一次调用一个的 memset）。

需要更细的归因（按流、按路径、空闲构成）时，导出 sqlite 自己查；`scripts/nsys_attr.py` 封装了这些查询。**归因规则只有一条**（kernel 归给发起它的那个 CPU 时刻所在的最内层区间），见 SKILL.md 的 L2——不要用 CPU 窗口去框 GPU 的执行。

### 导出 sqlite 的两个坑

1. `nsys stats` 会复用同名 `.sqlite`。若它比 `.nsys-rep` 旧，nsys 会**拒绝**并退出（`Existing SQLite export found ... Use --force-export=true to update`）。**不要用 `--force-export` 覆写**——那可能是别人的产物；把报告复制到临时目录再导，或直接换输出路径。
2. `--format csv` 的输出前几行是 `NOTICE: ...`，**不是表头**。按「行首是否为数字」过滤。

### `cuda_gpu_kern_sum:nvtx-name` 的行名

格式是「区间名/kernel名」**拼接**。区间名自身可能含 `/`，所以不能按第一个 `/` 切分，应拿已知的区间词表做**最长前缀匹配**。

### 三张表各有什么

| 表 | 关键列 |
| :--- | :--- |
| `CUPTI_ACTIVITY_KIND_KERNEL` | `start`/`end`（**GPU 执行区间**）、`streamId`、`correlationId`、`shortName`/`demangledName`、`gridX/Y/Z`、`blockX/Y/Z`、`registersPerThread`、`staticSharedMemory`、`dynamicSharedMemory` |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | `start`/`end`（**CPU 上这次调用的起止**）、`globalTid`、`correlationId`、`nameId` |
| `NVTX_EVENTS` | `start`/`end`、`globalTid`、`text`/`textId`、`domainId`、`eventType`、`rangeId` |

时间戳单位都是纳秒，且 CPU 与 GPU 事件在**同一根轴上**（nsys 做过时钟对齐），可以直接比较。

**归因链**：`kernel.correlationId` → `runtime.correlationId` → 拿到 `(globalTid, CPU 时刻)` → 该线程该时刻最内层的 `NVTX_EVENTS`。

**两个坑**：名字列（`shortName` / `demangledName` / `text` 为空时的 `textId`）存的是 `StringIds` 的 **id 而不是字符串**，要 join 才能拿到名字；框架自己发的区间名字落在 `textId`，只读 `text` 会整批漏掉。

## 读 timeline

stats 报告是聚合结果，timeline 是原始事件。两者源自同一份数据，但很容易读出不同结论——因为 timeline 上一行行画的**不是同一类东西**。

### 一根轴，两根时钟

| 区域 | 时钟 | 画的是什么 | 对应数据 |
| :--- | :--- | :--- | :--- |
| Threads | CPU | 代码在 CPU 上执行 | 用户代码 / OS 运行时 |
| Threads → **NVTX** | CPU | 代码里 push/pop 的区间 | `NVTX_EVENTS` |
| Threads → **CUDA API** | CPU | 每次 CUDA 调用的主机侧耗时 | `CUPTI_ACTIVITY_KIND_RUNTIME` |
| CUDA HW | GPU | 设备上真正在跑什么 | `CUPTI_ACTIVITY_KIND_KERNEL` / `MEMCPY` |
| CUDA HW → **NVTX** | GPU | 上面那批区间的**投影** | `nvtx_gpu_proj_*` |

**每一条 bar 只属于一根时钟。** 同一个 kernel 在图上出现三次——发起它的 API 调用（CPU）、它自己的执行（GPU）、被投进的那条 NVTX——**三者水平位置都不同**。这不是画错，是异步执行本身。

### CUDA API 行

**每次 kernel 发射就是一次独立的 CPU 调用**，所以这一行一个 kernel 一条 bar。**条的宽度是 CPU 发射这次调用的耗时，不是 kernel 的执行时间**（通常几微秒）。

- 条窄且密 → 发射得快，不是瓶颈
- 条很宽 → 发射本身贵（host-bound）。此时优化算子没有意义，要减少发射次数
- 这一行没有空隙 → CPU 一直在忙，GPU 大概率在等它

若 bar 上标着 kernel 名，那是 nsys 用 `correlationId` 关联后被发射的 kernel 名；**宽度仍属于发射调用**。

### 两条 NVTX 行为什么不对齐

| | Threads → NVTX | CUDA HW → NVTX |
| :--- | :--- | :--- |
| 是什么 | 原始 push/pop 括号 | **投影** |
| 时间轴 | CPU 的 `NVTX_EVENTS.start/end` | 该区间**发起**的 GPU op 从第一个到最后一个的跨度 |
| 语义 | 「CPU 执行这段代码花了多久」 | 「这段代码在 GPU 上造成了多大范围」 |
| 报告里的名字 | `Range Time` | `Proj Time` |

**不对齐是必然的**：水平偏移 = 队列延迟（CPU 提前入队、GPU 滞后执行）；宽度差 = 两个口径本身的差。嵌套区间投影到 GPU 上还会互相重叠，所以 `Proj Time` **不可相加**。

投影的语义是 **launched within**（这段 CPU 区间里**发起**了哪些 GPU 活动），不是 executed within。nsys 自己用的也是发起时刻口径——`cuda_gpu_kern_sum:nvtx-name` 的结果与按归因规则手算的**逐项一致**（实测同一采集内全部区间与 kernel 数量、时长都吻合）。

### 同名区间会出现在多条 CUDA HW 行上

CUDA HW 行按 stream 分行，投影**也按流画**：一个区间的工作跨了几条流，就在那几行上各画一条，**每条只覆盖那条流上的那部分**。

所以看到同一个区间名在两条 GPU 行上各有一条，含义是**这段代码的 GPU 工作被拆到了多条流**，不是测量出错。（框架的分片搬运、通信通常有自己的流，于是「分片单元」那层区间就横跨两条行——这个现象能直接定位搬运落在哪个区间上。）

**反之**：一个区间只出现在一条 CUDA HW 行上，可以确定它的 GPU 工作没有跨流。

### 问题 → 看哪一行

| 你想知道 | 看哪一行 |
| :--- | :--- |
| GPU 有没有闲着 | **只看 CUDA HW 行**的空隙。空隙 = GPU 无事可做 |
| 这个 kernel 是谁发起的 | 从 kernel 条找同 `correlationId` 的 **CUDA API 条**，它在哪个 NVTX 括号内就归谁 |
| GPU 跑这个 kernel 时 CPU 在干嘛 | 垂直对齐到 Threads 行（这是窗口视角，答的是另一个问题） |
| 队列有多深 | 同一个 kernel 的 API 条与 kernel 条之间的**水平距离** |
| CPU 是不是瓶颈 | Threads 行是否连续、API 条是否很宽、CUDA HW 行是否大片空白 |

**不要用「NVTX 条压在 kernel 条上面」判断归属**——那是 CPU 窗口法，规则见 SKILL.md 的 L2。

## 权限

- CUDA 追踪与 NVTX 捕获**无需提权**。`perf_event_paranoid` 只影响 CPU 采样与上下文切换追踪。
- 硬件指标采样与 `ncu` **需要**放宽 `perf_event_paranoid`（或 root）。不可用时，「受算力还是带宽约束」只能靠解析值（算术强度）与间接判据（耗时随规模变化的斜率）。

## 区间怎么划

划在**计算单元的边界**上：框架的分片单元、模块的 forward 边界、循环的单步。

一条经验：**父层不要为了「完整」而多包一层**。如果外层区间已经覆盖了它，再嵌一层会让子区间看起来像是要从某个东西里减掉，反而制造理解负担。

另一条：观察对象不是可包裹的方法时（局部闭包、内联表达式、一串元素算子），可以走「复制方法体再手动插桩」这条路——它的代价与防脱节的机制见 SKILL.md 的「探针怎么放」。
