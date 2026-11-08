---
title: "BF16 dense GEMM small M：逐实现、逐 Point 的本地验收边界"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1, 2, 4, 8, 16, 32, 64, 128"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1 whole-call per implementation
measured:
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n1024_k7168, UUID 4c0839a9bd82123e, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_4c0839a9bd82123e/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n2048_k2048, UUID 697eeb2233f25b25, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_697eeb2233f25b25/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n2048_k7168, UUID 1f5a93c79dbd12bf, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_1f5a93c79dbd12bf/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "3/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n3584_k512, UUID 449a025fc7aa0de6, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_449a025fc7aa0de6/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n4096_k2048, UUID 37c909c4b629ad59, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_37c909c4b629ad59/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "7/7 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n512_k4096, UUID e42f6c833c101f69, M 1, 2, 4, 8, 16, 32, 64", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_e42f6c833c101f69/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n6144_k1536, UUID 1d3a1fcaf77631e1, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_1d3a1fcaf77631e1/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n6144_k4096, UUID 3e2804d64ce6b4c7, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_3e2804d64ce6b4c7/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n6144_k6144, UUID bfae0e0137d0d54d, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_bfae0e0137d0d54d/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "6/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n64_k4096, UUID 8bec4c4d3db66ba0, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_8bec4c4d3db66ba0/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "6/8 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n7168_k512, UUID d8bd1ca29ef0a990, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_d8bd1ca29ef0a990/validation/report.json", "date": "2026-11-08"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n1024_k7168/4c0839a9bd82123e.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/697eeb2233f25b25.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/449a025fc7aa0de6.json
  - aklr/ops/gemm/impls/flydsl/gemm_n4096_k2048/37c909c4b629ad59.json
  - aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/e42f6c833c101f69.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/1d3a1fcaf77631e1.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k4096/3e2804d64ce6b4c7.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k6144/bfae0e0137d0d54d.json
  - aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/8bec4c4d3db66ba0.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/d8bd1ca29ef0a990.json
---

# BF16 dense GEMM：小 M逐实现本地证据

## Applies when

只覆盖 `gemm@1.0`、`row_major`、A=(M,K)、B=(N,K)，bf16输入/输出、trans_b=True、无bias；compare为bitwise、fixed element order、零violation fraction。
下表逐实现列出**离散** authored且已测M；没有把相邻值外推为连续范围，也没有声称未列出的Point已测。调用/服务规则见 [AKLR算子入口](../../../../aklr/ops/README.md)。

| UUID | name（N/K固定） | authored且已测 M |
|---|---|---|
| `4c0839a9bd82123e` | `gemm_n1024_k7168` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `697eeb2233f25b25` | `gemm_n2048_k2048` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `1f5a93c79dbd12bf` | `gemm_n2048_k7168` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `449a025fc7aa0de6` | `gemm_n3584_k512` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `37c909c4b629ad59` | `gemm_n4096_k2048` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `e42f6c833c101f69` | `gemm_n512_k4096` | 1, 2, 4, 8, 16, 32, 64 |
| `1d3a1fcaf77631e1` | `gemm_n6144_k1536` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `3e2804d64ce6b4c7` | `gemm_n6144_k4096` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `bfae0e0137d0d54d` | `gemm_n6144_k6144` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `8bec4c4d3db66ba0` | `gemm_n64_k4096` | 1, 2, 4, 8, 16, 32, 64, 128 |
| `d8bd1ca29ef0a990` | `gemm_n7168_k512` | 1, 2, 4, 8, 16, 32, 64, 128 |

## Counter-signals

- 本地独立测后审稿支持所列Point的整调用correctness和timing；**没有独立组件/单因子对照**。Mechanism仅是原始源码事实，不能用整调用倍数证明DMA、prefetch、tile order、priority、MFMA原子或LLVM hint各自的收益。
- 测试report的head/base都是 `a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`，但候选是untracked新source；不是“该commit已包含kernel”。绑定依据是schema SHA256、kernel content fingerprint、report/receipt hash，逐项在Evidence及映射归档中列出。不是CI，也不是线上已选择证明。
- 按固定版本默认规则复算的资格不等于线上阈值：worker环境原件仅记录版本/GPU，完整runtime配置及环境变量未归档，不能认证 `AKLR_MIN_SPEEDUP` 的实际值；task receipt仅记录group/path/sha256/id，不是环境凭据；当前payload尚未写accuracy/speedup，因此本页不声称当前未发布schema已默认服务。
- 若复用跨工作组split-K/tag/counter路径，另核发布/读取的memory order、tag碰撞/回绕、未初始化scratch、多stream并发及graph重放；整调用通过和源码中的固定加法顺序都不足以证明协议安全。
- tile重排只记录源码的逻辑分组；不把block_id模8当作各stream下不变的绝对XCD编号，不声称已确认实际cache命中。
- 本地证据文件可能失效；发布前归档报告和fingerprint，并重新走官方验证。
- 下列Point正确但**不具默认服务资格**：`449a025fc7aa0de6` M=1，1.020 *(field, E4)*；`449a025fc7aa0de6` M=2，1.011 *(field, E4)*；`449a025fc7aa0de6` M=4，1.014 *(field, E4)*；`449a025fc7aa0de6` M=8，1.020 *(field, E4)*；`449a025fc7aa0de6` M=16，1.022 *(field, E4)*；`8bec4c4d3db66ba0` M=1，1.026 *(field, E10)*；`8bec4c4d3db66ba0` M=2，1.027 *(field, E10)*；`d8bd1ca29ef0a990` M=1，1.023 *(field, E11)*；`d8bd1ca29ef0a990` M=128，0.980 *(field, E11)*。
- N=3584/K=512 wrapper依赖固定AITER的非公开编译/launch符号与空槽约定；该源码未声明requires版本约束。本次固定版本已测不表示升级兼容，不能称独立FlyDSL实现。

## Mechanism

可借用的是按离散M选body、分清wave split与跨工作组split、按数据布局选择寄存器或LDS路径；迁移前按自己的Point重新定价。以下每段只陈述源码，不给组件收益。

### `4c0839a9bd82123e` — `gemm_n1024_k7168`

M=1,2 为每wave一输出列的寄存器GEMV；M=4,8,16,32,64,128 为四波split-K MFMA，B作为首操作数，wave-private XOR LDS DMA环；:268–288 依wave顺序归约再转bf16。不能将最小两档的GEMV称作MFMA收益。（S1，行号均为该schema内 `kernel.py`。）

### `697eeb2233f25b25` — `gemm_n2048_k2048`

M=1,2,4,8 调用 `_compile_skw8`，四波分K、每波完整K片驻留LDS；M=16,32,64,128 调用 `_compile_skw_av`，四波各有DMA环，按wave索引顺序在LDS归约后转bf16。一调用一次launch，无跨工作组workspace。归约代码 :295–319、:398–418。（S2，行号均为该schema内 `kernel.py`。）

### `1f5a93c79dbd12bf` — `gemm_n2048_k7168`

M=1,2,4,8 四波在一个工作组内分K，经私有XOR LDS环；M=16,32,64,128 再跨工作组分K四路。per-call workspace保存fp32 partial与各split的(queue指针、dispatch id)派生tag；:368–381 固定split顺序fold，:397–423 发布tag、读取所有tag并条件fold，不是atomic最后到达计数。整调用审稿不证明tag碰撞、发布可见性、并发或唯一fold者。（S3，行号均为该schema内 `kernel.py`。）

### `449a025fc7aa0de6` — `gemm_n3584_k512`

这是 **AITER配置wrapper，不是自包含GPU kernel**：依赖 `aiter.ops.flydsl.kernels.splitk_hgemm.compile_hgemm_kernel` 和 `tensor_shim._run_compiled/ptr_arg`。M=1,2,4,8,16 配16×32×256；M=32,64 配32×32×256；M=128 配32×64×128、STAGES=3、B_TO_LDS=True。wrapper未传split-K参数，不能仅凭此证明外部kernel内部协议。:30 将B指针填入bias/semaphore/signal槽，依赖固定AITER在无bias/无split-K路径不读取它们；不能移植成通用ABI承诺。（S4，行号均为该schema内 `kernel.py`。）

### `37c909c4b629ad59` — `gemm_n4096_k2048`

独立A/B LDS DMA环，四波布局按M改变：M=1,2,4,8,16 为(1,1,4)，M=32 为(1,2,2)，M=64,128 为(2,2,1)，分别沿M/N/K分波。B作为MFMA首操作数；只有wk>1才有K片LDS归约，不能把所有M都称作wave split-K。:279–298 为收尾/归约，:308–339 编译并单次调用。（S5，行号均为该schema内 `kernel.py`。）

### `e42f6c833c101f69` — `gemm_n512_k4096`

M=1,2,4,8,16,32 `_compile_packed`将真实行与K片编码进MFMA操作数，仅匹配slice的对角partial有效；:147–190 先按slice顺序再wave顺序LDS归约。M=64调用独立 `_compile_m64`(:210–367)，不可把它当同一个packed body。都是单launch，不把packing或归约单独归因成收益。（S6，行号均为该schema内 `kernel.py`。）

### `1d3a1fcaf77631e1` — `gemm_n6144_k1536`

M=1,2,4,8 两波沿N且wk=1；M=16,32 四波布局(1,2,2)；M=64 八波(2,2,2)；M=128 十六波(2,4,2)。A/B独立DMA环及BF16 MFMA；wk>1时LDS归约。:252–354 还有非输出工作组的B预取分支和常量配置；源码写了预取，不表示已独立量出其收益或确认真实L2命中。（S7，行号均为该schema内 `kernel.py`。）

### `3e2804d64ce6b4c7` — `gemm_n6144_k4096`

单launch的B列strip，K在工作组波间分片，A/B经XOR LDS DMA环到16×16×32 BF16 MFMA。`_SKINNY`选择环深、K起点旋转、row split；:153–166是tile索引重排。这些索引只表示布局提示，不证明绝对XCD编号或旋转的性能收益。（S8，行号均为该schema内 `kernel.py`。）

### `bfae0e0137d0d54d` — `gemm_n6144_k6144`

M=1,2,4,8,16,32 为strip DMA环单launch；M=64 为四路跨工作组split-K、per-call未初始化workspace、dispatch-tag到达计数，最后工作组按split索引顺序归约(:535–615)；M=128 为四路split-K加独立fold第二launch(:623–651)。不能把M=128描述为tag单launch，不能用整调用通过证明M=64协议安全。（S9，行号均为该schema内 `kernel.py`。）

### `8bec4c4d3db66ba0` — `gemm_n64_k4096`

各M从 `_CONFIG` 选小输出tile，A/B直接buffer-load到寄存器，`fdot2_f32_bf16`累加(:179–206)，不是MFMA。lane的固定xor/butterfly归约后，W>1才在LDS按wave顺序归约(:209–242)，一launch、无跨工作组计数。（S10，行号均为该schema内 `kernel.py`。）

### `d8bd1ca29ef0a990` — `gemm_n7168_k512`

M=1,2,4,8,16 配16×64；M=32 配32×32；M=64 配32×64；M=128 配64×64。整个K=512按64深chunk预先DMA进XOR LDS，逐chunk wait后MFMA(:177–193)，波沿M/N而非跨工作组分K，单launch。（S11，行号均为该schema内 `kernel.py`。）

## Result

基准树固定为 `a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`；baseline是 `aklr.ops.gemm.baseline.gemm_baseline` 调用的AITER tuned `gemm_a16w16`，不是泛指任意vendor GEMM。
CI声明的完整AITER pin为 `dbd8bf5bd624120197a7a26780a8c72201824f0f`；本地worker `environment.json`实际记录短版本 `0.1.21.dev66+gdbd8bf5bd`、FlyDSL 0.3.2、Torch `2.11.0+rocm10.0.0`、gfx950，`install_aiter.log`记录该短版本wheel安装成功；report platform记录MI355X及ROCm `7.15.26333`。短版本与CI pin前缀一致，但不据此认证实际源码完整commit或完整runtime配置；这些版本不来自task receipt。各原件见对应E行，baseline/CI pin来源见 S12。
profile v18：CUDA graph、逐调用flush L2，seeded shuffle rounds，16/32两次look，alpha每look=0.025；整调用timer累加GPU duration，mean within replay、median across replay，trial为mean log-ratio的Student-t下界。N=1表示每实现一个独立验证任务，不将任务内rounds称作多次独立组件实验。
本组所有Point正确。默认资格按 [官方规则](../../../../aklr/ops/README.md) 和固定版本审稿复算；entry列基于未舍入lowerbound，published speedup是三位小数，故显示1.020的两个Point可以具有不同entry资格。
**Incumbent单独列示为N/A**：审稿逐完整Point核对固定base，没有默认可服务incumbent（或同kernel），所以没有incumbent timing；不能声称击败其他base、其他服务kernel。以下baseline数不冒充incumbent比值。

| UUID | M | baseline下界（未舍入值展示6位） | published speedup | baseline entry | incumbent ratio | 默认资格（复算） |
|---|---:|---:|---:|---|---|---|
| `4c0839a9bd82123e` | 1 | 1.477749 | 1.478 *(field, E1)* | 是 | N/A | 是 |
| `4c0839a9bd82123e` | 2 | 1.405065 | 1.405 *(field, E1)* | 是 | N/A | 是 |
| `4c0839a9bd82123e` | 4 | 1.414082 | 1.414 *(field, E1)* | 是 | N/A | 是 |
| `4c0839a9bd82123e` | 8 | 1.258848 | 1.259 *(field, E1)* | 是 | N/A | 是 |
| `4c0839a9bd82123e` | 16 | 1.238280 | 1.238 *(field, E1)* | 是 | N/A | 是 |
| `4c0839a9bd82123e` | 32 | 1.089918 | 1.090 *(field, E1)* | 是 | N/A | 是 |
| `4c0839a9bd82123e` | 64 | 1.156854 | 1.157 *(field, E1)* | 是 | N/A | 是 |
| `4c0839a9bd82123e` | 128 | 1.084762 | 1.085 *(field, E1)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 1 | 1.464990 | 1.465 *(field, E2)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 2 | 1.558844 | 1.559 *(field, E2)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 4 | 1.543911 | 1.544 *(field, E2)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 8 | 1.559993 | 1.560 *(field, E2)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 16 | 1.578391 | 1.578 *(field, E2)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 32 | 1.891897 | 1.892 *(field, E2)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 64 | 1.771382 | 1.771 *(field, E2)* | 是 | N/A | 是 |
| `697eeb2233f25b25` | 128 | 1.438038 | 1.438 *(field, E2)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 1 | 1.186341 | 1.186 *(field, E3)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 2 | 1.181218 | 1.181 *(field, E3)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 4 | 1.224266 | 1.224 *(field, E3)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 8 | 1.174560 | 1.175 *(field, E3)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 16 | 1.126163 | 1.126 *(field, E3)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 32 | 1.213094 | 1.213 *(field, E3)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 64 | 1.163149 | 1.163 *(field, E3)* | 是 | N/A | 是 |
| `1f5a93c79dbd12bf` | 128 | 1.240476 | 1.240 *(field, E3)* | 是 | N/A | 是 |
| `449a025fc7aa0de6` | 1 | 1.020051 | 1.020 *(field, E4)* | 是 | N/A | 否 |
| `449a025fc7aa0de6` | 2 | 1.011053 | 1.011 *(field, E4)* | 否 | N/A | 否 |
| `449a025fc7aa0de6` | 4 | 1.013806 | 1.014 *(field, E4)* | 否 | N/A | 否 |
| `449a025fc7aa0de6` | 8 | 1.019516 | 1.020 *(field, E4)* | 否 | N/A | 否 |
| `449a025fc7aa0de6` | 16 | 1.021835 | 1.022 *(field, E4)* | 是 | N/A | 否 |
| `449a025fc7aa0de6` | 32 | 1.171552 | 1.172 *(field, E4)* | 是 | N/A | 是 |
| `449a025fc7aa0de6` | 64 | 1.298128 | 1.298 *(field, E4)* | 是 | N/A | 是 |
| `449a025fc7aa0de6` | 128 | 1.274794 | 1.275 *(field, E4)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 1 | 1.361648 | 1.362 *(field, E5)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 2 | 1.375412 | 1.375 *(field, E5)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 4 | 1.366573 | 1.367 *(field, E5)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 8 | 1.419866 | 1.420 *(field, E5)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 16 | 1.464402 | 1.464 *(field, E5)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 32 | 1.385102 | 1.385 *(field, E5)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 64 | 1.316348 | 1.316 *(field, E5)* | 是 | N/A | 是 |
| `37c909c4b629ad59` | 128 | 1.163228 | 1.163 *(field, E5)* | 是 | N/A | 是 |
| `e42f6c833c101f69` | 1 | 1.232662 | 1.233 *(field, E6)* | 是 | N/A | 是 |
| `e42f6c833c101f69` | 2 | 1.209180 | 1.209 *(field, E6)* | 是 | N/A | 是 |
| `e42f6c833c101f69` | 4 | 1.216808 | 1.217 *(field, E6)* | 是 | N/A | 是 |
| `e42f6c833c101f69` | 8 | 1.227066 | 1.227 *(field, E6)* | 是 | N/A | 是 |
| `e42f6c833c101f69` | 16 | 1.241318 | 1.241 *(field, E6)* | 是 | N/A | 是 |
| `e42f6c833c101f69` | 32 | 1.298193 | 1.298 *(field, E6)* | 是 | N/A | 是 |
| `e42f6c833c101f69` | 64 | 1.273375 | 1.273 *(field, E6)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 1 | 1.636794 | 1.637 *(field, E7)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 2 | 1.631079 | 1.631 *(field, E7)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 4 | 1.628818 | 1.629 *(field, E7)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 8 | 1.606261 | 1.606 *(field, E7)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 16 | 1.544889 | 1.545 *(field, E7)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 32 | 1.497920 | 1.498 *(field, E7)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 64 | 1.457703 | 1.458 *(field, E7)* | 是 | N/A | 是 |
| `1d3a1fcaf77631e1` | 128 | 1.473621 | 1.474 *(field, E7)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 1 | 1.307260 | 1.307 *(field, E8)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 2 | 1.308391 | 1.308 *(field, E8)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 4 | 1.321339 | 1.321 *(field, E8)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 8 | 1.230177 | 1.230 *(field, E8)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 16 | 1.296701 | 1.297 *(field, E8)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 32 | 1.264235 | 1.264 *(field, E8)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 64 | 1.228344 | 1.228 *(field, E8)* | 是 | N/A | 是 |
| `3e2804d64ce6b4c7` | 128 | 1.141700 | 1.142 *(field, E8)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 1 | 1.161614 | 1.162 *(field, E9)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 2 | 1.104085 | 1.104 *(field, E9)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 4 | 1.197271 | 1.197 *(field, E9)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 8 | 1.095102 | 1.095 *(field, E9)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 16 | 1.111169 | 1.111 *(field, E9)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 32 | 1.096910 | 1.097 *(field, E9)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 64 | 1.234531 | 1.235 *(field, E9)* | 是 | N/A | 是 |
| `bfae0e0137d0d54d` | 128 | 1.309959 | 1.310 *(field, E9)* | 是 | N/A | 是 |
| `8bec4c4d3db66ba0` | 1 | 1.025736 | 1.026 *(field, E10)* | 是 | N/A | 否 |
| `8bec4c4d3db66ba0` | 2 | 1.026609 | 1.027 *(field, E10)* | 是 | N/A | 否 |
| `8bec4c4d3db66ba0` | 4 | 1.036068 | 1.036 *(field, E10)* | 是 | N/A | 是 |
| `8bec4c4d3db66ba0` | 8 | 1.349139 | 1.349 *(field, E10)* | 是 | N/A | 是 |
| `8bec4c4d3db66ba0` | 16 | 1.166338 | 1.166 *(field, E10)* | 是 | N/A | 是 |
| `8bec4c4d3db66ba0` | 32 | 1.173173 | 1.173 *(field, E10)* | 是 | N/A | 是 |
| `8bec4c4d3db66ba0` | 64 | 1.180002 | 1.180 *(field, E10)* | 是 | N/A | 是 |
| `8bec4c4d3db66ba0` | 128 | 1.273336 | 1.273 *(field, E10)* | 是 | N/A | 是 |
| `d8bd1ca29ef0a990` | 1 | 1.022538 | 1.023 *(field, E11)* | 是 | N/A | 否 |
| `d8bd1ca29ef0a990` | 2 | 1.095757 | 1.096 *(field, E11)* | 是 | N/A | 是 |
| `d8bd1ca29ef0a990` | 4 | 1.103884 | 1.104 *(field, E11)* | 是 | N/A | 是 |
| `d8bd1ca29ef0a990` | 8 | 1.119674 | 1.120 *(field, E11)* | 是 | N/A | 是 |
| `d8bd1ca29ef0a990` | 16 | 1.146986 | 1.147 *(field, E11)* | 是 | N/A | 是 |
| `d8bd1ca29ef0a990` | 32 | 1.256190 | 1.256 *(field, E11)* | 是 | N/A | 是 |
| `d8bd1ca29ef0a990` | 64 | 1.376860 | 1.377 *(field, E11)* | 是 | N/A | 是 |
| `d8bd1ca29ef0a990` | 128 | 0.979974 | 0.980 *(field, E11)* | 否 | N/A | 否 |

本组 87 个Point全部accuracy passed，78 个默认资格复算为是；不聚合不同Point的speedup。 *(field, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11)*

## Reproduce

在MI355X及上述固定依赖、基准树和完全匹配fingerprint的source下，逐schema执行；这是复现实验说明，本次KB写作没有运行GPU/网络。

```bash
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n1024_k7168/4c0839a9bd82123e.json --base a12866b1e40e86ea6aff7ac24cbe5634132f6b2e
```

其余实现逐个替换为front matter的source路径。检查完整Point、audit/bench findings、baseline/incumbent分母、未舍入bounds及环境；报告按E行归档，hash核对后才与本页比较。不要只看进程rc或bench passed。
published schema以后可能包含测量字段，schema字节hash会随之改变；kernel content fingerprint用于区分源码身份，不能拿不同source的同名kernel报告代替。

## Rollback

回滚对应UUID的新增schema和本页该UUID段（未commit时移除该候选文件），让dispatch重新选择固定base已有实现或baseline；没有默认资格的Point不能强行放开阈值。
不要把整contract disable当作逐实现回滚。涉及tag/counter路径发现协议疑点时停用该候选并独立验证，不修改原KB来宣称所有类似协议安全。

## Evidence

以下均为本地独立AKLR audit/bench及测后审稿，非CI；`<schema>.json#kernel.py:<line>` 指 `kernel.sources[0].content` 的原始源码行。源码与交付原byte一致，未用docstring替代执行分支核查。
- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `4c0839a9bd82123e` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_4c0839a9bd82123e/validation/report.json，report SHA256 `b8844ae33a597943f167f6ce389a1aca29678503f93c01c92fe16f3630685647`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_4c0839a9bd82123e/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_4c0839a9bd82123e]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S1 aklr/ops/gemm/impls/flydsl/gemm_n1024_k7168/4c0839a9bd82123e.json#kernel.py:301 上述机制及run分支；source SHA256 `729e1459ca1d91f2ed18598d62495249c1fa7708a501aaf260526d44d2fc9c09`；schema SHA256 `a75082067eaab07883160e2da0256ada3ff6f5071fff96346dd5b837f9da21ad`；official kernel content fingerprint `131553682bb7ab996af86ae9a365b24157b78125c0405df55344f36a3cef275f`（未commit候选，不使用final KF commit冒充PR source commit）。
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `697eeb2233f25b25` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_697eeb2233f25b25/validation/report.json，report SHA256 `104e6aa88eb9ab8e233fc0049b9fa60efb0cd09adef7ae931548eb245f03ede6`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_697eeb2233f25b25/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_697eeb2233f25b25]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S2 aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/697eeb2233f25b25.json#kernel.py:435 上述机制及run分支；source SHA256 `06b986e887db6a69e48ba8fb81ccacc302da78a7a3df0134f4d26faa5c92d19b`；schema SHA256 `8fe3330e787ec9c286fcd4aa78e8114fb772727b95c033172bd675076d11408f`；official kernel content fingerprint `3b56dda781c2125efa5d92240ade3715d771941c8e9b41ae95d0b2ec4e3232f6`（未commit候选，不使用final KF commit冒充PR source commit）。
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `1f5a93c79dbd12bf` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_1f5a93c79dbd12bf/validation/report.json，report SHA256 `4766837485517dd3f45f93e2eb2ef18ac50e72ac00777779fd4b78e7a757cab7`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_1f5a93c79dbd12bf/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_1f5a93c79dbd12bf]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S3 aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json#kernel.py:441 上述机制及run分支；source SHA256 `e611d57b3d859a9c817b5d834d95c1dd8e96b0e8e059a2f0f20d9a49734c00d7`；schema SHA256 `6701af9c3879df9e6aa3275a44ebd5cbc0e16bc81c9f9f8c4a545ad60cda32bd`；official kernel content fingerprint `e542953f5a06396d7ca63d62a4dcc5585587b6c80bc04e46dea3744227312a94`（未commit候选，不使用final KF commit冒充PR source commit）。
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `449a025fc7aa0de6` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_449a025fc7aa0de6/validation/report.json，report SHA256 `127dd05e98c7a9d8c45a18ef73285ef189544b6e757c59082ed98643de2d64e0`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_449a025fc7aa0de6/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_449a025fc7aa0de6]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S4 aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/449a025fc7aa0de6.json#kernel.py:10 上述机制及run分支；source SHA256 `89844b4ccc4529d9bd0960318901e3b86d653710d2a19c1082ce83987ddcae28`；schema SHA256 `62e53fd6c6635cd086c9e27143faba90a9e0dd91b2d9754a4f9cb60accfd1fd2`；official kernel content fingerprint `8f95eca19e17081c6d73d6a8bfd4e7341264f5ff44cdb5b932ebbd486e14d21b`（未commit候选，不使用final KF commit冒充PR source commit）。
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `37c909c4b629ad59` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_37c909c4b629ad59/validation/report.json，report SHA256 `47d2a326f7892d436aec394dd1be9e3e21efefdb1fbe6bba477debfac5252eac`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_37c909c4b629ad59/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_37c909c4b629ad59]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S5 aklr/ops/gemm/impls/flydsl/gemm_n4096_k2048/37c909c4b629ad59.json#kernel.py:45 上述机制及run分支；source SHA256 `fa159d7235b5e78084087189e4852c43f689712944bbb195a9685afe1bb279b0`；schema SHA256 `697d42713792e4f0878040f46b0b9b2c2bd45e70337c3dd8f4285100be0be300`；official kernel content fingerprint `6cf9e18c5d8dbdcb7fc58fab117ee719d72a5e003e324b9a2481519d3b51fdb2`（未commit候选，不使用final KF commit冒充PR source commit）。
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `e42f6c833c101f69` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_e42f6c833c101f69/validation/report.json，report SHA256 `bb0c4af1c60370257ab03a611a649641887b6f41e08b9ce9afa0f59e2d912044`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_e42f6c833c101f69/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_e42f6c833c101f69]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S6 aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/e42f6c833c101f69.json#kernel.py:44 上述机制及run分支；source SHA256 `4a8416852a0c3bfc4848c7605a376773981a0d6dacff7ea42de2dbc6611d907a`；schema SHA256 `5e08414c7bca11883d11cc487a9dd5e890539fc9476ca7315b6544301eb513b6`；official kernel content fingerprint `7b48a582da92276a4fe6adb089f98080c76f4974faa2f88f3588ffe5acf14877`（未commit候选，不使用final KF commit冒充PR source commit）。
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `1d3a1fcaf77631e1` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_1d3a1fcaf77631e1/validation/report.json，report SHA256 `cb117b3fe62f25f8356d82bd7b5beb9c0ed6f06828dfe7b701f4bb829a444aea`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_1d3a1fcaf77631e1/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_1d3a1fcaf77631e1]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S7 aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/1d3a1fcaf77631e1.json#kernel.py:47 上述机制及run分支；source SHA256 `8be69008f5e55c3de52671e09697d97530ac15b3aa96a475fb8e6edfbb0f5fab`；schema SHA256 `0c032938a444c0f50d09202d204be764abae77e88689c43ae5e13c30107a2f64`；official kernel content fingerprint `95579edeaab3105e572000f4a6c339bb6af6fd57235ea730bd678cabcffd6c2d`（未commit候选，不使用final KF commit冒充PR source commit）。
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `3e2804d64ce6b4c7` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_3e2804d64ce6b4c7/validation/report.json，report SHA256 `d69b9fe579fa33654b498c50a87215fea46c238e03d9163e9685282481b26c7b`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_3e2804d64ce6b4c7/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_3e2804d64ce6b4c7]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S8 aklr/ops/gemm/impls/flydsl/gemm_n6144_k4096/3e2804d64ce6b4c7.json#kernel.py:99 上述机制及run分支；source SHA256 `ed0ae1c5b45f8080c4c6333342a1babb8b5cc4361e145d2d860bf49b902b6edd`；schema SHA256 `8560f7344ff9357b4d04278dcea51ecbc68848ba25c7aec8fee757eda4c06279`；official kernel content fingerprint `8301c14ce910616f46999f56de1568ff96cbccdaa5111b0d3721caf2376b22ef`（未commit候选，不使用final KF commit冒充PR source commit）。
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `bfae0e0137d0d54d` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_bfae0e0137d0d54d/validation/report.json，report SHA256 `8a0a797e93a7dfa446da45e21ab641a207530b105c35f7486005c1b595fba65b`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_bfae0e0137d0d54d/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_bfae0e0137d0d54d]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S9 aklr/ops/gemm/impls/flydsl/gemm_n6144_k6144/bfae0e0137d0d54d.json#kernel.py:29 上述机制及run分支；source SHA256 `1c3da2da9e12d7eced81f81abc8b9c0972aafc5c3b50ee1d6654bad56c1f4abd`；schema SHA256 `701cbac8de32b2fabaf9145b5804a2eb2a6ea1236c2b92edf816cd228c1965ea`；official kernel content fingerprint `a3c10d217a82355d67b3a2685de62d435f6f54aa379085248bb6805ee4d7f3aa`（未commit候选，不使用final KF commit冒充PR source commit）。
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `8bec4c4d3db66ba0` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_8bec4c4d3db66ba0/validation/report.json，report SHA256 `57795d5468785df1664354cc86f6e8941ae54734b8df3cdb35a767d31326748e`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_8bec4c4d3db66ba0/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_8bec4c4d3db66ba0]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S10 aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/8bec4c4d3db66ba0.json#kernel.py:109 上述机制及run分支；source SHA256 `cab5ec5646efb52ffa851888a8dc20244e256ad999d8e63e1387de564f2124c7`；schema SHA256 `6c7429c90c11c64c4c1479339cd125396d389f10ae16f083473b946b627b3614`；official kernel content fingerprint `1cf2ec272b86ca6a194f6e5b259339e2941a88b306f89a8e922298f59d63fa6c`（未commit候选，不使用final KF commit冒充PR source commit）。
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `d8bd1ca29ef0a990` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_d8bd1ca29ef0a990/validation/report.json，report SHA256 `a257e80909725090cdb47203c434e1a1c36fa0bc85ba9ce5901dd3c2475991b3`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_d8bd1ca29ef0a990/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_small_d8bd1ca29ef0a990]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S11 aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/d8bd1ca29ef0a990.json#kernel.py:23 上述机制及run分支；source SHA256 `1aa1cde59b9b7a65aa90f499ce171812407860ccf587b117b8e05c7e2c03a4cc`；schema SHA256 `f916a412704128ce6633be3a10a9564479561f7ad9ef8ed69d2a8e6eeabba5cd`；official kernel content fingerprint `a41d31c6c9fcbdede12656f56c57396d5ef67637fefccae9b54dddf081c90cd0`（未commit候选，不使用final KF commit冒充PR source commit）。
- S12 aklr/ops/gemm/baseline.py:15 baseline wrapper；.github/workflows/ci.yaml:7 pinned AITER；aklr/core/validation/stages/_measure.py:47 entry rule；aklr/ops/README.md:338 default serving rule。
