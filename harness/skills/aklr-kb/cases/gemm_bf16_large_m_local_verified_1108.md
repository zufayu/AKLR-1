---
title: "BF16 dense GEMM large M：逐实现、逐 Point 的本地验收边界"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 256, 512, 1024, 2048, 4096"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1 whole-call per implementation
measured:
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "4/4 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n128_k6144, UUID 71234a12ada11ac5, M 256, 512, 1024, 2048", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_71234a12ada11ac5/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n2048_k2048, UUID 9fe8f6480f07d235, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_9fe8f6480f07d235/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "3/3 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n256_k6144, UUID 3da7088c64723d7c, M 256, 512, 1024", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_3da7088c64723d7c/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n28672_k512, UUID ce4647f84b8de43b, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_ce4647f84b8de43b/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n3584_k512, UUID 1fbaf19b1c4fdb45, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_1fbaf19b1c4fdb45/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "3/5 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n6144_k1536, UUID d5c0cc4ef7575b67, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_d5c0cc4ef7575b67/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n64_k4096, UUID f39f8ba0861eecef, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_f39f8ba0861eecef/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; exact ratios in Result", "gpu": "MI355X", "workload": "gemm_n7168_k512, UUID 04da38be4d3af890, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g3/bf16_large_04da38be4d3af890/validation/report.json", "date": "2026-11-08"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/71234a12ada11ac5.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/9fe8f6480f07d235.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/3da7088c64723d7c.json
  - aklr/ops/gemm/impls/flydsl/gemm_n28672_k512/ce4647f84b8de43b.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/1fbaf19b1c4fdb45.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/d5c0cc4ef7575b67.json
  - aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/f39f8ba0861eecef.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/04da38be4d3af890.json
---

# BF16 dense GEMM：大 M逐实现本地证据

## Applies when

只覆盖 `gemm@1.0`、`row_major`、A=(M,K)、B=(N,K)，bf16输入/输出、trans_b=True、无bias；compare为bitwise、fixed element order、零violation fraction。
下表逐实现列出**离散** authored且已测M；没有把相邻值外推为连续范围，也没有声称未列出的Point已测。调用/服务规则见 [AKLR算子入口](../../../../aklr/ops/README.md)。

| UUID | name（N/K固定） | authored且已测 M |
|---|---|---|
| `71234a12ada11ac5` | `gemm_n128_k6144` | 256, 512, 1024, 2048 |
| `9fe8f6480f07d235` | `gemm_n2048_k2048` | 256, 512, 1024, 2048, 4096 |
| `3da7088c64723d7c` | `gemm_n256_k6144` | 256, 512, 1024 |
| `ce4647f84b8de43b` | `gemm_n28672_k512` | 256, 512, 1024, 2048, 4096 |
| `1fbaf19b1c4fdb45` | `gemm_n3584_k512` | 256, 512, 1024, 2048, 4096 |
| `d5c0cc4ef7575b67` | `gemm_n6144_k1536` | 256, 512, 1024, 2048, 4096 |
| `f39f8ba0861eecef` | `gemm_n64_k4096` | 256, 512, 1024, 2048, 4096 |
| `04da38be4d3af890` | `gemm_n7168_k512` | 256, 512, 1024, 2048, 4096 |

## Counter-signals

- 本地独立测后审稿支持所列Point的整调用correctness和timing；**没有独立组件/单因子对照**。Mechanism仅是原始源码事实，不能用整调用倍数证明DMA、prefetch、tile order、priority、MFMA原子或LLVM hint各自的收益。
- 测试report的head/base都是 `a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`，但候选是untracked新source；不是“该commit已包含kernel”。绑定依据是schema SHA256、kernel content fingerprint、report/receipt hash，逐项在Evidence及映射归档中列出。不是CI，也不是线上已选择证明。
- 按固定版本默认规则复算的资格不等于线上阈值：worker环境原件仅记录版本/GPU，完整runtime配置及环境变量未归档，不能认证 `AKLR_MIN_SPEEDUP` 的实际值；task receipt仅记录group/path/sha256/id，不是环境凭据；当前payload尚未写accuracy/speedup，因此本页不声称当前未发布schema已默认服务。
- 若复用跨工作组split-K/tag/counter路径，另核发布/读取的memory order、tag碰撞/回绕、未初始化scratch、多stream并发及graph重放；整调用通过和源码中的固定加法顺序都不足以证明协议安全。
- tile重排只记录源码的逻辑分组；不把block_id模8当作各stream下不变的绝对XCD编号，不声称已确认实际cache命中。
- 本地证据文件可能失效；发布前归档报告和fingerprint，并重新走官方验证。

- N=6144/K=1536 的 M=2048 是library fallback整调用，正确、0.999、entry否，不能归因于custom DMA/MFMA机制；M=4096正确、1.028、entry是，但两档默认资格均否。 *(field, E6)*

## Mechanism

以下是原始源码事实，不是独立组件因果收益。

### `71234a12ada11ac5` — `gemm_n128_k6144`

M=256,512 八波分K，M=1024 四波、M=2048 三波；每wave私有DMA环。:416–448 依配置用固定pairwise tree或wave顺序LDS归约，最后转bf16；不是跨工作组计数协议。M=4096没有authored/test Point，不外推最大bucket fallback。（S1，行号均为该schema内 `kernel.py`。）

### `9fe8f6480f07d235` — `gemm_n2048_k2048`

M=256 四波分K、3-stage环并配置额外prefetch波；M=512,1024,2048,4096 KW=1，分波沿M/N，分段且带调度fence的ping-pong(:307–355)。所有路径A/B DMA至XOR LDS、16×16×32 BF16 MFMA；只有第一档是wave split-K。（S2，行号均为该schema内 `kernel.py`。）

### `3da7088c64723d7c` — `gemm_n256_k6144`

四wave各有2-stage LDS环；M=256 KS=1，M=512 KS=2，M=1024 KS=4。后两档跨工作组partial/dispatch-tag计数workspace(:409–470)。KS=2使用own+peer二项和；KS=4按split索引顺序累加(:478–504)。计数协议存在是源码事实，不是已证安全；M=2048,4096无authored/test Point。（S3，行号均为该schema内 `kernel.py`。）

### `ce4647f84b8de43b` — `gemm_n28672_k512`

短K的persistent工作组walk head tile及较短row tail item；:67–101主机plan，:298–339 tile map，:341–343 B首操作数MFMA，A/B DMA到LDS。无跨工作组split-K；M=1024的 `_SPLIT_PLAN` 指head/tail布局，不是K split。计划里的tail cost常量不是本次独立组件实测。（S4，行号均为该schema内 `kernel.py`。）

### `1fbaf19b1c4fdb45` — `gemm_n3584_k512`

M=256,512,1024,2048,4096 分别用64×64、64×128、128×128、128×224、256×224 tile。A/B DMA多stage环、B首操作数MFMA(:616–618)；per-M配置选择LDS staging或register C store及overlap(:942–997)。`split_stage/compute_split`是stage内片段调度，不据其名字推断跨工作组split-K。（S5，行号均为该schema内 `kernel.py`。）

### `d5c0cc4ef7575b67` — `gemm_n6144_k1536`

M=256、512、1024、4096 有 `_CONFIGS` 项，走custom A/B async-LDS tile pipeline；**M=2048没有配置项**，:78–88 的 `_launcher` 返回 `_library`，执行 `torch.matmul(a, b.t(), out=out)`，不是custom DMA/MFMA路径。custom各档 :223–230 由MFMA32配置切换16×16×32或32×32×16 BF16 MFMA。C经pad过的LDS按C_SPLIT分段输出(:457–513)，EARLY路径在末K步先存部分行(:374–412)。无跨工作组K split；C_SPLIT不是split-K。（S6，行号均为该schema内 `kernel.py`。）

### `f39f8ba0861eecef` — `gemm_n64_k4096`

M=256,512,1024 八波、M=2048 四波在工作组内分K，wave-private LDS DMA环；M=4096切 `_build_fold`，四路跨工作组split-K及per-call未初始化workspace，tagged counter CAS/atomic add决定fold(:383–426)。:438–446按绝对split索引顺序选择partial再累加，虽peer load是旋转顺序，sum不是到达顺序。协议正确性仍需另证。（S7，行号均为该schema内 `kernel.py`。）

### `04da38be4d3af890` — `gemm_n7168_k512`

M=256,512 调 `_build_large`一shot tiled body；M=1024,2048,4096 调 `_build_pipe` persistent多tile walk。A/B DMA到XOR LDS后16×16×32 BF16 MFMA，:941–947按bucket编译调度提示与kernarg preload。hint存在不证明被编译器按意图实现，更不证明单因子收益。（S8，行号均为该schema内 `kernel.py`。）

## Result

`d5c0cc4ef7575b67` 的M=2048测到的是 `_library` → `torch.matmul` 回退整调用，其0.999不是custom DMA/MFMA实现收益；M=256、512、1024、4096才走custom body。 *(field, E6)*

基准树固定为 `a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`；baseline是 `aklr.ops.gemm.baseline.gemm_baseline` 调用的AITER tuned `gemm_a16w16`，不是泛指任意vendor GEMM。
CI声明的完整AITER pin为 `dbd8bf5bd624120197a7a26780a8c72201824f0f`；本地worker `environment.json`实际记录短版本 `0.1.21.dev66+gdbd8bf5bd`、FlyDSL 0.3.2、Torch `2.11.0+rocm10.0.0`、gfx950，`install_aiter.log`记录该短版本wheel安装成功；report platform记录MI355X及ROCm `7.15.26333`。短版本与CI pin前缀一致，但不据此认证实际源码完整commit或完整runtime配置；这些版本不来自task receipt。各原件见对应E行，baseline/CI pin来源见 S9。
profile v18：CUDA graph、逐调用flush L2，seeded shuffle rounds，16/32两次look，alpha每look=0.025；整调用timer累加GPU duration，mean within replay、median across replay，trial为mean log-ratio的Student-t下界。N=1表示每实现一个独立验证任务，不将任务内rounds称作多次独立组件实验。
本组所有Point正确。默认资格按 [官方规则](../../../../aklr/ops/README.md) 和固定版本审稿复算；entry列基于未舍入lowerbound，published speedup是三位小数，资格按原始值判定，不按展示舍入值判定。
**Incumbent单独列示为N/A**：审稿逐完整Point核对固定base，没有默认可服务incumbent（或同kernel），所以没有incumbent timing；不能声称击败其他base、其他服务kernel。以下baseline数不冒充incumbent比值。

| UUID | M | baseline下界（原始值展示6位） | published speedup | baseline entry | incumbent ratio | 默认资格（复算） |
|---|---:|---:|---:|---|---|---|
| `71234a12ada11ac5` | 256 | 1.293070 | 1.293 *(field, E1)* | 是 | N/A | 是 |
| `71234a12ada11ac5` | 512 | 1.337250 | 1.337 *(field, E1)* | 是 | N/A | 是 |
| `71234a12ada11ac5` | 1024 | 1.263918 | 1.264 *(field, E1)* | 是 | N/A | 是 |
| `71234a12ada11ac5` | 2048 | 1.205937 | 1.206 *(field, E1)* | 是 | N/A | 是 |
| `9fe8f6480f07d235` | 256 | 1.217558 | 1.218 *(field, E2)* | 是 | N/A | 是 |
| `9fe8f6480f07d235` | 512 | 1.233715 | 1.234 *(field, E2)* | 是 | N/A | 是 |
| `9fe8f6480f07d235` | 1024 | 1.225520 | 1.226 *(field, E2)* | 是 | N/A | 是 |
| `9fe8f6480f07d235` | 2048 | 1.066559 | 1.067 *(field, E2)* | 是 | N/A | 是 |
| `9fe8f6480f07d235` | 4096 | 1.040005 | 1.040 *(field, E2)* | 是 | N/A | 是 |
| `3da7088c64723d7c` | 256 | 1.376877 | 1.377 *(field, E3)* | 是 | N/A | 是 |
| `3da7088c64723d7c` | 512 | 1.623740 | 1.624 *(field, E3)* | 是 | N/A | 是 |
| `3da7088c64723d7c` | 1024 | 1.442374 | 1.442 *(field, E3)* | 是 | N/A | 是 |
| `ce4647f84b8de43b` | 256 | 1.371169 | 1.371 *(field, E4)* | 是 | N/A | 是 |
| `ce4647f84b8de43b` | 512 | 1.421874 | 1.422 *(field, E4)* | 是 | N/A | 是 |
| `ce4647f84b8de43b` | 1024 | 1.266952 | 1.267 *(field, E4)* | 是 | N/A | 是 |
| `ce4647f84b8de43b` | 2048 | 1.443782 | 1.444 *(field, E4)* | 是 | N/A | 是 |
| `ce4647f84b8de43b` | 4096 | 1.167431 | 1.167 *(field, E4)* | 是 | N/A | 是 |
| `1fbaf19b1c4fdb45` | 256 | 1.630782 | 1.631 *(field, E5)* | 是 | N/A | 是 |
| `1fbaf19b1c4fdb45` | 512 | 1.518101 | 1.518 *(field, E5)* | 是 | N/A | 是 |
| `1fbaf19b1c4fdb45` | 1024 | 1.360221 | 1.360 *(field, E5)* | 是 | N/A | 是 |
| `1fbaf19b1c4fdb45` | 2048 | 1.528573 | 1.529 *(field, E5)* | 是 | N/A | 是 |
| `1fbaf19b1c4fdb45` | 4096 | 1.391125 | 1.391 *(field, E5)* | 是 | N/A | 是 |
| `d5c0cc4ef7575b67` | 256 | 1.257317 | 1.257 *(field, E6)* | 是 | N/A | 是 |
| `d5c0cc4ef7575b67` | 512 | 1.170410 | 1.170 *(field, E6)* | 是 | N/A | 是 |
| `d5c0cc4ef7575b67` | 1024 | 1.160032 | 1.160 *(field, E6)* | 是 | N/A | 是 |
| `d5c0cc4ef7575b67` | 2048 | 0.999334 | 0.999 *(field, E6)* | 否 | N/A | 否 |
| `d5c0cc4ef7575b67` | 4096 | 1.027530 | 1.028 *(field, E6)* | 是 | N/A | 否 |
| `f39f8ba0861eecef` | 256 | 1.521796 | 1.522 *(field, E7)* | 是 | N/A | 是 |
| `f39f8ba0861eecef` | 512 | 1.528113 | 1.528 *(field, E7)* | 是 | N/A | 是 |
| `f39f8ba0861eecef` | 1024 | 1.526578 | 1.527 *(field, E7)* | 是 | N/A | 是 |
| `f39f8ba0861eecef` | 2048 | 1.524845 | 1.525 *(field, E7)* | 是 | N/A | 是 |
| `f39f8ba0861eecef` | 4096 | 1.401987 | 1.402 *(field, E7)* | 是 | N/A | 是 |
| `04da38be4d3af890` | 256 | 1.163546 | 1.164 *(field, E8)* | 是 | N/A | 是 |
| `04da38be4d3af890` | 512 | 1.367619 | 1.368 *(field, E8)* | 是 | N/A | 是 |
| `04da38be4d3af890` | 1024 | 1.786689 | 1.787 *(field, E8)* | 是 | N/A | 是 |
| `04da38be4d3af890` | 2048 | 1.351206 | 1.351 *(field, E8)* | 是 | N/A | 是 |
| `04da38be4d3af890` | 4096 | 1.313838 | 1.314 *(field, E8)* | 是 | N/A | 是 |

本组37个Point全部accuracy passed，35个默认资格复算为是；不聚合不同Point的speedup。 *(field, E1, E2, E3, E4, E5, E6, E7, E8)*

## Reproduce

在MI355X及上述固定依赖、基准树和完全匹配fingerprint的source下，逐schema执行；这是复现实验说明，本次KB写作没有运行GPU/网络。

```bash
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/71234a12ada11ac5.json --base a12866b1e40e86ea6aff7ac24cbe5634132f6b2e
```

其余实现逐个替换为front matter的source路径。检查完整Point、audit/bench findings、baseline/incumbent分母、未舍入bounds及环境；报告按E行归档，hash核对后才与本页比较。不要只看进程rc或bench passed。
published schema以后可能包含测量字段，schema字节hash会随之改变；kernel content fingerprint用于区分源码身份，不能拿不同source的同名kernel报告代替。

## Rollback

回滚对应UUID的新增schema和本页该UUID段（未commit时移除该候选文件），让dispatch重新选择固定base已有实现或baseline；没有默认资格的Point不能强行放开阈值。
不要把整contract disable当作逐实现回滚。涉及tag/counter路径发现协议疑点时停用该候选并独立验证，不修改原KB来宣称所有类似协议安全。

## Evidence

以下均为本地独立AKLR audit/bench及测后审稿，非CI；`<schema>.json#kernel.py:<line>` 指 `kernel.sources[0].content` 的原始源码行。
- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `71234a12ada11ac5` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_71234a12ada11ac5/validation/report.json，report SHA256 `044c011519b3eb0a2c0ffb472344e82c976d05608de87ee54ee5bec4247a1073`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_71234a12ada11ac5/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_71234a12ada11ac5]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S1 aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/71234a12ada11ac5.json#kernel.py:22 上述机制及run分支；source SHA256 `5a26038aaf3eaa7cc88df149867c05b76781cf97ca7fac4a6639ed9678e30e50`；schema SHA256 `c0f9584a9b63dd8642fad9cc5ef89b04f8bb50aa10276c552f316e83a99852b7`；official kernel content fingerprint `6d38d051155179c947334d8dd7bc9f887859017ee43e586797dbb7a933525f1b`（未commit候选）。
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `9fe8f6480f07d235` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_9fe8f6480f07d235/validation/report.json，report SHA256 `fefd100cc255ea28a977cc38b211856770cbdffd58962d959156cd47b2812ac2`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_9fe8f6480f07d235/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_9fe8f6480f07d235]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S2 aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/9fe8f6480f07d235.json#kernel.py:28 上述机制及run分支；source SHA256 `138cd2cef48cc44aca08a81418800af27e1f500061b46b2734edae97a66781df`；schema SHA256 `86602629f95bcdfe9f5c7d2a2293ed2390ad37a09bc05813ae6391246cffb1f6`；official kernel content fingerprint `59e885a53a143b7f1064f69caa6561a0c24e484903ea3d56a711ce7940c70179`（未commit候选）。
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `3da7088c64723d7c` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_3da7088c64723d7c/validation/report.json，report SHA256 `8cea7569820c3630e4c88f944410ad2c209b29931730993ebf48d6b3da6c65b3`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_3da7088c64723d7c/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_3da7088c64723d7c]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S3 aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/3da7088c64723d7c.json#kernel.py:23 上述机制及run分支；source SHA256 `d69b9cb7ace4b41afd6e5b7e5cd0b3cb6ec67791610d1b44b8e389f9c78b1cde`；schema SHA256 `42dee2d5eaa1ae624140e7c17665755150e0645373853923a708b1ac5831e542`；official kernel content fingerprint `c4eb3c9f885fb9f96ff5efa2b2fb5e32b39083a7bb61d030db2ac22bec2ecf73`（未commit候选）。
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `ce4647f84b8de43b` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_ce4647f84b8de43b/validation/report.json，report SHA256 `871fbec4aa195ff0c799df1669e74ea8dc2cafa630ea8a0eb81e81464a082cd1`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_ce4647f84b8de43b/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_ce4647f84b8de43b]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S4 aklr/ops/gemm/impls/flydsl/gemm_n28672_k512/ce4647f84b8de43b.json#kernel.py:161 上述机制及run分支；source SHA256 `9a507b4a73b918b73a6239996fd2154f93b07dbd0ab109eda432b0107dcfb179`；schema SHA256 `105749041aa52bfd56a860567ad4c1b75ce011347133813894559412fd2a60f5`；official kernel content fingerprint `887fba39c8d5719be6ecb78327321a623396eaaae834fd38720b7115a9ef2f85`（未commit候选）。
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `1fbaf19b1c4fdb45` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_1fbaf19b1c4fdb45/validation/report.json，report SHA256 `7f7e25ac4f3ef5e9483179cdc04fb2fa6b8550183bc71b06e2279dc3272b2199`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_1fbaf19b1c4fdb45/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_1fbaf19b1c4fdb45]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g3/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S5 aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/1fbaf19b1c4fdb45.json#kernel.py:34 上述机制及run分支；source SHA256 `ab4f7c71d17688e43c0edbadfecde5de8e2925ee5ca300197e43ff92b68ef511`；schema SHA256 `fd3b2897823d815caf78efad6663bacbe25781689c1d8c585ddcbdec6f489905`；official kernel content fingerprint `5f40da7545581c61cbfa58920f6a2a9d7c14fee0a203f1ffbfb45fc365ec2bc7`（未commit候选）。
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `d5c0cc4ef7575b67` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_d5c0cc4ef7575b67/validation/report.json，report SHA256 `9e7d8f70b1bfc6177d83a0d5b03293fd7ccb17577f8619b436000b374a45d1e7`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_d5c0cc4ef7575b67/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_d5c0cc4ef7575b67]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g4/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S6 aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/d5c0cc4ef7575b67.json#kernel.py:23 上述机制及run分支；source SHA256 `540ffda92fac89cb364cbab2b52b7f725d275015ad9b7f1924883d657fad13aa`；schema SHA256 `d237d932ea2f527d7daa668d77dc7806179280ababd66ea1583ee577da8937c9`；official kernel content fingerprint `956115ab4022a5ccd3a1e528b433c0b70f736f0635c3d9244ec8869da151e2e9`（未commit候选）。
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `f39f8ba0861eecef` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_f39f8ba0861eecef/validation/report.json，report SHA256 `1586d0a34d0a70b22282f493d3e31a56e141b87e7c70f4d07d81cc841eb4fb01`；receipt local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_f39f8ba0861eecef/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_f39f8ba0861eecef]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0142/results/g5/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S7 aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/f39f8ba0861eecef.json#kernel.py:21 上述机制及run分支；source SHA256 `f3c5ebc39545e9c8a376413e633ffa9310956cda2a8cb9a65f8d476f5a4bf03c`；schema SHA256 `805931cad8f2a61ebb2c0f0c8d4c4a9ff09d36218666767462176bf73e7fa281`；official kernel content fingerprint `86c2c459e1569e5ab69c7f0ff5927972702fc3545fbe064d075fc1c5005accac`（未commit候选）。
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `04da38be4d3af890` 所列M整调用accuracy passed、逐Point baseline ratios/entry/default复算见Result；无incumbent timing。report local:/tmp/inf3_validation_checkpoint_0147/results/g3/bf16_large_04da38be4d3af890/validation/report.json，report SHA256 `47c9461d2ae1841cef97d764a2843199e179aa4c2c5f4d6aa9d81089fc656ec5`；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g3/bf16_large_04da38be4d3af890/receipt.json；测后审稿 local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json 的 schemas[id=bf16_large_04da38be4d3af890]。 worker版本原件 local:/tmp/inf3_validation_checkpoint_0147/results/g3/environment.json:1；安装原件 local:/tmp/inf3_validation_checkpoint_0147/results/g3/install_aiter.log:1–7。task receipt只绑定group/path/sha256/id，不包含环境。
- S8 aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/04da38be4d3af890.json#kernel.py:29 上述机制及run分支；source SHA256 `d5171067a306b2f9da3db2cabc811c396956fad77bdb684aaee88d6500b19a3b`；schema SHA256 `c3df47d1b664b5586a3b4d384ce8204cd5ec5855802c977b4cadd8c939e2fd74`；official kernel content fingerprint `6c1bedc1abb52d8b7fb4aeddb804cb24aad7e81a7a06fb3edfa29592175ecfe7`（未commit候选）。
- S9 aklr/ops/gemm/baseline.py:15 baseline wrapper；.github/workflows/ci.yaml:7 pinned AITER；aklr/core/validation/stages/_measure.py:47 entry rule；aklr/ops/README.md:338 default serving rule。
