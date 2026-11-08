---
title: "Flash MoE 四份最终实现：仅20个已测Point"
kind: "case"
ops: ["moe"]
dtypes: ["mxfp4", "bf16"]
regimes: ["flash_ep32: [{\"n_tokens\":16},{\"n_tokens\":32},{\"n_tokens\":64},{\"n_tokens\":128},{\"n_tokens\":256}]", "flash_ep64: [{\"n_tokens\":16},{\"n_tokens\":32},{\"n_tokens\":64},{\"n_tokens\":128},{\"n_tokens\":256}]", "flash_tp256: [{\"n_tokens\":16},{\"n_tokens\":32},{\"n_tokens\":64},{\"n_tokens\":128},{\"n_tokens\":256}]", "flash_tp512: [{\"n_tokens\":16},{\"n_tokens\":32},{\"n_tokens\":64},{\"n_tokens\":128},{\"n_tokens\":256}]"]
contract: bounded
gens: ["gfx950"]
flydsl: "0.3.2"
basis: "measured N=16; actual graph timing rounds per comparison; source mechanisms not causally attributed"
measured: [{"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":16}=1.607; {\"n_tokens\":32}=1.486; {\"n_tokens\":64}=1.424; {\"n_tokens\":128}=1.291; {\"n_tokens\":256}=1.303", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256/d8335ff9da7e16df.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g4/flash_d8335ff9da7e16df/validation/report.json sha256=0be8873b7279a3b81a8da0ac6c2023569ca48714b2bc816cb363c79ec4cc07f3", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":16}=1.392; {\"n_tokens\":32}=1.364; {\"n_tokens\":64}=1.26; {\"n_tokens\":128}=1.207; {\"n_tokens\":256}=1.202", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256/69c33e38a0f91a72.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g5/flash_69c33e38a0f91a72/validation/report.json sha256=120953376967dc836ee882fd50d8764cc43c5d4d88641015651d023c0ef787fa", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":16}=1.969; {\"n_tokens\":32}=1.643; {\"n_tokens\":64}=1.396; {\"n_tokens\":128}=1.323; {\"n_tokens\":256}=1.342", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256/d637cbb8575c3d72.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g3/flash_d637cbb8575c3d72/validation/report.json sha256=927cc32442c17beb18a663fdfc988306fa2798c89a95e0b8a3d614d1b6f76509", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":16}=1.643; {\"n_tokens\":32}=1.377; {\"n_tokens\":64}=1.312; {\"n_tokens\":128}=1.229; {\"n_tokens\":256}=1.199", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/9e158753082c37e4.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g4/flash_9e158753082c37e4/validation/report.json sha256=1fbbd0644bf54bb268ad5be97023936b26d05ae352649c6a6d1eb77cf13103f6", "date": "2026-11-08"}]
sources: ["aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256/d8335ff9da7e16df.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256/69c33e38a0f91a72.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256/d637cbb8575c3d72.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/9e158753082c37e4.json"]
---

# Flash MoE 四份最终实现：仅20个已测Point

这是已交付到本地PR准备树的具体案例范围和测试事实，非通则。原有case的其他测点不被本页替代。

## Applies when

`n_tokens` 是本次 hidden_states 第一维，不是路由后每expert GEMM的M；报告未记录逐expert实际M，不能按tokens×topk均摊。
逻辑输入形状：hidden_states[T,D]、w1[E,2I,D]、w2[E,D,I]、routing tensors[T,K]，out[T,D]；MXFP4存储和scale物理布局由下列exact Point布局决定，而非直接套行主序公式。
topk_ids每行distinct：无mask为local id，有mask为global id；expert_mask按全局expert顺序映射到本rank权重，输出是本rank份额，不包括跨rank collective。
GLM separated布局使用SiLU(gate)·up、MXFP4 activation路径；Flash/Pro interleave布局使用clamped SwiGLU，gate上限limit、up限制在±limit。contract reference保留bf16 activation，但本页部分实现内部用fp8；SQNR验收不意味着两条数值路径等价。所有本页MoE doweight_stage1=false，routing weight在stage2或最终route sum使用。

### flash_ep32 — `d8335ff9da7e16df`

- Exact schema: `fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256/d8335ff9da7e16df.json`。
- 固定轴：`{"model_dim":4096,"n_experts":32,"topk":6,"inter_dim":2048,"global_experts":256,"gate_up_rows":4096,"model_blocks":128,"inter_blocks":64}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"interleave","has_expert_mask":true,"has_swiglu_limit":true,"act_quant":4,"has_num_local_tokens":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","expert_mask":"int32","out":"bfloat16","num_local_tokens":null}`。
- layout：`gate_up_interleaved`；scalars：`{"swiglu_limit":10.0}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"gate_up_interleaved":{"layouts":{"w1":{"kind":"gate_up_interleaved_weight","gate_up":true},"w2":{"kind":"gate_up_interleaved_weight","gate_up":false},"w1_scale":{"kind":"gate_up_interleaved_scales","gate_up":true},"w2_scale":{"kind":"gate_up_interleaved_scales","gate_up":false}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":2}]}}}`。
- 测试仅包含：`[{"n_tokens":16},{"n_tokens":32},{"n_tokens":64},{"n_tokens":128},{"n_tokens":256}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页interleave/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### flash_ep64 — `69c33e38a0f91a72`

- Exact schema: `fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256/69c33e38a0f91a72.json`。
- 固定轴：`{"model_dim":4096,"n_experts":64,"topk":6,"inter_dim":2048,"global_experts":256,"gate_up_rows":4096,"model_blocks":128,"inter_blocks":64}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"interleave","has_expert_mask":true,"has_swiglu_limit":true,"act_quant":4,"has_num_local_tokens":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","expert_mask":"int32","out":"bfloat16","num_local_tokens":null}`。
- layout：`gate_up_interleaved`；scalars：`{"swiglu_limit":10.0}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"gate_up_interleaved":{"layouts":{"w1":{"kind":"gate_up_interleaved_weight","gate_up":true},"w2":{"kind":"gate_up_interleaved_weight","gate_up":false},"w1_scale":{"kind":"gate_up_interleaved_scales","gate_up":true},"w2_scale":{"kind":"gate_up_interleaved_scales","gate_up":false}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":2}]}}}`。
- 测试仅包含：`[{"n_tokens":16},{"n_tokens":32},{"n_tokens":64},{"n_tokens":128},{"n_tokens":256}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页interleave/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### flash_tp256 — `d637cbb8575c3d72`

- Exact schema: `fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256/d637cbb8575c3d72.json`。
- 固定轴：`{"model_dim":4096,"n_experts":256,"topk":6,"inter_dim":256,"gate_up_rows":512,"model_blocks":128,"inter_blocks":8,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"interleave","has_expert_mask":false,"has_swiglu_limit":true,"act_quant":4,"has_num_local_tokens":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`gate_up_interleaved`；scalars：`{"swiglu_limit":10.0}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"gate_up_interleaved":{"layouts":{"w1":{"kind":"gate_up_interleaved_weight","gate_up":true},"w2":{"kind":"gate_up_interleaved_weight","gate_up":false},"w1_scale":{"kind":"gate_up_interleaved_scales","gate_up":true},"w2_scale":{"kind":"gate_up_interleaved_scales","gate_up":false}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":2}]}}}`。
- 测试仅包含：`[{"n_tokens":16},{"n_tokens":32},{"n_tokens":64},{"n_tokens":128},{"n_tokens":256}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页interleave/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### flash_tp512 — `9e158753082c37e4`

- Exact schema: `fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/9e158753082c37e4.json`。
- 固定轴：`{"model_dim":4096,"n_experts":256,"topk":6,"inter_dim":512,"gate_up_rows":1024,"model_blocks":128,"inter_blocks":16,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"interleave","has_expert_mask":false,"has_swiglu_limit":true,"act_quant":4,"has_num_local_tokens":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`gate_up_interleaved`；scalars：`{"swiglu_limit":10.0}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"gate_up_interleaved":{"layouts":{"w1":{"kind":"gate_up_interleaved_weight","gate_up":true},"w2":{"kind":"gate_up_interleaved_weight","gate_up":false},"w1_scale":{"kind":"gate_up_interleaved_scales","gate_up":true},"w2_scale":{"kind":"gate_up_interleaved_scales","gate_up":false}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":2}]}}}`。
- 测试仅包含：`[{"n_tokens":16},{"n_tokens":32},{"n_tokens":64},{"n_tokens":128},{"n_tokens":256}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页interleave/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

## Counter-signals

- 不外推未列出的tokens、slots、scalar、dtype、layout或路由分布；原报告输入抽样不能替代真实流量。
- 正确性及速度均为整次contract调用；不证明跨WG协议、重入/多stream安全、实际occupancy、单项kernel性能或某一机制的因果收益。
- 本地source-bound audit/bench通过不是CI通过或线上serve已上线；payload未写入accuracy/speedup。本页仅引用归档事实，不改变dispatch。
- 实际运行AKLR_MIN_SPEEDUP未归档，表中默认资格为固定base默认值下的重建，不是当时部署env证明。

## Mechanism

以下逐源码核对的是代码事实，不是收益归因。可选分支和metadata依赖不会因whole-call通过就被单独证成。

- `d8335ff9da7e16df`（flash_ep32）：`_plan` 依静态 route 数选择 `_TIERS`，16/32/64/128 tokens 把路由、激活量化及输出清零折进 stage 1；256 tokens 单独调用 prelude。stage 1/2 调用 AITER FlyDSL 混合 MoE builder，经本文件源码替换调整 K 遍历起点和 stage-2 地址；调用处明确使用 fp8 激活、fp4 权重及 stage-2 atomic 路由权重累加。 见 S1。
- `69c33e38a0f91a72`（flash_ep64）：`_CFG` 对16/32/64/128 tokens 启用 fused route；256 tokens 执行 prologue。自有 stage1 从 bf16 输入生成 fp8 中间值和 E8M0 scales；stage2 读取 sorted weights 写调用方 out。源码含 K rotation 及 xcd 网格设置，设置本身不是实际 XCD 对应关系或性能因果证据。 见 S2。
- `d637cbb8575c3d72`（flash_tp256）：`_build` 中 stage 1 按 DIRECT 阈值选择 per-route 或 per-expert 路径；stage 2 在 routes≥384 时选择 `_stage2_ex`，本页64/128/256 tokens 命中该分支。256 tokens 还命中 SPEC，输出 tile-N=512；其他 expert-stage2 为256。两阶段以 bf16 Y 交接；更大 route 数或 expert mask 的 baseline 回退不属于此页测点。 见 S3。
- `9e158753082c37e4`（flash_tp512）：`_fold(m)` 对本页16/32/64/128 tokens 返回 true，stage 1 折入排序、量化及 out 清零，接自有 stage2。256 tokens 走独立 prep 及两个 projection 调用。stage1 接收 swiglu_limit，stage2 接收 sorted weights。此页没有把这些代码选择的速度贡献单独量出。 见 S4。

## Result

逐Point列出本地原报告结果；不是Points均值。倍率为报告可发表speedup字段；baseline是 `aklr.ops.fused_moe.baseline.fused_moe_baseline`。
profile v18：AKLR CUDA graph、Kineto gpu_duration_sum、每call前1GiB读flush、每replay内mean、replay间median、每round seeded shuffle，Student-t mean-log-ratio下界。以下N取timings实际n_rounds，不把round_looks上限32或measurement_replays=10当N。

全部本页MoE完整Point在固定base没有默认served incumbent；对baseline的胜出不代表击败另一份incumbent。

| UUID | exact变量轴 | scalar | accuracy / worst violation | baseline倍率 | incumbent倍率 / UUID | N baseline/incumbent | entry / 默认资格 |
|---|---|---|---|---|---|---|---|
| `d8335ff9da7e16df` | `{"n_tokens":16}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.607 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d8335ff9da7e16df` | `{"n_tokens":32}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.486 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d8335ff9da7e16df` | `{"n_tokens":64}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.424 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d8335ff9da7e16df` | `{"n_tokens":128}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.291 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d8335ff9da7e16df` | `{"n_tokens":256}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.303 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `69c33e38a0f91a72` | `{"n_tokens":16}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.392 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `69c33e38a0f91a72` | `{"n_tokens":32}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.364 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `69c33e38a0f91a72` | `{"n_tokens":64}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.26 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `69c33e38a0f91a72` | `{"n_tokens":128}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.207 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `69c33e38a0f91a72` | `{"n_tokens":256}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.202 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d637cbb8575c3d72` | `{"n_tokens":16}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.969 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d637cbb8575c3d72` | `{"n_tokens":32}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.643 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d637cbb8575c3d72` | `{"n_tokens":64}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.396 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d637cbb8575c3d72` | `{"n_tokens":128}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.323 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `d637cbb8575c3d72` | `{"n_tokens":256}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.342 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `9e158753082c37e4` | `{"n_tokens":16}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.643 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `9e158753082c37e4` | `{"n_tokens":32}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.377 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `9e158753082c37e4` | `{"n_tokens":64}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.312 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `9e158753082c37e4` | `{"n_tokens":128}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.229 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `9e158753082c37e4` | `{"n_tokens":256}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.199 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |

上表倍率与gate/default资格分别保留；表里的通过不是CI或线上服务状态。

## Reproduce

固定base：`a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`；pinned AITER：`dbd8bf5bd624120197a7a26780a8c72201824f0f`；报告记录的 `amd-aiter` 为 `0.1.21.dev66+gdbd8bf5bd`。MoE baseline 对 gate_mode=interleave 设置 `AITER_BF16_FP8_MOE_BOUND=16`，这是activation路径的行数切换设置，不是SQNR dB阈值；DSA baseline不使用此MoE设置。未归档实际sglang版本、实际tuned配置及完整运行env，因此不宣称这些条件已锁定。

本次未运行任何GPU、网络或新测量。将来在gfx950、FlyDSL 0.3.2及同base上，用完整导出schema而非裸kernel复测，逐点核对scalars、dtype、attrs和incumbent；不能以重放结果替换本页原报告。

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256/d8335ff9da7e16df.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256/69c33e38a0f91a72.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256/d637cbb8575c3d72.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/9e158753082c37e4.json
```

原始本地报告和receipt路径及SHA256见Evidence；local文件可能不长存。源码原文件SHA256、payload/schema SHA256、官方kernel content hash及逐Point结果另归档于 `local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/moe_dsa_kernel_kb_mapping.json`，不将这些本地归档冒充CI永久证据。

## Rollback

当前只有本地准备文件，未声明已serve。撤回时只删除本页列出的新UUID实现及其新增case（通过正常review流程）；旧实现保持。恢复选择由base原有Point与baseline决定，不声称无条件回退到某一赢家。

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `d8335ff9da7e16df` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g4/flash_d8335ff9da7e16df/validation/report.json sha256=0be8873b7279a3b81a8da0ac6c2023569ca48714b2bc816cb363c79ec4cc07f3；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g4/flash_d8335ff9da7e16df/receipt.json sha256=2817da05eae331f03469fef1a4061735202966fd99fc9dcc707be921479374e5。
- S1 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/flash_ep32/tx/flash_ep32_kernel_final.py:393 最终原源码，sha256=031134df9950cb08acb0e6a8ac9283e1f0482eb757d94c9b7e38425b53ed06df；核对行段 393–398, 1451–1503, 1548–1619；与导出kernel.py字节完全相同。
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `69c33e38a0f91a72` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g5/flash_69c33e38a0f91a72/validation/report.json sha256=120953376967dc836ee882fd50d8764cc43c5d4d88641015651d023c0ef787fa；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g5/flash_69c33e38a0f91a72/receipt.json sha256=95d3b5478256ac9b52dff581f647bb3cbe96b1c247d4880a86533b9c87be1928。
- S2 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/flash_ep64/tx/flash_ep64_kernel_final.py:50 最终原源码，sha256=374078232f5737bbfc6ed45f38eccb88e1861b8ea8127edcdb4f1f13694ace50；核对行段 50–56, 3307–3385；与导出kernel.py字节完全相同。
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `d637cbb8575c3d72` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g3/flash_d637cbb8575c3d72/validation/report.json sha256=927cc32442c17beb18a663fdfc988306fa2798c89a95e0b8a3d614d1b6f76509；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g3/flash_d637cbb8575c3d72/receipt.json sha256=1f57cb53876a06d94846b67c0f8b38bc2f06705c31c5f4960f617ff3ddebb5ef。
- S3 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/flash_tp256/tx/flash_tp256_kernel_final.py:749 最终原源码，sha256=8494e45412ef7ed84ed677e30f3c192ebb80e7de5b7378a11cac5a5286b54271；核对行段 749–869, 893–950；与导出kernel.py字节完全相同。
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `9e158753082c37e4` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g4/flash_9e158753082c37e4/validation/report.json sha256=1fbbd0644bf54bb268ad5be97023936b26d05ae352649c6a6d1eb77cf13103f6；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g4/flash_9e158753082c37e4/receipt.json sha256=63b30fbfbcdb09fc6b4c7803b919ebbc5e19e72bff8dbb8b743a65b73cecfdc8。
- S4 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/flash_tp512/tx/flash_tp512_kernel_final.py:4381 最终原源码，sha256=2b376e4a98d01dc32c5c8d481ccb333eb7076c4cadb6ef7a32f0052cb7500f98；核对行段 4381–4430, 4486–4571；与导出kernel.py字节完全相同。
- S5 aklr/ops/fused_moe/contract.py:75 axes/attrs/tensor/layout/scalar/output契约；本页只声明exact Points，不声明完整contract域。
- S6 aklr/ops/fused_moe/baseline.py:9 baseline实际调用；MoE interleave设置AITER_BF16_FP8_MOE_BOUND=16；DSA baseline调用sglang Triton split-K。
- S7 aklr/ops/fused_moe/reference.py:77 global→local mask映射；:88–99 activation量化；:107–127 clamp、激活、stage2 routing weight及topk求和。
