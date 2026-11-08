---
title: "GLM MoE 五份最终实现：仅20个已测Point"
kind: "case"
ops: ["moe"]
dtypes: ["mxfp4", "bf16"]
regimes: ["glm_ep4_l: [{\"n_tokens\":256},{\"n_tokens\":2048},{\"n_tokens\":4096}]", "glm_ep4_s: [{\"n_tokens\":4},{\"n_tokens\":8},{\"n_tokens\":16},{\"n_tokens\":32},{\"n_tokens\":64},{\"n_tokens\":128}]", "glm_tp8_l: [{\"n_tokens\":512},{\"n_tokens\":1024},{\"n_tokens\":2048},{\"n_tokens\":4096}]", "glm_l: [{\"n_tokens\":128},{\"n_tokens\":512},{\"n_tokens\":1024},{\"n_tokens\":2048},{\"n_tokens\":4096}]", "glm_s: [{\"n_tokens\":32},{\"n_tokens\":64}]"]
contract: bounded
gens: ["gfx950"]
flydsl: "0.3.2"
basis: "measured N=16; actual graph timing rounds per comparison; source mechanisms not causally attributed"
measured: [{"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":256}=1.172; {\"n_tokens\":2048}=1.361; {\"n_tokens\":4096}=1.258", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/40937d155ca6aa98.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_40937d155ca6aa98/validation/report.json sha256=bddb9f8e50ce814b42fe7644b168dd8e07cb249d76fb0dfe035980332b6100a2", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":4}=1.185; {\"n_tokens\":8}=1.145; {\"n_tokens\":16}=1.126; {\"n_tokens\":32}=1.168; {\"n_tokens\":64}=1.183; {\"n_tokens\":128}=1.159", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/2c7c5bacd06c8e29.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_2c7c5bacd06c8e29/validation/report.json sha256=2198069dd4f4f8093d31b643d167003d10ea25258d78db864fa470538162c34f", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":512}=1.266; {\"n_tokens\":1024}=1.503; {\"n_tokens\":2048}=1.61; {\"n_tokens\":4096}=1.843", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/aa1aaac0bdd0068f.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_aa1aaac0bdd0068f/validation/report.json sha256=968368f53cff8c5b1b4eba61528513b95cd5582d18be8421fc79658b6ef850a0", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":128}=1.008; {\"n_tokens\":512}=1.063; {\"n_tokens\":1024}=1.164; {\"n_tokens\":2048}=1.41; {\"n_tokens\":4096}=1.674", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/f2e2b374be1c9f8b.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g4/glm_f2e2b374be1c9f8b/validation/report.json sha256=16976d1fb17a8accdbdde1ad5a0e7585dc91bbee2b94973cdceee5e6bab620e1", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":32}=1.041; {\"n_tokens\":64}=1.032", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/1ddb007af161404d.json", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_1ddb007af161404d/validation/report.json sha256=46247d656a8e6f22b694af3c8fa20fa483be6faeee33a1fac7a6a079ca280bd4", "date": "2026-11-08"}]
sources: ["aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/40937d155ca6aa98.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/2c7c5bacd06c8e29.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/aa1aaac0bdd0068f.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/f2e2b374be1c9f8b.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/1ddb007af161404d.json"]
---

# GLM MoE 五份最终实现：仅20个已测Point

这是已交付到本地PR准备树的具体案例范围和测试事实，非通则。原有case的其他测点不被本页替代。

## Applies when

`n_tokens` 是本次 hidden_states 第一维，不是路由后每expert GEMM的M；报告未记录逐expert实际M，不能按tokens×topk均摊。
逻辑输入形状：hidden_states[T,D]、w1[E,2I,D]、w2[E,D,I]、routing tensors[T,K]，out[T,D]；MXFP4存储和scale物理布局由下列exact Point布局决定，而非直接套行主序公式。
topk_ids每行distinct：无mask为local id，有mask为global id；expert_mask按全局expert顺序映射到本rank权重，输出是本rank份额，不包括跨rank collective。
GLM separated布局使用SiLU(gate)·up、MXFP4 activation路径；Flash/Pro interleave布局使用clamped SwiGLU，gate上限limit、up限制在±limit。contract reference保留bf16 activation，但本页部分实现内部用fp8；SQNR验收不意味着两条数值路径等价。所有本页MoE doweight_stage1=false，routing weight在stage2或最终route sum使用。

### glm_ep4_l — `40937d155ca6aa98`

- Exact schema: `fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/40937d155ca6aa98.json`。
- 固定轴：`{"model_dim":6144,"n_experts":65,"topk":9,"inter_dim":2048,"gate_up_rows":4096,"model_blocks":192,"inter_blocks":64,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"separated","has_expert_mask":false,"act_quant":4,"has_num_local_tokens":false,"has_swiglu_limit":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`shuffled_16x16`；scalars：`{}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"shuffled_16x16":{"layouts":{"w1":{"kind":"preshuffle","tile":[16,16]},"w2":{"kind":"preshuffle","tile":[16,16]},"w1_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true},"w2_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":3}]}}}`。
- 测试仅包含：`[{"n_tokens":256},{"n_tokens":2048},{"n_tokens":4096}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页separated/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### glm_ep4_s — `2c7c5bacd06c8e29`

- Exact schema: `fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/2c7c5bacd06c8e29.json`。
- 固定轴：`{"model_dim":6144,"n_experts":65,"topk":9,"inter_dim":2048,"gate_up_rows":4096,"model_blocks":192,"inter_blocks":64,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"separated","has_expert_mask":false,"act_quant":4,"has_num_local_tokens":false,"has_swiglu_limit":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`shuffled_16x16`；scalars：`{}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"shuffled_16x16":{"layouts":{"w1":{"kind":"preshuffle","tile":[16,16]},"w2":{"kind":"preshuffle","tile":[16,16]},"w1_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true},"w2_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":3}]}}}`。
- 测试仅包含：`[{"n_tokens":4},{"n_tokens":8},{"n_tokens":16},{"n_tokens":32},{"n_tokens":64},{"n_tokens":128}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页separated/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### glm_tp8_l — `aa1aaac0bdd0068f`

- Exact schema: `fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/aa1aaac0bdd0068f.json`。
- 固定轴：`{"model_dim":6144,"n_experts":257,"topk":9,"inter_dim":256,"gate_up_rows":512,"model_blocks":192,"inter_blocks":8,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"separated","has_expert_mask":false,"act_quant":4,"has_num_local_tokens":false,"has_swiglu_limit":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`shuffled_16x16`；scalars：`{}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"shuffled_16x16":{"layouts":{"w1":{"kind":"preshuffle","tile":[16,16]},"w2":{"kind":"preshuffle","tile":[16,16]},"w1_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true},"w2_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":3}]}}}`。
- 测试仅包含：`[{"n_tokens":512},{"n_tokens":1024},{"n_tokens":2048},{"n_tokens":4096}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页separated/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### glm_l — `f2e2b374be1c9f8b`

- Exact schema: `fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/f2e2b374be1c9f8b.json`。
- 固定轴：`{"model_dim":6144,"n_experts":257,"topk":9,"inter_dim":512,"gate_up_rows":1024,"model_blocks":192,"inter_blocks":16,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"separated","has_expert_mask":false,"act_quant":4,"has_num_local_tokens":false,"has_swiglu_limit":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`shuffled_16x16`；scalars：`{}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"shuffled_16x16":{"layouts":{"w1":{"kind":"preshuffle","tile":[16,16]},"w2":{"kind":"preshuffle","tile":[16,16]},"w1_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true},"w2_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":3}]}}}`。
- 测试仅包含：`[{"n_tokens":128},{"n_tokens":512},{"n_tokens":1024},{"n_tokens":2048},{"n_tokens":4096}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页separated/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### glm_s — `1ddb007af161404d`

- Exact schema: `fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/1ddb007af161404d.json`。
- 固定轴：`{"model_dim":6144,"n_experts":257,"topk":9,"inter_dim":512,"gate_up_rows":1024,"model_blocks":192,"inter_blocks":16,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"separated","has_expert_mask":false,"act_quant":4,"has_num_local_tokens":false,"has_swiglu_limit":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`shuffled_16x16`；scalars：`{}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"shuffled_16x16":{"layouts":{"w1":{"kind":"preshuffle","tile":[16,16]},"w2":{"kind":"preshuffle","tile":[16,16]},"w1_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true},"w2_scale":{"kind":"e8m0_scales","block_size":32,"pad_to":[256,8],"shuffled":true}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":3}]}}}`。
- 测试仅包含：`[{"n_tokens":32},{"n_tokens":64}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页separated/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

## Counter-signals

- 不外推未列出的tokens、slots、scalar、dtype、layout或路由分布；原报告输入抽样不能替代真实流量。
- 正确性及速度均为整次contract调用；不证明跨WG协议、重入/多stream安全、实际occupancy、单项kernel性能或某一机制的因果收益。
- 本地source-bound audit/bench通过不是CI通过或线上serve已上线；payload未写入accuracy/speedup。本页仅引用归档事实，不改变dispatch。
- 实际运行AKLR_MIN_SPEEDUP未归档，表中默认资格为固定base默认值下的重建，不是当时部署env证明。
- `f2e2b374be1c9f8b` 的tokens128正确性通过，但entry与默认资格均未通过；不把该点隐藏于整组收益。

## Mechanism

以下逐源码核对的是代码事实，不是收益归因。可选分支和metadata依赖不会因whole-call通过就被单独证成。

- `40937d155ca6aa98`（glm_ep4_l）：本页256/2048/4096 tokens 均在 `_OWNED`，因此直接调用 `build_moe_module(..., cfg_m=2048)`，并非 run 中后面的 AITER two_stages 回退。自有模块具备 route/gather、两投影、SiLU乘up、量化及 route reduction 的源码路径；入口把全部七个张量地址传入模块，并直接写 out。不依据注释中的平均专家行数解释收益。 见 S1。
- `2c7c5bacd06c8e29`（glm_ep4_s）：4/8/16 tokens 在 `_FOLD` 中，stage1 自读路由及 hidden rows；32/64/128 tokens 使用 prelude，尾部 stage1 具备六份 K partial 和 arrival-counter 合并路径。stage1 输出 packed fp4 中间值及 scales，stage2 调用 `_flydsl_moe_stage2_impl`，配置 fp4/fp4、bf16 out、atomic 以及 sorted routing weights。 见 S2。
- `aa1aaac0bdd0068f`（glm_tp8_l）：`_plan` 从 AITER metadata 取配置，再替换本文件 stage1、sort 和 stage2 实现；四个本页 tokens 均列在 `_TUNED`，源码有 fp4 route-result packing/reduction、sort与activation quant 融合及可选 W1 prefetch。run 在 atomic 与 route-reduce 两分支分别传 sorted weights 或 topk_weights；动态 metadata 的具体结果仍应按报告与生成代码复核，不把所有可选分支都声称为每点实际执行。 见 S3。
- `f2e2b374be1c9f8b`（glm_l）：`_DECODE` 仅列1/2/4 tokens，本页128/512/1024/2048/4096 不命中该两核 decode 分支。`run` 在契约 dtype/layout 条件匹配时取 `_aiter_plan`，再由 `_aiter_into_out` 组合自有 route/quant 或 AITER sort、两阶段投影与可选自有 fp4 reduction，直接写 out；不将源码存在的每个可选分支当作每点命中证明。128 tokens 的正确性通过但性能门槛未通过。 见 S4。
- `1ddb007af161404d`（glm_s）：本页32/64 tokens 不在 `_DECODE`，但均在 `_FOLD`。shuffled 布局且入口 dtype/attr 条件满足时 `_fold_run` 调用 `_fold_stage1` 与 `_fold_stage2`；stage1 生成路由/packed fp4 中间值及 scales，stage2 使用 routing weights 和 atomic bf16 epilogue。32 tokens 网格布尔设置为(True,True)，64为(False,False)；不能把网格设置称为实际绝对 XCD 编号。 见 S5。

## Result

逐Point列出本地原报告结果；不是Points均值。倍率为报告可发表speedup字段；baseline是 `aklr.ops.fused_moe.baseline.fused_moe_baseline`。
profile v18：AKLR CUDA graph、Kineto gpu_duration_sum、每call前1GiB读flush、每replay内mean、replay间median、每round seeded shuffle，Student-t mean-log-ratio下界。以下N取timings实际n_rounds，不把round_looks上限32或measurement_replays=10当N。

全部本页MoE完整Point在固定base没有默认served incumbent；对baseline的胜出不代表击败另一份incumbent。

| UUID | exact变量轴 | scalar | accuracy / worst violation | baseline倍率 | incumbent倍率 / UUID | N baseline/incumbent | entry / 默认资格 |
|---|---|---|---|---|---|---|---|
| `40937d155ca6aa98` | `{"n_tokens":256}` | `{}` | passed / 0.0 | 1.172 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `40937d155ca6aa98` | `{"n_tokens":2048}` | `{}` | passed / 0.0 | 1.361 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `40937d155ca6aa98` | `{"n_tokens":4096}` | `{}` | passed / 0.0 | 1.258 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `2c7c5bacd06c8e29` | `{"n_tokens":4}` | `{}` | passed / 0.0 | 1.185 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `2c7c5bacd06c8e29` | `{"n_tokens":8}` | `{}` | passed / 0.0 | 1.145 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `2c7c5bacd06c8e29` | `{"n_tokens":16}` | `{}` | passed / 0.0 | 1.126 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `2c7c5bacd06c8e29` | `{"n_tokens":32}` | `{}` | passed / 0.0 | 1.168 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `2c7c5bacd06c8e29` | `{"n_tokens":64}` | `{}` | passed / 0.0 | 1.183 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `2c7c5bacd06c8e29` | `{"n_tokens":128}` | `{}` | passed / 0.0 | 1.159 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `aa1aaac0bdd0068f` | `{"n_tokens":512}` | `{}` | passed / 0.0 | 1.266 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `aa1aaac0bdd0068f` | `{"n_tokens":1024}` | `{}` | passed / 0.0 | 1.503 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `aa1aaac0bdd0068f` | `{"n_tokens":2048}` | `{}` | passed / 0.0 | 1.61 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `aa1aaac0bdd0068f` | `{"n_tokens":4096}` | `{}` | passed / 0.0 | 1.843 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `f2e2b374be1c9f8b` | `{"n_tokens":128}` | `{}` | passed / 0.0 | 1.008 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 未通过 / 未通过 |
| `f2e2b374be1c9f8b` | `{"n_tokens":512}` | `{}` | passed / 0.0 | 1.063 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `f2e2b374be1c9f8b` | `{"n_tokens":1024}` | `{}` | passed / 0.0 | 1.164 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `f2e2b374be1c9f8b` | `{"n_tokens":2048}` | `{}` | passed / 0.0 | 1.41 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `f2e2b374be1c9f8b` | `{"n_tokens":4096}` | `{}` | passed / 0.0 | 1.674 *(field, E4)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `1ddb007af161404d` | `{"n_tokens":32}` | `{}` | passed / 0.0 | 1.041 *(field, E5)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `1ddb007af161404d` | `{"n_tokens":64}` | `{}` | passed / 0.0 | 1.032 *(field, E5)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |

上表倍率与gate/default资格分别保留；表里的通过不是CI或线上服务状态。

## Reproduce

固定base：`a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`；pinned AITER：`dbd8bf5bd624120197a7a26780a8c72201824f0f`；报告记录的 `amd-aiter` 为 `0.1.21.dev66+gdbd8bf5bd`。MoE baseline 对 gate_mode=interleave 设置 `AITER_BF16_FP8_MOE_BOUND=16`，这是activation路径的行数切换设置，不是SQNR dB阈值；DSA baseline不使用此MoE设置。未归档实际sglang版本、实际tuned配置及完整运行env，因此不宣称这些条件已锁定。

本次未运行任何GPU、网络或新测量。将来在gfx950、FlyDSL 0.3.2及同base上，用完整导出schema而非裸kernel复测，逐点核对scalars、dtype、attrs和incumbent；不能以重放结果替换本页原报告。

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/40937d155ca6aa98.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/2c7c5bacd06c8e29.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/aa1aaac0bdd0068f.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/f2e2b374be1c9f8b.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/1ddb007af161404d.json
```

原始本地报告和receipt路径及SHA256见Evidence；local文件可能不长存。源码原文件SHA256、payload/schema SHA256、官方kernel content hash及逐Point结果另归档于 `local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/moe_dsa_kernel_kb_mapping.json`，不将这些本地归档冒充CI永久证据。

## Rollback

当前只有本地准备文件，未声明已serve。撤回时只删除本页列出的新UUID实现及其新增case（通过正常review流程）；旧实现保持。恢复选择由base原有Point与baseline决定，不声称无条件回退到某一赢家。

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `40937d155ca6aa98` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_40937d155ca6aa98/validation/report.json sha256=bddb9f8e50ce814b42fe7644b168dd8e07cb249d76fb0dfe035980332b6100a2；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_40937d155ca6aa98/receipt.json sha256=84187951363ef041aa4a2e9e4a51745411df69b58f96741aca4af91ccff38510。
- S1 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_l/tx/glm_ep4_l_kernel_after_refinalize.py:2532 最终原源码，sha256=d1fa451c8a92024960193cfe5453c4ef428ea1724ef8170bbe90cd517d29f2cf；核对行段 2532–2605, 3065–3183, 3270–3308；与导出kernel.py字节完全相同。
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `2c7c5bacd06c8e29` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_2c7c5bacd06c8e29/validation/report.json sha256=2198069dd4f4f8093d31b643d167003d10ea25258d78db864fa470538162c34f；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_2c7c5bacd06c8e29/receipt.json sha256=a2e1760098febb807d9fceb3aeef9a6e7ea0abe698a7e9dd2fa4111945f56124。
- S2 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_s/tx/glm_ep4_s_kernel_final.py:57 最终原源码，sha256=0fdbd1d9d931eba74691858c75fc70ba86bf31024eea7d581da3ca873b659de9；核对行段 57–66, 326–389, 423–502；与导出kernel.py字节完全相同。
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `aa1aaac0bdd0068f` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_aa1aaac0bdd0068f/validation/report.json sha256=968368f53cff8c5b1b4eba61528513b95cd5582d18be8421fc79658b6ef850a0；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_aa1aaac0bdd0068f/receipt.json sha256=e55617fe94659e69ba0a75ab9cdf69b8dcf9f431dedc591485288dd0a7081bc9。
- S3 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_tp8_l/tx/glm_tp8_l_kernel_final.py:2363 最终原源码，sha256=a8f4c3f4c72a0956e637dd63bcedc819c5f1633c7de0b2d6142c6d880207f9ae；核对行段 2363–2372, 2376–2489, 2492–2679；与导出kernel.py字节完全相同。
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `f2e2b374be1c9f8b` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g4/glm_f2e2b374be1c9f8b/validation/report.json sha256=16976d1fb17a8accdbdde1ad5a0e7585dc91bbee2b94973cdceee5e6bab620e1；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g4/glm_f2e2b374be1c9f8b/receipt.json sha256=83fcb0c0cbce307a0f7a39c990e829207bd3967aed08f9e66d8aa53decaffc74。
- S4 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_l/tx/glm_l_kernel_final.py:362 最终原源码，sha256=462a01e0285a3a07e2fe2ad41b02540328a05cd0cbd90d4b1d7a0df46d18d181；核对行段 362–397, 2026–2241, 2264–2327；与导出kernel.py字节完全相同。
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `1ddb007af161404d` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_1ddb007af161404d/validation/report.json sha256=46247d656a8e6f22b694af3c8fa20fa483be6faeee33a1fac7a6a079ca280bd4；receipt local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_1ddb007af161404d/receipt.json sha256=0fb0405af81cf84bd8caf8e3ccca89c14be8519a6e6b0cecf5d8584819420e5a。
- S5 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_s/tx/glm_s_kernel_final.py:396 最终原源码，sha256=3943d69f65c792048624979d5a6616a1a67e3a357d4be07c3e773fd075e50b53；核对行段 396–431, 970–970, 1920–1989, 2034–2058；与导出kernel.py字节完全相同。
- S6 aklr/ops/fused_moe/contract.py:75 axes/attrs/tensor/layout/scalar/output契约；本页只声明exact Points，不声明完整contract域。
- S7 aklr/ops/fused_moe/baseline.py:9 baseline实际调用；MoE interleave设置AITER_BF16_FP8_MOE_BOUND=16；DSA baseline调用sglang Triton split-K。
- S8 aklr/ops/fused_moe/reference.py:77 global→local mask映射；:88–99 activation量化；:107–127 clamp、激活、stage2 routing weight及topk求和。
