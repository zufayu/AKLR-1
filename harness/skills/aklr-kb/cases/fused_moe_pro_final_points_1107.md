---
title: "Pro MoE 三份最终实现：仅19个已测Point"
kind: "case"
ops: ["moe"]
dtypes: ["mxfp4", "bf16"]
regimes: ["pro_ep4: [{\"n_tokens\":1},{\"n_tokens\":2},{\"n_tokens\":16},{\"n_tokens\":32}]", "pro_tp384: [{\"n_tokens\":1},{\"n_tokens\":2},{\"n_tokens\":4},{\"n_tokens\":8},{\"n_tokens\":16},{\"n_tokens\":32},{\"n_tokens\":64},{\"n_tokens\":128},{\"n_tokens\":256}]", "pro_tp768: [{\"n_tokens\":1},{\"n_tokens\":2},{\"n_tokens\":4},{\"n_tokens\":16},{\"n_tokens\":32},{\"n_tokens\":64}]"]
contract: bounded
gens: ["gfx950"]
flydsl: "0.3.2"
basis: "measured N=16; actual graph timing rounds per comparison; source mechanisms not causally attributed"
measured: [{"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":1}=5.103; {\"n_tokens\":2}=1.879; {\"n_tokens\":16}=1.236; {\"n_tokens\":32}=1.129", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384/ef4b4c3360749dd8.json", "run_ref": "local:/tmp/inf3_validation_final_1108/results/g4/pro_ef4b4c3360749dd8/validation/report.json sha256=43145a211424bdd0e71e3ad4091897d26548c5759905c11ba49e55e1eedfee02", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":1}=3.797; {\"n_tokens\":2}=3.173; {\"n_tokens\":4}=2.444; {\"n_tokens\":8}=1.863; {\"n_tokens\":16}=1.539; {\"n_tokens\":32}=1.201; {\"n_tokens\":64}=1.087; {\"n_tokens\":128}=1.03; {\"n_tokens\":256}=1.041", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384/98d3a5be0bf0cdd0.json", "run_ref": "local:/tmp/inf3_validation_final_1108/results/g5/pro_98d3a5be0bf0cdd0/validation/report.json sha256=111e33bb864ed781176c5857f783f81d33ef73162512631328cecdf127e9bf7f", "date": "2026-11-08"}, {"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":1}=3.154; {\"n_tokens\":2}=2.321; {\"n_tokens\":4}=1.839; {\"n_tokens\":16}=1.234; {\"n_tokens\":32}=1.146; {\"n_tokens\":64}=1.151", "gpu": "MI355X", "workload": "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768/6ea61fb782ff0801.json", "run_ref": "local:/tmp/inf3_validation_final_1108/results/g3/pro_6ea61fb782ff0801/validation/report.json sha256=d904abf2da53b53c9777862239f264b23fcaa37a7255d96f703a95248b92eae8", "date": "2026-11-08"}]
sources: ["aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384/ef4b4c3360749dd8.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384/98d3a5be0bf0cdd0.json", "aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768/6ea61fb782ff0801.json"]
---

# Pro MoE 三份最终实现：仅19个已测Point

这是已交付到本地PR准备树的具体案例范围和测试事实，非通则。原有case的其他测点不被本页替代。

## Applies when

`n_tokens` 是本次 hidden_states 第一维，不是路由后每expert GEMM的M；报告未记录逐expert实际M，不能按tokens×topk均摊。
逻辑输入形状：hidden_states[T,D]、w1[E,2I,D]、w2[E,D,I]、routing tensors[T,K]，out[T,D]；MXFP4存储和scale物理布局由下列exact Point布局决定，而非直接套行主序公式。
topk_ids每行distinct：无mask为local id，有mask为global id；expert_mask按全局expert顺序映射到本rank权重，输出是本rank份额，不包括跨rank collective。
GLM separated布局使用SiLU(gate)·up、MXFP4 activation路径；Flash/Pro interleave布局使用clamped SwiGLU，gate上限limit、up限制在±limit。contract reference保留bf16 activation，但本页部分实现内部用fp8；SQNR验收不意味着两条数值路径等价。所有本页MoE doweight_stage1=false，routing weight在stage2或最终route sum使用。

### pro_ep4 — `ef4b4c3360749dd8`

- Exact schema: `fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384/ef4b4c3360749dd8.json`。
- 固定轴：`{"model_dim":7168,"n_experts":96,"topk":6,"inter_dim":3072,"global_experts":384,"gate_up_rows":6144,"model_blocks":224,"inter_blocks":96}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"interleave","has_expert_mask":true,"has_swiglu_limit":true,"act_quant":4,"has_num_local_tokens":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","expert_mask":"int32","out":"bfloat16","num_local_tokens":null}`。
- layout：`gate_up_interleaved`；scalars：`{"swiglu_limit":10.0}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"gate_up_interleaved":{"layouts":{"w1":{"kind":"gate_up_interleaved_weight","gate_up":true},"w2":{"kind":"gate_up_interleaved_weight","gate_up":false},"w1_scale":{"kind":"gate_up_interleaved_scales","gate_up":true},"w2_scale":{"kind":"gate_up_interleaved_scales","gate_up":false}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":2}]}}}`。
- 测试仅包含：`[{"n_tokens":1},{"n_tokens":2},{"n_tokens":16},{"n_tokens":32}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页interleave/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### pro_tp384 — `98d3a5be0bf0cdd0`

- Exact schema: `fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384/98d3a5be0bf0cdd0.json`。
- 固定轴：`{"model_dim":7168,"n_experts":384,"topk":6,"inter_dim":384,"gate_up_rows":768,"model_blocks":224,"inter_blocks":12,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"interleave","has_expert_mask":false,"has_swiglu_limit":true,"act_quant":4,"has_num_local_tokens":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`gate_up_interleaved`；scalars：`{"swiglu_limit":10.0}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"gate_up_interleaved":{"layouts":{"w1":{"kind":"gate_up_interleaved_weight","gate_up":true},"w2":{"kind":"gate_up_interleaved_weight","gate_up":false},"w1_scale":{"kind":"gate_up_interleaved_scales","gate_up":true},"w2_scale":{"kind":"gate_up_interleaved_scales","gate_up":false}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":2}]}}}`。
- 测试仅包含：`[{"n_tokens":1},{"n_tokens":2},{"n_tokens":4},{"n_tokens":8},{"n_tokens":16},{"n_tokens":32},{"n_tokens":64},{"n_tokens":128},{"n_tokens":256}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页interleave/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

### pro_tp768 — `6ea61fb782ff0801`

- Exact schema: `fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768`; contract `fused_moe@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768/6ea61fb782ff0801.json`。
- 固定轴：`{"model_dim":7168,"n_experts":384,"topk":6,"inter_dim":768,"gate_up_rows":1536,"model_blocks":224,"inter_blocks":24,"global_experts":1}`。
- attrs：`{"activation":"silu","quant_type":"per_1x32","doweight_stage1":false,"gate_mode":"interleave","has_expert_mask":false,"has_swiglu_limit":true,"act_quant":4,"has_num_local_tokens":false}`。
- dtypes：`{"hidden_states":"bfloat16","w1":"float4_e2m1","w2":"float4_e2m1","topk_weights":"float32","topk_ids":"int32","w1_scale":"float8_e8m0","w2_scale":"float8_e8m0","out":"bfloat16","expert_mask":null,"num_local_tokens":null}`。
- layout：`gate_up_interleaved`；scalars：`{"swiglu_limit":10.0}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"gate_up_interleaved":{"layouts":{"w1":{"kind":"gate_up_interleaved_weight","gate_up":true},"w2":{"kind":"gate_up_interleaved_weight","gate_up":false},"w1_scale":{"kind":"gate_up_interleaved_scales","gate_up":true},"w2_scale":{"kind":"gate_up_interleaved_scales","gate_up":false}},"detect_facts":[{"location":"tensor_attr","name":"w1.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w2.is_shuffled","expected_value":true},{"location":"tensor_attr","name":"w1_scale.ndim","expected_value":2}]}}}`。
- 测试仅包含：`[{"n_tokens":1},{"n_tokens":2},{"n_tokens":4},{"n_tokens":16},{"n_tokens":32},{"n_tokens":64}]`。
- compare：`{"max_violation_fraction":0.001,"determinism":"bounded","element_order":"fixed"}`；固定base合同对本页interleave/silu使用whole-output SQNR bound=13 dB（contract.py:265–272）；与interleave baseline行数切换设置16不同。

## Counter-signals

- 不外推未列出的tokens、slots、scalar、dtype、layout或路由分布；原报告输入抽样不能替代真实流量。
- 正确性及速度均为整次contract调用；不证明跨WG协议、重入/多stream安全、实际occupancy、单项kernel性能或某一机制的因果收益。
- 本地source-bound audit/bench通过不是CI通过或线上serve已上线；payload未写入accuracy/speedup。本页仅引用归档事实，不改变dispatch。
- 实际运行AKLR_MIN_SPEEDUP未归档，表中默认资格为固定base默认值下的重建，不是当时部署env证明。
- pro_ep8最终源码缺失，准备入口列为blocker；本页只交付上述三份，不借旧源码补第四份。

## Mechanism

以下逐源码核对的是代码事实，不是收益归因。可选分支和metadata依赖不会因whole-call通过就被单独证成。

- `ef4b4c3360749dd8`（pro_ep4）：1/2 tokens 用 DIRECT per-route stage1，16/32 tokens 用 tile path；stage1 K份数分别依入口配置选择。1/2 tokens 且 limit=10 时 SCY=true，中间量使用固定尺度 fp8路径；stage2 用 `_stage2_tm`，其他两点用 `_stage2_cr`。global id 经 expert_mask 映射 local id，路由权重只在stage2使用。没有将 SCY 数值范围推导升级为独立验证通则。 见 S1。
- `98d3a5be0bf0cdd0`（pro_tp384）：本页1/2/4/8/16/32/64 tokens 走自有两投影路径，按 route 数选择多行处理和可选 K partial，Y 使用bf16。128/256 tokens 进入 `_aiter_plan`/`_front_into`，自有 front 写 sorted routes、fp8 A1、scales 与 out，接 AITER metadata 投影，stage2 读 sorted weights。整次调用包含这些辅助步骤，不是单项 GEMM 计时。 见 S2。
- `6ea61fb782ff0801`（pro_tp768）：1/2 tokens 选择 split stage1与 `_stage2r`；4 tokens 选择 per-route 两阶段；16 tokens 选择 persistent grouped stage1并直接接 `_stage2g`；32/64 tokens 先 `_groups` 再 grouped 两阶段。group数由 `_G` 定义，不是报告中的实际每expert M；源码 `_TOKENS` 还列3，但此页不包含该未测点。 见 S3。

## Result

逐Point列出本地原报告结果；不是Points均值。倍率为报告可发表speedup字段；baseline是 `aklr.ops.fused_moe.baseline.fused_moe_baseline`。
profile v18：AKLR CUDA graph、Kineto gpu_duration_sum、每call前1GiB读flush、每replay内mean、replay间median、每round seeded shuffle，Student-t mean-log-ratio下界。以下N取timings实际n_rounds，不把round_looks上限32或measurement_replays=10当N。

全部本页MoE完整Point在固定base没有默认served incumbent；对baseline的胜出不代表击败另一份incumbent。

| UUID | exact变量轴 | scalar | accuracy / worst violation | baseline倍率 | incumbent倍率 / UUID | N baseline/incumbent | entry / 默认资格 |
|---|---|---|---|---|---|---|---|
| `ef4b4c3360749dd8` | `{"n_tokens":1}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 5.103 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `ef4b4c3360749dd8` | `{"n_tokens":2}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.879 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `ef4b4c3360749dd8` | `{"n_tokens":16}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.236 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `ef4b4c3360749dd8` | `{"n_tokens":32}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.129 *(field, E1)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":1}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 3.797 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":2}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 3.173 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":4}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 2.444 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":8}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.863 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":16}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.539 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":32}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.201 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":64}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.087 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":128}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.03 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `98d3a5be0bf0cdd0` | `{"n_tokens":256}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.041 *(field, E2)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `6ea61fb782ff0801` | `{"n_tokens":1}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 3.154 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `6ea61fb782ff0801` | `{"n_tokens":2}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 2.321 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `6ea61fb782ff0801` | `{"n_tokens":4}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.839 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `6ea61fb782ff0801` | `{"n_tokens":16}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.234 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `6ea61fb782ff0801` | `{"n_tokens":32}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.146 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |
| `6ea61fb782ff0801` | `{"n_tokens":64}` | `{"swiglu_limit":10.0}` | passed / 0.0 | 1.151 *(field, E3)* | 无（base无默认served incumbent） | 16/– | 通过 / 通过 |

Pro大倍率的gate下界可能截断于2，但speedup字段使用未截断统计；两者不是同一个数。各原值和门槛均在映射JSON逐Point保留。

## Reproduce

固定base：`a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`；pinned AITER：`dbd8bf5bd624120197a7a26780a8c72201824f0f`；报告记录的 `amd-aiter` 为 `0.1.21.dev66+gdbd8bf5bd`。MoE baseline 对 gate_mode=interleave 设置 `AITER_BF16_FP8_MOE_BOUND=16`，这是activation路径的行数切换设置，不是SQNR dB阈值；DSA baseline不使用此MoE设置。未归档实际sglang版本、实际tuned配置及完整运行env，因此不宣称这些条件已锁定。

本次未运行任何GPU、网络或新测量。将来在gfx950、FlyDSL 0.3.2及同base上，用完整导出schema而非裸kernel复测，逐点核对scalars、dtype、attrs和incumbent；不能以重放结果替换本页原报告。

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384/ef4b4c3360749dd8.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384/98d3a5be0bf0cdd0.json
```

```bash
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768/6ea61fb782ff0801.json
```

原始本地报告和receipt路径及SHA256见Evidence；local文件可能不长存。源码原文件SHA256、payload/schema SHA256、官方kernel content hash及逐Point结果另归档于 `local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/moe_dsa_kernel_kb_mapping.json`，不将这些本地归档冒充CI永久证据。

## Rollback

当前只有本地准备文件，未声明已serve。撤回时只删除本页列出的新UUID实现及其新增case（通过正常review流程）；旧实现保持。恢复选择由base原有Point与baseline决定，不声称无条件回退到某一赢家。

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `ef4b4c3360749dd8` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_final_1108/results/g4/pro_ef4b4c3360749dd8/validation/report.json sha256=43145a211424bdd0e71e3ad4091897d26548c5759905c11ba49e55e1eedfee02；receipt local:/tmp/inf3_validation_final_1108/results/g4/pro_ef4b4c3360749dd8/receipt.json sha256=46c274c47f2c29d2c396095dc3c830cd82ba577a141746eddb57cb30003bac6a。
- S1 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/pro_ep4/tx/pro_ep4_kernel_final.py:726 最终原源码，sha256=77ab645d8eab103169a3817971f4c89737b9453dab90bc41edece8368f80d6b9；核对行段 726–823, 847–863；与导出kernel.py字节完全相同。
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `98d3a5be0bf0cdd0` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_final_1108/results/g5/pro_98d3a5be0bf0cdd0/validation/report.json sha256=111e33bb864ed781176c5857f783f81d33ef73162512631328cecdf127e9bf7f；receipt local:/tmp/inf3_validation_final_1108/results/g5/pro_98d3a5be0bf0cdd0/receipt.json sha256=87b6daf5645fac5a975df7dcbbc756c6ce85002b2594eca8fb3c94a1bf4d9caf。
- S2 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/pro_tp384/tx/pro_tp384_kernel_final.py:19 最终原源码，sha256=e0a55f98514b083cb783491d39030dd519cb7bd50f2293db73aae75e00c58d04；核对行段 19–19, 449–537, 829–907, 931–980；与导出kernel.py字节完全相同。
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `6ea61fb782ff0801` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_final_1108/results/g3/pro_6ea61fb782ff0801/validation/report.json sha256=d904abf2da53b53c9777862239f264b23fcaa37a7255d96f703a95248b92eae8；receipt local:/tmp/inf3_validation_final_1108/results/g3/pro_6ea61fb782ff0801/receipt.json sha256=63ae4bc7373ac612f23ac79d9437c0594abc8bdb56663d9e0b75e5ba027d9345。
- S3 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/pro_tp768/tx/pro_tp768_kernel_final.py:872 最终原源码，sha256=f6f3e7b81e21dfdfd9410d902fbcf9e421352782de6e948d92030263bddec6cf；核对行段 872–883, 887–990, 1014–1025；与导出kernel.py字节完全相同。
- S4 aklr/ops/fused_moe/contract.py:75 axes/attrs/tensor/layout/scalar/output契约；本页只声明exact Points，不声明完整contract域。
- S5 aklr/ops/fused_moe/baseline.py:9 baseline实际调用；MoE interleave设置AITER_BF16_FP8_MOE_BOUND=16；DSA baseline调用sglang Triton split-K。
- S6 aklr/ops/fused_moe/reference.py:77 global→local mask映射；:88–99 activation量化；:107–127 clamp、激活、stage2 routing weight及topk求和。
