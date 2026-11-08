---
title: "DSA head8最终实现：仅三个配对tokens/slots Point"
kind: "case"
ops: ["dsa_backend"]
dtypes: ["fp8", "bf16"]
regimes: ["dsa8: [{\"n_tokens\":32,\"n_slots\":65536},{\"n_tokens\":128,\"n_slots\":262144},{\"n_tokens\":256,\"n_slots\":524288}]"]
contract: bitwise
gens: ["gfx950"]
flydsl: "0.3.2"
basis: "measured N=16; actual graph timing rounds per comparison; source mechanisms not causally attributed"
measured: [{"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "{\"n_tokens\":32,\"n_slots\":65536}=1.671; {\"n_tokens\":128,\"n_slots\":262144}=1.596; {\"n_tokens\":256,\"n_slots\":524288}=1.238", "gpu": "MI355X", "workload": "aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/4ec6dbc4f20a30a5.json", "run_ref": "local:/tmp/inf3_validation_final_1108/results/g4/dsa_4ec6dbc4f20a30a5/validation/report.json sha256=d51b645c6750952942f17b2f57305b7873c3255b58533ae5d52ca715b0bf359d", "date": "2026-11-08"}]
sources: ["aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/4ec6dbc4f20a30a5.json"]
---

# DSA head8最终实现：仅三个配对tokens/slots Point

这是已交付到本地PR准备树的具体案例范围和测试事实，非通则。原有case的其他测点不被本页替代。

## Applies when

本rank absorbed KV-b-projection的稀疏MLA decode：fp8 q与KV不额外反量化，score乘sm_scale，softmax加权取KV前512 latent；后64是RoPE。topk_indices=-1表示不读，全部无有效slot时输出零。
三组(tokens,slots)是配对点，不能做笛卡尔积或宣称连续域；head=8、kv_dim=576、topk=2048、sm_scale=0.0625均固定。

### dsa8 — `4ec6dbc4f20a30a5`

- Exact schema: `dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048`; contract `dsa_sparse_mla_decode@1.0`，entry `kernel.py::run`。
- 实现文件：`aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/4ec6dbc4f20a30a5.json`。
- 固定轴：`{"n_heads":8,"kv_dim":576,"topk":2048,"kv_lora_rank":512}`。
- attrs：`{}`。
- dtypes：`{"q":"float8_e4m3fn","kv_cache":"float8_e4m3fn","topk_indices":"int32","output":"bfloat16"}`。
- layout：`row_major`；scalars：`{"sm_scale":0.0625}`。
- 布局合同事实（含scale物理布局与detect attrs）：`{"by_name":{"row_major":{"layouts":{},"detect_facts":[]}}}`。
- 测试仅包含：`[{"n_tokens":32,"n_slots":65536},{"n_tokens":128,"n_slots":262144},{"n_tokens":256,"n_slots":524288}]`。
- compare：`{"max_violation_fraction":0.0,"determinism":"bitwise","element_order":"fixed"}`；output SQNR bound=20 dB。

## Counter-signals

- 不外推未列出的tokens、slots、scalar、dtype、layout或路由分布；原报告输入抽样不能替代真实流量。
- 正确性及速度均为整次contract调用；不证明跨WG协议、重入/多stream安全、实际occupancy、单项kernel性能或某一机制的因果收益。
- 本地source-bound audit/bench通过不是CI通过或线上serve已上线；payload未写入accuracy/speedup。本页仅引用归档事实，不改变dispatch。
- 实际运行AKLR_MIN_SPEEDUP未归档，表中默认资格为固定base默认值下的重建，不是当时部署env证明。

## Mechanism

以下逐源码核对的是代码事实，不是收益归因。可选分支和metadata依赖不会因whole-call通过就被单独证成。

- `4ec6dbc4f20a30a5`（dsa8）：固定NH=8、D=576、DV=512、TOPK=2048。32/128/256 tokens 的 `_config` 分别取 splits=8/2/1、每WG4 waves。producer 用 buffer-to-LDS DMA 双槽 ring gather，fp8 MFMA 与 online softmax，随后 LDS 合并 waves；S>1 时写bf16 partial与fp32 maxima/sums，使用dispatch-tag arrival words和atomic cmpswap/or决定同launch合并，S=1直接写output。只有一个 producer launch；源码事实并不证明跨WG协议在所有并发/复用条件下安全，也不证明各步骤的独立收益或实际occupancy。 见 S1。

## Result

逐Point列出本地原报告结果；不是Points均值。倍率为报告可发表speedup字段；timings中 versus=baseline 的实际对手是 callable `aklr.plugins.sglang.layers.dsa_backend.baseline.dsa_sparse_mla_decode_baseline`。report workload record 的 baseline 元数据在这三个Point分别记为Triton incumbent schema路径，它不等于 timings中 versus=baseline 的实际计时对手；timed_versus_identity 将二者区分。
profile v18：AKLR CUDA graph、Kineto gpu_duration_sum、每call前1GiB读flush、每replay内mean、replay间median、每round seeded shuffle，Student-t mean-log-ratio下界。以下N取timings实际n_rounds，不把round_looks上限32或measurement_replays=10当N。

baseline是sglang Triton split-K sparse MLA。下列incumbent是固定base三个不同的真实AKLR Triton实现，分别计时；不以旧schema speedup常数冒充本次对比。

| UUID | exact变量轴 | scalar | accuracy / worst violation | baseline倍率 | incumbent倍率 / UUID | N baseline/incumbent | entry / 默认资格 |
|---|---|---|---|---|---|---|---|
| `4ec6dbc4f20a30a5` | `{"n_tokens":32,"n_slots":65536}` | `{"sm_scale":0.0625}` | passed / 0.0 | 1.671 *(field, E1)* | 1.555 / 43ffd5e3e5b67298 | 16/16 | 通过 / 通过 |
| `4ec6dbc4f20a30a5` | `{"n_tokens":128,"n_slots":262144}` | `{"sm_scale":0.0625}` | passed / 0.0 | 1.596 *(field, E1)* | 1.326 / ae32cc132679692c | 16/16 | 通过 / 通过 |
| `4ec6dbc4f20a30a5` | `{"n_tokens":256,"n_slots":524288}` | `{"sm_scale":0.0625}` | passed / 0.0 | 1.238 *(field, E1)* | 1.171 / fbe6ddab5dac5958 | 16/16 | 通过 / 通过 |

上表倍率与gate/default资格分别保留；表里的通过不是CI或线上服务状态。

## Reproduce

固定base：`a12866b1e40e86ea6aff7ac24cbe5634132f6b2e`；pinned AITER：`dbd8bf5bd624120197a7a26780a8c72201824f0f`；报告记录的 `amd-aiter` 为 `0.1.21.dev66+gdbd8bf5bd`。MoE baseline 对 gate_mode=interleave 设置 `AITER_BF16_FP8_MOE_BOUND=16`，这是activation路径的行数切换设置，不是SQNR dB阈值；DSA baseline不使用此MoE设置。未归档实际sglang版本、实际tuned配置及完整运行env，因此不宣称这些条件已锁定。

本次未运行任何GPU、网络或新测量。将来在gfx950、FlyDSL 0.3.2及同base上，用完整导出schema而非裸kernel复测，逐点核对scalars、dtype、attrs和incumbent；不能以重放结果替换本页原报告。

```bash
python -m aklr validate aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/4ec6dbc4f20a30a5.json
```

原始本地报告和receipt路径及SHA256见Evidence；local文件可能不长存。源码原文件SHA256、payload/schema SHA256、官方kernel content hash及逐Point结果另归档于 `local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/moe_dsa_kernel_kb_mapping.json`，不将这些本地归档冒充CI永久证据。

## Rollback

当前只有本地准备文件，未声明已serve。撤回时只删除本页列出的新UUID实现及其新增case（通过正常review流程）；旧实现保持。恢复选择由base原有Point与baseline决定，不声称无条件回退到某一赢家。

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `4ec6dbc4f20a30a5` 上表exact Points的整次调用accuracy/倍率/资格；非CI。原report local:/tmp/inf3_validation_final_1108/results/g4/dsa_4ec6dbc4f20a30a5/validation/report.json sha256=d51b645c6750952942f17b2f57305b7873c3255b58533ae5d52ca715b0bf359d；receipt local:/tmp/inf3_validation_final_1108/results/g4/dsa_4ec6dbc4f20a30a5/receipt.json sha256=2874fbe58ef36cba4da2bed485008b51da25179bba51b4d256acc64e513041d3。
- S1 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/dsa8/tx/dsa8_kernel_final.py:16 最终原源码，sha256=8109c10b4a8916ca06e454ebc7c2637807fc6733da61ef857886a57ee0c62268；核对行段 16–45, 93–151, 492–588, 630–714, 735–765, 768–792；与导出kernel.py字节完全相同。
- S2 aklr/plugins/sglang/layers/dsa_backend/contract.py:37 axes/attrs/tensor/layout/scalar/output契约；本页只声明exact Points，不声明完整contract域。
- S3 aklr/plugins/sglang/layers/dsa_backend/baseline.py:14 baseline实际调用sglang Triton split-K，区别于三个AKLR Triton incumbent。
