---
title: "DSA sparse MLA decode with eight heads"
kind: "case"
ops: ["dsa_backend"]
dtypes: ["fp8", "bf16"]
regimes: ["eight-head decode; exact paired token/slot sizes in case"]
contract: bitwise
gens: ["gfx950"]
flydsl: "0.3.2"
basis: "measured N=16; actual graph timing rounds per comparison; source mechanisms not causally attributed"
measured: [{"metric": "local AKLR graph per-Point baseline / candidate speedup field (not CI)", "value": "1.671 over baseline and 1.555 over incumbent at tokens 32 / slots 65536; see E1 and mapping for all paired Points", "gpu": "MI355X", "workload": "aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/4ec6dbc4f20a30a5.json", "run_ref": "local:/tmp/inf3_validation_final_1108/results/g4/dsa_4ec6dbc4f20a30a5/validation/report.json sha256=d51b645c6750952942f17b2f57305b7873c3255b58533ae5d52ca715b0bf359d", "date": "2026-11-08"}]
sources: ["aklr/plugins/sglang/layers/dsa_backend/impls/flydsl/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/4ec6dbc4f20a30a5.json"]
---

# DSA sparse MLA decode with eight heads

This case combines gather, attention and split merging in one producer launch, but changes the split count with token count. Treat the in-launch merge as a separate concurrency/reuse validation obligation, not as a broadly safe replacement for a second merge launch.

This section covers the implementations and measured token counts below. Local results are not CI acceptance or deployed dispatch. This is dsa_sparse_mla_decode@1.0, not the padded-pages flash_mla_with_kvcache@1.0 case in [mla_sparse_fp8_decode.md](mla_sparse_fp8_decode.md): the cache, query dtype, comparison and baseline differ.

## Applies when

Per-rank sparse absorbed-KV MLA: q[T,8,576] and kv_cache[S,576] are row-major float8_e4m3fn, topk_indices[T,2048] is int32, and output[T,8,512] is bf16. No extra dequantization scale is applied. Scores use sm_scale=0.0625; values are the first 512 latent dimensions and the final 64 are RoPE. Index -1 is skipped; no valid slot yields zero output. Attrs are empty; kv_lora_rank=512. The three (tokens,slots) pairs are not a Cartesian product or a continuous domain. Compare requires bitwise determinism, fixed element order, zero violation fraction and output SQNR >= 20 dB; bitwise determinism is not bitwise equality to the reference.

| Implementation | Configuration | Measured tokens |
|---|---|---|
| dsa8 | 8 heads, KV 576, top-k 2048; paired slots 65536 / 262144 / 524288 | 32, 128, 256 |

## Counter-signals

- Do not extrapolate to other tokens, slots, scalars, dtypes, layouts or routing/index distributions. The sampled inputs do not establish serving-traffic performance.
- Whole-contract correctness/timing does not prove cross-workgroup protocol safety, reentrancy/multistream safety, occupancy, individual-kernel performance or causal gains from a mechanism. Optional source branches are not independently validated by a whole-call pass.
- Payload accuracy/speedup fields were not populated; these archived local results change no dispatch. Actual AKLR_MIN_SPEEDUP was not archived: default qualification is reconstructed under fixed-base defaults, not proof of the runtime environment.

## Mechanism

Source facts, not ablation-based gain attribution.

- **dsa8**: NH=8, D=576, DV=512 and TOPK=2048 are fixed. _config uses splits=8/2/1 at 32/128/256 tokens and four waves per workgroup. One producer launch uses a double-slot buffer-to-LDS DMA ring gather, fp8 MFMA, online softmax and an LDS wave merge. With splits > 1 it writes bf16 partials and fp32 maxima/sums; dispatch-tag arrival words and atomic cmpswap/or select an in-launch merge. One split writes output directly. This does not establish protocol safety under arbitrary concurrency/reuse or actual occupancy. See S1.

## Result

Local whole-call AKLR graph measurements (profile v18), with a 1 GiB cache flush before each call and 16 actual timing rounds per comparison. Representative values are the published per-Point speedup fields, not averages. Full timing and statistical details remain in the reports and mapping.

The timed baseline callable is aklr.plugins.sglang.layers.dsa_backend.baseline.dsa_sparse_mla_decode_baseline (sglang Triton split-K). The workload record labels its baseline metadata with incumbent schema paths; those are not the callable timed under versus=baseline. timed_versus_identity records the distinction. The three fixed-base AKLR Triton incumbents were separately timed here; their identities are preserved in the mapping. These ratios are not old schema constants.

| Implementation | Representative speedup over baseline |
|---|---|
| dsa8 | baseline 1.671 / 1.596 / 1.238 and separately timed incumbent 1.555 / 1.326 / 1.171, at paired (tokens,slots) 32/65536, 128/262144, 256/524288 respectively *(field, E1)* |

These are selected per-Point results, not group averages. All 3 Points pass accuracy; all qualify under fixed-base defaults. The complete per-Point scalars, accuracy, timings, incumbent identities and gates remain in the original reports and mapping archive, not repeated here. Local qualification does not establish CI or serving.

## Reproduce

Fixed base a12866b1e40e86ea6aff7ac24cbe5634132f6b2e; pinned AITER dbd8bf5bd624120197a7a26780a8c72201824f0f; reported amd-aiter 0.1.21.dev66+gdbd8bf5bd; gfx950, MI355X, FlyDSL 0.3.2. DSA does not use the MoE activation-path row switch. Actual sglang version, tuned configuration and full runtime environment were not archived.

Future revalidation must use each full exported schema recorded in the mapping, not a bare kernel, and check exact Points, scalars, attrs, dtypes and incumbent identity. It must not replace these archived results. From the repository root, use the published_path for the implementation in the mapping:

```bash
python -m aklr validate <implementation-path>
```

Evidence retains original report/receipt paths and SHA256. Local files may not persist. Source/payload/schema hashes, official kernel content hashes and raw per-Point results remain in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/moe_dsa_kernel_kb_mapping.json; this archive is not durable CI evidence.

## Rollback

These are local preparation artifacts, not a claim of serving. Through normal review, remove only these new UUID implementations and this section (the standalone DSA page for DSA); keep old implementations and earlier case text. Selection then follows the base Points and baseline, with no unconditional named winner.

## Evidence

The separately timed incumbent UUIDs are `43ffd5e3e5b67298` at 32/65536, `ae32cc132679692c` at 128/262144, and `fbe6ddab5dac5958` at 256/524288. Exact identities and per-Point records are retained in the mapping and original report below.

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `4ec6dbc4f20a30a5` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_final_1108/results/g4/dsa_4ec6dbc4f20a30a5/validation/report.json sha256=d51b645c6750952942f17b2f57305b7873c3255b58533ae5d52ca715b0bf359d; receipt local:/tmp/inf3_validation_final_1108/results/g4/dsa_4ec6dbc4f20a30a5/receipt.json sha256=2874fbe58ef36cba4da2bed485008b51da25179bba51b4d256acc64e513041d3.
- S1 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/dsa8/tx/dsa8_kernel_final.py:16 final original source, sha256=8109c10b4a8916ca06e454ebc7c2637807fc6733da61ef857886a57ee0c62268; checked line ranges 16–45, 93–151, 492–588, 630–714, 735–765, 768–792; byte-identical to exported kernel.py.
- S2 aklr/plugins/sglang/layers/dsa_backend/contract.py:37 axes/attrs/tensors/layout/scalars/output contract; only the exact Points are claimed, not the full contract domain.
- S3 aklr/plugins/sglang/layers/dsa_backend/baseline.py:14 baseline calls sglang Triton split-K, distinct from the three AKLR Triton incumbents.
