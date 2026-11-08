---
title: "MXFP4 fused MoE with clamped SwiGLU and gate/up-interleaved weights"
kind: case
ops: [moe]
dtypes: [mxfp4, bf16]
regimes: ["small-token direct routes", "larger-token fused or grouped pipelines; exact sizes in sections"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: "measured N=1 (original CI); local section N=16 per comparison, not CI"
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "3.194–4.035; 2 of 2 Points ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256 file df5c661e4ce848dc, m 1, 2", run_ref: "5571f4589 (CI run 40613440602); file added by 39a51ac02 (#71)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "3.128–3.359; 2 of 2 Points ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256 file 5db89e7db4f729ff, m 1, 2", run_ref: "5571f4589 (CI run 40613440602); file added by 39a51ac02 (#71)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.917–3.015; 4 of 4 Points ≥ 1.03 (m 1, 2, 4, 8)", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256 file a5a343437ee191a7, m 1, 2, 4, 8", run_ref: "102877ca0 (CI run 40154665266); file added by 39a51ac02 (#71)", date: "2026-10-25"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.489–3.089; 4 of 4 Points ≥ 1.03 (m 1, 2, 4, 8)", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512 file 9dcb4d796b597d0b, m 1, 2, 4, 8", run_ref: "5571f4589 (CI run 40613440602); file added by 39a51ac02 (#71)", date: "2026-10-28"}
  - {"metric": "local whole-call baseline speedup, representative Point; not CI", "value": "5.103 at 1 token; see E100-E102 and mapping for complete coverage", "gpu": "MI355X", "workload": "pro_ep4", "run_ref": "local:/tmp/inf3_validation_final_1108/results/g4/pro_ef4b4c3360749dd8/validation/report.json", "date": "2026-11-08"}
sources:
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256/df5c661e4ce848dc.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256/5db89e7db4f729ff.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256/a5a343437ee191a7.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/9dcb4d796b597d0b.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384/ef4b4c3360749dd8.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384/98d3a5be0bf0cdd0.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768/6ea61fb782ff0801.json
---

# MXFP4 fused MoE with clamped SwiGLU and gate/up-interleaved weights

## Pro specialized implementations

The practical split is direct routes for very small token counts versus grouped or metadata-selected pipelines at larger counts. The strong one-token results narrow toward baseline as tokens grow; preserve per-regime decisions rather than promoting one path for all sizes. The masked configuration has 96 local experts out of 384 global experts.

This section covers the implementations and measured token counts below. Local results are not CI acceptance or deployed dispatch. The Historical implementations section describes the original UUIDs and CI snapshot only; its coverage, serving, activation and reduction statements do not apply to these implementations.

### Applies when

Clamped SwiGLU, top-6, MXFP4 gate/up-interleaved weights and 2-D E8M0 scales, swiglu_limit=10.0. Gate is capped at the limit and up at +/-limit. The masked variants use global ids mapped by expert_mask and return only the rank-local contribution, without a collective; unmasked variants use local ids. The reference retains bf16 activations, while some implementations use fp8 internally: SQNR acceptance is not numerical equivalence. Hidden states and output are bf16; routing weights enter stage 2 or the final sum. These use fused_moe@1.0 with bounded comparison and whole-output SQNR >= 13 dB. T denotes input tokens, not actual per-expert GEMM M; the latter was not recorded. Each token selects distinct expert ids; num_local_tokens is absent. Exact attrs, dtypes, layouts and Points remain in the exported schemas and mapping.

| Implementation | Configuration | Measured tokens |
|---|---|---|
| pro_ep4 | hidden 7168, experts 96, inter 3072; masked | 1, 2, 16, 32 |
| pro_tp384 | hidden 7168, experts 384, inter 384; no mask | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| pro_tp768 | hidden 7168, experts 384, inter 768; no mask | 1, 2, 4, 16, 32, 64 |

### Counter-signals

- Do not extrapolate to other tokens, slots, scalars, dtypes, layouts or routing/index distributions. The sampled inputs do not establish serving-traffic performance.
- Whole-contract correctness/timing does not prove cross-workgroup protocol safety, reentrancy/multistream safety, occupancy, individual-kernel performance or causal gains from a mechanism. Optional source branches are not independently validated by a whole-call pass.
- Payload accuracy/speedup fields were not populated; these archived local results change no dispatch. Actual AKLR_MIN_SPEEDUP was not archived: default qualification is reconstructed under fixed-base defaults, not proof of the runtime environment.
- pro_ep8 remains blocked: the exact final source is absent in the bounded local search. No older source substitutes for it.

### Mechanism

Source facts, not ablation-based gain attribution.

- **pro_ep4**: DIRECT per-route stage 1 runs at 1/2 tokens; the tile path runs at 16/32, with K parts selected by entry configuration. At 1/2 and limit=10, SCY uses fixed-scale fp8 intermediates and _stage2_tm; the other Points use _stage2_cr. expert_mask maps global to local ids; routing weights enter stage 2 only. SCY range reasoning is not an independently validated rule. See S100.
- **pro_tp384**: 1/2/4/8/16/32/64 tokens use custom projections with route-count-dependent row grouping and optional K partials, exchanging bf16 Y. At 128/256, _aiter_plan/_front_into uses a custom front to write sorted routes, fp8 A1, scales and out, followed by metadata-selected AITER projections with sorted stage-2 weights. Timing includes the helper steps, not just GEMM. See S101.
- **pro_tp768**: 1/2 tokens use split stage 1 and _stage2r; 4 uses per-route projections; 16 uses persistent grouped stage 1 directly followed by _stage2g; 32/64 use _groups before grouped projections. _G defines group counts, not measured per-expert M. _TOKENS also lists 3, which is not a measured Point here. See S102.

### Result

Local whole-call AKLR graph measurements (profile v18), with a 1 GiB cache flush before each call and 16 actual timing rounds per comparison. Representative values are the published per-Point speedup fields, not averages. Full timing and statistical details remain in the reports and mapping.

The baseline is aklr.ops.fused_moe.baseline.fused_moe_baseline. No measured Point in this section has a default-served incumbent at the fixed base. A baseline win therefore does not establish a win against another incumbent.

| Implementation | Representative speedup over baseline |
|---|---|
| pro_ep4 | 5.103 at 1 token; 1.129 at 32 *(field, E100)* |
| pro_tp384 | 3.797 at 1; 1.030 at 128; 1.041 at 256 *(field, E101)* |
| pro_tp768 | 3.154 at 1; 1.151 at 64 *(field, E102)* |

These are selected per-Point results, not group averages. All 19 Points pass accuracy; all qualify under fixed-base defaults. The complete per-Point scalars, accuracy, timings, incumbent identities and gates remain in the original reports and mapping archive, not repeated here. Local qualification does not establish CI or serving.
The gate lower bound may be capped at 2 for large Pro gains; the published speedup field uses uncapped statistics. Raw values and gates remain in the mapping JSON.

### Reproduce

Fixed base a12866b1e40e86ea6aff7ac24cbe5634132f6b2e; pinned AITER dbd8bf5bd624120197a7a26780a8c72201824f0f; reported amd-aiter 0.1.21.dev66+gdbd8bf5bd; gfx950, MI355X, FlyDSL 0.3.2. For interleave only, the MoE baseline sets AITER_BF16_FP8_MOE_BOUND=16: this is an activation-path row switch, not an SQNR dB bound. Actual sglang version, tuned configuration and full runtime environment were not archived.

Future revalidation must use each full exported schema recorded in the mapping, not a bare kernel, and check exact Points, scalars, attrs, dtypes and incumbent identity. It must not replace these archived results. From the repository root, use the published_path for the implementation in the mapping:

```bash
python -m aklr validate <implementation-path>
```

Evidence retains original report/receipt paths and SHA256. Local files may not persist. Source/payload/schema hashes, official kernel content hashes and raw per-Point results remain in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/moe_dsa_kernel_kb_mapping.json; this archive is not durable CI evidence.

### Rollback

These are local preparation artifacts, not a claim of serving. Through normal review, remove only these new UUID implementations and this section (the standalone DSA page for DSA); keep old implementations and earlier case text. Selection then follows the base Points and baseline, with no unconditional named winner.

## Historical implementations

The following original four schema files, one per schema name, embed the same `kernel.py` (identical `kernel` sections): DeepSeek-V4-Flash's
MoE under expert parallelism with 32 or 64 local experts of 256 and inter dim 2048, and with all 256 experts local at
inter dim 256 or 512. The function names below are in that `kernel.py`. "m" is the Point's `n_tokens`; a route is one
(token, expert) pair, 6m per call.

## Applies when

Historical scope: the four original hidden-4096 implementations and their recorded CI Points only.

- Contract `fused_moe@1.0`, layout variant `gate_up_interleaved`: `w1` (experts, 2 × inter, 4096) with gate and up rows
  interleaved in 16-row tiles and fp4 e2m1 in 16 × 64-byte blocks, `w2` (experts, 4096, inter), E8M0 scales per 1×32
  stored 2-D (`aklr/ops/fused_moe/contract.py`, the `gate_up_interleaved` variant). `gate_mode` interleave with
  `swiglu_limit` (gate clamped to at most the limit, up to ± the limit, then SiLU(gate)·up); activations stay bf16
  in this gate mode (the `act_quant` attr's description). bf16 `hidden_states`, fp32 `topk_weights` (asserted),
  int32 or int64 `topk_ids`, routing weight in stage 2, bf16 `out`.
- Expert parallel names: `expert_mask` over 256 global experts, 32 or 64 local; ids are global. The other two names
  have all 256 experts local and no mask.
- Every workload passes `swiglu_limit` 10.0, the start of the contract's interval.
- Compare: `determinism: bounded`, `max_violation_fraction` 1e-3, whole-output SQNR ≥ 13 dB (`SqnrBound`). Every
  reduction in this kernel at a served Point runs in a fixed order, with no atomics.
- Served Points: m 1, 2 (expert parallel) and m 1, 2, 4, 8 (all local): every Point of the four files. Each schema
  name also has a kernel-less file (`550200e0…`, `58ad3640…`, `af9db8f6…`, `ff293e2a…`) holding m 16–256, a
  definition that is not served (`aklr/ops/README.md` § Schema fields).
- The baseline is aiter's `fused_moe` with `AITER_BF16_FP8_MOE_BOUND=16`, sglang's DeepSeek-V4 setting
  (`aklr/ops/fused_moe/baseline.py`): below 16 rows aiter keeps bf16 activations. Each file is the only measured
  file of its name, so the yardstick is aiter.

## Counter-signals

Historical scope: the four original hidden-4096 implementations and their recorded CI Points only.

- **m 16–256 are not covered.** The commit that adds the kernel splits each name at those Points into a kernel-less
  definition because aiter's tuned fp8 kernels are faster from 16 tokens (commit 39a51ac02 message); no number is
  recorded for that. The kernel's path for more than 64 routes (`_prep`, `_route`, routes binned per local expert into
  16-row tiles, four launches) is reached by no Point: the largest route count is 48 (m 8, all local). CI measures
  only the per-route path described below.
- **15 of 16 MFMA rows are discarded.** Each route sits alone in its 16-row MFMA tile in both stages (every A row is
  the same token or route; only row 0 is stored). What that costs at these sizes is not measured.
- **Expert-parallel grids carry routes that are not local.** Stage 1 launches a workgroup row for every route; a
  route to an expert off this rank finds no local slot and its workgroups exit. Stage 2 takes a data-dependent branch:
  a token with exactly one local route spreads that route's K over all 6 waves, otherwise each local route's wave
  walks its own K. Which branch the bench's inputs exercise is not recorded.
- **The baseline does not clamp.** aiter's CK-Tile decode path does not apply `swiglu_limit` (`contract.py`, the
  comment at `swiglu_limit`); the contract's interval starts at 10 so that a limit binding on the bench's inputs
  does not leave the comparison without a baseline. Whether the clamp binds on the bench's draws at 10 is not
  established.
- **Stale docstring.** It describes routes binned per local expert into 16-row tiles as the main path and says rows at
  or past `num_local_tokens` contribute nothing; at every Point the per-route path runs, and `run` takes no
  `num_local_tokens` (it passes `HAS_NLT=False`), so those parts are reached by no Point.
- **Per-shape scratch.** `_build` (a `functools.cache` compile function) allocates `C`, `SLOT`, `LOC`, `ROUTES`,
  `MAP` and `Y` once per shape. On the per-route path each call writes `LOC` for every route and `Y` for every local
  route before stage 2 reads them; the other buffers serve only the unreached path. Two streams calling one Point at
  once would share them.

## Mechanism

Historical scope: the four original hidden-4096 implementations and their recorded CI Points only.

Two launches at every served Point, stage 1 then stage 2 (`_build`: `DIRECT = R <= 64`); no routing kernel, no
atomics, no workspace counters.

**Stage 1 (`_stage1`, per-route path).** Grid (inter / 16, routes): one workgroup per (route, 16 intermediate
columns), 4 waves; 96–1536 workgroups (inter 256: 96, 192, 384, 768 at m 1–8; inter 512: 192–1536; inter 2048 with 6
or 12 routes: 768, 1536).
- The expert: all local, the route's id; expert parallel, `_local_expert` builds the mask's four 64-expert words
  with wave `ballot`s and counts the set bits below the id with `ctpop`, which gives the local slot (or −1: the
  workgroup exits). The workgroup at column block 0 writes the slot to `LOC` for stage 2.
- K = 4096 is split over the 4 waves, 1024 each (8 chunks of 128). Each wave stages its own slice of the token's row
  in LDS with its own lanes, so no barrier; then issues every scale and weight load of its 8 chunks before the first
  MFMA, in the order row, scales, weights, pinned with `sched_barrier` so `vmcnt` retires them in that order.
- Weights: one 16-byte load per lane per chunk for the gate tile and one for the up tile of the interleaved pair; each
  dword of 8 fp4 values becomes 8 bf16 with the block scale applied by `cvt_scalef32_pk_bf16_fp4`; one scale dword
  holds the gate and up bytes of two chunks. Non-temporal loads under expert parallelism, cached otherwise (routes that
  share an expert then reuse it from L2, per the code's note) (`../moe/grouped_moe_inference.md` §4).
- `mfma_f32_16x16x32_bf16`, 4 per chunk for gate and 4 for up, into fp32 (`../gemm/mainloop.md` §5).
- `_reduce_waves`: the 4 waves' partials through LDS, summed in wave order (`../gemm/small_m.md` §4).
  Wave 0 clamps, applies SiLU with `exp2` and `rcp`, rounds to bf16 and stores the route's 16 values to `Y`.

**Stage 2 (`_stage2_tp`, `_stage2_ep`).** Grid (4096 / 16, m): one workgroup per (token, 16 output columns), 6 waves,
one per route; 256–2048 workgroups.
- Each wave walks its route's K = inter (2, 4 or 16 chunks of 128) with `mfma_f32_16x16x32_bf16` on `w2` converted the
  same way. The unrolled walk (`_w2_walk`) takes chunks in blocks of 4 whose `Y`, weight and scale loads all issue
  before the block's first MFMA. Expert parallel, it always runs; all local, it runs when m × inter / 128 ≤ 8, and a
  runtime loop of one chunk at a time runs above (inter 256 at m 8, inter 512 at m 4, 8).
- Row 0 of each wave's tile goes to LDS; 16 threads add the 6 routes × routing weight in route order, skipping routes
  that are not local, and store bf16 (`../gemm/correctness.md` §8).

How this relates to the decode MoE cards: no sort and one route per workgroup set, as the 1–2-token kernels of
[fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md); `../moe/decode.md` §1 on why the
launch count matters at these sizes.

## Result

Historical scope: the four original hidden-4096 implementations and their recorded CI Points only.

`speedup` as CI writes it into each schema's `workloads`: a one-sided lower bound (α = 0.05) of the aiter / kernel time
ratio over 32 rounds of CUDA-graph replays with L2 flushed before each call (`aklr/core/validation/stages/_measure.py`).
A Point is served from 1.03 (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment variables). Bold: served.

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | Served |
|---|---|---|---|---|---|
| 32 local of 256, inter 2048 `df5c661e…` *(field, E1)* | **3.194** | **4.035** | – | – | 2 of 2 |
| 64 local of 256, inter 2048 `5db89e7d…` *(field, E2)* | **3.359** | **3.128** | – | – | 2 of 2 |
| 256 local, inter 256 `a5a34343…` *(field, E3)* | **3.015** | **2.923** | **2.409** | **1.917** | 4 of 4 |
| 256 local, inter 512 `9dcb4d79…` *(field, E4)* | **3.089** | **2.590** | **1.990** | **1.489** | 4 of 4 |

"–": no workload in that file. 12 of the 12 kernel Points are served, all with `accuracy: passed`; the 20 Points of
the kernel-less files (m 16–256) run aiter. The three files measured twice moved by at most 0.13 between the two
runs *(field, E5)*.

## Reproduce

Historical scope: the four original hidden-4096 implementations and their recorded CI Points only.

On an MI355X, from the repository root, one schema file at a time:

```
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/9dcb4d796b597d0b.json
```

and the same for the other three paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands). Add `--base origin/main`
only on a branch that edits the file: with the base set, bench measures only schemas that are not at the base, so on
an unchanged checkout it skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The
local result does not count; CI's measurement is the one dispatch reads (`aklr/ops/README.md` § Workflow).

## Rollback

Historical scope: the four original hidden-4096 implementations and their recorded CI Points only.

Delete the schema file (in a PR): its Points go unserved, since the other file of each name has no kernel.
`AKLR_DISABLE=fused_moe@1.0` stops serving the whole contract, every other MoE kernel included. Roll back for a wrong
result CI does not catch, for example after a FlyDSL or LLVM upgrade.

## Evidence

`kernel.py:<line>` is a line of the `kernel.py` the four schema files embed (byte-identical `kernel` sections).

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 3.194 at m 1, 4.035 at m 2 (…_n_experts32_topk6_inter_dim2048_global_experts256, file df5c661e4ce848dc; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 3.359 at m 1, 3.128 at m 2 (…_n_experts64_topk6_inter_dim2048_global_experts256, file 5db89e7db4f729ff; 5571f4589, CI run 40613440602)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 3.015, 2.923, 2.409, 1.917 at m 1, 2, 4, 8 (…_n_experts256_topk6_inter_dim256, file a5a343437ee191a7; 102877ca0, CI run 40154665266)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 3.089, 2.590, 1.990, 1.489 at m 1, 2, 4, 8 (…_n_experts256_topk6_inter_dim512, file 9dcb4d796b597d0b; 5571f4589, CI run 40613440602)
- E5 [2026-10, gfx950, FlyDSL 0.3.4.1, N=2, AKLR CI] the two committed measurements, in commit order: df5c661e4ce848dc 3.324, 4.133 then 3.194, 4.035; 5db89e7db4f729ff 3.410, 3.044 then 3.359, 3.128; 9dcb4d796b597d0b 2.970, 2.611, 1.961, 1.481 then 3.089, 2.590, 1.990, 1.489 (102877ca0, CI run 40154665266; 5571f4589, CI run 40613440602)
- S1 kernel.py:532 `DIRECT = R <= 64`; launches at :560–574: `_prep` and `_route` only when not `DIRECT`
- S2 kernel.py:55 `_local_expert`: `ballot` at :70, `ctpop` at :71 and :74; `LOC` written by column block 0 at :255–256
- S3 kernel.py:251 `XL` (the token row staged in LDS by each wave's own lanes, :282–302); loads issued row, scales, weights between `sched_barrier`s at :287–297; `PF1` = all of a wave's chunks at :535
- S4 kernel.py:86 `_fp4x8` with `cvt_scalef32_pk_bf16_fp4` (:91); `_mfma` `mfma_f32_16x16x32_bf16` at :96–97; non-temporal under expert parallelism `NT1` at :540 (comment :538–539)
- S5 kernel.py:210 `_reduce_waves` sums in wave order (:222–223); clamp, SiLU and the row-0 store at :311–325
- S6 kernel.py:401 `_stage2_tp`, route sum in route order with the `LOC >= 0` select at :448–455; `_stage2_ep` at :462, one-local-route split at :488–497 (`_split_chunks`, :333); `UNROLL2` at :547
- S7 kernel.py:1 the docstring (binning into 16-row tiles, `num_local_tokens` at :11–12); `run` passes `HAS_NLT=False` at :596–597; scratch `ws` per shape at :548–552
- S8 aklr/ops/fused_moe/baseline.py:10 `AITER_BF16_FP8_MOE_BOUND=16` for gate_mode interleave; aklr/ops/fused_moe/contract.py:253 aiter's CK-Tile decode path does not clamp, interval from 10.0 at :255
- E100 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `ef4b4c3360749dd8` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_final_1108/results/g4/pro_ef4b4c3360749dd8/validation/report.json sha256=43145a211424bdd0e71e3ad4091897d26548c5759905c11ba49e55e1eedfee02; receipt local:/tmp/inf3_validation_final_1108/results/g4/pro_ef4b4c3360749dd8/receipt.json sha256=46c274c47f2c29d2c396095dc3c830cd82ba577a141746eddb57cb30003bac6a.
- S100 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/pro_ep4/tx/pro_ep4_kernel_final.py:726 final original source, sha256=77ab645d8eab103169a3817971f4c89737b9453dab90bc41edece8368f80d6b9; checked line ranges 726–823, 847–863; byte-identical to exported kernel.py.
- E101 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `98d3a5be0bf0cdd0` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_final_1108/results/g5/pro_98d3a5be0bf0cdd0/validation/report.json sha256=111e33bb864ed781176c5857f783f81d33ef73162512631328cecdf127e9bf7f; receipt local:/tmp/inf3_validation_final_1108/results/g5/pro_98d3a5be0bf0cdd0/receipt.json sha256=87b6daf5645fac5a975df7dcbbc756c6ce85002b2594eca8fb3c94a1bf4d9caf.
- S101 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/pro_tp384/tx/pro_tp384_kernel_final.py:19 final original source, sha256=e0a55f98514b083cb783491d39030dd519cb7bd50f2293db73aae75e00c58d04; checked line ranges 19–19, 449–537, 829–907, 931–980; byte-identical to exported kernel.py.
- E102 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `6ea61fb782ff0801` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_final_1108/results/g3/pro_6ea61fb782ff0801/validation/report.json sha256=d904abf2da53b53c9777862239f264b23fcaa37a7255d96f703a95248b92eae8; receipt local:/tmp/inf3_validation_final_1108/results/g3/pro_6ea61fb782ff0801/receipt.json sha256=63ae4bc7373ac612f23ac79d9437c0594abc8bdb56663d9e0b75e5ba027d9345.
- S102 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/pro_tp768/tx/pro_tp768_kernel_final.py:872 final original source, sha256=f6f3e7b81e21dfdfd9410d902fbcf9e421352782de6e948d92030263bddec6cf; checked line ranges 872–883, 887–990, 1014–1025; byte-identical to exported kernel.py.
- S103 aklr/ops/fused_moe/contract.py:75 axes/attrs/tensors/layout/scalars/output contract; only the exact Points are claimed, not the full contract domain.
- S104 aklr/ops/fused_moe/baseline.py:9 baseline callable; MoE interleave sets AITER_BF16_FP8_MOE_BOUND=16; DSA calls sglang Triton split-K.
- S105 aklr/ops/fused_moe/reference.py:77 global-to-local mask mapping; :88–99 activation quantization; :107–127 clamp, activation, stage-2 routing weight and topk sum.
