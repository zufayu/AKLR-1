---
title: "MXFP4 fused MoE, hidden 6144, top-9, 16×16-shuffled weights (257, 65 and 33 experts) — no sort and two launches at 1 token, the sort folded into stage 1 at 8–64, sorted row tiles with fixed-order route sums from 64 tokens"
kind: case
ops: [moe]
dtypes: [mxfp4]
regimes: ["tokens 1-2", "tokens 4-64", "tokens 128-4096", "GLM exact tokens 4-4096; see Point table"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: "measured N=1 (original CI); local section N=16 per comparison, not CI"
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "0.138–2.472; 2 of 13 Points ≥ 1.03 (m 1, 2), both served", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256 file 1fee59821837c26f, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.087–1.129; 4 of 4 Points ≥ 1.03 (m 8, 16, 32, 64), all served", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256 file 4fbd9d02b4c053c7, m 8, 16, 32, 64", run_ref: "102877ca0 (CI run 40154665266); file added by 0310eba03 (#96)", date: "2026-10-25"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.991–1.609; 11 of 13 Points ≥ 1.03 (all but m 8, 32)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048 file ef15e87894a8ae06, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file at this path since fdac7c716 (#52)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.801–1.365; 3 of 13 Points ≥ 1.03 (m 1, 512, 1024)", gpu: MI355X, workload: "fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048 file 1bf41ccba6750300, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file at this path since fdac7c716 (#52)", date: "2026-10-28"}
  - {"metric": "local whole-call baseline speedup, representative Point; not CI", "value": "1.361 at 2048 tokens; see E9-E13 and mapping for complete coverage", "gpu": "MI355X", "workload": "glm_ep4_l", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_40937d155ca6aa98/validation/report.json", "date": "2026-11-08"}
sources:
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/1fee59821837c26f.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048/ef15e87894a8ae06.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/1bf41ccba6750300.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/40937d155ca6aa98.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/2c7c5bacd06c8e29.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/aa1aaac0bdd0068f.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/f2e2b374be1c9f8b.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/1ddb007af161404d.json
---

# MXFP4 fused MoE with 16×16-shuffled weights, hidden 6144, top-9

Four FlyDSL kernel files for `fused_moe@1.0` under three schema names: 257 experts with inter dim 256 (two files,
`1fee5982…` and `4fbd9d02…`), 65 experts and 33 experts with inter dim 2048 (`1bf41ccb…`, `ef15e878…`). Each schema
embeds its kernel in `kernel.sources` (`4fbd9d02…` as eight files, the others as one `kernel.py`); the function
names below are in those sources. "m" is the Point's `n_tokens`; a route is one (token, expert) pair, 9m per call.

## Applies when

- Contract `fused_moe@1.0`, layout variant `shuffled_16x16`: `w1` (experts, 2 × inter, 6144) and `w2` (experts,
  6144, inter) as fp4 e2m1 in aiter's 16×16 preshuffle, one E8M0 scale per 1×32, shuffled and padded to (256, 8);
  bf16 `hidden_states`, fp32 `topk_weights`, int32 `topk_ids`, no expert mask; SiLU on separate gate and up rows,
  routing weight applied in stage 2, `act_quant` 4 (activations quantized to MXFP4); bf16 `out`
  (`aklr/ops/fused_moe/contract.py`, the `shuffled_16x16` variant).
- Compare: `determinism: bounded`, `max_violation_fraction` 1e-3, and one bound on the whole output: SQNR ≥ 13 dB
  against the reference (float32 in validation) rounded to bf16, with no per-element bound and no bias test (`SqnrBound`,
  `aklr/core/contract/semantics/_compare.py`). Atomic accumulation is allowed.
- The commit that adds the 257-expert `4fbd9d02…` names it GLM-5.2's MXFP4 MoE (256 routed experts plus the fused
  shared one, top-9) at decode sizes; the 65- and 33-expert names have no model named in their files.
- Served Points (Result): 257 experts m 1, 2 (`1fee5982…`) and m 8–64 (`4fbd9d02…`); 33 experts m 1, 2, 4, 16 and
  64–4096; 65 experts m 1, 512, 1024.
- The yardstick at every Point is aiter: where the two 257-expert files overlap (m 8–64), `1fee5982…` reads below 1.

## Counter-signals

- **257 experts at m 4 and m ≥ 128: not served.** `4fbd9d02…` has no Point there and `1fee5982…` reads 0.138–0.862
  *(field, E1)*. At m 4, 16, 128, 512 and 1024 `1fee5982…` repacks `w1`, `w2` and their scales into a K-major record
  layout on every call (`pack_weights`, two extra `pack_weight_scales` launches): about 0.64 GB read and as much
  written per call (estimated from the tensor sizes). Those five Points read 0.138–0.425, its other unserved Points
  0.743–0.903 *(field, E1)*. A kernel may not keep a repacked weight between calls (`../SKILL.md`, Stop table), and
  the `shuffled_16x16` layout is the one its stages read without repacking.
- **At m 8–64 the two 257-expert files compete:** `4fbd9d02…` serves at 1.087–1.129 *(field, E2)*; `1fee5982…` reads
  0.184–0.903 there *(field, E1)*, so `4fbd9d02…` is 1.25–5.9× faster (estimated from the two CI ratios).
- **`4fbd9d02…` adds each token's 9 weighted rows into `out` with bf16x2 atomics in arrival order**, so its output bits
  differ between runs: run-to-run SQNR 48.4–51.9 dB *(field, E8)*. Under a bitwise contract it would need a
  fixed-order stage 2 (`../moe/decode.md`, Counter-signals).
- **33 experts: m 8 (1.019) and m 32 (0.991) are not served, and m 64 is served at exactly 1.030** *(field, E3)*.
  Across the five committed measurements: m 8 0.994, 1.031, 1.009, 1.012, 1.019; m 32 0.974, 1.024, 1.003, 1.002,
  0.991; m 64 1.008, 1.033, 1.025, 1.022, 1.030; m 2 1.011, 1.039, 1.033, 1.046, 1.049; m 16 0.983, 1.045, 1.036,
  1.043, 1.039 *(field, E5)*. Which of these Points dispatch serves depends on the run that wrote the file. m 1 read
  0.921 in the first of the five and 1.258–1.279 in the other four.
- **65 experts: 10 of 13 Points are below 1.03** *(field, E4)*. m 2 1.024 and m 256 1.023 sit at the threshold
  (five runs: 0.794, 1.049, 0.989, 1.014, 1.024 and 1.034, 0.968, 0.917, 1.026, 1.023); m 512 read 1.012 in one of the
  five *(field, E5)*. m 4–128 (0.909–0.991) run four launches with 16-row tiles (32 at m 128); m 2048 and 4096 read
  0.873 and 0.801.
- **Precision is spent where the SQNR bound allows it.** `ef15e878…` stores its stage-2 rows as fp8 e4m3 with no
  scale at m 16, 32, 64 and 128–4096 (`cvt_pk_fp8_f32`) before the route sum; `1bf41ccb…` truncates, rather than
  rounds, its routing-weighted bf16 rows and its bf16 output at every m (`bits >> 16`); `1fee5982…` at m ≥ 512, where
  it is not served, stores int8 rows with one fp32 scale per 32 values. The contract notes that these 33- and
  65-expert kernels keep partials far off aiter's error (`contract.py`, the FIXME at `out`'s `error_bounds`). CI
  records `accuracy: passed`, not the SQNR; the error itself is not measured here. Under a tighter bound these
  Points need wider partials and rounding (code read).
- **Not the compile-function form in three files.** `1fee5982…`, `ef15e878…` and `1bf41ccb…` keep built launches in
  a module-level dict `_RUN_CACHE` filled from `run`, not in a `functools.cache` compile function, allocate their
  per-shape scratch on device `"cuda"` rather than an argument's device, and `1fee5982…` hashes its own source file
  (`Path(__file__).read_bytes()`) on a Point's first call: file I/O in launch code (`aklr/ops/README.md` § The kernel
  section and ABI). Their scratch is written by every call before it is read (`1fee5982…` zeroes its expert counters
  in `quant_input` at m ≥ 32; the others store their counts per call); two streams calling one Point at once would
  share it. `4fbd9d02…` uses `functools.cache` and allocates its workspace per call.
- **Code no Point reaches:** in `ef15e878…`, `_route` and `_route_block` (route ranks taken with atomics),
  `_route_fused`, `_quantize` and `_activate_split`; in `1bf41ccb…`, the torch sort fallback, `quant_kernel` and
  `_make_split_quant`; in `4fbd9d02…`, the split-K exchange, n-fusion, item-direct head and k-rotation of `g1_fold.py`
  and the non-atomic epilogues of `g2_port.py`. CI measures none of them.
- **Docstrings and names.** `ef15e878…`'s docstring promises "reference-compatible rounding" while its stage-2 rows
  are fp8 at most Points; `1bf41ccb…`'s names a fallback torch sort no Point runs. The builders are named for inter
  128 (`build_mxfp4_moe_e257_i128_module`) and 1024 (`…_e33_i1024_…`, `…_e65_i1024_…`) on shapes with 256 and 2048.

## Mechanism

What the four kernels share:

- **Scaled fp4 MFMA.** `mfma_scale_f32_16x16x128_f8f6f4` with fp4 on both operands and the E8M0 scale bytes as
  operands (`ef15e878…` also its 32×32×64 form in stage 2 at m 64, 256, 512), fp32 accumulation
  (`../gemm/mainloop.md` §5). The preshuffled weights go from global straight into MFMA operand
  registers with 16-byte loads (`../gemm/datapath.md` §10, §11).
- **Activations quantized twice on the GPU:** the input rows to MXFP4 (amax of each 32-value group across lanes, an
  E8M0 exponent from it, `cvt_scalef32_pk_fp4_*`), and the SiLU(gate)·up intermediate again before stage 2: in stage
  1's epilogue, or, in `1fee5982…` at m 1 and 2, at the start of stage 2.
- **At m = 1 the three files with that Point run two launches and no routing sort.** Each route gets its own
  workgroups, so the expert is read straight from `topk_ids`; the token row is quantized inside each stage-1
  workgroup; stage 1 splits K over waves or workgroups and adds the parts in a fixed order; stage 2 runs one wave per
  route and adds the 9 weighted rows of a token in LDS in route order. Every row of the 16-row MFMA tile is the same
  token and only row 0 is kept. This is the decode shape of `../moe/grouped_moe_inference.md` §3
  (the N blocks are the parallelism) taken to one route per workgroup.
- **Configuration per m from constants in the source**, compiled on a Point's first call.

Per file (functions are in the sources each schema embeds):

**`1fee5982…`, 257 experts, inter 256 — served at m 1, 2.** Two launches (flags at `kernel.py:655–707`):
- `stage1`: 8 column blocks of 64 × 9m routes × 4 K parts (288 and 576 workgroups of 4 waves). The workgroup
  quantizes its token's 1536-value K quarter into LDS once (`_quant8`), then each wave runs 12 scaled MFMAs on 16
  gate or up columns and stores row 0 as an fp32 partial per K part.
- `stage2_reduce`: 192 blocks of 32 output columns × m tokens, 9 waves, one per route. Each wave adds its route's 4
  partials in part order, applies SiLU·up, rounds to bf16, quantizes 4 values per lane (`_quant4`) and moves the packed
  words into MFMA operand order with `ds_bpermute`, so the intermediate never goes to memory; 32 threads then add
  the 9 routes × routing weight in route order and store bf16.
- At m ≥ 32 the file switches to sorted 16- to 96-row tiles (`group_routes`, `grouped_gemm`) and a route reduce, 5–8
  launches; none of those Points is served (Counter-signals).

**`4fbd9d02…`, 257 experts, inter 256 — served at m 8, 16, 32, 64.** Two launches with the routing sort folded into
stage 1: `../moe/decode.md` describes the fold.
- Stage 1 (`g1_fold.py`): 16-row expert blocks × 2 blocks of 128 gate + 128 up columns, 4 waves of 4 16-column
  tiles; K = 6144 in 24 tiles of 256. Weights non-temporal into a 2-stage register ring; the 16 hidden rows gathered
  by token id two loop bodies ahead, quantized inline into a 3-stage XOR-swizzled LDS ring; 8 scaled MFMAs per tile
  and wave at raised priority (`s_setprio`); accumulators through LDS, SiLU·up, MXFP4 with shuffled E8M0 scales in
  sorted-row order. At m = 32 the tile order is grouped per XCD after the bound test
  (`../gemm/schedule_and_locality.md` §2). 144–554 workgroups.
- Stage 2 (`g2_port.py`, aiter's `mxfp4_gemm2.py` with a CK-style tile remap `_g2_spart_tile`, the expert id read
  before the A loads, and `amdgpu-kernarg-preload-count` 16, `../gemm/small_m.md` §6): per 16-row block,
  24 workgroups of 256 output columns, K = 256 in one tile, the MXFP4 intermediate by buffer→LDS DMA, weights
  non-temporal into registers, × routing weight, bf16x2 `atomicrmw fadd` into `out`. Upper-bound grids of
  1728–6648 workgroups; those past the device bound exit (`../moe/grouped_moe_inference.md` §2, §6).

**`ef15e878…`, 33 experts, inter 2048 — served at m 1, 2, 4, 16, 64–4096.** Every GEMM maps its workgroup id with
`_chiplet_order`, a contiguous slice of the tile order per XCD (`../gemm/schedule_and_locality.md` §2);
one aligned scale dword serves two 16-column halves and two K slices (`PACKED_SCALES1/2`); the intermediate is
rounded to bf16 before its MXFP4 quantization. Stage-1 weights load non-temporal at m 2–256, cached at m 1 and
m ≥ 512 (`CACHE1`). Per m (`build_mxfp4_moe_e33_i1024_module`):

| m | Launches | Routing | Stage 1 | Stage 2 rows and route sum |
|---|---|---|---|---|
| 1 | 2 | none (`DIRECT`) | `_gemm1_split_fused`: 144 workgroups of 16 waves, 8 per K half; each quantizes the token row into LDS; halves added in order, SiLU, MXFP4 | `_gemm2_route_reduce`: a wave per route, the weighted rows added in LDS in route order |
| 2 | 4 | `_quantize_route_fused`: the grid quantizes the input; workgroup 0 ranks each expert's routes with ballots in input order and writes the 16-row tile table | `_gemm1_grouped_split_fused`: 16-row tiles, 2 K halves × 4 waves | `_gemm2`, fp32 rows; `_reduce` |
| 4, 8, 16, 32 | 4 | the same | `_gemm1`, 16-row tiles gathered from the quantized input by route; 64 + 64 columns and 4 waves (m 4), 128 + 128 and 8 waves (8, 16), 4 waves with K steps in pairs (32) | fp32 rows and `_reduce` at m 4, 8; fp8 rows through LDS and `_reduce_fp8x4` at 16, 32 |
| 64–1024 | 5 | `_route_expert`: one workgroup per expert compacts its routes in input order while the rest of the grid quantizes the input | `_gather_quant` (a coalesced XOR-swizzled LDS transpose from m 256) copies each tile's rows in tile-major order; `_gemm1` with 32-, 48- or 96-row tiles (m 64, 128, 256–1024) staged through LDS two K steps at a time, the next weights prefetched at m 64, 128, the quantized output staged in LDS for 16-byte stores except at m 128; groups of 2–3 row tiles of one expert per column sweep at m 512, 1024 (`GROUP_M`) | `_gemm2`, fp8 rows, `_reduce_fp8x4`; 32×32×64 MFMA at m 64, 256, 512 |
| 2048, 4096 | 6 | the same plus `_route_tiles` | 96- and 128-row tiles, `GROUP_M` 3 and 9 | the same, 16×16 MFMA |

**`1bf41ccb…`, 65 experts, inter 2048 — served at m 1, 512, 1024.**
- m 1, two launches: stage 1 with 9 routes × 64 tiles of 32 gate + 32 up columns (576 workgroups), each workgroup
  quantizing the token row into LDS and its 4 waves taking a K quarter each; the quarters are added in LDS in quarter
  order before SiLU·up and MXFP4 (`FUSE_SPLIT_QUANT`). Stage 2 `direct_reduce`: 192 workgroups of 32 columns, 9 waves,
  one per route, K = 2048, the weighted rows added in LDS in route order.
- m 512, 1024, five launches: `collect` (per expert, ballot ranks in input order; extra workgroups quantize the input)
  and `compact` (expert-major 64-row blocks padded with a sentinel row); stage 1 on 64-row tiles, 4 waves of 32 + 32
  columns sharing the tile's activation rows through two alternating LDS pages, the next K chunk loaded through a
  loop-carried `range(..., init=...)`, SiLU·up through LDS and one lane per row quantizing its 32 values; stage 2 on
  64-row tiles, 4 waves × 128 columns, with the tile's whole K = 2048 activation panel loaded into LDS once so the MFMA
  loop has no cross-wave barrier (`FULL_STAGE_A`), MFMA operands swapped so a lane holds 4 adjacent columns, 8-byte
  stores of the weighted rows; `reduce` adds each token's 9 rows in route order.
- No atomics on data at any served Point (the m 4–128 routing takes route ranks with LDS atomics; results stay
  indexed by route).

## Result

`speedup` as CI writes it into each schema's `workloads`: a one-sided lower bound (α = 0.05) of the aiter / kernel time
ratio over 32 rounds of CUDA-graph replays with L2 flushed before each call (`aklr/core/validation/stages/_measure.py`).
Dispatch serves each Point with the file of the highest `speedup` among the files of its schema name, from 1.03
(`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment variables). Bold: the file served there. Per Point:

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | Served |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 257 experts `1fee5982…` *(field, E1)* | **2.472** | **1.990** | 0.138 | 0.903 | 0.184 | 0.815 | 0.838 | 0.317 | 0.862 | 0.369 | 0.425 | 0.787 | 0.743 | 2 of 13 |
| 257 experts `4fbd9d02…` *(field, E2)* | – | – | – | **1.129** | **1.087** | **1.117** | **1.090** | – | – | – | – | – | – | 4 of 4 |
| 33 experts `ef15e878…` *(field, E3)* | **1.258** | **1.049** | **1.053** | 1.019 | **1.039** | 0.991 | **1.030** | **1.147** | **1.109** | **1.257** | **1.609** | **1.081** | **1.070** | 11 of 13 |
| 65 experts `1bf41ccb…` *(field, E4)* | **1.353** | 1.024 | 0.980 | 0.991 | 0.970 | 0.952 | 0.956 | 0.909 | 1.023 | **1.113** | **1.365** | 0.873 | 0.801 | 3 of 13 |

"–": no workload in that file. By schema name: 257 experts 6 of 13 Points served (2 by `1fee5982…`, 4 by
`4fbd9d02…`), 33 experts 11 of 13, 65 experts 3 of 13: 20 of 39. Every workload reads `accuracy: passed`; no kernel
serves the other Points.

## Reproduce

On an MI355X, from the repository root, one schema file at a time:

```
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7.json
```

and the same for the other three paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands). Add `--base origin/main`
only on a branch that edits the file: with the base set, bench measures only schemas that are not at the base, so on
an unchanged checkout it skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The
local result does not count; CI's measurement is the one dispatch reads (`aklr/ops/README.md` § Workflow).

## Rollback

Delete a schema file (in a PR): its Points fall back to the other file of the same name where it qualifies, else go
unserved. Deleting `4fbd9d02…` leaves m 8–64 of the 257-expert name unserved, since `1fee5982…` is below 1.03
there. `AKLR_DISABLE=fused_moe@1.0` stops serving the whole contract, every other MoE kernel included. Points CI
measures below 1.03 are not served without any rollback. Remove `4fbd9d02…` if the contract becomes bitwise, and
`ef15e878…` and `1bf41ccb…` at their fp8- or truncation-affected Points if the error bound tightens.

## Evidence

`<file>#kernel.py:<line>` is a line of the `kernel.py` source the schema file embeds; `g1_fold.py` and `g2_port.py`
are sources of `4fbd9d02b4c053c7.json`.

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point m 1–4096: 2.472, 1.990, 0.138, 0.903, 0.184, 0.815, 0.838, 0.317, 0.862, 0.369, 0.425, 0.787, 0.743; 2 of 13 ≥ 1.03 (fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256, file 1fee59821837c26f; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter at m 8, 16, 32, 64: 1.129, 1.087, 1.117, 1.090 (the same schema name, file 4fbd9d02b4c053c7; 102877ca0, CI run 40154665266)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point m 1–4096: 1.258, 1.049, 1.053, 1.019, 1.039, 0.991, 1.030, 1.147, 1.109, 1.257, 1.609, 1.081, 1.070; 11 of 13 ≥ 1.03 (fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048, file ef15e87894a8ae06; 5571f4589, CI run 40613440602)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point m 1–4096: 1.353, 1.024, 0.980, 0.991, 0.970, 0.952, 0.956, 0.909, 1.023, 1.113, 1.365, 0.873, 0.801; 3 of 13 ≥ 1.03 (fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048, file 1bf41ccba6750300; 5571f4589, CI run 40613440602)
- E5 [2026-10, gfx950, FlyDSL ?, N=5, AKLR CI; FlyDSL of the runs: not recorded, 0.3.1, then 0.3.4.1 three times] the five committed measurements of an unchanged kernel, in commit order: 33 experts m 1 0.921, 1.279, 1.261, 1.278, 1.258; m 2 1.011, 1.039, 1.033, 1.046, 1.049; m 8 0.994, 1.031, 1.009, 1.012, 1.019; m 16 0.983, 1.045, 1.036, 1.043, 1.039; m 32 0.974, 1.024, 1.003, 1.002, 0.991; m 64 1.008, 1.033, 1.025, 1.022, 1.030; 65 experts m 2 0.794, 1.049, 0.989, 1.014, 1.024; m 256 1.034, 0.968, 0.917, 1.026, 1.023; m 512 1.147, 1.092, 1.012, 1.107, 1.113 (files ef15e87894a8ae06 and 1bf41ccba6750300; baa858028, cc6e3b96c, cca5b12df, 102877ca0, 5571f4589)
- E6 [2026-10, gfx950, FlyDSL 0.3.4.1, N=2, AKLR CI] 4fbd9d02b4c053c7 at m 8, 16, 32, 64: 1.114, 1.056, 1.117, 1.089, then 1.129, 1.087, 1.117, 1.090 (cca5b12df, CI run 40058570127; 102877ca0, CI run 40154665266)
- E7 [2026-10, gfx950, FlyDSL 0.3.4.1, N=3, AKLR CI] 1fee59821837c26f m 1 2.514, 2.495, 2.472; m 2 1.964, 1.997, 1.990; m 4 0.135, 0.137, 0.138; every repacking Point (m 4, 16, 128, 512, 1024) ≤ 0.426 in all three (cca5b12df, 102877ca0, 5571f4589)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=3, three of the bench's input sets per token count, SQNR of the bf16 output between two runs] 48.35–51.87 dB at m 8, 16, 32, 64 (the 4fbd9d02b4c053c7 sources, byte-identical; local:/workspace/code/gpt_oss_docker/scratch/glm52_sikl_port/evidence/moe_check2_result.json)
- S1 aklr/ops/fused_moe/contract.py:269 `SqnrBound(min_sqnr_db=13.0)` under the FIXME at :267–268 (e33/e65 kernels keep fp8/MXFP4 partials); compare at :298–300; the `shuffled_16x16` variant at :277–285
- S2 fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/1fee59821837c26f.json#kernel.py:655 `pack_weights` at m 4, 16, 128, 512, 1024, launched at :763–785 (`pack_weight_scales`, :138); stage 1 at :245, stage-2 gather by `ds_bpermute` at :374–381, route sum in route order at :394–400; `_RUN_CACHE` at :801, `read_bytes()` at :653, device `"cuda"` at :708; counters zeroed per call at :234–236
- S3 fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7.json#kernel.py:58 `_plan` (functools.cache) builds the fold for m in {8, 16, 32, 64}, XCD grouping at m 32 (:37); workspace per call at :144; g1_fold.py:518 non-temporal weights, :1143 `s_setprio(1)`; g2_port.py:328 kernarg preload 16, :899–915 bf16x2 `atomicrmw fadd`
- S4 fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048/ef15e87894a8ae06.json#kernel.py:1340 per-m flags (to :1493) and launch chain at :1527–1574; fp8 rows at :1231 and :1238, `_reduce_fp8x4` at :1320; `_chiplet_order` at :156; docstring at :1; `_RUN_CACHE` at :1593, device `'cuda'` at :1494
- S5 fused_moe_shuffled_16x16_model_dim6144_n_experts65_topk9_inter_dim2048/1bf41ccba6750300.json#kernel.py:1380 per-m flags in `_Runner` (to :1472); bf16 truncation of the stage-2 rows at :777 (m ≥ 2) and :974 (m 1), of the output at :982 (m 1) and :1080 (m ≥ 2); the other stage-2 store paths (:782–803, :843–867) are reached by no Point; `FULL_STAGE_A` at :293 and :455–467; torch sort fallback at :1493–1509 behind `fused_sort` (:1435); `_RUN_CACHE` at :1550
- S6 aklr/core/runtime/dispatch/_dispatcher.py:111 a workload qualifies at `speedup >= AKLR_MIN_SPEEDUP` (`aklr/core/contract/spec/_workloads.py:38`); the highest-ranked file serves a Point (:113–116)
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `40937d155ca6aa98` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_40937d155ca6aa98/validation/report.json sha256=bddb9f8e50ce814b42fe7644b168dd8e07cb249d76fb0dfe035980332b6100a2; receipt local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_40937d155ca6aa98/receipt.json sha256=84187951363ef041aa4a2e9e4a51745411df69b58f96741aca4af91ccff38510.
- S7 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_l/tx/glm_ep4_l_kernel_after_refinalize.py:2532 final original source, sha256=d1fa451c8a92024960193cfe5453c4ef428ea1724ef8170bbe90cd517d29f2cf; checked line ranges 2532–2605, 3065–3183, 3270–3308; byte-identical to exported kernel.py.
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `2c7c5bacd06c8e29` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_2c7c5bacd06c8e29/validation/report.json sha256=2198069dd4f4f8093d31b643d167003d10ea25258d78db864fa470538162c34f; receipt local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_2c7c5bacd06c8e29/receipt.json sha256=a2e1760098febb807d9fceb3aeef9a6e7ea0abe698a7e9dd2fa4111945f56124.
- S8 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_ep4_s/tx/glm_ep4_s_kernel_final.py:57 final original source, sha256=0fdbd1d9d931eba74691858c75fc70ba86bf31024eea7d581da3ca873b659de9; checked line ranges 57–66, 326–389, 423–502; byte-identical to exported kernel.py.
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `aa1aaac0bdd0068f` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_aa1aaac0bdd0068f/validation/report.json sha256=968368f53cff8c5b1b4eba61528513b95cd5582d18be8421fc79658b6ef850a0; receipt local:/tmp/inf3_validation_checkpoint_0147/results/g5/glm_aa1aaac0bdd0068f/receipt.json sha256=e55617fe94659e69ba0a75ab9cdf69b8dcf9f431dedc591485288dd0a7081bc9.
- S9 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/ma8/glm_tp8_l/tx/glm_tp8_l_kernel_final.py:2363 final original source, sha256=a8f4c3f4c72a0956e637dd63bcedc819c5f1633c7de0b2d6142c6d880207f9ae; checked line ranges 2363–2372, 2376–2489, 2492–2679; byte-identical to exported kernel.py.
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `f2e2b374be1c9f8b` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_checkpoint_0147/results/g4/glm_f2e2b374be1c9f8b/validation/report.json sha256=16976d1fb17a8accdbdde1ad5a0e7585dc91bbee2b94973cdceee5e6bab620e1; receipt local:/tmp/inf3_validation_checkpoint_0147/results/g4/glm_f2e2b374be1c9f8b/receipt.json sha256=83fcb0c0cbce307a0f7a39c990e829207bd3967aed08f9e66d8aa53decaffc74.
- S10 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_l/tx/glm_l_kernel_final.py:362 final original source, sha256=462a01e0285a3a07e2fe2ad41b02540328a05cd0cbd90d4b1d7a0df46d18d181; checked line ranges 362–397, 2026–2241, 2264–2327; byte-identical to exported kernel.py.
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR local graph profile v18] `1ddb007af161404d` whole-call accuracy/speedup/qualification for the exact Points above; not CI. Original report local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_1ddb007af161404d/validation/report.json sha256=46247d656a8e6f22b694af3c8fa20fa483be6faeee33a1fac7a6a079ca280bd4; receipt local:/tmp/inf3_validation_checkpoint_0147/results/g3/glm_1ddb007af161404d/receipt.json sha256=0fb0405af81cf84bd8caf8e3ccca89c14be8519a6e6b0cecf5d8584819420e5a.
- S11 /workspace/code/gpt_oss_docker/work/sikl/kb_distill/moe6/glm_s/tx/glm_s_kernel_final.py:396 final original source, sha256=3943d69f65c792048624979d5a6616a1a67e3a357d4be07c3e773fd075e50b53; checked line ranges 396–431, 970–970, 1920–1989, 2034–2058; byte-identical to exported kernel.py.
- S12 aklr/ops/fused_moe/contract.py:75 axes/attrs/tensors/layout/scalars/output contract; only the exact Points are claimed, not the full contract domain.
- S13 aklr/ops/fused_moe/baseline.py:9 baseline callable; MoE interleave sets AITER_BF16_FP8_MOE_BOUND=16; DSA calls sglang Triton split-K.
- S14 aklr/ops/fused_moe/reference.py:77 global-to-local mask mapping; :88–99 activation quantization; :107–127 clamp, activation, stage-2 routing weight and topk sum.

## GLM specialized implementations

Keep small and large token regimes separate: the small kernels fold routing into stage 1, while the large kernels use owned or metadata-selected projection pipelines and explicit reduction choices. The weak 128-token large case is a concrete boundary, not evidence that the large path wins everywhere.

This section covers the implementations and measured token counts below. Local results are not CI acceptance or deployed dispatch. The preceding body describes its original UUIDs and CI snapshot only; its coverage, serving, activation and reduction statements do not apply to these implementations.

### Applies when

Hidden 6144, top-9, separated SiLU gate/up and 16x16-shuffled MXFP4 weights. E8M0 scales are shuffled and padded to (256,8); activations are quantized to MXFP4. No expert mask or SwiGLU limit is used. Hidden states and output are bf16; routing weights enter stage 2 or the final sum. These use fused_moe@1.0 with bounded comparison and whole-output SQNR >= 13 dB. T denotes input tokens, not actual per-expert GEMM M; the latter was not recorded. Each token selects distinct expert ids; num_local_tokens is absent. Exact attrs, dtypes, layouts and Points remain in the exported schemas and mapping.

| Implementation | Configuration | Measured tokens |
|---|---|---|
| glm_ep4_l | hidden 6144, experts 65, inter 2048; no mask | 256, 2048, 4096 |
| glm_ep4_s | hidden 6144, experts 65, inter 2048; no mask | 4, 8, 16, 32, 64, 128 |
| glm_tp8_l | hidden 6144, experts 257, inter 256; no mask | 512, 1024, 2048, 4096 |
| glm_l | hidden 6144, experts 257, inter 512; no mask | 128, 512, 1024, 2048, 4096 |
| glm_s | hidden 6144, experts 257, inter 512; no mask | 32, 64 |

### Counter-signals

- Do not extrapolate to other tokens, slots, scalars, dtypes, layouts or routing/index distributions. The sampled inputs do not establish serving-traffic performance.
- Whole-contract correctness/timing does not prove cross-workgroup protocol safety, reentrancy/multistream safety, occupancy, individual-kernel performance or causal gains from a mechanism. Optional source branches are not independently validated by a whole-call pass.
- Payload accuracy/speedup fields were not populated; these archived local results change no dispatch. Actual AKLR_MIN_SPEEDUP was not archived: default qualification is reconstructed under fixed-base defaults, not proof of the runtime environment.
- f2e2b374be1c9f8b at 128 tokens passes correctness but fails both entry and default performance qualification.

### Mechanism

Source facts, not ablation-based gain attribution.

- **glm_ep4_l**: All listed token counts are in _OWNED and call build_moe_module(..., cfg_m=2048), not the later AITER two_stages fallback. The owned module performs route/gather, two projections, SiLU times up, quantization and route reduction; all seven tensor addresses enter the module and it writes out directly. See S7.
- **glm_ep4_s**: _FOLD covers 4/8/16 tokens; 32/64/128 use a prelude and stage-1 code with six K partials and arrival-counter merging. Stage 1 emits packed fp4 and scales; _flydsl_moe_stage2_impl uses fp4/fp4, bf16 out, atomics and sorted routing weights. See S8.
- **glm_tp8_l**: _plan starts from AITER metadata and replaces stage 1, sort and stage 2. All four token counts are in _TUNED. Source includes fp4 route-result packing/reduction, fused sort/activation quantization and optional W1 prefetch. Atomic and route-reduce branches take sorted weights and topk_weights respectively; metadata-dependent branch selection is not established for every Point. See S9.
- **glm_l**: None of these Points is in _DECODE (1/2/4 tokens). Matching dtype/layout conditions select _aiter_plan, then _aiter_into_out combines custom route/quant or AITER sort, two projections and optional custom fp4 reduction into caller-owned out. Optional code is not proof of per-Point execution. At 128 tokens correctness passes but the performance gates fail. See S10.
- **glm_s**: 32/64 tokens are outside _DECODE but inside _FOLD. Matching shuffled dtype/attrs select _fold_stage1 and _fold_stage2: routing, packed fp4 and scales, then weighted atomic bf16 output. Grid flags are (True,True) at 32 and (False,False) at 64, not evidence of absolute XCD numbers. See S11.

### Result

Local whole-call AKLR graph measurements (profile v18), with a 1 GiB cache flush before each call and 16 actual timing rounds per comparison. Representative values are the published per-Point speedup fields, not averages. Full timing and statistical details remain in the reports and mapping.

The baseline is aklr.ops.fused_moe.baseline.fused_moe_baseline. No measured Point in this section has a default-served incumbent at the fixed base. A baseline win therefore does not establish a win against another incumbent.

| Implementation | Representative speedup over baseline |
|---|---|
| glm_ep4_l | 1.172 at 256 tokens; 1.361 at 2048; 1.258 at 4096 *(field, E9)* |
| glm_ep4_s | 1.185 at 4; 1.159 at 128 *(field, E10)* |
| glm_tp8_l | 1.266 at 512; 1.843 at 4096 *(field, E11)* |
| glm_l | 1.008 at 128 (performance gates fail); 1.674 at 4096 *(field, E12)* |
| glm_s | 1.041 at 32; 1.032 at 64 *(field, E13)* |

These are selected per-Point results, not group averages. All 20 Points pass accuracy; GLM large at 128 tokens fails performance qualification. The complete per-Point scalars, accuracy, timings, incumbent identities and gates remain in the original reports and mapping archive, not repeated here. Local qualification does not establish CI or serving.

### Reproduce

Fixed base a12866b1e40e86ea6aff7ac24cbe5634132f6b2e; pinned AITER dbd8bf5bd624120197a7a26780a8c72201824f0f; reported amd-aiter 0.1.21.dev66+gdbd8bf5bd; gfx950, MI355X, FlyDSL 0.3.2. For interleave only, the MoE baseline sets AITER_BF16_FP8_MOE_BOUND=16: this is an activation-path row switch, not an SQNR dB bound. Actual sglang version, tuned configuration and full runtime environment were not archived.

Future revalidation must use each full exported schema recorded in the mapping, not a bare kernel, and check exact Points, scalars, attrs, dtypes and incumbent identity. It must not replace these archived results. From the repository root, use the published_path for the implementation in the mapping:

```bash
python -m aklr validate <implementation-path>
```

Evidence retains original report/receipt paths and SHA256. Local files may not persist. Source/payload/schema hashes, official kernel content hashes and raw per-Point results remain in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/moe_dsa_kernel_kb_mapping.json; this archive is not durable CI evidence.

### Rollback

These are local preparation artifacts, not a claim of serving. Through normal review, remove only these new UUID implementations and this section (the standalone DSA page for DSA); keep old implementations and earlier case text. Selection then follows the base Points and baseline, with no unconditional named winner.
