---
title: "bf16 GEMM, narrow N (32–256) × K 6144 — whole dot products per wave or workgroup at M ≤ 8, split-K with a fixed-order fold at M ≥ 2048; additional exact large-M cases: N 128 × K 6144; N 256 × K 6144"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 2-8", "M 16-1024", "M 2048-4096", "N=128 K=6144 M={256,512,1024,2048}", "N=256 K=6144 M={256,512,1024}"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "0.565–1.195; 2 of 13 Points ≥ 1.03 (m 2, 4)", gpu: MI355X, workload: "gemm_n128_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.490–1.201; 5 of 13 Points ≥ 1.03 (m 2, 4, 8, 2048, 4096)", gpu: MI355X, workload: "gemm_n256_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.694–1.841; 3 of 13 Points ≥ 1.03 (m 4, 8, 2048)", gpu: MI355X, workload: "gemm_n32_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "4/4 Points qualify under default rule; representative ratios in large-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n128_k6144, UUID 71234a12ada11ac5, M 256, 512, 1024, 2048", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_71234a12ada11ac5/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "3/3 Points qualify under default rule; representative ratios in large-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n256_k6144, UUID 3da7088c64723d7c, M 256, 512, 1024", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_3da7088c64723d7c/validation/report.json", "date": "2026-11-08"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0bac580d3d76a3df.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/0d33213684622a71.json
  - aklr/ops/gemm/impls/flydsl/gemm_n32_k6144/87f4f44793d84725.json
  - aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/71234a12ada11ac5.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/3da7088c64723d7c.json
---

# bf16 GEMM at narrow N, K 6144: one dot product per wave at small M — historical collection and additional exact large-M cases

The original collection below describes the historical kernels and Points. The additional exact large-M shapes are scoped separately in the large-M implementation evidence section; they do not expand the historical measurements or imply Kimi-K3 model provenance.


Three FlyDSL kernels for the `gemm@1.0` contract on GLM-5.2 shapes (PR #43), one schema file per N. Each schema
embeds its kernel as `kernel.sources[0].content` (`kernel.py`); the function names below are in that file. "m" is
the Point's M.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0;
  `GEMM` in `aklr/ops/gemm/contract.py`). Every reduction these kernels run at a served Point adds in a fixed order.
- K = 6144 with N = 32, 128 or 256: a weight of 384 KiB, 1.5 MiB or 3 MiB, a narrow-N row bound by launch,
  reduction and memory latency rather than by reading `b` (`../gemm/small_m.md` §2).
- Served Points: n128 m 2, 4; n256 m 2, 4, 8, 2048, 4096; n32 m 4, 8, 2048. Each file holds 13 Points,
  m = 1, 2, 4, …, 4096.
- Each schema name has only this file, so at every Point the yardstick is aiter.

## Counter-signals

- **The middle of the M range is served in none of the three files:** n128 at m ≥ 8 0.565–0.977 *(field, E1)*,
  n256 at m 16–1024 0.490–0.970 *(field, E2)*, n32 at m 16–1024 0.694–1.012 *(field, E3)*. These Points run split-K
  MFMA paths, and all but n256 at m = 1024 first pack `b` into MFMA fragment order on every call (n128
  `_make_pack_b` through `prepare_b`; n256 `_run_pack_wave_b` on the path for m 16–512; n32
  `_compile_weight_pack` for m ≥ 16), a second launch over the whole weight. The one served Point that packs is
  n32 at m = 2048, at 1.062 *(field, E3)*.
- **m = 1 is not served in any file** (0.993–1.002 *(field, E1, E2, E3)*), nor n32 at m = 2: its 4×4×4 MFMA path
  with 16 K partitions across 128 workgroups reads 0.997 *(field, E3)*.
- **A reduction in arrival order at n128, m = 16.** 24 K partitions add their fp32 MFMA tiles into an fp64
  workspace with `BufferAtomicAdd(Float64)` (`_epilogue_atomic_sum`); the last producer converts to bf16. Adding
  fp32 partials in fp64 is exact only while all partials and running sums fit one 53-bit window, which
  cancellation or a wide magnitude spread breaks (estimated), so the output can depend on arrival order under a
  bitwise contract (`../gemm/correctness.md` §8). CI's check passed at that Point, which is not
  served (0.631 *(field, E1)*). Do not copy it.
- **Counters that live across calls.** n32's split paths (m = 2 and m ≥ 16, the served m = 2048 included) zero
  their arrival counters once per shape and rely on `atomicrmw uinc_wrap` returning each to 0 at its last
  arrival; n128 at m ≥ 512 advances a counter by one generation per call and never resets it. `SKILL.md`'s Stop
  table asks for scratch rewritten by every call before it is read.
- **Not the compile-function form.** `run` keeps built launches in a module-level dict (`_RUN_CACHE`), not in a
  `functools.cache` function, so compilation and scratch allocation run inside `run` on a Point's first call
  (`aklr/ops/README.md` § The kernel section and ABI).
- **Stale docstrings.** n128's says the packed weight is refreshed for a new or modified weight, n32's that `b` is
  packed on first use and repacked when replaced or mutated; both kernels pack on every call.

## Mechanism

**m ≤ 8 (n128 m 2, 4; n256 m 2, 4, 8; n32 m 4, 8): the whole K of an output element stays in one workgroup.** No
MFMA, no split across workgroups, `b` read in its row-major layout with 16 B buffer loads straight into
registers, fp32 accumulation, one launch:

| | n128 `_make_tiny_gemm` | n256 `_gemv` | n32 `_compile_dot` |
|---|---|---|---|
| Work per workgroup | one column, 1 row (m 2) or 2 rows (m 4); 4 waves | one output element; 4 waves | one output element; 1 wave |
| Workgroups | 128 × m/rows = 256 at m 2 and 4 | m × 256: 512, 1024, 2048 | m × 32: 128, 256 |
| Inner product | each thread loads its 24 bf16 of the `b` row once and reuses them for the workgroup's rows; m 2: fp32 products summed in the thread; m 4: `v_dot2_f32_bf16` in 4 independent chains | 3 steps of 16 B of `a` and `b` per thread; `v_dot2_f32_bf16` into 4 fp32 accumulators | 12 steps of 16 B per lane; `v_dot2_f32_bf16` into 4 fp32 accumulators |
| Fold | 6-step xor shuffle per wave, then thread 0 sums the 4 waves' partials from LDS in wave order | the same | 6-step xor shuffle; lane 0 stores |

With N this narrow, an MFMA tile covers few columns and the grid is small (`../gemm/grid_pricing.md` §1,
`../gemm/small_m.md` §3); here every output element gets its own wave or workgroup and the fold never
leaves it (`../gemm/small_m.md` §5: splitting N across warps needs no LDS reduction, splitting K
does).

**m ≥ 2048: split-K MFMA with a fold in a fixed order.**

- n256, m 2048 and 4096 (`_gemm_pipeline`): 64×64 tile with 4 waves (m 2048) or 128×128 with 8 waves (m 4096);
  `a` and `b`, unpacked, go by `cdna4.BufferLoadAsyncLDS128b` into a 2-stage LDS ring and from there by 16 B reads
  into `mfma_f32_16x16x32_bf16`. Split-K 4 writes fp32 partials in fragment order; a second launch
  (`_reduce_packed_parts`) sums partitions 0–3 in order and stores bf16. 512 and 256 workgroups.
- n32, m 2048 (`_compile_hierarchical`): a 32×32 output tile per workgroup, K split over its 8 waves and over 4
  workgroups (256 workgroups of 512 threads); `b` packed per call (384 KiB); `mfma_f32_16x16x32_bf16`. The 8 wave
  partials are summed through LDS in wave order; owner waves store an fp32 plane and take a ticket from a wrapping
  counter (`atomicrmw uinc_wrap`, agent scope); the last of the 4 sums planes 0–3 in order and stores bf16, all in
  the GEMM launch.

Fold forms: a separate reduce launch against an in-launch last-arriver (`../gemm/grid_pricing.md` §2,
`../gemm/small_m.md` §4); both are bitwise-stable because the order is fixed
(`../gemm/correctness.md` §8).

## Result

`speedup` as CI writes it into each schema's `workloads`: a lower bound of the CUDA-graph time ratio aiter /
kernel (`aklr/core/validation/stages/_bench/_report.py`). Bold: served (≥ 1.03, `AKLR_MIN_SPEEDUP`,
`aklr/ops/README.md` § Environment variables). Per Point:

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | Served |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n128 *(field, E1)* | 1.002 | **1.195** | **1.182** | 0.600 | 0.631 | 0.565 | 0.584 | 0.662 | 0.650 | 0.747 | 0.738 | 0.942 | 0.977 | 2 of 13 |
| n256 *(field, E2)* | 0.997 | **1.065** | **1.201** | **1.125** | 0.490 | 0.500 | 0.494 | 0.531 | 0.595 | 0.784 | 0.970 | **1.047** | **1.069** | 5 of 13 |
| n32 *(field, E3)* | 0.993 | 0.997 | **1.257** | **1.841** | 0.694 | 0.695 | 0.752 | 1.012 | 0.979 | 0.947 | 0.974 | **1.062** | 0.811 | 3 of 13 |

10 of the 39 Points are served.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0bac580d3d76a3df.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/0d33213684622a71.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n32_k6144/87f4f44793d84725.json
```

The report goes to `aklr_validation/report.md`; a local run checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: dispatch stops serving that file's Points. `AKLR_DISABLE=gemm@1.0` stops serving the
whole contract, every other bf16 GEMM schema included. A re-measured Point below 1.03 stops being served
without a rollback, and a `failed` accuracy is never served. Roll back for a wrong result CI does not catch, for
example after a FlyDSL or LLVM upgrade, or a counter left off its expected value by an aborted launch (n32 at
m = 2048).

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.565–1.195, 2 of 13 Points ≥ 1.03 (m 2, 4) (gemm_n128_k6144, m 1–4096; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.490–1.201, 5 of 13 Points ≥ 1.03 (m 2, 4, 8, 2048, 4096) (gemm_n256_k6144, m 1–4096; 5571f4589, CI run 40613440602)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.694–1.841, 3 of 13 Points ≥ 1.03 (m 4, 8, 2048) (gemm_n32_k6144, m 1–4096; 5571f4589, CI run 40613440602)

### large-M evidence and source references

Local post-measurement review: `schemas[id=<group>_<UUID>]` in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json. Complete Point records, source/schema/report/receipt SHA256s, environment/install-log locators, and the source-byte identity checks are retained in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/bf16_kernel_kb_mapping.json.

- E212 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `71234a12ada11ac5` / `gemm_n128_k6144`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_71234a12ada11ac5/validation/report.json; report SHA256 `044c011519b3eb0a2c0ffb472344e82c976d05608de87ee54ee5bec4247a1073`. Qualification is reconstructed; no incumbent timing.
- S212 aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/71234a12ada11ac5.json#kernel.py:22 executable branches; official kernel-content fingerprint `6d38d051155179c947334d8dd7bc9f887859017ee43e586797dbb7a933525f1b`. Untracked at measurement, not source contained in the report head commit.
- E214 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `3da7088c64723d7c` / `gemm_n256_k6144`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_3da7088c64723d7c/validation/report.json; report SHA256 `8cea7569820c3630e4c88f944410ad2c209b29931730993ebf48d6b3da6c65b3`. Qualification is reconstructed; no incumbent timing.
- S214 aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/3da7088c64723d7c.json#kernel.py:23 executable branches; official kernel-content fingerprint `c4eb3c9f885fb9f96ff5efa2b2fb5e32b39083a7bb61d030db2ac22bec2ecf73`. Untracked at measurement, not source contained in the report head commit.

## large-M implementation evidence

Only the discrete authored/tested Points below are covered; older kernels and results above are unchanged. These are source mechanisms and whole-call representative results, not component ablations or proof of deployment selection. See the [large-M contract, denominator, toolchain, measurement limits, reproduction, and rollback](k3_bf16_band_stream.md#large-m-implementation-evidence). Exact Point records remain in the reports and local mapping cited under Evidence.

### large-M N=128, K=6144

M=256,512 split K over eight waves; M=1024 uses four waves and M=2048 three, with a private DMA ring per wave. Lines 416–448 select a fixed pairwise tree or a wave-order LDS fold before bf16 conversion. There is no cross-workgroup counter. M=4096 has no authored or tested Point; the largest-bucket fallback is not evidence for it. *(source, S212; embedded `kernel.py` line numbers)*

**Representative result:** M=512 reached 1.337×; M=2048 reached 1.206×. All authored/tested Points for this candidate qualify under the reconstructed default rule; unlisted M values remain untested. *(field, E212)*

### large-M N=256, K=6144

Four waves each have a two-stage LDS ring. M=256 uses KS=1; M=512 uses KS=2; M=1024 uses KS=4. The latter two use cross-workgroup partials and dispatch-tagged counter workspace (409–470). KS=2 adds own+peer; KS=4 adds in split-index order (478–504). The source establishes that the protocol exists, not that it is safe. M=2048,4096 have no authored or tested Points. *(source, S214; embedded `kernel.py` line numbers)*

**Representative result:** M=512 reached 1.624×; M=256 reached 1.377×. All authored/tested Points for this candidate qualify under the reconstructed default rule; unlisted M values remain untested. *(field, E214)*
