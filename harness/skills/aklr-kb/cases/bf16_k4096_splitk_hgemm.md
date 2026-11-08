---
title: "bf16 GEMM, N 256–2048 × K 4096 — K split over waves or workgroups, folded in a fixed order; additional exact small-M cases: N 2048 × K 2048; N 512 × K 4096; N 64 × K 4096"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-256", "M 512-4096", "N=2048 K=2048 M={1,2,4,8,16,32,64,128}", "N=512 K=4096 M={1,2,4,8,16,32,64}", "N=64 K=4096 M={1,2,4,8,16,32,64,128}"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "0.858–1.400; 9 of 13 Points ≥ 1.03 (m 1–256)", gpu: MI355X, workload: "gemm_n2048_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by fdac7c716 (#52)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.584–1.290; 5 of 13 Points ≥ 1.03 (m 1, 2, 64, 128, 256)", gpu: MI355X, workload: "gemm_n1024_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.992–1.207; 10 of 13 Points ≥ 1.03 (m 4–64, 256–4096)", gpu: MI355X, workload: "gemm_n256_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by fdac7c716 (#52)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.793–1.237; 5 of 13 Points ≥ 1.03 (m 128, 256, 512, 2048, 4096)", gpu: MI355X, workload: "gemm_n512_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by fdac7c716 (#52)", date: "2026-10-28"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "8/8 Points qualify under default rule; representative ratios in small-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n2048_k2048, UUID 697eeb2233f25b25, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_697eeb2233f25b25/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "7/7 Points qualify under default rule; representative ratios in small-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n512_k4096, UUID e42f6c833c101f69, M 1, 2, 4, 8, 16, 32, 64", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_e42f6c833c101f69/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "6/8 Points qualify under default rule; representative ratios in small-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n64_k4096, UUID 8bec4c4d3db66ba0, M 1, 2, 4, 8, 16, 32, 64, 128", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_8bec4c4d3db66ba0/validation/report.json", "date": "2026-11-08"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k4096/095a7d687f48faac.json
  - aklr/ops/gemm/impls/flydsl/gemm_n1024_k4096/4f306c3bd2f6b0c1.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k4096/9200d6d7a56e5076.json
  - aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/09c4acd65577d824.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/697eeb2233f25b25.json
  - aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/e42f6c833c101f69.json
  - aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/8bec4c4d3db66ba0.json
---

# bf16 GEMM at N 256–2048, K 4096: split K, fold in a fixed order — historical collection and additional exact small-M cases

The original collection below describes the historical kernels and Points. The additional exact small-M shapes are scoped separately in the small-M implementation evidence section; they do not expand the historical measurements or imply Kimi-K3 model provenance.


Four FlyDSL kernels for the `gemm@1.0` contract on DeepSeek-V4 shapes (PR #49), one schema file per N. Each
schema embeds its kernel as `kernel.sources[0].content` (`kernel.py`); the function names below are in that
file. "m" is the Point's M.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise: `determinism: bitwise`, `max_violation_fraction` 0.0
  (`CompareSpec` defaults, `GEMM` in `aklr/ops/gemm/contract.py`). Every reduction these kernels run at a served
  Point adds in a fixed order.
- K = 4096 with N = 256, 512, 1024 or 2048: a weight of 2, 4, 8 or 16 MiB. Each file holds 13 Points,
  m = 1, 2, 4, …, 4096.
- Served Points: n2048 m 1–256; n1024 m 1, 2, 64–256; n256 m 4–64 and 256–4096; n512 m 128–512, 2048, 4096.
- Each schema name has only this file, so at every Point the yardstick is aiter.

## Counter-signals

- **Repacking the weight on every call does not pay at N = 1024.** At m = 4, 512, 1024 and 2048, n1024's
  launch runs `_build_weight_pack` (the whole 8 MiB `b` into MFMA lane order, a second launch) before
  `_build_packed_splitk_launcher`: 0.584, 0.819, 0.884, 0.973 *(field, E2)*. The module docstring still
  describes a cached relayout invalidated by tensor identity or version; the code
  (`build_gemm_a16w16_nt_n1024_k4096_module`) repacks on every call.
- **64–128 workgroups with the K split only inside the workgroup stay below aiter.** n512 at m ≤ 64 (16×16 tiles,
  4 waves splitting K, 32, 64 and 128 workgroups): 0.793–0.941 *(field, E4)*. n1024 at m 8–32 (16×16 tiles, 4–8
  waves splitting K, 64 and 128 workgroups): 0.907–0.980 *(field, E2)*. The count alone does not decide: n256 at
  m 4–64 runs 64 workgroups with split-K 8 or 4 across workgroups and is served at 1.036–1.130 *(field, E3)*.
- **No gain at large M without a K split:** n2048 0.858–1.026 at m ≥ 512 *(field, E1)*; n1024 0.819–0.973 at
  m ≥ 512 *(field, E2)*.
- **Arrival counters that live across calls.** n2048's counters are zeroed once per shape in the cached builder;
  n256's are allocated on the first call inside `launch` and kept in a closure list (with per-address pointer
  caches). The reducing block resets each counter, so the next call finds zeros. `SKILL.md`'s Stop table asks
  for scratch rewritten by every call before it is read: for a new kernel, zero counters on every call
  (`../gemm/small_m.md` §4).
- **System-scope partials.** n2048 stores and reloads its fp32 partials with `sc0 sc1` (`CPOL_SYS`);
  `../gemm/small_m.md` §4 found agent scope sufficient across XCDs on other kernels. Not raced here.
- **Code that no Point runs:** n512's cross-workgroup split-K with a separate reduce kernel (`_build_reduce`; the
  in-workgroup split sets `SPLITK = 1` at m ≤ 1024 and m 2048, 4096 have split 1), although its docstring
  describes it; n1024's cross-workgroup last-arriver path (`_build_splitk_launcher`); n256's bf16-atomic split-K
  (`IS_SPLIT_K` is false whenever `SPLIT_FP32` is set, which the builder does for every split). CI measures none
  of them.

## Mechanism

Shared by the four kernels:

- **MFMA `mfma_f32_16x16x32_bf16` into fp32 accumulators** wherever an MFMA runs (`../gemm/mainloop.md`
  §5); n1024 at m 1 and 2 runs none.
- **Operands into LDS by buffer→LDS DMA** (`buffer_load_dwordx4 … lds`, 16 B per lane) in XOR-swizzled 16 B
  chunks, read back as 16 B fragments (`../gemm/datapath.md` §2, §5).
- **K split for parallelism at small M, partials kept in fp32 and summed in a fixed order, one bf16 rounding**
  (`../gemm/grid_pricing.md` §2, `../gemm/small_m.md` §4, §5,
  `../gemm/correctness.md` §8). n2048 ports aiter's split-K HGEMM but replaces its bf16 atomic
  adds into C, which follow arrival order, with an fp32 workspace and a last-arriver fold.
- **Every served Point runs 64, 256 or 512 workgroups** (`../gemm/grid_pricing.md` §1,
  `../gemm/small_m.md` §3).
- **A per-m configuration table**, compiled on a Point's first call inside the `functools.lru_cache` function
  `_aklr_launch`.

At the served Points:

| | n2048_k4096 | n1024_k4096 | n256_k4096 | n512_k4096 |
|---|---|---|---|---|
| Path, tile, waves | m 1–8: `_compile_skw8`, one 8×8 output tile per workgroup, 8 waves. m 16, 32: `_compile_skw_av`, 16×8 / 16×16 tile, 4 waves, a private 6-slot LDS ring per wave. m 64–256: `compile_hgemm`, 32×32 / 32×64 / 64×64 tile, 4 waves, 2–4-stage LDS ring | m 1: `_build_gemv_launcher`, one output column per wave, 4 waves; 16 B loads of `a` and `b` per 512-wide K chunk, `v_dot2_f32_bf16` into one fp32 chain, a 6-step `ds_bpermute` butterfly. m 2: `_build_multi_gemv_launcher`, each `b` vector feeds both rows' chains. m 64–256: `_build_lds_splitk_launcher`, 16×16 / 16×32 / 32×32 tile, 4 waves | m 4–64: 16×32 tile, 1 wave (m ≤ 16), or 32×32, 2 waves. m 256, 512: 32×32, 2 waves. m 1024: 32×32, 2 waves. m 2048, 4096: 32×64 / 64×64, 2 waves along N × 2 along K | m 128–512: 16×16 / 16×32 / 32×32 tile, 4 waves along K, a private 4-stage LDS ring per wave. m 2048: 64×64, 2×2 waves, 5 stages; m 4096: 128×64, 4×2 waves, 4 stages; both an unrolled ping-pong with fenced next-tile fragment reads |
| K split and fold | Inside the workgroup (m ≤ 32): wave w takes K slice w, partials summed through LDS in slice order. Across workgroups (m 64–256, split 4): each slice stores its fp32 tile to a workspace, then takes a ticket from a per-tile counter (agent-scope `atomicrmw add`); the slice holding ticket 3 sums slices 0–3 in index order (its own from registers), rounds once, stores, resets the counter | m 1, 2: none (one wave holds all of K). m 64–256: each wave owns a contiguous quarter of K; the K loop waits only on that wave's own DMA, no `s_barrier`; partials parked in LDS, summed over the 4 waves in fixed order | Split-K 8 (m ≤ 32) or 4 (m 64–512) across workgroups (`SPLIT_FP32`): fp32 partials in MFMA fragment order, `sc1` stores, one counter per (tile, wave), or per tile at m 32; the last arriver loads slices 0…s−1 in index order, sums, resets the counter. m ≥ 2048: the two K waves fold in fp32 through LDS in fixed order | m 128–512: wave 0 sums the 4 partials through LDS in wave order. m 2048, 4096: none |
| Workgroups | 256 (m ≤ 32), 512 (m 64–256) | 256 at each served Point | 64 (m 4–64), 256 (m 256, 1024–4096), 512 (m 512) | 256 at each served Point |
| Tile order | m 32: the two workgroups that share a `b` tile on one XCD. Otherwise linear; the split-K path asserts no remap | linear | m ≥ 512: each XCD a contiguous eighth of the M tiles with all N tiles (`swizzle_for_cache_reuse`) | m 128–512: each XCD a box of 8 M tiles × 4 N tiles (`_XCD_REMAP`); m 2048, 4096 linear |

XCD order: `../gemm/schedule_and_locality.md` §2, `../gemm/small_m.md` §6.

## Result

`speedup` as CI writes it into each schema's `workloads`: a lower bound of the CUDA-graph time ratio aiter /
kernel (`aklr/core/validation/stages/_bench/_report.py`). Bold: served (≥ 1.03, `AKLR_MIN_SPEEDUP`,
`aklr/ops/README.md` § Environment variables). Per Point:

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | Served |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n2048 *(field, E1)* | **1.394** | **1.395** | **1.400** | **1.373** | **1.278** | **1.197** | **1.205** | **1.151** | **1.076** | 1.026 | 1.002 | 0.988 | 0.858 | 9 of 13 |
| n1024 *(field, E2)* | **1.288** | **1.290** | 0.584 | 0.980 | 0.907 | 0.973 | **1.163** | **1.190** | **1.208** | 0.819 | 0.884 | 0.973 | 0.944 | 5 of 13 |
| n256 *(field, E3)* | 1.000 | 1.002 | **1.101** | **1.081** | **1.130** | **1.071** | **1.036** | 0.992 | **1.109** | **1.207** | **1.153** | **1.172** | **1.057** | 10 of 13 |
| n512 *(field, E4)* | 0.925 | 0.898 | 0.877 | 0.850 | 0.793 | 0.851 | 0.941 | **1.187** | **1.194** | **1.237** | 1.025 | **1.116** | **1.059** | 5 of 13 |

29 of the 52 Points are served.


## small-M implementation evidence

Only the discrete authored/tested Points below are covered; older kernels and results above are unchanged. These are source mechanisms and whole-call representative results, not component ablations or proof of deployment selection. See the [small-M contract, denominator, toolchain, measurement limits, reproduction, and rollback](k3_bf16_band_stream.md#small-m-implementation-evidence). Exact Point records remain in the reports and local mapping cited under Evidence.

### small-M N=2048, K=2048

M=1,2,4,8 select `_compile_skw8`: four K-split waves, each keeping its entire K slice in LDS. M=16,32,64,128 select `_compile_skw_av`: one DMA ring per wave, then an LDS fold in wave-index order and bf16 conversion (295–319, 398–418). Every call has one launch and no cross-workgroup workspace. This extends the wave-split mechanism of the K=4096 cases, not their shape coverage. *(source, S102; embedded `kernel.py` line numbers)*

**Representative result:** M=32 reached 1.892×; M=128 reached 1.438×. All authored/tested Points for this candidate qualify under the reconstructed default rule; unlisted M values remain untested. *(field, E102)*

### small-M N=512, K=4096

At M=1,2,4,8,16,32, `_compile_packed` encodes real rows and K slices in MFMA operands; only matching-slice diagonal partials are valid. Lines 147–190 reduce in slice order, then wave order, through LDS. M=64 uses the distinct `_compile_m64` body (210–367), not the packed body. Both are single-launch paths; packing and reduction were not isolated as causal benefits. *(source, S106; embedded `kernel.py` line numbers)*

**Representative result:** M=32 reached 1.298×; M=2 reached 1.209×. All authored/tested Points for this candidate qualify under the reconstructed default rule; unlisted M values remain untested. *(field, E106)*

### small-M N=64, K=4096

Each M selects a small output tile from `_CONFIG`. A/B buffer loads go directly to registers and accumulate with `fdot2_f32_bf16` (179–206), not MFMA. A fixed lane xor/butterfly fold is followed, only when W>1, by an LDS fold in wave order (209–242). One launch, no cross-workgroup counter. This is the register-dot alternative to the K=4096 MFMA paths above; N=64 is an additional exact shape, not an extrapolation of their N range. *(source, S110; embedded `kernel.py` line numbers)*

**Representative result:** M=8 reached 1.349×; M=1 passed accuracy but reached only 1.026× and does not qualify under the reconstructed default rule. M=2 also does not qualify. *(field, E110)*

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n2048_k4096/095a7d687f48faac.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n1024_k4096/4f306c3bd2f6b0c1.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n256_k4096/9200d6d7a56e5076.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/09c4acd65577d824.json
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
example after a FlyDSL or LLVM upgrade, or a split-K counter left non-zero by an aborted launch.

## Evidence

### small-M evidence and source references

Local post-measurement review: `schemas[id=<group>_<UUID>]` in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json. Complete Point records, source/schema/report/receipt SHA256s, environment/install-log locators, and the source-byte identity checks are retained in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/bf16_kernel_kb_mapping.json.

- E102 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `697eeb2233f25b25` / `gemm_n2048_k2048`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_small_697eeb2233f25b25/validation/report.json; report SHA256 `104e6aa88eb9ab8e233fc0049b9fa60efb0cd09adef7ae931548eb245f03ede6`. Qualification is reconstructed; no incumbent timing.
- S102 aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/697eeb2233f25b25.json#kernel.py:435 executable branches; official kernel-content fingerprint `3b56dda781c2125efa5d92240ade3715d771941c8e9b41ae95d0b2ec4e3232f6`. Untracked at measurement, not source contained in the report head commit.
- E106 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `e42f6c833c101f69` / `gemm_n512_k4096`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_small_e42f6c833c101f69/validation/report.json; report SHA256 `bb0c4af1c60370257ab03a611a649641887b6f41e08b9ce9afa0f59e2d912044`. Qualification is reconstructed; no incumbent timing.
- S106 aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/e42f6c833c101f69.json#kernel.py:44 executable branches; official kernel-content fingerprint `7b48a582da92276a4fe6adb089f98080c76f4974faa2f88f3588ffe5acf14877`. Untracked at measurement, not source contained in the report head commit.
- E110 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `8bec4c4d3db66ba0` / `gemm_n64_k4096`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_small_8bec4c4d3db66ba0/validation/report.json; report SHA256 `57795d5468785df1664354cc86f6e8941ae54734b8df3cdb35a767d31326748e`. Qualification is reconstructed; no incumbent timing.
- S110 aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/8bec4c4d3db66ba0.json#kernel.py:109 executable branches; official kernel-content fingerprint `1cf2ec272b86ca6a194f6e5b259339e2941a88b306f89a8e922298f59d63fa6c`. Untracked at measurement, not source contained in the report head commit.

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.858–1.400, 9 of 13 Points ≥ 1.03 (m 1–256) (gemm_n2048_k4096, m 1–4096; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.584–1.290, 5 of 13 Points ≥ 1.03 (m 1, 2, 64, 128, 256) (gemm_n1024_k4096, m 1–4096; 5571f4589, CI run 40613440602)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.992–1.207, 10 of 13 Points ≥ 1.03 (m 4–64, 256–4096) (gemm_n256_k4096, m 1–4096; 5571f4589, CI run 40613440602)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.793–1.237, 5 of 13 Points ≥ 1.03 (m 128, 256, 512, 2048, 4096) (gemm_n512_k4096, m 1–4096; 5571f4589, CI run 40613440602)
