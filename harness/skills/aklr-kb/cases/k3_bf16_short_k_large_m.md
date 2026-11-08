---
title: "bf16 GEMM, short K or large M only (Kimi-K3: N 1536 × K 128, N 163840 × K 256, N 2304 × K 1536, N 3072 × K 512) — the whole K in one tile or one workgroup's waves, persistent grids at the largest M; additional exact large-M cases: N 28672 × K 512; N 3584 × K 512; N 6144 × K 1536; N 7168 × K 512"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-14", "M 8-112", "M 893-16384", "M 35107-342073", "N=28672 K=512 M={256,512,1024,2048,4096}", "N=3584 K=512 M={256,512,1024,2048,4096}", "N=6144 K=1536 M={256,512,1024,2048,4096}", "N=7168 K=512 M={256,512,1024,2048,4096}"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.023–1.716; 12 of 17 Points ≥ 1.03 (not m 8, 12, 16, 32, 64); 0 served", gpu: MI355X, workload: "gemm_n1536_k128, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2_local/report.md:67", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.527–1.628; 11 of 11 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n163840_k256, m 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14", run_ref: "e44184aec (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.087–1.322; 4 of 4 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n2304_k1536, m 893, 2732, 9397, 16384", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:39", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.113–1.294; 3 of 3 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n3072_k512, m 35107, 104093, 342073", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:43", date: "2026-10-26"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; representative ratios in large-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n28672_k512, UUID ce4647f84b8de43b, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_ce4647f84b8de43b/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; representative ratios in large-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n3584_k512, UUID 1fbaf19b1c4fdb45, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_1fbaf19b1c4fdb45/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "3/5 Points qualify under default rule; representative ratios in large-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n6144_k1536, UUID d5c0cc4ef7575b67, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_d5c0cc4ef7575b67/validation/report.json", "date": "2026-11-08"}
  - {"metric": "whole-call per-Point baseline speedup; local AKLR audit/bench, not CI", "value": "5/5 Points qualify under default rule; representative ratios in large-M implementation evidence; full Point records in local mapping", "gpu": "MI355X", "workload": "gemm_n7168_k512, UUID 04da38be4d3af890, M 256, 512, 1024, 2048, 4096", "run_ref": "local:/tmp/inf3_validation_checkpoint_0147/results/g3/bf16_large_04da38be4d3af890/validation/report.json", "date": "2026-11-08"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n1536_k128/f0579ac60c343047.json
  - aklr/ops/gemm/impls/flydsl/gemm_n163840_k256/1220a995dd7d23c0.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2304_k1536/0c3c675002f2fa94.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3072_k512/4c6a228ee0e2d0ed.json
  - aklr/ops/gemm/impls/flydsl/gemm_n28672_k512/ce4647f84b8de43b.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/1fbaf19b1c4fdb45.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/d5c0cc4ef7575b67.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/04da38be4d3af890.json
---

# bf16 GEMM at short K or large M only (Kimi-K3): no K loop to split, a persistent grid where M is large — historical collection and additional exact large-M cases

The original collection below describes the historical Kimi-K3 kernels and Points. The additional exact large-M shapes are scoped separately in the large-M implementation evidence section; they do not expand the historical measurements or imply Kimi-K3 model provenance.


Four FlyDSL kernels for the `gemm@1.0` contract on Kimi-K3 shapes (definitions #69, kernels #115, merge
a8fe490b1), one schema file per shape, written by forge-loop campaigns and reviewed. Each schema embeds its
kernel as `kernel.sources[0].content` (`kernel.py`); function names below are in that file. "m" is the Point's M.
None of the four splits K across workgroups.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0,
  the `CompareSpec` defaults), with the fp32 dot-product error bound of `_dot_k()` (`GEMM` in
  `aklr/ops/gemm/contract.py`). Every output is one fp32 sum in a fixed order.
- Shapes:
  - N 1536 × K 128 (a 384 KiB weight) at m 8–16384: K fits one LDS tile.
  - N 163840 × K 256 (80 MiB) at m 1–14: a read-once weight over a very wide N.
  - N 2304 × K 1536 at m 893–16384 and N 3072 × K 512 at m 35107–342073: only large-M Points, compute-bound
    tiles; the second has 8 K blocks of 64.
- Each schema name has only this file (no file of another language or an older kernel under the same name on
  main), so at every Point the yardstick is aiter.
- **No Point of these files is served today.** No CI run has written `speedup` or `accuracy` into their
  `workloads` (main at 0cef4f2e3); dispatch serves a workload only from `AKLR_MIN_SPEEDUP` 1.03 and falls back with
  `not_measured` (`aklr/ops/README.md` § Environment variables; `aklr/core/contract/spec/_workloads.py:38`).

## Counter-signals

- **n1536_k128 at m ≤ 112 sits on the threshold.** Its 13 decode Points read 1.023–1.038 in the latest local run,
  5 of them below 1.03 *(field, E1)*; the same code read 1.010–1.038 in the commit's run and 1.021–1.123 in a third
  run's two arms *(field, E2)*. Which of them CI serves will depend on its run. These Points run a 16-column,
  one-wave tile on 96 workgroups for 256 CUs (code read).
- **n1536_k128: a retune of the three largest m buckets did not pay.** Run beside the delivered kernel in one
  window, it passed the local validate's bar (a lower bound of 1.02) at 14 and 12 of 17 Points against the delivered
  kernel's 14 and 16 in two runs *(field, E11)*; the campaign's ablation file still scores that version as its best.
- **n2304's two-tile walk requires `m % 256 == 0`** (asserted when the persistent kernel is built); m 16384 is the
  only Point that takes it. n3072 sizes its persistent grid from the CU count of the current device at compile time
  and starts half of each XCD's workgroups late only up to m 200000 (`_STAGGER_MAX_M`), so m 342073 runs without
  the delay.
- **Code no Point reaches:** n1536_k128's `_CONFIGS` row 2731 (no Point between 893 and 2732).
- Nothing these kernels compute crosses a workgroup, so there are no flags, counters or workspace: no state lives
  between calls, and no reduction depends on arrival order (code read).

## Mechanism

Shared by the four kernels:

- **MFMA `mfma_f32_16x16x32_bf16` with B as the first operand**, so a lane holds four consecutive output columns
  of one row (`../gemm/epilogue.md` §3), fp32 accumulators.
- **Kernel pointer arguments preloaded into SGPRs** (`amdgpu-kernarg-preload-count` 6 or 16;
  `../gemm/small_m.md` §6).
- **A per-m configuration** compiled in a `functools.cache` function on a Point's first call.

Per file:

**n1536_k128** (`_build_tiled` at every m).
- The whole K = 128 is one LDS tile: one round of buffer→LDS DMAs for A and B (`buffer_load_dwordx4 … lds`,
  XOR-swizzled 16-byte chunks), one `s_waitcnt vmcnt(0)` + barrier, four MFMA K steps; no ring, no K loop
  (`../gemm/datapath.md` §2, §5).
- m ≤ 112: a TILE_M × 16 tile with TILE_M the m bucket (16 … 112), one wave, 96 workgroups, B `nt`.
- m 893: 32×192, 4 waves, grouped XCD order (`GROUP_M` 4), 224 workgroups. m 2732: 64×96, 4 waves, `GROUP_M` 8,
  688 workgroups. m 9397: 128×128, `GROUP_M` 16, 888 workgroups. m 16384: 128×192, `GROUP_M` 8, 1024 workgroups.
- Stores: C staged through LDS and written as 16-byte rows with `sc1`; at m 9397 and 16384 B's rows are fetched
  permuted so a lane's two adjacent 16-column fragments are contiguous output columns, and C goes out as 16-byte
  stores straight from registers (`C_DIRECT`).

**n163840_k256** (`_build_skinny`; m ≤ 16 asserted).
- One 16-row MFMA block (all m rows) by 16 or 32 columns per workgroup; the waves split K and an LDS fold sums
  their partials in wave order:
  - m 1: 4 waves, each a quarter of K, 16 columns per workgroup, 10240 workgroups;
  - m 2, 3: 8 waves, each an eighth of K, 32 columns, 5120 workgroups;
  - m 4–14: 8 waves as 2 column halves × 4 K quarters, 32 columns, 5120 workgroups.
- B goes straight from global into registers, every K step's load issued before the first MFMA; each B element
  feeds one wave, so LDS would add no reuse. A goes to registers directly at m ≤ 3 and through LDS (row stride
  padded by 8 elements) from m 4.
- Adjacent column tiles are dealt to one XCD in groups of 8 (`_grouped_tile`), so a C row's neighbouring tiles
  finish in one L2 (`../gemm/small_m.md` §6); C leaves as 16-byte row stores from m 2.

**n2304_k1536** (`_build`, per-m keyword table).

| m | Tile, waves | Stages, K tile | Workgroups | Also |
|---|---|---|---|---|
| 893 | 64×128, 8 (2×4) | 3, 128 | 252 | grouped XCD order (`GM` 7), fragment prefetch, K loop unrolled, C `sc1` |
| 2732 | 144×192, 12 (3×4) | 3, 64 | 228 | `GM` 7, refills issued inside the compute after the first fragment reads, unrolled, C `nt` |
| 9397 | 448×192, 8 (4×2) | 2, 64 | 252, one round | C staged and stored in two passes, `sc1`; LLVM scheduler `iterative-ilp` |
| 16384 | 256×288, 8 (4×2) | 2, 64 | 256 for 512 tiles | two tiles per workgroup: the second walks K in reverse, starting on the slices the first just left in L2; tile 0's C leaves through stage 1 while stage 0 loads tile 1; scheduler `iterative-minreg`; C `sc1` |

Grids sized to whole rounds of the 256 CUs (`../gemm/grid_pricing.md` §1, §10); the scheduler
strategy goes in through `amdgpu-sched-strategy` (`../gemm/schedule_and_locality.md` §3).

**n3072_k512** (`_build`, one configuration for every m).
- 256×256 tiles, 8 waves (2×4), two LDS stages of K 64, `GROUP_M` 2 grouped order, `waves_per_eu` 2.
- A persistent grid of 256 workgroups (`G`, from the device's CU count) walks head tiles pid, pid + G, … over
  the first P whole 256-row panels, then the remaining rows as 128-row tail items, handed first to the workgroups
  with one head tile fewer (`_plan`):
  - m 35107: P 128 panels, 1536 head tiles (6 per workgroup) and 228 tail items;
  - m 104093: P 405, 4860 head tiles (18 or 19 per workgroup) and 48 tail items;
  - m 342073: P 1337, 16044 tiles (62 or 63 per workgroup), no tail.
- No workgroup barrier between items: each K block's barrier sits mid-block, the block after next is filled behind
  it, and each wave stages its own C rows (32 per pass) through a private LDS area and stores them `nt`.
- At m ≤ 200000, odd workgroups of each XCD sleep before their first fill, so the two halves reach their
  epilogues out of phase (`../gemm/grid_pricing.md` §10).

**What each part bought** (campaign ablations; each kept round timed beside the version before it):

- n1536_k128: the single-pass kernel with per-m tiles and preloaded pointers +0.6–4.4% at m ≤ 112 and
  +22.3–42.4% at m ≥ 893 over the campaign's starting kernel; `sc1` C stores +3.1–5.0% at m ≤ 112 and +8.3–20.3%
  above *(field, E6)*; the register-direct C store +5.9% (16384) and +3.8% (9397) *(field, E7)*.
- n163840_k256: the K-split-over-waves port +18.8–41.3%; kernel-argument preload +1.6–3.0% at every Point; the
  XCD grouping +0.0–1.6%; the two-half workgroup for m ≥ 4 +0.1–2.6% at m 4–14. Against the starting kernel the
  delivered one is +49.7–62.7% *(field, E8)*.
- n2304_k1536: grouping, vector C staging and `nt` C +2.5–8.1%; BK 128 with 3 stages +6.2% at m 893; the unrolled
  K loop +1.2% / +3.2% at m 893 / 2732; the scheduler strategy +2.3% / +0.7% at m 9397 / 16384; the persistent
  two-tile walk +0.9% at m 16384. Against the starting kernel: +9.5–23.2% *(field, E9)*.
- n3072_k512: the 256×256 port with grouped order and `nt` C +6.4–16.7%; the persistent loop with per-wave C
  +5.3–10.0%; tail items +2.3% at m 35107; the start delay +2.9% / +1.1% at m 35107 / 104093; the mid-block barrier
  +0.6–2.2%. Against the starting kernel: +13.8–31.5% *(field, E10)*.

## Result

`speedup` is absent from every workload of these files: CI has not measured them, so dispatch serves none of
their 35 Points. When CI does measure them it runs the FlyDSL of its image (0.3.4.1 in
the image of the latest perf runs), not the 0.3.2 of the local numbers below
(`../pitfalls/measured_a_different_build.md`).

The only per-Point measurements are the campaigns' own `python -m sikl validate` runs on MI355X (the repository
before its rename to AKLR), FlyDSL 0.3.2: a lower bound of the CUDA-graph time ratio aiter / kernel, one run each;
"≥ 1.03" counts them against the dispatch threshold (that validate itself demoted below 1.02).

| File | Points | Lower bound per Point | ≥ 1.03 locally | Served |
|---|---|---|---|---|
| n1536_k128 *(field, E1)* | m 8–112 | 1.023–1.038 | 8 of 13 | 0 |
| | m 893, 2732, 9397, 16384 | 1.243, 1.716, 1.432, 1.435 | 4 of 4 | 0 |
| n163840_k256 *(field, E3)* | m 1–14 | 1.527–1.628 | 11 of 11 | 0 |
| n2304_k1536 *(field, E4)* | m 893, 2732, 9397, 16384 | 1.322, 1.165, 1.153, 1.087 | 4 of 4 | 0 |
| n3072_k512 *(field, E5)* | m 35107, 104093, 342073 | 1.294, 1.248, 1.113 | 3 of 3 | 0 |

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root, one schema at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n1536_k128/f0579ac60c343047.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n163840_k256/1220a995dd7d23c0.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n2304_k1536/0c3c675002f2fa94.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n3072_k512/4c6a228ee0e2d0ed.json
```

The report goes to `aklr_validation/report.md`; a local run checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). n3072's m 342073 compares a 2 GB output
against the reference; validate scales the trial's time limit with the output size.
To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: its Points go unserved, since no other file serves them. `AKLR_DISABLE=gemm@1.0`
stops serving the whole contract, every other bf16 GEMM schema included. Until CI writes `speedup`, nothing is
served and nothing needs rolling back; afterwards a Point measured below 1.03 is not served, and a `failed`
accuracy never is. Roll back for a wrong result CI does not catch, for example after a FlyDSL or LLVM upgrade.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`; "local validate"
is `python -m sikl validate` on MI355X in the campaign containers, the ratio lower bound over aiter per Point.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1536_k128: m 8 1.025, 12 1.023, 16 1.024, 24 1.031, 32 1.023, 40 1.032, 48 1.030, 56 1.038, 60 1.037, 64 1.024, 80 1.033, 96 1.033, 112 1.032, 893 1.243, 2732 1.716, 9397 1.432, 16384 1.435 (gemm_n1536_k128; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2_local/report.md:67)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1536_k128 at m ≤ 112: 1.010–1.038 in the commit's run of the first delivered version; 1.048–1.123 and 1.021–1.103 for that version (602fb113) and the large-m group's module (437aa0ab) side by side in another run; the delivery check compared the final ISA at m ≤ 893 (gemm_n1536_k128; commit message of fb552046c; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab_local/report.md:118; local:/workspace/code/gpt_oss_docker/.claude/memory/project_k3_bf16_gemm_agent_kernels.md:131)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n163840_k256: m 1 1.628, 2 1.588, 3 1.591, 4 1.573, 5 1.575, 6 1.573, 7 1.569, 8 1.566, 10 1.562, 12 1.551, 14 1.527 (gemm_n163840_k256; commit message of e44184aec)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n2304_k1536: m 893 1.322, 2732 1.165, 9397 1.153, 16384 1.087 (gemm_n2304_k1536; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:39)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n3072_k512: m 35107 1.294, 104093 1.248, 342073 1.113 (gemm_n3072_k512; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:43)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench, recorded rounds] n1536_k128: single-pass kernel +2.0, 0.6, 1.2, 1.8, 1.6, 2.1, 2.1, 4.4, 3.5, 2.7, 2.8, 2.9, 3.5% at m 8–112, +22.3, 42.4, 26.4, 25.4% at 893, 2732, 9397, 16384; `sc1` C +3.1–5.0% at m ≤ 112, +10.0, 20.3, 8.7, 8.3% above (gemm_n1536_k128; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1536_k128/forge_experiments/ablation.json)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop paired bench, group campaign] n1536_k128 C from registers with permuted B rows: +5.9% at m 16384, +3.8% at 9397, +1.3% at 2732 (r2_n7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/r2_n7168/forge_experiments/ablation.json)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n163840_k256: port +18.8–41.3% (recorded); preload +1.6–3.0% (recorded); XCD grouping +0.0–1.6%; two-half workgroup +0.1–2.6% at m 4–14; all changes off +49.7–62.7% (gemm_n163840_k256; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/ablation.json)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n2304_k1536: grouping + vector C staging + `nt` C +7.9, 8.1, 2.5, 2.6% (recorded); BK 128 +6.2% at 893; unroll +1.2 / +3.2% at 893 / 2732; scheduler strategy +2.3 / +0.7% at 9397 / 16384; persistent +0.9% at 16384; all changes off +23.2, 15.8, 15.6, 9.5% at m 893, 2732, 9397, 16384 (gemm_n2304_k1536; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2304_k1536/forge_experiments/ablation.json; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_docs/gemm_n2304_k1536/ablation.md)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n3072_k512 at m 35107 / 104093 / 342073: port +9.2 / 16.7 / 6.4% (recorded); persistent loop +10.0 / 7.3 / 5.3% (recorded); tail items +2.3 / 0.1 / 0.0%; start delay +2.9 / 1.1 / 0.0%; mid-block barrier +0.6 / 2.2 / 1.1%; all changes off +28.2 / 31.5 / 13.8% (gemm_n3072_k512; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/ablation.json; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_docs/gemm_n3072_k512/ablation.md)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=2, local validate, one window per run] n1536_k128 delivered beside the dropped last round: 14 and 16 of 17 Points past the local bar against 14 and 12 (gemm_n1536_k128; commit message of fb552046c)
- S1 aklr/ops/gemm/contract.py:128 `GEMM` with `_UNSHUFFLED_VARIANTS` (:124) and `_dot_k()` (:95); `CompareSpec` defaults bitwise at aklr/core/contract/spec/_args.py:217
- S2 aklr/ops/gemm/impls/flydsl/gemm_n1536_k128/f0579ac60c343047.json#kernel.py:26 `_CONFIGS`; whole-K DMA and one wait at :260; `C_DIRECT` stores at :281; `sc1` stores at :20 and :58
- S3 aklr/ops/gemm/impls/flydsl/gemm_n163840_k256/1220a995dd7d23c0.json#kernel.py:20 `_CONFIGS`; `_grouped_tile` at :47; B loads first at :134; B-first MFMA at :154; wave-order fold at :182 and :197
- S4 aklr/ops/gemm/impls/flydsl/gemm_n2304_k1536/0c3c675002f2fa94.json#kernel.py:35 `_CONFIGS`; scheduler strategy at :114; two-tile assertion at :173; second tile in reverse K at :521
- S5 aklr/ops/gemm/impls/flydsl/gemm_n3072_k512/4c6a228ee0e2d0ed.json#kernel.py:49 `_plan`; grid from the CU count at :110; start delay at :476; tail items at :492; `nt` C at :454

### large-M evidence and source references

Local post-measurement review: `schemas[id=<group>_<UUID>]` in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/operator_validation_all32_reviewed.json. Complete Point records, source/schema/report/receipt SHA256s, environment/install-log locators, and the source-byte identity checks are retained in local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/bf16_kernel_kb_mapping.json.

- E215 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `ce4647f84b8de43b` / `gemm_n28672_k512`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g5/bf16_large_ce4647f84b8de43b/validation/report.json; report SHA256 `871fbec4aa195ff0c799df1669e74ea8dc2cafa630ea8a0eb81e81464a082cd1`. Qualification is reconstructed; no incumbent timing.
- S215 aklr/ops/gemm/impls/flydsl/gemm_n28672_k512/ce4647f84b8de43b.json#kernel.py:161 executable branches; official kernel-content fingerprint `887fba39c8d5719be6ecb78327321a623396eaaae834fd38720b7115a9ef2f85`. Untracked at measurement, not source contained in the report head commit.
- E216 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `1fbaf19b1c4fdb45` / `gemm_n3584_k512`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g3/bf16_large_1fbaf19b1c4fdb45/validation/report.json; report SHA256 `7f7e25ac4f3ef5e9483179cdc04fb2fa6b8550183bc71b06e2279dc3272b2199`. Qualification is reconstructed; no incumbent timing.
- S216 aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/1fbaf19b1c4fdb45.json#kernel.py:34 executable branches; official kernel-content fingerprint `5f40da7545581c61cbfa58920f6a2a9d7c14fee0a203f1ffbfb45fc365ec2bc7`. Untracked at measurement, not source contained in the report head commit.
- E217 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `d5c0cc4ef7575b67` / `gemm_n6144_k1536`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0142/results/g4/bf16_large_d5c0cc4ef7575b67/validation/report.json; report SHA256 `9e7d8f70b1bfc6177d83a0d5b03293fd7ccb17577f8619b436000b374a45d1e7`. Qualification is reconstructed; no incumbent timing.
- S217 aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/d5c0cc4ef7575b67.json#kernel.py:23 executable branches; official kernel-content fingerprint `956115ab4022a5ccd3a1e528b433c0b70f736f0635c3d9244ec8869da151e2e9`. Untracked at measurement, not source contained in the report head commit.
- E219 [2026-11, gfx950, FlyDSL 0.3.2, N=1, local AKLR profile-v18 whole-call] `04da38be4d3af890` / `gemm_n7168_k512`: all authored Points accuracy passed; report local:/tmp/inf3_validation_checkpoint_0147/results/g3/bf16_large_04da38be4d3af890/validation/report.json; report SHA256 `47c9461d2ae1841cef97d764a2843199e179aa4c2c5f4d6aa9d81089fc656ec5`. Qualification is reconstructed; no incumbent timing.
- S219 aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/04da38be4d3af890.json#kernel.py:29 executable branches; official kernel-content fingerprint `6c1bedc1abb52d8b7fb4aeddb804cb24aad7e81a7a06fb3edfa29592175ecfe7`. Untracked at measurement, not source contained in the report head commit.

## large-M implementation evidence

Only the discrete authored/tested Points below are covered; older kernels and results above are unchanged. These are source mechanisms and whole-call representative results, not component ablations or proof of deployment selection. See the [large-M contract, denominator, toolchain, measurement limits, reproduction, and rollback](k3_bf16_band_stream.md#large-m-implementation-evidence). Exact Point records remain in the reports and local mapping cited under Evidence.

### large-M N=28672, K=512

Persistent workgroups walk short-K head tiles and shorter row-tail items: host plan at 67–101, tile mapping at 298–339, B-first MFMA at 341–343, A/B DMA into LDS. There is no cross-workgroup split-K. `_SPLIT_PLAN` at M=1024 describes head/tail layout, not K splitting. Tail-cost constants in the plan are not independent component measurements from this validation. *(source, S215; embedded `kernel.py` line numbers)*

**Representative result:** M=2048 reached 1.444×; M=4096 reached 1.167×. All authored/tested Points for this candidate qualify under the reconstructed default rule; unlisted M values remain untested. *(field, E215)*

### large-M N=3584, K=512

M=256,512,1024,2048,4096 use 64×64, 64×128, 128×128, 128×224, and 256×224 tiles respectively. A/B multistage DMA rings feed B-first MFMA (616–618); per-M configuration selects LDS C staging or register-direct C stores and overlap (942–997). `split_stage/compute_split` schedule fragments within a stage: the names do not imply cross-workgroup split-K. *(source, S216; embedded `kernel.py` line numbers)*

**Representative result:** M=256 reached 1.631×; M=1024 reached 1.360×. All authored/tested Points for this candidate qualify under the reconstructed default rule; unlisted M values remain untested. *(field, E216)*

### large-M N=6144, K=1536

**M=2048 is library fallback, not custom DMA/MFMA.** `_CONFIGS` has M=256,512,1024,4096 only. At M=2048, `_launcher` (78–88) returns `_library`, which executes `torch.matmul(a, b.t(), out=out)`. Custom Points use an A/B async-LDS tile pipeline; MFMA32 selects 16×16×32 or 32×32×16 BF16 MFMA (223–230). Padded LDS C staging emits segments selected by C_SPLIT (457–513); EARLY stores some rows during the last K step (374–412). No cross-workgroup K split; C_SPLIT is not split-K. M=2048 passed accuracy at 0.999 and fails baseline entry; M=4096 passed at 1.028 and passes entry. Neither qualifies under the reconstructed default serving rule; the linked report supplies their complete Point evidence. *(source, S217; embedded `kernel.py` line numbers)*

**Representative result:** M=256 reached 1.257×; M=2048 passed accuracy but reached only 0.999× and does not qualify under the reconstructed default rule. *(field, E217)*

### large-M N=7168, K=512

M=256,512 select `_build_large`, a one-shot tiled body. M=1024,2048,4096 select `_build_pipe`, a persistent multi-tile walk. A/B DMA into XOR-swizzled LDS feeds 16×16×32 BF16 MFMA. Lines 941–947 select bucket-specific compiler scheduling hints and kernarg preload. A hint’s presence proves neither compiler realization nor an isolated performance benefit. *(source, S219; embedded `kernel.py` line numbers)*

**Representative result:** M=1024 reached 1.787×; M=256 reached 1.164×. All authored/tested Points for this candidate qualify under the reconstructed default rule; unlisted M values remain untested. *(field, E219)*
