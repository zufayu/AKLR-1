---
title: "Block-scaled FP8 GEMM at N7168 K384"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M Points 1,2,4,8,16,32,64,128"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=16–32
measured:
  - {metric: "local AKLR baseline speed-ratio lower bounds", value: "see exact Point table", gpu: MI355X, workload: "N7168/K384; M1/2/4/8/16/32/64/128", run_ref: "local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn7168_k384.json", date: "2026-11-07"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384/2c481a926208c5cd.json
---

# Block-scaled FP8 GEMM at N7168 K384

## Applies when

- Contract `gemm_a8w8_blockwise_scaled@1.0`, existing `b_shuffled` scenario
  `gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384`, implementation `2c481a926208c5cd`.
  Only M1/2/4/8/16/32/64/128 were tested, not a continuous range and not M256.
- A and AITER 16x16-preshuffled B are FP8 e4m3fn, output is BF16, no bias, transposed B,
  K-major 128-wide scales. A-scale payload uses `kb*M + row`; B scales use `column//128, kb`.
- Baseline: `aklr.ops.gemm.baseline.gemm_a8w8_blockwise_scaled_baseline`, AKLR main `a12866b`,
  AITER `dbd8bf5bd624120197a7a26780a8c72201824f0f`. Local results do not establish upstream CI or deployed selection.

## Counter-signals

- Only M8/64/128 reached the stated 1.03 lower-bound qualification. Other measured Points passed recorded
  accuracy, but are not claimed default-serving qualified. Unlisted Points have no validation here.
- Bounded contract comparison is not bitwise reproducibility. Zero recorded violation does not cover
  arbitrary inputs or toolchains. The source's larger M limits are not validation coverage.
- No independent component ablation here establishes tile, cache, staging or occupancy causality;
  do not apply this case to other N/K, grouped GEMM or different scale layouts without new checks.

## Mechanism

The source's `_pick_config` chooses 16x16/one wave at M1/2/4/8/16, 16x64/four waves at M32/64,
and 32x64/four waves at M128. A is direct only at M1/2/4; C is direct through M64 and staged at M128.
The kernel processes three 128-wide K scale blocks and writes the caller-provided output.
These are source-level configuration choices, not isolated explanations of the measured gain.
Source is embedded in the implementation JSON; borrowing a path requires destination-contract checks and remeasurement.

## Result

*(field, E1)* Local `audit,bench` completed at all eight Points with zero recorded violation.
Published source and UUID match the tested artifact. Complete-call baseline lower bounds are:

| M | Paired rounds | Baseline speed-ratio lower bound | 1.03 lower-bound qualified |
|---:|---:|---:|---|
| 1 | 32 | 1.028771 | no |
| 2 | 16 | 1.020397 | no |
| 4 | 16 | 1.020032 | no |
| 8 | 32 | 1.030872 | yes |
| 16 | 32 | 1.027845 | no |
| 32 | 16 | 1.025670 | no |
| 64 | 16 | 1.055182 | yes |
| 128 | 16 | 1.154112 | yes |

These qualification labels describe this measurement only; CI remains authoritative.

## Reproduce

On gfx950 with the recorded versions, run `python -m aklr export check`, then
`python -m aklr validate --base origin/main --stages audit,bench --devices 0`
with `aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384/2c481a926208c5cd.json`.
Retain the report and exact source fingerprint, and compare by Point rather than a mean over M.

The recorded baseline configuration matters: `baseline_config_receipt.json` archives both CSV byte strings
and the module-resolved paths in `/tmp/aiter_configs/`. The preshuffle table
`a8w8_blockscale_bpreshuffle_tuned_gemm.csv` has SHA256
`53c4d8057ce42ede156990613d45b313e180d04a79f3ac2b156108393dab3fa2`; the other blockscale table has
SHA256 `6bc517bbd0e368c5f3088451af686d437881b1d9c84cc90d578ae1b62bb6d988`.
The receipt is at `local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/baseline_config_receipt.json`.
Restore those archived CSV bytes and verify that AITER resolves the same configuration files before
comparing the reproduced ratios; installing the AITER commit alone does not fix the denominator.
These reports compare against the baseline only, not an incumbent implementation.

## Rollback

Remove only the `2c481a926208c5cd.json` implementation; previous implementations and the contract stay intact.
Do not hand-fill accuracy or speedup fields. CI writes final measurements and determines acceptance.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16–32, local AKLR audit/bench graph timing per Point, paired baseline comparisons] Whole-call accuracy and lower bounds in the Result table; no component-causal claim (local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn7168_k384.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/point_verdicts.json).
