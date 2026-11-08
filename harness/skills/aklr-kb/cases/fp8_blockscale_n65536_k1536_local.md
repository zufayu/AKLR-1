---
title: "Block-scaled FP8 GEMM at N65536 K1536"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M Points 1, 2, 4, 8, 16, 32"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=16
measured:
  - {metric: "local AKLR baseline speed-ratio lower bounds", value: "see exact Point table", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536; M=1, 2, 4, 8, 16, 32", run_ref: "local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn65536_k1536.json", date: "2026-11-07"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/e5f9fd89c5a994fa.json
---

# Block-scaled FP8 GEMM at N65536 K1536

## Applies when

- Contract `gemm_a8w8_blockwise_scaled@1.0`, existing `b_shuffled` scenario `gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536`;
  implementation `e5f9fd89c5a994fa`. Only the exact M Points **1, 2, 4, 8, 16, 32** were tested; not a continuous range.
- FP8 e4m3fn A and AITER 16x16-preshuffled B, K-major scales, transposed B, no bias, BF16 output.
  A-scale raw payload uses `kb*M + row`, B scales use `column//128, kb`; do not substitute row-major A scales.
- Baseline: `aklr.ops.gemm.baseline.gemm_a8w8_blockwise_scaled_baseline`, with CI-pinned
  AITER `dbd8bf5bd624120197a7a26780a8c72201824f0f`, AKLR main `a12866b`.
  This is local validation, not upstream CI or a claim of deployed selection.

## Counter-signals

- Default performance qualification was observed only at M=1, 2, 4, 8, 16, 32. Other listed Points are not
  claimed default-serving qualified; unlisted Points have no validation here.
- The contract comparison is bounded, not a promise of bitwise reproducibility. The recorded violation
  fraction is zero at all listed Points, not evidence across arbitrary inputs or toolchains.
- No component ablation here proves cache, tile, occupancy or compiler causality. Do not use this case
  to choose another N/K, grouped GEMM, a different scale layout or a different activation type.

## Mechanism

`_skinny_cfg` selects BM16/BN64 with W_INFLIGHT=8 at M1/2/4/8/16 and BM32/BN128 with W_INFLIGHT=4 at M32. The source sets weight-load B_CPOL=2. The measured result belongs to the whole implementation; it does not establish a generally superior non-temporal policy.
The run entry writes the contract's caller-provided output. The source is embedded in the implementation
JSON above; reuse those parts only after checking the destination contract and remeasuring the first correct build.

## Result

*(field, E1)* Local `audit,bench` completed for each listed Point. Source content and implementation UUID
were checked against the report. The lower bounds below compare the complete call against its recorded baseline;
passing audit/bench does not mean every Point reaches the default 1.03 threshold.

| M | Baseline speed-ratio lower bound | Default 1.03 qualified |
|---:|---:|---|
| 1 | 1.045325 | yes |
| 2 | 1.054079 | yes |
| 4 | 1.061123 | yes |
| 8 | 1.065742 | yes |
| 16 | 1.093687 | yes |
| 32 | 1.034602 | yes |

## Reproduce

Use the existing contract scenario and the implementation UUID above on gfx950, with the recorded AITER
and FlyDSL versions. Run `python -m aklr export check`, then
`python -m aklr validate --base origin/main --stages audit,bench --devices 0 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/e5f9fd89c5a994fa.json`.
Keep both the report and exact source fingerprint. Results from another timer are not substituted for AKLR's.

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

Remove only `aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/e5f9fd89c5a994fa.json` to leave the previous implementations intact. Do not alter the contract or hand-fill
workload performance fields; CI decides final acceptance and writes its measurements.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, local AKLR audit/bench graph timing per Point, paired baseline comparison] Complete-call accuracy and speed-ratio lower bounds in the Result table; no component-causal claim (local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn65536_k1536.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/point_verdicts.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/new_fp8_deliveries.json).
