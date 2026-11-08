---
title: "FP8 block-scaled GEMM with shuffled B — N4096 K sweep and large-N / short-K kernels; CI and local evidence separated"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M 1-64", "M 128-512", "M 1024-4096"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1 CI; local paired rounds N=16–32
measured:
  - {metric: "local baseline speed-ratio lower bound, per Point", value: "1.034602–1.093687; 6 of 6 Points ≥ 1.03", gpu: MI355X, workload: "N65536 K1536; M 1, 2, 4, 8, 16, 32", run_ref: "local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn65536_k1536.json", date: "2026-11-07"}
  - {metric: "local baseline speed-ratio lower bound, per Point", value: "1.333043–1.966142; 5 of 5 Points ≥ 1.03", gpu: MI355X, workload: "N16384 K1536; M 1, 2, 4, 16, 32", run_ref: "local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn16384_k1536.json", date: "2026-11-07"}
  - {metric: "local baseline speed-ratio lower bound, per Point", value: "1.020032–1.154112; 3 of 8 Points ≥ 1.03", gpu: MI355X, workload: "N7168 K384; M 1, 2, 4, 8, 16, 32, 64, 128", run_ref: "local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn7168_k384.json", date: "2026-11-07"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.034–1.849; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.973–1.531; 4 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.816–1.125; 5 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.064–1.293; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k2048, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "102877ca0 (CI run 40154665266)", date: "2026-10-25"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.148–2.250; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.094–2.557; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "102877ca0 (CI run 40154665266)", date: "2026-10-25"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.378–1.892; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n8192_k1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.015–2.229; 12 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/e5f9fd89c5a994fa.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n16384_k1536/ef050f07296547ba.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384/2c481a926208c5cd.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256/5fcd9f0ebdced65c.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512/9a8319559bcb2f29.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024/e5d65066d6572075.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k2048/0610d6f957904254.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096/8c907984278eefc4.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n8192_k1024/41f2a05011466520.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024/42ffbb4ee1697e03.json
---

# FP8 block-scaled GEMM with shuffled B: K sweep and large-N / short-K kernels

The eight CI-measured kernels below and the three implementations in
[Local large-N and short-K implementations](#local-large-n-and-short-k-implementations) share the same
column-major A-scale contract and preshuffled B layout. The local N16384/N65536 K1536 activation-slab
paths extend the large-N skinny mechanism; N7168 K384 extends the short-K tiled/direct path.
They belong here rather than in the [row-major A-scale case](a8w8_row_major_a_scale.md) or the
[small-N / long-K wave-split case](a8w8_small_n_large_k.md). Each UUID remains a separate implementation.

**Evidence boundary:** Applies when through Rollback below describe the original eight CI kernels only
(E1–E9, S1–S8). The separate local section has its own exact Points, baseline configuration, toolchain
and paired-round counts (E10–E13, S9–S11). Do not combine its qualification counts with the CI serving count or interpret
its baseline-only ratios as improvements over an incumbent AKLR implementation.

## Applies when

- Contract `gemm_a8w8_blockwise_scaled@1.0`, layout variant `b_shuffled` (`operand_format` `none:preshuffled`,
  `a_scale.is_row_major` false): a and b fp8 e4m3fn, b in aiter's 16×16 `shuffle_weight` order, one fp32 a-scale
  per 1×128 of a in the raw column-major payload (`S_a[i, kb] = a_scale_flat[kb·m + i]`), one fp32 b-scale per
  128×128 of b, `trans_b`, `scale_major_mode` K, no bias, bf16 out.
- N = 4096 with K = 256, 512, 1024, 2048, 4096 or 8192 (2 to 64 scale blocks along K), and N = 8192 or 32768 with
  K = 1024. Each (N, K) pair is its own schema; each serves m = 1, 2, 4, …, 4096.
- Compare: each schema's `compare` block is `determinism: bounded`, `max_violation_fraction: 0.0`, against the
  contract's tolerance (atol = rtol = 1e-2 for bf16 out; `aklr/ops/gemm/contract.py`). One file's split-K
  depends on it (Counter-signals).
- Read the file for your K before reusing a design: the eight files are eight different kernels, not one kernel
  tuned eight ways.

## Counter-signals

- **N 4096 with K 512 or 1024 at small and mid m is at or below aiter.** n4096_k512 serves 4 of 13 Points
  (m ≥ 512); its m ≤ 256 Points read 0.973–1.006 *(field, E2)*. It splits K nowhere: at m ≤ 64 one 16×64 tile holds
  all of K (64 workgroups at m ≤ 16). n4096_k1024 serves 5 of 13 *(field, E3)*: m = 4, 8, 16 at 1.064–1.081, the
  three Points on its eight-wave in-workgroup K split, and m = 2048, 4096 at 1.096 and 1.125. Its unsplit four-wave
  body reads 0.816–1.018 at every other m from 1 to 1024 *(field, E3)*. K 256 holds a thin margin at m ≤ 256:
  1.034–1.052 *(field, E1)*.
- **With one CI measurement per file, Points within a few percent of 1.03 move across it.** Across the five committed
  measurements: n4096_k1024 m = 32 1.014, 1.023, 0.999, 1.030, 1.018 and m = 512 1.062, 1.057, 1.018, 0.990, 1.002;
  n4096_k256 m = 8 1.006, 1.024, 1.025, 1.029, 1.047; n4096_k512 m = 256 1.084, 1.053, 1.007, 0.995, 1.006;
  n32768_k1024 m = 64 1.008, 1.071, 1.017, 1.020, 1.015 *(field, E9)*. Which of these Points dispatch serves depends
  on the run that wrote the file. The five runs span at least two CI images; the last three share one image and FlyDSL
  0.3.4.1, and n4096_k1024 m = 32 and n4096_k256 m = 8 cross 1.03 within those three as well.
- **n32768_k1024 at m = 64: 1.015, not served** *(field, E8)*. That Point runs the skinny path with a 64-row
  activation slab and 128 columns per workgroup (8 waves, 256 workgroups); at m ≤ 32 the same path with 16- or
  32-row slabs and 64 columns (512 workgroups) reads 1.272–1.443 *(field, E8)*.
- **n4096_k8192 at m = 64 and 128 reduces its split-K partials in arrival order.** Split-K 4 across workgroups: each
  split stores its fp32 partial, bumps an agent-scope counter, and the last to arrive adds its own partial first and
  then the others from split index (own + 1) mod 4 onwards. The association depends on which split arrives last, so
  the output bits can differ between runs. The contract is `bounded`, so this passes; under `bitwise` these Points
  need a fixed order (`../gemm/correctness.md` §8, `../gemm/small_m.md` §4). At m = 256
  (split 2) the two-term sum does not depend on the order.
- **The same split-K keeps its counters between calls**: the builder zeroes them once per shape (`torch.zeros`) and
  the last arriver subtracts the split count. No AKLR call may depend on what an earlier call left (`../SKILL.md`,
  Stop table), and two streams calling the same Point at once would share the counters.

## Mechanism

What the eight kernels share:

- **One 16×16×128 fp8 MFMA per 128-wide K block**, into a zeroed fp32 partial, folded into the fp32 accumulator
  with S_a·S_b, by one FMA in most bodies (`../gemm/mainloop.md` §5). n4096_k256 keeps its two K-block partials
  apart and combines them once at the end.
- **B straight to registers** from the 16×16 preshuffled layout, two 16-byte loads per lane per K block
  (`../gemm/datapath.md` §10), except where noted below (n4096_k4096 at m ≥ 64 and n32768_k1024 at
  m ≥ 128 DMA B into LDS, where several waves share it: `../gemm/datapath.md` §7).
- **A through XOR-swizzled LDS** for most Points (`../gemm/datapath.md` §2), written from registers
  or by DMA (`../gemm/datapath.md` §5).
- **S_a·S_b staged in LDS once per workgroup**, in the prologue, at some or all Points of five files (k512, k2048,
  k8192, n8192, n32768), so the K loop reads scales with LDS loads only.
- **No K split at K ≤ 2048** except n4096_k1024 at m 4–16. K 4096 splits inside the workgroup at m ≤ 256; K 8192
  splits inside the workgroup at m ≤ 32 and 256–512, and across workgroups at m 64–256
  (`../gemm/grid_pricing.md` §2).
- **Configuration per m from a table**, compiled in an `lru_cache` function on a Point's first call (`../SKILL.md`,
  Stop table); XCD-aware tile order in five files (`../gemm/schedule_and_locality.md` §2).

Per file (functions are in the `kernel.py` each schema embeds):

**n4096_k256** (`_pick_tiles`) — the whole K = 256 in one tile.
- Tile 16×16 with one wave (m ≤ 16), 16×64 (32, 64), 32×64 (128, 256), 32×128 (512, 1024), 64×128 (2048, 4096)
  with four waves; 128 to 2048 workgroups.
- MFMA roles swapped (weights are the A operand): a lane holds 4 adjacent output columns.
- A straight into registers at m ≤ 4, else written from registers into XOR-swizzled LDS. With 128-column tiles,
  S_a is staged in LDS behind the A tile and S_b read by scalar loads.
- Stores: m ≤ 64 direct 8-byte stores of 4 columns; above, C transposed through XOR-swizzled LDS (its own LDS tile
  at m 512, 1024) and written as row-contiguous 16-byte stores; cache policy sc0|sc1 up to m = 1024, nt above
  (`../gemm/epilogue.md` §3, §5). From m = 2048 consecutive workgroups take adjacent column tiles.

**n4096_k512** (`_pick_tiles`, `_pick_g3_opts`) — no K split at any m.
- m ≤ 64: 16×64 tile holding all of K in one LDS stage (64, 128, 256 workgroups at m ≤ 16, 32, 64). m = 128:
  32×64, two 256-deep stages. m 256, 512: 32×128, every A K tile DMA'd (`BufferCopyLDS128b`) into its own LDS
  buffer in the prologue, B in 4 register stages. m 1024–4096: 64×128 the same way; at m 2048 and 4096 each
  workgroup walks 2 and 4 adjacent 128-column subtiles over its resident A and issues the next subtile's first MFMA
  before this subtile's epilogue. 512 workgroups from m = 512.
- S_a·S_b pre-multiplied into an LDS table in the prologue at m ≥ 4 (m 1, 2 load S_a per tile).
- Standard MFMA orientation. Stores: 2-byte on the register path (m ≤ 128); on the DMA path the bf16 subtile goes
  row-major through XOR-swizzled LDS and out as 16-byte stores with sc1.

**n4096_k1024** (`_select_tiles`; `_compile_wave_splitk_gemm` at m 4, 8, 16).
- m 4, 8, 16: a 16×16 tile per workgroup, 8 waves, one K block per wave, A and B straight into registers; wave 0
  sums the 8 partials from LDS in K-block order. 256 workgroups. m 4, 8 swap the MFMA operands for 8-byte stores;
  m = 16 stores 2 bytes per value.
- Other m: four waves along N, A written from registers into a 2-stage XOR-swizzled LDS ping-pong (async DMA,
  `BufferLoadAsyncLDS128b`, at m = 256), B into registers double-buffered, FlyDSL `preshuffle_gemm`-style
  MFMA/VMEM/DS interleave hints (`../gemm/schedule_and_locality.md` §3,
  `../flydsl/gemm.md` § mainloop §8). Tiles 16×64×256 (m ≤ 32), 16×128×256 (64, 128), 32×128×256 (256),
  64×128×256 (512), 64×256×128 (m ≥ 1024); 64 to 1024 workgroups; 2-byte stores.

**n4096_k2048** (`_choose_tiles`) — no K split.
- Tiles 16×64×512 (m ≤ 64), 32×64×512 (128, 256), 64×64×512 (512), 64×128×256 (1024), 128×128×128 (2048,
  4096); four waves along N; 64 to 1024 workgroups.
- A written from registers into a 2-stage XOR-swizzled LDS ping-pong, B into registers; S_a·S_b for all 16 K
  blocks in LDS; interleave hints at m ≤ 1024. At 32 ≤ m ≤ 1024 the N tiles run along grid x, so workgroups
  sharing a B panel land on one XCD.
- Epilogue: lane pairs swap bf16 halves with one DPP `quad_perm`, then each lane stores 4 bytes (2 adjacent
  columns) per row (`../gemm/epilogue.md` §3).

**n4096_k4096** (`_pick_config`, `_SK2_CFG_SMALL`, `_SK2_CFG`, `_pick_large_config`) — every path issues the MFMA
transposed and stores 8 bytes (4 adjacent columns).
- m ≤ 32: 16×16 (m ≤ 16) or 16×32 (32) workgroup tile, A and B straight into registers, K split over 8 waves
  (m 1, 2) or 4 (m 4–32); tile i is reduced by wave i mod KS from the others' LDS slots in a fixed order.
- m 64–256: two independent halves of K per workgroup, each with its own async-DMA LDS ring for A and B
  (`BufferLoadAsyncLDS128b`); the halves hand partials over through the then-idle B ring. Tiles 32×32, 32×64,
  64×64.
- m ≥ 512: 64×128 (512) or 128×128 tile, A and B DMA'd into a 5-, 3-, 2-, 2-stage LDS ring (m 512, 1024, 2048,
  4096), A XOR-swizzled on the global side so each lane's DMA slot stays linear; at m 2048 and 4096 each XCD gets a
  compact patch of tiles.
- 256 workgroups up to m = 1024, 512 at 2048, 1024 at 4096.

**n4096_k8192** (`_pick_tiles`, `_compile_skinny` at m ≤ 32).
- m ≤ 32: 16×16 or 32×16 tile, four waves each taking 16 contiguous K blocks; A by async DMA
  (`RawPtrBufferLoadAsyncLdsOp`) into a wave-private swizzled LDS ring at m ≥ 2, B into registers; the waves'
  partials summed from LDS in wave order. 256 workgroups.
- m 64, 128: 64×64 / 64×128 tiles with split-K 4 across workgroups (Counter-signals). m 256: split-K 2 with two
  4-wave groups per workgroup taking alternate K blocks of the workgroup's share, summed through LDS. m 512: the two groups
  without the cross-workgroup split. m 1024: 64×128 unsplit; m 2048, 4096: 64×256, its two 128-column halves
  sharing the A reads.
- Main body: A written from registers into 2-stage LDS, B in a register ring, S_a·S_b in LDS, and a block order
  that gives each XCD contiguous (N tile, split) panels so all K slices of a tile share one L2.

**n8192_k1024** (`_pick_tiles`) — no K split; four waves.
- Tiles 32×64 (m ≤ 64), 64×64 (128), 64×128 (256, 512), 128×128 (m ≥ 1024); 128 to 2048 workgroups.
- MFMA transposed at m ≤ 64 and m ≥ 256 (a lane's 4 adjacent columns share one combined scale); standard at
  m = 128. At m ≤ 16 only the first 16-row fragment is computed.
- A from registers into LDS at m ≤ 64; from m = 128 async DMA into a 2- or 3-slot LDS ring whose rows are permuted
  and XOR-swizzled; B into registers, two K blocks ahead below m = 1024, one from m = 1024.
- Combined S_a·S_b per (K block, row) staged in LDS in the prologue; XCD-aware block order. Stores: m ≥ 256
  through an XOR-swizzled LDS epilogue as 16-byte row stores with nt; m ≤ 64 4 adjacent bf16 per lane directly;
  m = 128 2 bytes per value.

**n32768_k1024** (`_skinny_cfg`, `_pick_config`, `_Runner`) — no K split.
- m ≤ 64: weights as the MFMA A operand loaded straight into registers; the activation slab (all of K for 16, 32 or
  64 rows) goes through LDS once with a row stride padded by 16 bytes, behind one barrier; each wave owns one
  16-column tile over all of K.
- m = 128: 8 waves, 128×128 tile, 3-stage LDS ring; the two wave groups alternate, one issuing loads and LDS reads
  while the other runs its MFMAs at raised priority (`s_setprio`). m = 256: 8 waves, 128×128 with the operands
  swapped; m ≥ 512: 4 waves, 256×128 swapped, two workgroups per CU.
- From m = 128 both operands are DMA'd into LDS (`BufferCopyLDS128b`): activations XOR-swizzled, weights in their
  preshuffled order. S_a·S_b for the workgroup tile is folded into an LDS table at kernel start. Swapped tiles pair
  adjacent tiles across lanes (`permlane16_swap`) into 16-byte stores of 8 columns
  (`../gemm/epilogue.md` §3).

## Result

Each file is the only schema at its Points, so the faster of AKLR's served kernel and aiter is aiter, and the
`speedup` CI wrote is the yardstick: a one-sided lower bound (α = 0.05) of the aiter / kernel time ratio over 32
rounds of CUDA-graph replays with L2 flushed before each call (`aklr/core/validation/stages/_measure.py`). A Point is
served from 1.03.

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

| File | m 1–64 | m 128–512 | m 1024–4096 | Served |
|---|---|---|---|---|
| n4096_k256 | 1.034–1.050 *(field, E1)* | 1.043–1.156 *(field, E1)* | 1.291–1.849 *(field, E1)* | 13 of 13 |
| n4096_k512 | 0.986–0.999 *(field, E2)* | 0.973 (128), 1.006 (256), 1.066 (512) *(field, E2)* | 1.171–1.531 *(field, E2)* | 4 of 13 |
| n4096_k1024 | 0.982–0.994 (1, 2), 1.064–1.081 (4–16), 1.018 (32), 0.943 (64) *(field, E3)* | 0.816–1.002 *(field, E3)* | 0.916 (1024), 1.096–1.125 (2048, 4096) *(field, E3)* | 5 of 13 |
| n4096_k2048 | 1.124–1.214 *(field, E4)* | 1.146–1.293 *(field, E4)* | 1.064–1.109 *(field, E4)* | 13 of 13 |
| n4096_k4096 | 1.451–2.250 *(field, E5)* | 1.204–1.416 *(field, E5)* | 1.148–1.414 *(field, E5)* | 13 of 13 |
| n4096_k8192 | 1.659–2.557 *(field, E6)* | 1.396–1.489 *(field, E6)* | 1.094–1.269 *(field, E6)* | 13 of 13 |
| n8192_k1024 | 1.398–1.618 *(field, E7)* | 1.378–1.489 *(field, E7)* | 1.797–1.892 *(field, E7)* | 13 of 13 |
| n32768_k1024 | 1.272–1.443 (1–32), 1.015 (64) *(field, E8)* | 1.593–1.996 *(field, E8)* | 1.796–2.229 *(field, E8)* | 12 of 13 |

86 of the 104 Points are served; all 104 pass accuracy (`accuracy: passed` in each workload).

## Reproduce

On an MI355X, in a checkout that holds these files with these numbers (main at 0cef4f2e3), one schema at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024/e5d65066d6572075.json
```

and the same for the other seven original CI schema paths under `sources` (N4096 across K and
N8192/N32768 K1024), not the three local additions. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands). Add `--base origin/main`
only on a branch that edits the file: with the base set, bench measures only schemas that are not at the base, so
on an unchanged checkout it skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`).
The local result does not count; CI's measurement is the one dispatch reads.

## Rollback

Delete the schema file (in a PR): its Points go unserved, since no other file serves them. `AKLR_DISABLE`
takes a whole contract id and would also drop every other `gemm_a8w8_blockwise_scaled@1.0` kernel. Points CI
measures below 1.03 are not served without any rollback (9 Points of n4096_k512, 8 of n4096_k1024, 1 of
n32768_k1024 now). Delete n4096_k8192's file, or give its m 64 and 128 Points a fixed-order reduction, if the
contract's `compare` becomes `bitwise`.

## Local large-N and short-K implementations

### Shared mechanism and shape boundaries

These implementations extend the same `gemm_a8w8_blockwise_scaled@1.0` / `b_shuffled` contract:
FP8 e4m3fn A and AITER 16×16-preshuffled B, K-major 128-wide scales, transposed B, no bias,
BF16 output. A-scale addressing remains `kb*M + row`, not row-major. All entries write the
caller-provided output. Row-major A-scale and small-N / long-K wave splitting remain separate cases.

**At large N with K1536, reuse the activation slab while streaming preshuffled weights into registers.**
The N16384 and N65536 bodies pass A through LDS with a padded K+16-byte row stride and use weights
as the MFMA A operand. This follows the large-N skinny path above, but the loading window, slab
partitioning and output staging are shape-specific; one configuration is not a rule for all large N.

- **N65536 K1536** (`e5f9fd89c5a994fa`, S9): BM16/BN64 with an eight-block weight window at
  M1/2/4/8/16; BM32/BN128 with a four-block window at M32. The wider output tile does not imply a
  larger gain: the baseline lower bound is **1.094x at M16 but 1.035x at M32** *(field, E10)*.
  This is a modest-margin implementation, not evidence that `B_CPOL=2` is generally superior.
- **N16384 K1536** (`ef050f07296547ba`, S10): BM16/BN64 at M1/2/4/16 and BM32/BN64 at M32;
  the larger-row paths partition the A slab along K, and M32 stages C through LDS. The complete-call
  lower bound reaches **1.966x at M16**, compared with **1.395x at M32** *(field, E11)*.
  Cache-policy and slab-part settings differ from N65536; the evidence supports this whole kernel,
  not an isolated benefit from either setting. M8 is outside the recorded coverage.

**Short K384 uses narrow direct paths at small M.**
N7168 K384 (`2c481a926208c5cd`, S11) processes just three scale blocks: a 16×16 / one-wave tile
at M1/2/4/8/16, 16×64 / four waves at M32/64 and 32×64 / four waves at M128. A is direct at
M1/2/4; C remains direct through M64 and is LDS-staged at M128. The lower bound is **1.154x at
M128**, but only **1.020x at M2**; **M8's 1.031x** is barely above the recorded threshold
*(field, E12)*. These displayed ratios are rounded; the threshold comparison uses the original
report's full-precision lower bound. Short K does not guarantee a useful margin at every small M. M256 is not covered.

These are local baseline-only comparisons, not improvements over an incumbent AKLR kernel or evidence
of deployed selection. The exact tested Points are M1/2/4/8/16/32 for N65536,
M1/2/4/16/32 for N16384, and M1/2/4/8/16/32/64/128 for N7168; they are not continuous ranges.
Bounded accuracy is not bitwise reproducibility or a guarantee across arbitrary inputs and toolchains.
There is no component ablation proving cache, staging, tile, occupancy or compiler causality.

### Reproduce against the recorded baseline

Use the UUID's JSON path in `sources`, gfx950, FlyDSL 0.3.2, AKLR main `a12866b` and AITER
`dbd8bf5bd624120197a7a26780a8c72201824f0f`. The denominator is
`aklr.ops.gemm.baseline.gemm_a8w8_blockwise_scaled_baseline`.
**The AITER commit alone does not fix that denominator:** restore both CSV byte strings from the
baseline configuration receipt and verify AITER's module-resolved paths under `/tmp/aiter_configs/`
(E13). Keep the original reports, source fingerprints and per-Point paired-round counts (E10–E13),
not the CI section's assumed measurement conditions or a mean over M.

For each local JSON, run `python -m aklr export check`, then:

```sh
python -m aklr validate --base origin/main --stages audit,bench --devices 0 <implementation-json-path>
```

CI remains authoritative ([AKLR operations](../../../../aklr/ops/README.md)); local reports are
machine-local evidence, not durable CI artifacts. To roll back, remove only the affected UUID's JSON,
leaving prior implementations and the contract intact. Do not hand-fill accuracy or performance fields.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` source a schema embeds in `kernel.sources`.

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 1.034–1.849 per Point, 13 of 13 ≥ 1.03; m 1–64 1.034–1.050, m 128 1.043, m 256 1.052, m 512 1.156, m 1024–4096 1.291–1.849 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256, m 1–4096; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 0.973–1.531 per Point, 4 of 13 ≥ 1.03; m 1–64 0.986–0.999, m 128 0.973, m 256 1.006, m 512 1.066, m 1024–4096 1.171–1.531 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512, m 1–4096; 5571f4589, CI run 40613440602)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 0.816–1.125 per Point, 5 of 13 ≥ 1.03; m 1 0.994, 2 0.982, 4 1.064, 8 1.081, 16 1.066, 32 1.018, 64 0.943, 128 0.816, 256 0.860, 512 1.002, 1024 0.916, 2048 1.096, 4096 1.125 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024, m 1–4096; 5571f4589, CI run 40613440602)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 1.064–1.293 per Point, 13 of 13 ≥ 1.03; m 1–64 1.124–1.214, m 128–512 1.146–1.293, m 1024–4096 1.064–1.109 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k2048, m 1–4096; 102877ca0, CI run 40154665266)
- E5 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 1.148–2.250 per Point, 13 of 13 ≥ 1.03; m 1–64 1.451–2.250, m 128–512 1.204–1.416, m 1024–4096 1.148–1.414 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096, m 1–4096; 5571f4589, CI run 40613440602)
- E6 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 1.094–2.557 per Point, 13 of 13 ≥ 1.03; m 1–64 1.659–2.557, m 128–512 1.396–1.489, m 1024–4096 1.094–1.269 (gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192, m 1–4096; 102877ca0, CI run 40154665266)
- E7 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 1.378–1.892 per Point, 13 of 13 ≥ 1.03; m 1–64 1.398–1.618, m 128–512 1.378–1.489, m 1024–4096 1.797–1.892 (gemm_a8w8_blockwise_scaled_b_shuffled_n8192_k1024, m 1–4096; 5571f4589, CI run 40613440602)
- E8 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter 1.015–2.229 per Point, 12 of 13 ≥ 1.03; m 1–32 1.272–1.443, m 64 1.015, m 128–512 1.593–1.996, m 1024–4096 1.796–2.229 (gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024, m 1–4096; 5571f4589, CI run 40613440602)
- E9 [2026-10, gfx950, FlyDSL ?, N=5, AKLR CI; FlyDSL of the runs: not recorded, 0.3.1, then 0.3.4.1 three times] near-threshold Points across the five committed measurements, in commit order: n4096_k1024 m 32 1.014, 1.023, 0.999, 1.030, 1.018 and m 512 1.062, 1.057, 1.018, 0.990, 1.002; n4096_k256 m 8 1.006, 1.024, 1.025, 1.029, 1.047; n4096_k512 m 256 1.084, 1.053, 1.007, 0.995, 1.006; n32768_k1024 m 64 1.008, 1.071, 1.017, 1.020, 1.015 (those four schemas; baa858028, cc6e3b96c, cca5b12df, 102877ca0, 5571f4589)
- S1 aklr/ops/gemm/contract.py:200 the contract notes that k8192's split-K order follows CTA arrival; compare `bounded` at :281
- S2 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json#kernel.py:962 `finish` sums its own partial, then peers (bid_z + d) & (split_k − 1) at :968; counter zeroed once at :1399, decremented by the last arriver at :155
- S3 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json#kernel.py:234 `_pick_tiles`: split 4 at m 64 and 128, split 2 with two groups at 256, two groups at 512
- S4 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k1024/e5d65066d6572075.json#kernel.py:638 m 4, 8, 16 take `_compile_wave_splitk_gemm`; wave 0 sums the K blocks in order at :596
- S5 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k512/9a8319559bcb2f29.json#kernel.py:771 `_pick_tiles`; the kernel has no split-K parameter; one LDS buffer per K tile on the DMA path at :143
- S6 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k256/5fcd9f0ebdced65c.json#kernel.py:101 `tile_k = K`
- S7 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096/8c907984278eefc4.json#kernel.py:304 owner-wave reduction; half exchange through the B ring at :1092
- S8 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n32768_k1024/42ffbb4ee1697e03.json#kernel.py:1139 skinny kernel; path choice in `_Runner` at :1415
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=16, local AKLR audit/bench graph timing per Point, paired baseline comparison] N65536 K1536, e5f9fd89c5a994fa: M1/2/4/8/16/32; zero recorded violation and baseline lower bounds ≥1.03 at all six; representative M16/M32 ratios rounded in the summary. Full-precision results remain in the report and mapping (local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn65536_k1536.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/point_verdicts.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/new_fp8_deliveries.json).
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=16, local AKLR audit/bench graph timing per Point, paired baseline comparison] N16384 K1536, ef050f07296547ba: M1/2/4/16/32, not M8; zero recorded violation and baseline lower bounds ≥1.03 at all five. Full-precision results remain in the report and mapping (local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn16384_k1536.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/point_verdicts.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/new_fp8_deliveries.json).
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=16–32, local AKLR audit/bench graph timing per Point, paired baseline comparison] N7168 K384, 2c481a926208c5cd: M1/2/4/8/16/32/64/128, not M256; 32 rounds at M1/8/16, 16 at the other Points; zero recorded violation at all eight, baseline lower bounds ≥1.03 only at M8/64/128. M8 is near threshold using the full-precision report, not its rounded display. Full-precision results remain in the report and mapping (local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/fn7168_k384.json; local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/point_verdicts.json).
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=16–32, local paired baseline configuration receipt] The denominator uses AITER dbd8bf5bd624120197a7a26780a8c72201824f0f and AKLR main a12866b. The receipt archives both CSV byte strings and module-resolved paths under /tmp/aiter_configs/: a8w8_blockscale_bpreshuffle_tuned_gemm.csv SHA256 53c4d8057ce42ede156990613d45b313e180d04a79f3ac2b156108393dab3fa2; the other blockscale table, identified by its receipt path, SHA256 6bc517bbd0e368c5f3088451af686d437881b1d9c84cc90d578ae1b62bb6d988 (local:/workspace/code/gpt_oss_docker/sync/inference3/postprocess_1107/baseline_config_receipt.json).
- S9 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n65536_k1536/e5f9fd89c5a994fa.json#kernel.py:85 slab/register kernel; `_skinny_cfg` at :284, B_CPOL at :292, caller-output entry at :301. Embedded kernel.py UTF-8 SHA256: 7509a1998934b84a9eeca0389fcec5f93755645049faf47a5bcbe8aae192d74b.
- S10 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n16384_k1536/ef050f07296547ba.json#kernel.py:218 slab-part staging; scaled partial accumulation at :274, `_skinny_cfg` at :343, caller-output entry at :393. Embedded kernel.py UTF-8 SHA256: 23229cdc65fff88f27391d95e57ad391e47931b92642ad848144e9a976215d5f.
- S11 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n7168_k384/2c481a926208c5cd.json#kernel.py:389 `_pick_config`; caller-output entry at :421. Embedded kernel.py UTF-8 SHA256: 227290c056d00f00965ce1701201dd102e9eca9d9b9bd6a23627d2379a2f5e2b.
