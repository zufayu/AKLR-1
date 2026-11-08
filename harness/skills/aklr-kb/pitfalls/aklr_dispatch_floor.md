---
title: "Profiled AKLR timing can mask differences between short kernels"
kind: pitfall
scope: [aklr validate and AKLR CI timing, any kernel or baseline shorter than about 4 µs]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=3
sources:
  - "AMD-AGI/AKLR@a29581b2d:aklr/core/validation/stages/_bench/_execution/_runner.py:475"
---

# AKLR's dispatch floor

## Symptom

A kernel and its baseline that differ a lot read about the same, both between 4.0 and 4.6 µs. A 1.7 µs bmm against
a 2.6 µs baseline read 4.09 against 4.10 µs in `aklr validate` and was demoted as `not_faster`
*(field, E2)*. An empty kernel (a one-element `zero_`) reads 4.19 µs on AKLR's graph timer and 1.70 µs replayed
back to back in a graph without profiling *(field, E1)*. A kernel several times longer than the floor is not held
to it.

## Root cause

In the recorded AKLR version, the bench runner set `GPU_FORCE_QUEUE_PROFILING=1` (S1).
The listed short-kernel comparisons read about 4.1 µs under that configuration. In one back-to-back
CUDA-event experiment with the runner still forcing queue profiling, readings remained 4.1–4.3 µs,
one kernel per call *(field, E3)*. These observations can explain masked differences in those runs;
they do not establish a universal per-dispatch minimum or the behavior of a different runner.

## How to detect

- Read `baseline.json` (forge_task) or a validate report: a baseline at 4–5 µs warrants a ruler check,
  not an automatic floor classification.
- Time the call outside AKLR, in a process without `GPU_FORCE_QUEUE_PROFILING`: replay it back to back in one CUDA
  graph, with CUDA events, the call count taken from a short graph rather than an eager call. If both sides drop
  well below 4 µs, the two measurement configurations differ; that alone does not isolate the cause or
  establish a universal floor.

## Fix or workaround

- Before a campaign, flag baselines near the historical 4–5 µs readings for a ruler check. Use paired
  measurements under the current AKLR runner to decide whether a gain is distinguishable; proximity alone
  does not predict CI's result or dispatch selection.
- Report an external-timer gain with that ruler, not as an AKLR speedup unless AKLR measured it.

## Scope

AKLR validate and CI at `AMD-AGI/AKLR@a29581b2d`, on MI355X. Kernels and baselines longer than about 5 µs are timed
as before. What a deployment's graph pays per call is the back-to-back time, not AKLR's reading.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=3, AKLR graph timer vs back-to-back graph replay with CUDA events] K3 top-k start kernel 4.27 vs 3.44 µs, aiter top-k 12.05 vs 10.54 µs, one-element `zero_` 4.19 vs 1.70 µs, at n_tokens=8 (local:/shared_nfs/kyle/moe/k3six/tasks2/probe_floor2.out)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=3, aklr validate] bmm_n128_k512 m=8: 4.09 vs 4.10 µs with the runner forcing queue profiling; 1.70 vs 2.59 µs, ratio lower bound 1.480, with the runner not forcing it (local:/shared_nfs/kyle/moe/k3six/deliver/fv3.log:41, local:/shared_nfs/kyle/moe/k3six/tasks2/step66.out:9)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3, aklr validate with back-to-back timing, kernels of the timed graph from Kineto] one `bmm_n128_k512_tiled_0` per call, the readings still 4.1-4.3 µs (local:/shared_nfs/kyle/moe/k3six/tasks2/step65.out:4)
- S1 AMD-AGI/AKLR@a29581b2d:aklr/core/validation/stages/_bench/_execution/_runner.py:475 `GPU_FORCE_QUEUE_PROFILING="1"`
