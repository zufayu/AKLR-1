###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench stage: every candidate workload checked for correctness and timed against what it replaces."""

from __future__ import annotations

import signal
import tempfile
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.seam import SeamRegistry
from aklr.core.schema import Schema, load_json_file
from aklr.core.utils.testing import kernel_tracer
from aklr.core.validation.stages._bench._base_revision import BaseRevision
from aklr.core.validation.stages._bench._batch import Batch
from aklr.core.validation.stages._bench._case import BenchCase
from aklr.core.validation.stages._bench._error import BenchError
from aklr.core.validation.stages._bench._report import BenchReport
from aklr.core.validation.stages._bench._task_executor import TaskExecutor
from aklr.core.validation.stages._report._finding import Finding
from aklr.core.validation.stages.stage import PipelineStage

if TYPE_CHECKING:
    from aklr.core.validation.pipeline import PipelineReport
    from aklr.core.validation.stages.validation_scope import ValidationScope

__all__ = ["BenchStage"]


class BenchStage(PipelineStage):
    name = "bench"
    report_type = BenchReport

    def __init__(self, devices: Sequence[str]) -> None:
        devices = list(devices)
        if not devices:
            raise ValueError("the bench needs at least one device")
        if len(set(devices)) != len(devices):
            raise ValueError(f"devices {devices} name one device twice")
        for device in devices:
            try:
                torch.device(device)
            except RuntimeError as error:
                raise ValueError(f"{device!r} is not a device: {error}") from error
        self.devices = devices

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> BenchReport:
        stopped = next((stage for stage in earlier.stages if not stage.is_passed), None)
        if stopped is not None:
            return BenchReport.skipped(self.name, f"the previous stage {stopped.stage_name} {stopped.status}")
        if not scope.bench_paths:
            return BenchReport.skipped(self.name, "no schema to bench")

        import aklr.seam  # noqa: F401 -- declares the adapters

        min_speedup = RuntimeEnv().min_speedup
        target = DeviceTarget.current(self.devices[0])
        findings: list[Finding] = []
        batches: list[Batch] = []
        for path in scope.bench_paths:
            schema_path = scope.relative_path(path)
            schema = load_json_file(Schema, path)
            if schema.kernel is None or not schema.workloads:
                continue
            try:
                base = BaseRevision.of(scope, schema_path, schema, min_speedup)
            except BenchError as error:
                findings.append(Finding(error.code, schema_path, None, error.detail))
                continue
            has_adapter = bool(SeamRegistry.default().adapters(schema.op_type))
            cases = []
            for workload in schema.workloads:
                incumbent = base.select_incumbent(workload.axes, schema.kernel, target, None)
                has_rival = incumbent is not None and incumbent[1].kernel != schema.kernel
                cases.append(BenchCase(schema_path, schema, workload.axes, is_timed=has_adapter or has_rival))
            batches.append(Batch(cases, base))

        kernel_tracer()  # built here once: every task only loads it
        n_workloads = sum(len(batch.cases) for batch in batches)
        self.progress.write(
            f"{n_workloads} workloads of {len(batches)} schemas on {len(self.devices)} devices"
        )
        with tempfile.TemporaryDirectory(prefix="aklr-bench-") as workdir:
            executor = TaskExecutor(self.devices, Path(workdir))
            threads = ThreadPoolExecutor(len(self.devices), thread_name_prefix="bench")

            def interrupt(signal_number: int, frame: object) -> None:
                raise KeyboardInterrupt(f"signal {signal_number}")

            is_main = threading.current_thread() is threading.main_thread()
            previous = signal.signal(signal.SIGTERM, interrupt) if is_main else None
            try:
                # the largest first: the last batch to finish is a short one
                ordered = sorted(batches, key=lambda b: -len(b.cases))
                futures = {threads.submit(batch.run, executor): batch for batch in ordered}
                for done, future in enumerate(as_completed(futures), 1):
                    future.result()
                    batch = futures[future]
                    self.progress.write(
                        f"{done}/{len(batches)} {batch.schema_path}: {len(batch.findings)} findings"
                    )
            finally:
                executor.close()
                threads.shutdown(wait=True, cancel_futures=True)
                if is_main:
                    signal.signal(signal.SIGTERM, previous)
        for batch in batches:
            findings += batch.findings
        return BenchReport.of(findings, [record for batch in batches for record in batch.records])
