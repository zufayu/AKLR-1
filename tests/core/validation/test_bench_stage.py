# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema import save_json_file
from aklr.core.validation.pipeline import PipelineReport, PlatformRecord
from aklr.core.validation.stages._bench import _stage
from aklr.core.validation.stages._bench._evaluation import (
    CandidateResult,
    CandidateWorkloadResult,
    ReferenceResult,
    ReferenceWorkloadResult,
    TrialRequest,
)
from aklr.core.validation.stages._bench._execution import ExecutionContext
from aklr.core.validation.stages._measure import RoundTimes
from aklr.core.validation.stages.validation_scope import ValidationScope
from tests.op_case import GEMM_KERNEL, OpCase


def _candidate(context: ExecutionContext, *, requests: tuple[TrialRequest, ...]) -> dict:
    assert context.devices == ("cpu",)
    (context.directory / "evidence").write_text(str(os.getpid()))
    Path(requests[0].schema_path).with_suffix(".candidate-dir").write_text(str(context.directory))
    result = CandidateResult(
        findings=(),
        workload_results=tuple(
            CandidateWorkloadResult(
                axes_text=request.axes_text,
                findings=(),
                rounds=tuple(
                    RoundTimes(
                        mode="graph",
                        round_index=i,
                        order=("baseline", "candidate"),
                        milliseconds={"baseline": 2.0, "candidate": 1.0},
                    )
                    for i in range(2)
                ),
                baseline="test::baseline",
                backend="test",
            )
            for request in requests
        ),
    )
    return TypeAdapter(CandidateResult).dump_python(result, mode="json")


def _reference(context: ExecutionContext, *, requests: tuple[TrialRequest, ...], candidate_dir: Path) -> dict:
    assert context.devices == ("cpu",)
    assert int((candidate_dir / "evidence").read_text()) != os.getpid()
    Path(requests[0].schema_path).with_suffix(".reference-dir").write_text(str(context.directory))
    result = ReferenceResult(
        workload_results=tuple(
            ReferenceWorkloadResult(
                axes_text=request.axes_text, findings=(), is_compared=True, is_complete=True
            )
            for request in requests
        )
    )
    return TypeAdapter(ReferenceResult).dump_python(result, mode="json")


def _failed_reference(context: ExecutionContext, **kwargs: object) -> dict:
    _reference(context, **kwargs)
    raise RuntimeError("reference unavailable")


@pytest.mark.parametrize("reference_fails", [False, True])
def test_stage_uses_process_results_and_cleans_task_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_fails: bool
) -> None:
    with monkeypatch.context() as setup:
        setup.setattr(DeviceTarget, "current", lambda: DeviceTarget(arch="gfx950", hardware="MI355X"))
        schema = OpCase.of("gemm_bf16_nt").served(GEMM_KERNEL, evaluations=[])
    path = tmp_path / "aklr/ops/gemm/impls/triton" / schema.name / f"{schema.uuid}.json"
    save_json_file(schema, path)
    scope = ValidationScope(tmp_path, "head", None, (path,), (path,))
    request = TrialRequest.of_workload(
        str(path),
        schema,
        schema.workloads[0],
        thresholds={"baseline": "entry"},
        incumbent=None,
        baseline=None,
    )
    stage = _stage.BenchStage(["cpu"])
    monkeypatch.setattr(stage, "_workloads_to_run", lambda *args: [(request,)])
    monkeypatch.setattr(_stage, "run_candidate", _candidate)
    monkeypatch.setattr(_stage, "run_reference", _failed_reference if reference_fails else _reference)
    earlier = PipelineReport(
        head_sha="head",
        base_sha=None,
        measurement_profile={},
        platform=PlatformRecord.current(),
        schema_paths=(scope.relative_path(path),),
        bench_paths=(scope.relative_path(path),),
        stages=(),
    )

    report = stage.run(scope, earlier)

    assert report.status == ("aborted" if reference_fails else "passed")
    (record,) = report.workload_records
    assert record.batch_stopped == reference_fails
    if reference_fails:
        assert {finding.code for finding in report.findings} == {"platform_failure"}
        assert "reference unavailable" in report.findings[0].detail
    else:
        assert record.is_compared
        assert record.speedup == 2.0
        assert (
            report.measurements(scope.relative_path(path), schema, "head", "gfx950/MI355X")[request.axes_text]
            .speedup["test"]
            .value
            == 2.0
        )
    for suffix in (".candidate-dir", ".reference-dir"):
        directory = Path(path.with_suffix(suffix).read_text())
        assert not directory.exists()
        assert not directory.parent.exists()
