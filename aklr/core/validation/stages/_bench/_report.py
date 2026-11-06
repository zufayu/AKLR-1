###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The bench's report: each workload's record, the findings, and the evaluations they justify."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import Field
from scipy.stats import t as student_t

from aklr.core.schema import Evaluation, Schema, Speedup, frozen_dataclass
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages._report._markdown import (
    Block,
    BulletList,
    Table,
    axes_text,
    schema_short_name,
)
from aklr.core.validation.stages.stage import StageReport

__all__ = ["BenchReport", "TimingRecord", "WorkloadRecord", "CaseResult"]


@frozen_dataclass
class TimingRecord:
    """One comparison's paired rounds and statistics; a ratio is the other's time over the candidate's."""

    THRESHOLDS: ClassVar[dict[str, float]] = {"entry": 1.02, "retain": 0.99}

    mode: Literal["graph"] = Field(default="graph", kw_only=True)
    versus: Literal["baseline", "incumbent"]
    threshold: Literal["entry", "retain"]
    n_rounds: int
    log_ratio_mean: float
    ratio_lower_bound: float  # of the log ratios clamped to ±ln 2: what the threshold judges
    min_ratio: float
    versus_milliseconds: tuple[float, ...]
    candidate_milliseconds: tuple[float, ...]

    @classmethod
    def measured(
        cls,
        versus_name: Literal["baseline", "incumbent"],
        threshold: Literal["entry", "retain"],
        versus: Sequence[float],
        candidate: Sequence[float],
    ) -> TimingRecord:
        """The statistics of raw rounds; ValueError when they are too few, unpaired or not positive and finite."""
        times = [*versus, *candidate]
        if (
            len(versus) < 2
            or len(versus) != len(candidate)
            or not all(math.isfinite(t) and t > 0 for t in times)
        ):
            raise ValueError(f"{len(versus)} and {len(candidate)} rounds of {versus_name} are not usable")
        logs = [math.log(v / c) for v, c in zip(versus, candidate)]
        clamped = [min(max(x, -math.log(2)), math.log(2)) for x in logs]
        return cls(
            versus=versus_name,
            threshold=threshold,
            n_rounds=len(logs),
            log_ratio_mean=statistics.fmean(logs),
            ratio_lower_bound=cls.lower_bound(clamped),
            min_ratio=math.exp(min(logs)),
            versus_milliseconds=tuple(versus),
            candidate_milliseconds=tuple(candidate),
        )

    @staticmethod
    def lower_bound(logs: Sequence[float]) -> float:
        """The one-sided 95% lower bound of the geometric mean ratio, by Student's t."""
        n = len(logs)
        margin = student_t.ppf(0.95, n - 1) * statistics.stdev(logs) / math.sqrt(n)
        bound = math.exp(statistics.fmean(logs) - margin)
        if not math.isfinite(bound):
            raise ValueError(f"the lower bound of {list(logs)} is {bound}")
        return bound

    @property
    def is_met(self) -> bool:
        return self.ratio_lower_bound >= self.THRESHOLDS[self.threshold]

    def describe(self) -> str:
        return f"{self.threshold} vs {self.versus}: {self.ratio_lower_bound:.3f} < {self.THRESHOLDS[self.threshold]}"


@frozen_dataclass
class CaseResult:
    """What a task process says of one workload: the candidate fills what it measured, the reference what it judged.

    It crossed from a less trusted process: the parent re-derives every statistic from the raw rounds.
    """

    axes: dict[str, int]
    findings: tuple[Finding, ...] = ()
    batch_stopped: bool = False  # the finding stopped the whole batch: every result carries it
    backend: str | None = None
    baseline: str | None = None
    timings: tuple[TimingRecord, ...] = ()
    is_complete: bool = False  # every check left every output; not that they are right
    is_compared: bool = False  # some check was judged against a comparable reference


@frozen_dataclass
class WorkloadRecord:
    schema_path: str
    axes: dict[str, int]
    kernel_content_hash: str
    baseline: (
        str | None
    )  # what versus="baseline" timed: an adapter entry, an incumbent's base path, or nothing
    backend: str | None  # the adapter's backend that served this workload's comparison
    is_timed: bool  # the plan, not that timings exist
    is_compared: bool  # a reference judged its outputs; not that they passed
    timings: tuple[TimingRecord, ...]
    batch_stopped: bool

    @property
    def speedup(self) -> float | None:
        """Over the baseline, from the unclamped rounds, at 3 decimals; 1.0 untimed, None without the data."""
        timing = next((t for t in self.timings if t.versus == "baseline"), None)
        if timing is not None:
            logs = [
                math.log(v / c) for v, c in zip(timing.versus_milliseconds, timing.candidate_milliseconds)
            ]
            return max(round(TimingRecord.lower_bound(logs), 3), 0.001)
        return 1.0 if not self.is_timed and not self.batch_stopped else None

    def performance_finding(self) -> Finding | None:
        """regressed when a retain bound falls short, else not_faster when an entry bound does; all in one."""
        missed = [t for t in self.timings if not t.is_met]
        if not missed:
            return None
        code = (
            FindingCode.REGRESSED if any(t.threshold == "retain" for t in missed) else FindingCode.NOT_FASTER
        )
        missed.sort(key=lambda t: t.threshold != "retain")
        return Finding(code, self.schema_path, self.axes, "; ".join(t.describe() for t in missed))


@frozen_dataclass
class BenchReport(StageReport):
    """Deciding findings decide the stage; demotions only mark their workloads."""

    # a workload's own failure: it demotes that workload and leaves the stage to the schema's acceptance
    DEMOTABLE: ClassVar = frozenset(
        {
            FindingCode.NUMERIC_MISMATCH, FindingCode.POST_TIMING_MISMATCH, FindingCode.INPUT_MUTATION,
            FindingCode.CAPTURE_UNSAFE, FindingCode.BUILD_FAILED, FindingCode.CANDIDATE_CRASH,
            FindingCode.CANDIDATE_TIMEOUT, FindingCode.NOT_FASTER, FindingCode.REGRESSED,
        }
    )  # fmt: skip
    PERFORMANCE: ClassVar = frozenset({FindingCode.NOT_FASTER, FindingCode.REGRESSED})

    workload_records: tuple[WorkloadRecord, ...] = ()
    demotions: tuple[Finding, ...] = ()

    @classmethod
    def of(cls, findings: Sequence[Finding], records: Sequence[WorkloadRecord]) -> BenchReport:
        """Split findings, then add no_qualifying_workload to each schema no workload of passes outright."""
        recorded = {(r.schema_path, axes_text(r.axes)) for r in records}
        is_demotion = [
            f.code in cls.DEMOTABLE and f.axes is not None and (f.schema_path, axes_text(f.axes)) in recorded
            for f in findings
        ]
        demotions = [f for f, demoted in zip(findings, is_demotion) if demoted]
        deciding = [f for f, demoted in zip(findings, is_demotion) if not demoted]
        flagged = {(f.schema_path, axes_text(f.axes)) for f in findings if f.axes is not None}
        for path in dict.fromkeys(r.schema_path for r in records):
            is_qualified = any(
                r.schema_path == path
                and r.is_compared
                and not r.batch_stopped
                and (path, axes_text(r.axes)) not in flagged
                for r in records
            )
            if not is_qualified and not any(f.schema_path in (path, None) for f in deciding):
                detail = "no workload passed every check without a finding"
                deciding.append(Finding(FindingCode.NO_QUALIFYING_WORKLOAD, path, None, detail))
        return cls.from_findings(
            "bench", deciding, workload_records=tuple(records), demotions=tuple(demotions)
        )

    def _demotion_codes(self, path: str) -> dict[str, list[FindingCode]]:
        codes: dict[str, list[FindingCode]] = {}
        for f in self.demotions:
            if f.schema_path == path and f.axes is not None:
                codes.setdefault(axes_text(f.axes), []).append(FindingCode(f.code))
        return codes

    def measurements(
        self, path: str, schema: Schema, commit: str, hardware: str
    ) -> dict[str, Evaluation] | None:
        """Each workload's evaluation by axes text; None when the schema's results are not whole and trusted."""
        if any(f.schema_path == path and f.code != FindingCode.NO_QUALIFYING_WORKLOAD for f in self.findings):
            return None
        records = {axes_text(r.axes): r for r in self.workload_records if r.schema_path == path}
        demoted = self._demotion_codes(path)
        evaluations = {}
        for workload in schema.workloads:
            key = axes_text(workload.axes)
            record = records.get(key)
            if record is None:
                return None
            is_failed = any(code not in self.PERFORMANCE for code in demoted.get(key, []))
            speedup = record.speedup
            if not is_failed and (speedup is None or not record.is_compared):
                return None
            has_speedup = record.backend is not None and record.is_timed and speedup is not None
            evaluations[key] = Evaluation(
                hardware=hardware,
                accuracy="failed" if is_failed else "passed",
                speedup={record.backend: Speedup(commit=commit, value=speedup)} if has_speedup else {},
            )
        return evaluations

    def record_blocks(self) -> list[Block]:
        """Per schema: its workloads, accuracy ok and failed (batch stops not counted), and median speedup."""
        rows = []
        for path in dict.fromkeys(r.schema_path for r in self.workload_records):
            records = [r for r in self.workload_records if r.schema_path == path]
            demoted = self._demotion_codes(path)
            failed = [
                r
                for r in records
                if not r.batch_stopped
                and any(c not in self.PERFORMANCE for c in demoted.get(axes_text(r.axes), []))
            ]
            ok = [r for r in records if r not in failed and r.is_compared and not r.batch_stopped]
            speedups = [s for r in records if (s := r.speedup) is not None]
            median = f"{statistics.median(speedups):.3f}" if speedups else "-"
            rows.append((schema_short_name(path), str(len(records)), str(len(ok)), str(len(failed)), median))
        return [
            Table(("schema", "workloads", "accuracy ok", "accuracy failed", "median speedup"), tuple(rows)),
            BulletList(tuple(f"demoted: {f.markdown_item()}" for f in self.demotions)),
        ]
