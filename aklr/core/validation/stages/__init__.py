###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from aklr.core.validation.stages._audit._stage import AuditStage
from aklr.core.validation.stages._bench import BenchReport, BenchStage, RoundTimer
from aklr.core.validation.stages._report._markdown import (
    Block,
    Heading,
    Paragraph,
    Table,
    escape,
)
from aklr.core.validation.stages.stage import PipelineStage, Progress, StageReport
from aklr.core.validation.stages.validation_scope import ValidationScope

STAGE_ORDER = tuple(stage.name for stage in (AuditStage, BenchStage))

__all__ = [
    "STAGE_ORDER", "AuditStage", "BenchReport", "BenchStage", "Block", "Heading", "Paragraph",
    "PipelineStage", "Progress", "StageReport", "Table", "RoundTimer", "ValidationScope", "escape",
]  # fmt: skip
