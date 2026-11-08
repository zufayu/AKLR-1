# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import threading
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from aklr.core.validation.stages._bench._execution import (
    ExecutionContext,
    Executor,
    NoResourceAvailable,
    ResourceRequest,
    TaskCancelled,
    TaskExecutionFailed,
    TaskExecutionTimeout,
)
from aklr.core.validation.stages._bench._execution._model import ResourceHealth, ResourceState
from aklr.core.validation.stages._bench._execution._resources import LocalAcceleratorCatalog


def _devices(context: ExecutionContext) -> dict:
    return {"device": context.devices[0], "devices": list(context.devices), "pid": os.getpid()}


class TestResourceGroups:
    @pytest.mark.parametrize("count", [0, -1, 1.5, True])
    def test_invalid_counts_are_rejected(self, count: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            ResourceRequest(count=count)

    @pytest.mark.parametrize("devices", [["cuda:0", "cuda:0"], ["cuda:0", "cuda:00"]])
    def test_duplicate_gpus_cannot_be_leased_as_distinct_resources(self, devices: list[str]) -> None:
        with pytest.raises(ValueError, match="distinct"):
            LocalAcceleratorCatalog(devices)

    def test_group_acquisition_is_atomic_and_stale_release_is_harmless(self) -> None:
        catalog = LocalAcceleratorCatalog(["cuda:0", "cuda:1", "cuda:2"])
        request = ResourceRequest(backend="cuda", count=2)
        first, second, third = catalog.matching(request)
        occupied = catalog.acquire(second.resource_id, "occupied")
        assert catalog.acquire((first.resource_id, second.resource_id), "blocked") is None
        assert catalog.matching(request)[0].state == ResourceState.AVAILABLE
        catalog.release(occupied, ResourceHealth.HEALTHY)
        group = catalog.acquire((first.resource_id, second.resource_id), "group")
        assert group.devices == ("cuda:0", "cuda:1")
        assert all(resource.state == ResourceState.LEASED for resource in group.resources)
        catalog.release(group, ResourceHealth.HEALTHY)
        replacement = catalog.acquire((second.resource_id, third.resource_id), "replacement")
        catalog.release(group, ResourceHealth.UNHEALTHY)
        assert [r.state for r in catalog.matching(request)] == [
            ResourceState.AVAILABLE,
            ResourceState.LEASED,
            ResourceState.LEASED,
        ]
        catalog.release(replacement, ResourceHealth.UNHEALTHY)
        assert [r.resource_id for r in catalog.matching(request)] == [first.resource_id]

    def test_insufficient_matching_resources_fail_without_running(self) -> None:
        with Executor.for_devices(["cpu", "cpu"], runner="thread") as executor:
            for request in [
                ResourceRequest(count=3),
                ResourceRequest(count=2, resource_ids=("0:cpu",)),
                ResourceRequest(backend="cuda"),
            ]:
                future = executor.submit(
                    lambda ctx: pytest.fail("must not run"), timeout_seconds=5, resource_request=request
                )
                with pytest.raises(NoResourceAvailable):
                    future.result(timeout=5)

    def test_waiting_group_does_not_hold_partial_resources(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def hold(context: ExecutionContext) -> None:
            entered.set()
            assert release.wait(5)

        with Executor.for_devices(["cpu"] * 3, runner="thread") as executor:
            first = executor.submit(
                hold, timeout_seconds=10, resource_request=ResourceRequest(resource_ids=("0:cpu",))
            )
            try:
                assert entered.wait(5)
                group = executor.submit(
                    _devices, timeout_seconds=5, resource_request=ResourceRequest(count=3)
                )
                # All other slots remain usable while the group waits for slot 0.
                smaller = executor.submit(
                    _devices,
                    timeout_seconds=5,
                    resource_request=ResourceRequest(count=2, resource_ids=("1:cpu", "2:cpu")),
                )
                assert len(smaller.result(timeout=5)["devices"]) == 2
                assert not group.done()
                assert group.resource_ids == ()
            finally:
                release.set()
            first.result(timeout=5)
            assert group.result(timeout=5)["devices"] == ["cpu"] * 3
            assert set(group.resource_ids) == {"0:cpu", "1:cpu", "2:cpu"}

    @pytest.mark.parametrize("runner", ["process", "thread"])
    def test_default_count_and_per_task_override_reach_the_runner(self, runner: str) -> None:
        with Executor.for_devices(["cpu"] * 2, runner=runner, resources_per_runner=2) as executor:
            group = executor.submit(_devices, timeout_seconds=10)
            result = group.result(timeout=60)
            assert result["devices"] == ["cpu", "cpu"]
            assert result["device"] == "cpu"
            assert group.devices == ("cpu", "cpu")
            assert len(group.resource_ids) == 2
            assert (result["pid"] == os.getpid()) == (runner == "thread")
            single = executor.submit(_devices, timeout_seconds=10, resource_request=ResourceRequest())
            assert single.result(timeout=60)["devices"] == ["cpu"]
            assert len(single.resource_ids) == 1


class TestThreadRunner:
    def test_shared_objects_and_parallel_threads_preserve_cwd(self) -> None:
        barrier = threading.Barrier(2)
        payload = object()
        cwd = Path.cwd()
        caller_thread = threading.get_ident()

        def task(context: ExecutionContext) -> object:
            assert threading.get_ident() != caller_thread
            assert Path.cwd() == cwd
            assert context.directory != cwd
            barrier.wait(timeout=5)
            return payload

        with Executor.for_devices(["cpu", "cpu"], runner="thread") as executor:
            futures = [executor.submit(task, timeout_seconds=10) for _ in range(2)]
            assert all(future.result(timeout=10) is payload for future in futures)
            assert futures[0].resource_ids != futures[1].resource_ids
        assert Path.cwd() == cwd

    def test_timeout_retains_resources_until_the_callable_returns(self) -> None:
        cancelled = threading.Event()
        release = threading.Event()

        def task(context: ExecutionContext) -> None:
            assert context.cancel_event.wait(5)
            cancelled.set()
            assert release.wait(5)

        with Executor.for_devices(["cpu"], runner="thread") as executor:
            future = executor.submit(task, timeout_seconds=0.1)
            try:
                assert cancelled.wait(5)
                next_task = executor.submit(_devices, timeout_seconds=5)
                assert not future.done()
                assert next_task.resource_ids == ()
            finally:
                release.set()
            with pytest.raises(TaskExecutionTimeout, match="exceeded"):
                future.result(timeout=5)
            assert not future.directory.exists()
            assert next_task.result(timeout=5)["device"] == "cpu"

    def test_shutdown_signals_cancellation_and_waits_for_task_cleanup(self) -> None:
        started = threading.Event()
        cancelled = threading.Event()
        release = threading.Event()

        def task(context: ExecutionContext) -> None:
            started.set()
            assert context.cancel_event.wait(5)
            cancelled.set()
            assert release.wait(5)

        with Executor.for_devices(["cpu"], runner="thread") as executor:
            future = executor.submit(task, timeout_seconds=10)
            try:
                assert started.wait(5)
                pending = executor.submit(_devices, timeout_seconds=5)
                executor.shutdown(wait=False)
                assert cancelled.wait(5)
                assert pending.cancelled()
                assert not future.done()
                assert future.directory.exists()
            finally:
                release.set()
            executor.close()
            with pytest.raises(TaskCancelled):
                future.result(timeout=5)
            assert not future.directory.parent.exists()

    def test_failure_includes_diagnostics_and_resources_are_reusable(self) -> None:
        def fail(context: ExecutionContext) -> None:
            context.log_path.write_text("task evidence\n")
            raise ValueError("broken task")

        with Executor.for_devices(["cpu"], runner="thread") as executor:
            future = executor.submit(fail, timeout_seconds=5)
            with pytest.raises(TaskExecutionFailed, match="task evidence") as caught:
                future.result(timeout=5)
            assert "broken task" in str(caught.value)
            assert not future.directory.exists()
            assert executor.submit(_devices, timeout_seconds=5).result(timeout=5)["device"] == "cpu"

    @pytest.mark.parametrize("fail_sync", [False, True])
    def test_gpu_group_is_synchronized_before_reuse_or_quarantined(
        self, monkeypatch: pytest.MonkeyPatch, fail_sync: bool
    ) -> None:
        selected, synchronized = [], []

        def select(device: str) -> object:
            selected.append(device)
            return nullcontext()

        def synchronize(device: str) -> None:
            synchronized.append(device)
            if fail_sync and device == "cuda:1":
                raise RuntimeError("GPU synchronization failed")

        monkeypatch.setattr(torch.cuda, "device", select)
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
        with Executor.for_devices(["cuda:0", "cuda:1"], runner="thread", resources_per_runner=2) as executor:
            future = executor.submit(_devices, timeout_seconds=5)
            if fail_sync:
                with pytest.raises(TaskExecutionFailed, match="synchronization failed"):
                    future.result(timeout=5)
                with pytest.raises(NoResourceAvailable):
                    executor.submit(_devices, timeout_seconds=5).result(timeout=5)
            else:
                assert future.result(timeout=5)["devices"] == ["cuda:0", "cuda:1"]
            assert selected == ["cuda:0"]
            assert synchronized == ["cuda:0", "cuda:1"]
