# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from unittest.mock import Mock

import pytest

from aklr.core.validation.stages._bench._execution import (
    ExecutionContext,
    ExecutionError,
    ExecutionFuture,
    Executor,
    ExecutorClosed,
    ExecutorState,
    ResourceRequest,
    RunnerStartupFailed,
    TaskCancelled,
    TaskExecutionFailed,
    TaskExecutionTimeout,
    TaskState,
)
from aklr.core.validation.stages._bench._execution._backend import RunnerExecutionBackend
from aklr.core.validation.stages._bench._execution._engine import ExecutionEngine
from aklr.core.validation.stages._bench._execution._model import ResourceHealth, ResourceState, RunnerHealth
from aklr.core.validation.stages._bench._execution._resources import LocalAcceleratorCatalog
from aklr.core.validation.stages._bench._execution._runner import (
    DisposableExecutionRunner,
    ExecutionRunner,
    RunnerFactory,
)
from aklr.core.validation.stages._bench._execution._shutdown import ShutdownGate


def _describe(context: ExecutionContext, value: int = 7) -> dict:
    (context.directory / "evidence").write_text(str(value))
    return {"pid": os.getpid(), "device": context.devices[0], "cwd": os.getcwd(), "value": value}


def _fail(context: ExecutionContext) -> None:
    print("task diagnostic", flush=True)
    raise ValueError("task failed")


def _sleep(context: ExecutionContext) -> None:
    marker = context.directory / "starting"
    marker.write_text(str(os.getpid()))
    marker.rename(context.directory / "started")
    time.sleep(60)


def _runner(tmp_path: Path, name: str) -> Mock:
    runner = Mock(spec=ExecutionRunner)
    runner.directory = tmp_path / name
    runner.directory.mkdir()
    runner.execute.return_value = "ok"
    runner.health.return_value = RunnerHealth.HEALTHY
    runner.diagnostics.return_value = None
    return runner


def _executor(resources: LocalAcceleratorCatalog, runners: list[Mock]) -> Executor:
    factory = Mock(spec=RunnerFactory)
    factory.create.side_effect = runners
    return Executor(RunnerExecutionBackend(resources, factory, ExecutionEngine()))


class TestExecutionFuture:
    def test_cancellation_notifies_waiters_once(self) -> None:
        future = ExecutionFuture("cancelled")
        assert future.assign("0:cpu", "cpu")
        assert future.cancel()
        assert future.cancel()
        assert not future.mark_running(None)
        assert future.state == TaskState.CANCELLED
        assert wait([future], timeout=0).done == {future}
        assert list(as_completed([future], timeout=0)) == [future]

    @pytest.mark.parametrize("already_done", [False, True])
    def test_callbacks_run_off_the_resolving_thread(self, already_done: bool) -> None:
        future = ExecutionFuture("callback")
        notified = threading.Event()
        callback_threads = []

        def callback(done: ExecutionFuture) -> None:
            callback_threads.append(threading.get_ident())
            assert done.result() == 7
            notified.set()

        if already_done:
            future.set_result(7)
        future.add_done_callback(callback)
        if not already_done:
            future.set_result(7)
        assert notified.wait(5)
        assert len(callback_threads) == 1
        assert callback_threads[0] != threading.get_ident()


class TestShutdownGate:
    def test_concurrent_callers_wait_for_one_cleanup_and_share_its_error(self) -> None:
        gate = ShutdownGate("test-shutdown")
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def close() -> None:
            calls.append(1)
            entered.set()
            assert release.wait(5)
            gate.run(close)  # reentrant shutdown must not wait on itself
            raise RuntimeError("cleanup failed")

        gate.run(close, wait=False)
        assert entered.wait(5)
        with ThreadPoolExecutor(4) as pool:
            callers = [pool.submit(gate.run, close) for _ in range(4)]
            try:
                assert all(not caller.done() for caller in callers)
            finally:
                release.set()
            for caller in callers:
                with pytest.raises(RuntimeError, match="cleanup failed"):
                    caller.result(timeout=5)
        assert calls == [1]


class TestSchedulingLifecycle:
    @pytest.mark.parametrize("capacity", [3, 4])
    def test_group_startup_retries_only_if_a_whole_healthy_group_remains(
        self, tmp_path: Path, capacity: int
    ) -> None:
        resources = LocalAcceleratorCatalog([f"cuda:{i}" for i in range(capacity)])
        failed, replacement = [_runner(tmp_path, str(i)) for i in range(2)]
        failed.start.side_effect = RunnerStartupFailed("group unavailable")
        failed.health.return_value = RunnerHealth.RESOURCE_FAILED
        with _executor(resources, [failed, replacement]) as executor:
            future = executor.submit(_describe, timeout_seconds=1, resource_request=ResourceRequest(count=2))
            if capacity == 4:
                assert future.result(timeout=5) == "ok"
                assert future.devices == ("cuda:2", "cuda:3")
                replacement.close.assert_called_once()
            else:
                with pytest.raises(RunnerStartupFailed, match="group unavailable"):
                    future.result(timeout=5)
                replacement.start.assert_not_called()
            assert len(resources.matching(ResourceRequest())) == capacity - 2
            assert all(r.state == ResourceState.AVAILABLE for r in resources.matching(ResourceRequest()))
            failed.close.assert_called_once()

    @pytest.mark.parametrize("failures", [1, 2])
    def test_startup_retries_once_on_a_different_healthy_resource(
        self, tmp_path: Path, failures: int
    ) -> None:
        resources = LocalAcceleratorCatalog(["cpu", "cuda:0", "cuda:1"])
        runners = [_runner(tmp_path, str(i)) for i in range(2)]
        for runner in runners[:failures]:
            runner.start.side_effect = RunnerStartupFailed("device unavailable")
            runner.health.return_value = RunnerHealth.RESOURCE_FAILED
        with _executor(resources, runners) as executor:
            future = executor.submit(_describe, timeout_seconds=1)
            if failures == 1:
                assert future.result(timeout=5) == "ok"
                assert future.state == TaskState.SUCCEEDED
                assert future.directory == runners[1].directory
            else:
                with pytest.raises(RunnerStartupFailed, match="device unavailable"):
                    future.result(timeout=5)
            assert len(resources.matching(ResourceRequest())) == 3 - failures
            for runner in runners:
                runner.start.assert_called_once()
                runner.close.assert_called_once()
            for runner in runners[:failures]:
                runner.execute.assert_not_called()
                assert not runner.directory.exists()

    def test_execution_failure_does_not_retry_or_remove_a_healthy_resource(self, tmp_path: Path) -> None:
        resources = LocalAcceleratorCatalog(["cpu", "cuda:0"])
        runner = _runner(tmp_path, "failed")
        runner.execute.side_effect = TaskExecutionFailed("task failed")
        with _executor(resources, [runner]) as executor:
            future = executor.submit(_describe, timeout_seconds=1)
            with pytest.raises(TaskExecutionFailed, match="task failed"):
                future.result(timeout=5)
            assert len(resources.matching(ResourceRequest())) == 2
            assert not runner.directory.exists()

    def test_result_waits_for_cleanup_and_resource_release(self, tmp_path: Path) -> None:
        resources = LocalAcceleratorCatalog(["cpu"])
        runner = _runner(tmp_path, "cleanup")
        closing = threading.Event()
        release = threading.Event()

        def close() -> None:
            closing.set()
            assert release.wait(5)

        runner.close.side_effect = close
        with _executor(resources, [runner]) as executor:
            future = executor.submit(_describe, timeout_seconds=1)
            try:
                assert closing.wait(5)
                assert not future.done()
                assert resources.matching(ResourceRequest())[0].state == ResourceState.LEASED
            finally:
                release.set()
            assert future.result(timeout=5) == "ok"
            assert resources.matching(ResourceRequest())[0].state == ResourceState.AVAILABLE

    def test_pending_cancellation_and_callback_shutdown(self, tmp_path: Path) -> None:
        resources = LocalAcceleratorCatalog(["cpu"])
        runner = _runner(tmp_path, "running")
        started = threading.Event()
        release = threading.Event()
        closed = threading.Event()

        def execute(task: object) -> str:
            started.set()
            assert release.wait(5)
            return "ok"

        runner.execute.side_effect = execute
        with _executor(resources, [runner]) as executor:
            first = executor.submit(_describe, timeout_seconds=1)
            try:
                assert started.wait(5)
                second = executor.submit(_describe, timeout_seconds=1)
                assert executor.cancel_pending() == 1
                assert second.cancelled()
                assert wait([second], timeout=0).done == {second}

                def shutdown(done: ExecutionFuture) -> None:
                    executor.shutdown()
                    closed.set()

                first.add_done_callback(shutdown)
            finally:
                release.set()
            assert first.result(timeout=5) == "ok"
            assert closed.wait(5)
            assert executor.state == ExecutorState.CLOSED
            with pytest.raises(ExecutorClosed):
                executor.submit(_describe, timeout_seconds=1).result()

    def test_cleanup_failure_is_reported_and_quarantines_the_resource(self, tmp_path: Path) -> None:
        resources = LocalAcceleratorCatalog(["cpu"])
        runner = _runner(tmp_path, "cleanup-failed")
        runner.close.side_effect = RuntimeError("cannot reap")
        with _executor(resources, [runner]) as executor:
            future = executor.submit(_describe, timeout_seconds=1)
            with pytest.raises(ExecutionError, match="runner cleanup failed.*cannot reap"):
                future.result(timeout=5)
            assert not resources.matching(ResourceRequest())
            assert not runner.directory.exists()


class TestResourceLease:
    def test_stale_release_cannot_release_a_new_lease(self) -> None:
        resources = LocalAcceleratorCatalog(["cpu", "cuda:0"])
        request = ResourceRequest(backend="cpu")
        (resource,) = resources.matching(request)
        first = resources.acquire(resource.resource_id, "first")
        resources.release(first, ResourceHealth.HEALTHY)
        second = resources.acquire(resource.resource_id, "second")
        resources.release(first, ResourceHealth.UNHEALTHY)
        assert resources.matching(request)[0].state == ResourceState.LEASED
        resources.release(second, ResourceHealth.HEALTHY)
        assert resources.matching(request)[0].state == ResourceState.AVAILABLE
        assert second.resource.state == ResourceState.LEASED  # the lease carries an immutable snapshot


class TestSpawnedExecution:
    def test_closing_during_startup_prevents_a_late_spawn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resources = LocalAcceleratorCatalog(["cpu"])
        (resource,) = resources.matching(ResourceRequest())
        lease = resources.acquire(resource.resource_id, "task")
        runner = DisposableExecutionRunner(lease, ExecutionEngine(), tmp_path)
        creating = threading.Event()
        release = threading.Event()
        channel, endpoint = Mock(), Mock()
        context = Mock()

        def create(context: object) -> tuple[Mock, Mock]:
            creating.set()
            assert release.wait(5)
            return channel, endpoint

        monkeypatch.setattr(
            "aklr.core.validation.stages._bench._execution._runner.MultiprocessingRunnerChannel.create",
            create,
        )
        monkeypatch.setattr(
            "aklr.core.validation.stages._bench._execution._runner.multiprocessing.get_context",
            lambda method: context,
        )
        with ThreadPoolExecutor(1) as pool:
            starting = pool.submit(runner.start)
            try:
                assert creating.wait(5)
                runner.close()
            finally:
                release.set()
            with pytest.raises(TaskCancelled):
                starting.result(timeout=5)
        context.Process.return_value.start.assert_not_called()
        channel.close.assert_called_once()
        endpoint.close.assert_called_once()

    def test_background_shutdown_cancels_a_running_task_and_reaps_its_process(self) -> None:
        with Executor.for_devices(["cpu"]) as executor:
            future = executor.submit(_sleep, timeout_seconds=60)
            deadline = time.monotonic() + 60
            while future.directory is None or not (future.directory / "started").exists():
                assert not future.done()
                assert time.monotonic() < deadline
                time.sleep(0.01)
            directory = future.directory
            pid = int((directory / "started").read_text())
            executor.shutdown(wait=False)
            with ThreadPoolExecutor(2) as pool:
                closers = [pool.submit(executor.close) for _ in range(2)]
                for closer in closers:
                    closer.result(timeout=15)
            with pytest.raises(TaskCancelled):
                future.result(timeout=0)
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert executor.state == ExecutorState.CLOSED
            assert not directory.parent.exists()

    def test_each_task_gets_a_fresh_process_and_preserves_evidence_until_shutdown(self) -> None:
        with Executor.for_devices(["cpu"]) as executor:
            first = executor.submit(_describe, timeout_seconds=10, value=42)
            result = first.result(timeout=60)
            directory = first.directory
            assert result == {"pid": result["pid"], "device": "cpu", "cwd": str(directory), "value": 42}
            assert (directory / "evidence").read_text() == "42"
            with pytest.raises(ProcessLookupError):
                os.kill(result["pid"], 0)
            second = executor.submit(_describe, timeout_seconds=10).result(timeout=60)
            assert second["pid"] != result["pid"]
        assert not directory.parent.exists()

    @pytest.mark.parametrize(
        ("task", "error", "message", "timeout_seconds"),
        [
            (_fail, TaskExecutionFailed, "task diagnostic", 10),
            (_sleep, TaskExecutionTimeout, "did not answer", 0.2),
        ],
    )
    def test_failed_task_is_cleaned_up_and_resource_can_be_reused(
        self, task: object, error: type[ExecutionError], message: str, timeout_seconds: float
    ) -> None:
        with Executor.for_devices(["cpu"]) as executor:
            future = executor.submit(task, timeout_seconds=timeout_seconds)
            with pytest.raises(error, match=message):
                future.result(timeout=60)
            assert future.state == TaskState.FAILED
            assert not future.directory.exists()
            assert executor.submit(_describe, timeout_seconds=10).result(timeout=60)["value"] == 7
