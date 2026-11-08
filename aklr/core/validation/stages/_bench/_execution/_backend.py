###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import shutil
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aklr.core.validation.stages._bench._execution._engine import ExecutionEngine
from aklr.core.validation.stages._bench._execution._errors import (
    ExecutionError,
    RunnerStartupFailed,
    TaskCancelled,
)
from aklr.core.validation.stages._bench._execution._future import ExecutionFuture
from aklr.core.validation.stages._bench._execution._model import (
    ExecutionTask,
    ResourceHealth,
    RunnerHealth,
)
from aklr.core.validation.stages._bench._execution._resources import (
    ResourceCatalog,
    ResourceLease,
)
from aklr.core.validation.stages._bench._execution._runner import (
    ExecutionRunner,
    RunnerFactory,
)
from aklr.core.validation.stages._bench._execution._scheduler import (
    ExecutionScheduler,
    RoundRobinStrategy,
    SchedulingStrategy,
)
from aklr.core.validation.stages._bench._execution._shutdown import ShutdownGate


class ExecutionBackend(ABC):
    @abstractmethod
    def submit(self, task: ExecutionTask) -> ExecutionFuture[Any]: ...

    @abstractmethod
    def cancel_pending(self) -> int: ...

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None: ...


class RunnerSupervisor:
    """Owns runner lifetimes and resolves tasks only after cleanup and resource release."""

    def __init__(
        self,
        runner_factory: RunnerFactory,
        engine: ExecutionEngine,
        on_finished: Callable[[ResourceLease, ResourceHealth], None],
        retry: Callable[[ExecutionTask, ExecutionFuture[Any]], bool],
        max_concurrency: int,
    ) -> None:
        self._runner_factory = runner_factory
        self._engine = engine
        self._on_finished = on_finished
        self._retry = retry
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_concurrency),
            thread_name_prefix="aklr-execution-runner",
        )
        self._active: dict[str, ExecutionRunner] = {}
        self._is_closing = False
        self._lock = threading.Lock()

    def dispatch(
        self,
        task: ExecutionTask,
        future: ExecutionFuture[Any],
        lease: ResourceLease,
    ) -> None:
        self._pool.submit(self._run, task, future, lease)

    def shutdown(self) -> None:
        with self._lock:
            self._is_closing = True
            runners = tuple(self._active.values())
        for runner in runners:
            try:
                runner.close()
            except Exception:  # noqa: BLE001, S110
                # the runner's own pool thread reads its health and reports it
                pass
        self._pool.shutdown(wait=True)

    def _run(
        self,
        task: ExecutionTask,
        future: ExecutionFuture[Any],
        lease: ResourceLease,
    ) -> None:
        if future.cancelled():
            self._on_finished(lease, ResourceHealth.HEALTHY)
            return

        runner = None
        directory = None
        result = None
        is_started = False
        is_returned = False
        failure = None
        runner_health = RunnerHealth.HEALTHY
        try:
            runner = self._runner_factory.create(lease, self._engine)
            directory = runner.directory
            with self._lock:
                if self._is_closing:
                    raise TaskCancelled(f"task {task.task_id} was cancelled: the backend is closing")
                self._active[task.task_id] = runner
            runner.start()
            is_started = True
        except BaseException as error:  # noqa: BLE001
            failure = self._as_execution_error(error, RunnerStartupFailed, "could not start execution runner")
        else:
            # marked running only once started: a start failure may retry the task elsewhere
            if future.mark_running(directory):
                try:
                    result = runner.execute(task)
                    is_returned = True
                except BaseException as error:  # noqa: BLE001
                    failure = self._as_execution_error(
                        error, ExecutionError, f"could not execute task {task.task_id}"
                    )
        finally:
            if runner is not None:
                try:
                    runner.close()
                    runner_health = runner.health()
                except BaseException as error:  # noqa: BLE001
                    runner_health = RunnerHealth.RESOURCE_FAILED
                    failure = self._append_cleanup_failure(failure, "runner cleanup failed", error)
                finally:
                    with self._lock:
                        self._active.pop(task.task_id, None)

                try:
                    diagnostics = runner.diagnostics()
                except BaseException as error:  # noqa: BLE001
                    diagnostics = None
                    failure = self._append_cleanup_failure(failure, "runner diagnostics failed", error)
                if diagnostics and failure is not None:
                    # the log first, so the failure's own text stays the last line
                    message = f"Runner log:\n{diagnostics}\n\n{failure}"
                    failure = (
                        type(failure)(message)
                        if isinstance(failure, ExecutionError)
                        else ExecutionError(message)
                    )

            resource_health = (
                ResourceHealth.UNHEALTHY
                if runner_health == RunnerHealth.RESOURCE_FAILED
                else ResourceHealth.HEALTHY
            )
            try:
                self._on_finished(lease, resource_health)
            except BaseException as error:  # noqa: BLE001
                failure = self._append_cleanup_failure(failure, "resource release failed", error)

        if failure is None and is_returned:
            future.publish_result(result)
            return
        # a failed or cancelled task's directory goes now; a succeeded one's is the caller's
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        if failure is None:
            return
        # the runner never started and its resource left: the task tries once more elsewhere
        if not is_started and resource_health == ResourceHealth.UNHEALTHY and self._retry(task, future):
            return
        future.publish_exception(failure)

    @staticmethod
    def _as_execution_error(
        error: BaseException,
        step_failure_type: type[ExecutionError],
        description: str,
    ) -> ExecutionError:
        if isinstance(error, ExecutionError):
            return error
        return step_failure_type(f"{description}: {type(error).__name__}: {error}")

    @staticmethod
    def _append_cleanup_failure(
        failure: BaseException | None,
        message: str,
        error: BaseException,
    ) -> ExecutionError:
        detail = f"{message}: {type(error).__name__}: {error}"
        if failure is not None:
            detail = f"{failure}\n\n{detail}"
            if isinstance(failure, ExecutionError):
                return type(failure)(detail)
        return ExecutionError(detail)


class RunnerExecutionBackend(ExecutionBackend):
    """Connects the scheduler to runner supervision and closes them in dependency order."""

    def __init__(
        self,
        resources: ResourceCatalog,
        runner_factory: RunnerFactory,
        engine: ExecutionEngine,
        strategy: SchedulingStrategy | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self._runner_factory = runner_factory
        self._scheduler: ExecutionScheduler
        self._supervisor = RunnerSupervisor(
            runner_factory=runner_factory,
            engine=engine,
            on_finished=lambda lease, health: self._scheduler.complete(lease, health),
            retry=lambda task, future: self._scheduler.retry(task, future),
            max_concurrency=max_concurrency or len(resources),
        )
        self._scheduler = ExecutionScheduler(
            resources=resources,
            strategy=strategy or RoundRobinStrategy(),
            dispatch=self._supervisor.dispatch,
        )
        self._shutdown_gate = ShutdownGate("aklr-execution-backend-shutdown")

    def submit(self, task: ExecutionTask) -> ExecutionFuture[Any]:
        return self._scheduler.submit(task)

    def cancel_pending(self) -> int:
        return self._scheduler.cancel_pending()

    def shutdown(self, wait: bool = True) -> None:
        # Stop accepting work before returning, including when cleanup runs in the background.
        self._scheduler.shutdown()
        self._shutdown_gate.run(self._close_supervisor, wait=wait)

    def _close_supervisor(self) -> None:
        try:
            self._supervisor.shutdown()
        finally:
            self._runner_factory.close()
