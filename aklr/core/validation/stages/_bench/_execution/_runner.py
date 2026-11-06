###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import itertools
import logging
import multiprocessing
import multiprocessing.connection
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final

from aklr.core.validation.stages._bench._execution._engine import ExecutionEngine
from aklr.core.validation.stages._bench._execution._errors import (
    ChannelFailure,
    ExecutionError,
    InvalidRunnerState,
    RunnerCrashed,
    RunnerExitedEarly,
    RunnerStartupFailed,
    TaskCancelled,
    TaskExecutionFailed,
    TaskExecutionTimeout,
)
from aklr.core.validation.stages._bench._execution._model import (
    ExecutionContext,
    ExecutionTask,
    RunnerHealth,
    RunnerState,
)
from aklr.core.validation.stages._bench._execution._resources import ResourceLease
from aklr.core.validation.stages._bench._execution._shutdown import ShutdownGate
from aklr.core.validation.stages._bench._execution._task_log import TaskLog
from aklr.core.validation.stages._bench._execution._transport import (
    ChannelReceiveTimeout,
    ChildRunnerEndpoint,
    ExecuteTask,
    MultiprocessingRunnerChannel,
    RunnerEvent,
    RunnerExited,
    RunnerFailed,
    RunnerStarted,
    ShutdownRunner,
    TaskCompleted,
    TaskFailed,
)
from aklr.core.validation.stages._process_group import ProcessGroup

DEFAULT_STARTUP_TIMEOUT_SECONDS: Final = 30.0
RUNNER_POLL_SECONDS: Final = 0.25
RUNNER_JOIN_SECONDS: Final = 2.0


class ExecutionRunner(ABC):
    """Starts a leased task, executes it synchronously, and closes its process."""

    @property
    @abstractmethod
    def runner_id(self) -> str: ...

    @property
    @abstractmethod
    def state(self) -> RunnerState: ...

    @property
    @abstractmethod
    def directory(self) -> Path: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def execute(self, task: ExecutionTask) -> Any: ...

    @abstractmethod
    def health(self) -> RunnerHealth: ...

    @abstractmethod
    def close(self) -> None:
        """Reap the process; concurrent callers must all wait until cleanup finishes."""

    def diagnostics(self) -> str | None:
        return None


class DisposableExecutionRunner(ExecutionRunner):
    """A freshly spawned child process that runs exactly one task."""

    def __init__(
        self,
        lease: ResourceLease,
        engine: ExecutionEngine,
        directory: Path,
        n_resources: int,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self._lease = lease
        self._engine = engine
        self._directory = directory
        self._n_resources = n_resources
        self._startup_timeout_seconds = startup_timeout_seconds
        self._runner_id = uuid.uuid4().hex
        self._state = RunnerState.CREATED
        self._health = RunnerHealth.HEALTHY
        self._state_lock = threading.Lock()
        self._process: Any | None = None
        self._channel: MultiprocessingRunnerChannel | None = None
        self._shutdown_gate = ShutdownGate("aklr-execution-runner-shutdown")

    @property
    def runner_id(self) -> str:
        return self._runner_id

    @property
    def state(self) -> RunnerState:
        with self._state_lock:
            return self._state

    @property
    def directory(self) -> Path:
        return self._directory

    def start(self) -> None:
        with self._state_lock:
            if self._state != RunnerState.CREATED:
                raise InvalidRunnerState(f"runner {self._runner_id} is {self._state.value}, expected created")
            self._state = RunnerState.STARTING

        if self._lease.resource.device != "cpu":
            from aklr.core.utils.testing import kernel_tracer

            kernel_tracer()  # built here, outside the child's startup timeout
        # spawn, not fork: torch refuses the GPU in a child forked from a parent that already used it
        context = multiprocessing.get_context("spawn")
        channel, child_endpoint = MultiprocessingRunnerChannel.create(context)
        process = context.Process(
            target=DisposableExecutionRunner._run_child,
            args=(
                child_endpoint,
                self._runner_id,
                self._lease.resource.device,
                self._engine,
                self._directory,
                self._n_resources,
            ),
            daemon=False,
        )
        try:
            # close() must either see this process or prevent it from being spawned.
            with self._state_lock:
                if self._state != RunnerState.STARTING:
                    channel.close()
                    raise TaskCancelled(f"runner {self._runner_id} was closed while starting")
                self._channel = channel
                self._process = process
                process.start()
            child_endpoint.close()
            event = self._receive_event(self._startup_timeout_seconds)
            if isinstance(event, RunnerFailed) and event.runner_id == self._runner_id:
                self._set_health(
                    RunnerHealth.RESOURCE_FAILED if event.resource_failed else RunnerHealth.FAILED
                )
                raise RunnerStartupFailed(event.reason)
            if not isinstance(event, RunnerStarted) or event.runner_id != self._runner_id:
                raise RunnerStartupFailed(
                    f"runner {self._runner_id} sent {type(event).__name__} while starting"
                )
            with self._state_lock:
                if self._state != RunnerState.STARTING:
                    raise InvalidRunnerState(
                        f"runner {self._runner_id} became {self._state.value} while starting"
                    )
                self._state = RunnerState.READY
        except Exception as error:
            is_closed = self.state == RunnerState.CLOSED
            if not is_closed:
                self._set_state(RunnerState.FAILED)
                if self.health() == RunnerHealth.HEALTHY:
                    # a resource whose child did not start leaves the pool
                    self._set_health(RunnerHealth.RESOURCE_FAILED)
            try:
                self.close()
            except ExecutionError as close_error:
                logging.getLogger(__name__).warning(
                    "runner %s did not close cleanly: %s", self._runner_id, close_error
                )
            if is_closed:
                raise TaskCancelled(f"runner {self._runner_id} was closed while starting") from error
            if isinstance(error, TaskExecutionTimeout):
                raise RunnerStartupFailed(
                    f"runner {self._runner_id} did not start within {self._startup_timeout_seconds} seconds"
                ) from error
            if isinstance(error, RunnerExitedEarly):
                raise RunnerStartupFailed(str(error)) from error
            if isinstance(error, ExecutionError):
                raise
            raise RunnerStartupFailed(
                f"runner {self._runner_id} could not start: {type(error).__name__}: {error}"
            ) from error
        finally:
            child_endpoint.close()

    def execute(self, task: ExecutionTask) -> Any:
        """Run the leased task and wait for its reply within the task's execution budget."""
        task_id = task.task_id
        if task.task_id != self._lease.task_id:
            raise InvalidRunnerState(f"task {task.task_id} does not own lease {self._lease.lease_id}")
        with self._state_lock:
            if self._state == RunnerState.CLOSED:
                raise TaskCancelled(f"task {task_id} was cancelled")
            if self._state != RunnerState.READY:
                raise InvalidRunnerState(f"runner {self._runner_id} is {self._state.value}, expected ready")
            self._require_channel().send(ExecuteTask(runner_id=self._runner_id, task=task))
            self._state = RunnerState.RUNNING

        try:
            event = self._receive_event(task.timeout_seconds)
        except (TaskExecutionTimeout, RunnerExitedEarly, ChannelFailure) as error:
            if self.state == RunnerState.CLOSED:
                raise TaskCancelled(f"task {task_id} was cancelled") from error
            self._set_state(RunnerState.FAILED)
            self._set_health(RunnerHealth.FAILED)
            if isinstance(error, TaskExecutionTimeout):
                raise
            if isinstance(error, RunnerExitedEarly):
                raise RunnerCrashed(str(error)) from error
            # an invalid message is the task's: its code ran in the process that wrote it
            raise TaskExecutionFailed(
                f"runner {self._runner_id} sent an invalid message for task {task_id}: {error}"
            ) from error

        if (
            isinstance(event, TaskCompleted)
            and event.runner_id == self._runner_id
            and event.task_id == task_id
        ):
            return event.result
        self._set_state(RunnerState.FAILED)
        self._set_health(RunnerHealth.FAILED)
        if isinstance(event, TaskFailed) and event.runner_id == self._runner_id and event.task_id == task_id:
            raise TaskExecutionFailed(event.reason)
        if isinstance(event, RunnerExited) and event.runner_id == self._runner_id:
            raise RunnerCrashed(f"runner {self._runner_id} exited before task {task_id} completed")
        raise RunnerCrashed(f"runner {self._runner_id} sent an invalid event for task {task_id}: {event}")

    def health(self) -> RunnerHealth:
        with self._state_lock:
            return self._health

    def close(self) -> None:
        self._shutdown_gate.run(self._close)

    def _close(self) -> None:
        with self._state_lock:
            previous_state = self._state
            self._state = RunnerState.CLOSED

        process = self._process
        channel = self._channel
        has_cleanup_failed = False
        is_shutdown_sent = False
        if channel is not None and previous_state in {
            RunnerState.STARTING,
            RunnerState.READY,
            RunnerState.RUNNING,
        }:
            try:
                channel.send(ShutdownRunner(runner_id=self._runner_id))
                is_shutdown_sent = True
            except Exception:  # noqa: BLE001
                has_cleanup_failed = True

        is_orphaned = False
        if process is not None and process.pid is not None:
            is_gone = False
            try:
                if is_shutdown_sent and previous_state != RunnerState.STARTING:
                    # a ready child, or one whose task answered, exits by itself
                    process.join(timeout=RUNNER_JOIN_SECONDS)
                process.kill()
                is_gone = ProcessGroup(process.pid).kill()
                process.join(timeout=RUNNER_JOIN_SECONDS)
                is_gone = is_gone and bool(multiprocessing.connection.wait([process.sentinel], 0))
            except Exception:  # noqa: BLE001
                # a process that cannot be reaped is assumed alive
                has_cleanup_failed = True
            if not is_gone:
                self._set_health(RunnerHealth.RESOURCE_FAILED)
                is_orphaned = True
            elif has_cleanup_failed and self.health() == RunnerHealth.HEALTHY:
                self._set_health(RunnerHealth.FAILED)

        if channel is not None:
            channel.close()
        if is_orphaned:
            raise RunnerCrashed(f"runner {self._runner_id}: the process group outlived its kill")

    def diagnostics(self) -> str | None:
        return TaskLog(self._directory / "task.log").tail() or None

    def _receive_event(self, timeout: float | None) -> RunnerEvent:
        process = self._process
        if process is None:
            raise InvalidRunnerState(f"runner {self._runner_id} has no process")
        channel = self._require_channel()
        deadline = None if timeout is None else time.monotonic() + timeout

        try:
            while True:
                wait_seconds = RUNNER_POLL_SECONDS
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        try:
                            return channel.receive(timeout=0)
                        except ChannelReceiveTimeout:
                            raise TaskExecutionTimeout(
                                f"runner {self._runner_id} did not answer within {timeout} seconds"
                            ) from None
                    wait_seconds = min(wait_seconds, remaining)
                try:
                    return channel.receive(wait_seconds)
                except ChannelReceiveTimeout:
                    if process.is_alive():
                        continue
                try:
                    return channel.receive(timeout=0)
                except ChannelReceiveTimeout:
                    break
        except EOFError:
            process.join(RUNNER_JOIN_SECONDS)
        raise RunnerExitedEarly(f"runner {self._runner_id} exited with code {process.exitcode}")

    def _set_state(self, state: RunnerState) -> None:
        with self._state_lock:
            if self._state != RunnerState.CLOSED:
                self._state = state

    def _set_health(self, health: RunnerHealth) -> None:
        with self._state_lock:
            self._health = health

    def _require_channel(self) -> MultiprocessingRunnerChannel:
        if self._channel is None:
            raise InvalidRunnerState(f"runner {self._runner_id} has no channel")
        return self._channel

    @staticmethod
    def _run_child(
        endpoint: ChildRunnerEndpoint,
        runner_id: str,
        device: str,
        engine: ExecutionEngine,
        directory: Path,
        n_resources: int,
    ) -> None:
        task_id: str | None = None
        is_gpu = device != "cpu"
        try:
            os.setsid()  # its pid becomes a group id: the parent kills the group, subprocesses included
            os.chdir(directory)
            task_log = TaskLog(directory / "task.log")
            with task_log.redirected_stdio():
                try:
                    import torch

                    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, n_resources)))
                    if is_gpu:
                        from aklr.core.utils.testing import kernel_tracer

                        # rocprofiler-sdk takes the timer's kernel tracer only before the first GPU call
                        kernel_tracer().configure()
                        torch.cuda.set_device(device)
                    endpoint.send(RunnerStarted(runner_id=runner_id))

                    command = endpoint.receive()
                    if isinstance(command, ShutdownRunner):
                        return
                    if not isinstance(command, ExecuteTask) or command.runner_id != runner_id:
                        raise RuntimeError(f"runner received invalid command: {command}")

                    task_id = command.task.task_id
                    result = engine.execute(
                        command.task,
                        ExecutionContext(
                            device=device,
                            log_path=task_log.path,
                            directory=directory,
                        ),
                    )
                    # the log is complete when the parent reads its tail
                    sys.stdout.flush()
                    sys.stderr.flush()
                    endpoint.send(TaskCompleted(runner_id=runner_id, task_id=task_id, result=result))
                except BaseException as error:  # noqa: BLE001
                    # after the task arrived, a failure is the task's; before, on a GPU, the device's
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    reason = f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}"
                    try:
                        endpoint.send(
                            TaskFailed(runner_id=runner_id, task_id=task_id, reason=reason)
                            if task_id is not None
                            else RunnerFailed(
                                runner_id=runner_id,
                                reason=reason,
                                resource_failed=is_gpu,
                            )
                        )
                    except Exception:  # noqa: BLE001, S110
                        # the parent sees the exit anyway
                        pass
        finally:
            try:
                endpoint.send(RunnerExited(runner_id=runner_id, task_id=task_id, exit_code=0))
            except Exception:  # noqa: BLE001, S110
                pass


class RunnerFactory(ABC):
    @abstractmethod
    def create(
        self,
        lease: ResourceLease,
        engine: ExecutionEngine,
    ) -> ExecutionRunner: ...

    @abstractmethod
    def close(self) -> None: ...


class DisposableRunnerFactory(RunnerFactory):
    def __init__(
        self,
        n_resources: int,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self._n_resources = n_resources
        self._startup_timeout_seconds = startup_timeout_seconds
        self._job_dir = Path(tempfile.mkdtemp(prefix="aklr-validation-"))
        self._task_numbers = itertools.count()
        self._lock = threading.Lock()

    def create(
        self,
        lease: ResourceLease,
        engine: ExecutionEngine,
    ) -> ExecutionRunner:
        with self._lock:
            task_number = next(self._task_numbers)
        directory = self._job_dir / f"{task_number}-{lease.resource.resource_id.replace(':', '-')}"
        directory.mkdir()
        return DisposableExecutionRunner(
            lease=lease,
            engine=engine,
            directory=directory,
            n_resources=self._n_resources,
            startup_timeout_seconds=self._startup_timeout_seconds,
        )

    def close(self) -> None:
        shutil.rmtree(self._job_dir, ignore_errors=True)
