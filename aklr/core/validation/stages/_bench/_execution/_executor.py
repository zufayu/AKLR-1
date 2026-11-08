###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import signal
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from aklr.core.validation.stages._bench._execution._backend import (
    ExecutionBackend,
    RunnerExecutionBackend,
)
from aklr.core.validation.stages._bench._execution._engine import ExecutionEngine
from aklr.core.validation.stages._bench._execution._errors import ExecutorClosed
from aklr.core.validation.stages._bench._execution._future import ExecutionFuture
from aklr.core.validation.stages._bench._execution._model import (
    ExecutionTask,
    ExecutorState,
    ResourceRequest,
)
from aklr.core.validation.stages._bench._execution._resources import (
    LocalAcceleratorCatalog,
)
from aklr.core.validation.stages._bench._execution._runner import (
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    RunnerFactory,
)
from aklr.core.validation.stages._bench._execution._shutdown import ShutdownGate

if TYPE_CHECKING:
    from typing import Self


class Executor:
    def __init__(self, backend: ExecutionBackend, *, resources_per_runner: int = 1) -> None:
        self._resource_request = ResourceRequest(count=resources_per_runner)
        self._backend = backend
        self._state = ExecutorState.OPEN
        self._state_lock = threading.Lock()
        self._shutdown_gate = ShutdownGate("aklr-execution-shutdown")
        self._previous_signal_handlers: dict[int, Any] = {}

    @classmethod
    def for_devices(
        cls,
        devices: Sequence[str],
        *,
        runner: Literal["process", "thread"] = "process",
        resources_per_runner: int = 1,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> Executor:
        """Run each task in a process or thread with an exclusive lease of one or more devices."""
        ResourceRequest(count=resources_per_runner)
        # the catalog first: a bad device raises before the job directory exists
        resources = LocalAcceleratorCatalog(devices)
        if resources_per_runner > len(resources):
            raise ValueError("resources_per_runner exceeds the available device count")
        return cls(
            RunnerExecutionBackend(
                resources=resources,
                runner_factory=RunnerFactory(
                    runner=runner,
                    startup_timeout_seconds=startup_timeout_seconds,
                ),
                engine=ExecutionEngine(),
            ),
            resources_per_runner=resources_per_runner,
        )

    @property
    def state(self) -> ExecutorState:
        with self._state_lock:
            return self._state

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *,
        timeout_seconds: float,
        resource_request: ResourceRequest | None = None,
        metadata: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> ExecutionFuture[Any]:
        """Run fn(context, **kwargs); an explicit resource request overrides the executor default."""
        task = ExecutionTask(
            call=fn,
            kwargs=kwargs,
            timeout_seconds=timeout_seconds,
            task_id=uuid.uuid4().hex,
            resource_request=resource_request or self._resource_request,
            metadata=metadata or {},
        )
        with self._state_lock:
            if self._state != ExecutorState.OPEN:
                future: ExecutionFuture[Any] = ExecutionFuture(task.task_id)
                future.set_exception(ExecutorClosed(f"execution executor is {self._state.value}"))
                return future
            return self._backend.submit(task)

    def cancel_pending(self) -> int:
        return self._backend.cancel_pending()

    def shutdown(self, wait: bool = True) -> None:
        with self._state_lock:
            if self._state == ExecutorState.OPEN:
                self._state = ExecutorState.CLOSING
        self._shutdown_gate.run(self._close_backend, wait=wait)

    def close(self) -> None:
        self.shutdown(wait=True)

    def __enter__(self) -> Self:
        # the children lead their own sessions: a signal must unwind through __exit__ to kill them
        if threading.current_thread() is threading.main_thread():
            for signal_number in (signal.SIGTERM, signal.SIGINT):
                self._previous_signal_handlers[signal_number] = signal.signal(
                    signal_number, self._exit_on_signal
                )
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            self.close()
        finally:
            for signal_number, handler in self._previous_signal_handlers.items():
                signal.signal(signal_number, handler)
            self._previous_signal_handlers.clear()

    def _close_backend(self) -> None:
        try:
            self._backend.shutdown(wait=True)
        finally:
            with self._state_lock:
                self._state = ExecutorState.CLOSED

    @staticmethod
    def _exit_on_signal(signal_number: int, frame: object) -> None:
        raise SystemExit(128 + signal_number)
