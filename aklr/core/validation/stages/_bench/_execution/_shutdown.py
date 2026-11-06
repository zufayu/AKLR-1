###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import threading
from collections.abc import Callable


class ShutdownGate:
    """Runs one shutdown body once, and holds every other caller until it ends."""

    def __init__(self, thread_name: str) -> None:
        self._thread_name = thread_name
        self._complete = threading.Event()
        self._error: BaseException | None = None
        self._owner_thread_id: int | None = None
        self._started = False
        self._lock = threading.Lock()

    def run(self, run_shutdown: Callable[[], None], *, wait: bool = True) -> None:
        with self._lock:
            initiate = not self._started
            self._started = True
        if initiate:
            if wait:
                self._run(run_shutdown)
            else:
                threading.Thread(
                    target=self._run,
                    args=(run_shutdown,),
                    name=self._thread_name,
                    daemon=True,
                ).start()
        # called from the body's own thread: waiting here would never end
        if wait and self._owner_thread_id != threading.get_ident():
            self._complete.wait()
            if self._error is not None:
                raise self._error

    def _run(self, run_shutdown: Callable[[], None]) -> None:
        self._owner_thread_id = threading.get_ident()
        try:
            run_shutdown()
        except BaseException as error:  # noqa: BLE001
            self._error = error
        finally:
            self._owner_thread_id = None
            self._complete.set()
