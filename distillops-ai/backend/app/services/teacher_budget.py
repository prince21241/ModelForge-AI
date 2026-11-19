"""Process-local teacher spending gate.

The lock covers the check and the increment together, so concurrent requests
cannot all pass a limit that only one of them should consume. The counter
lives in this process. It is not shared across machines or restarts.
"""

from __future__ import annotations

import threading
from decimal import Decimal

from backend.app.core.errors import TeacherBudgetError


class TeacherSpendGate:
    """Reserve a teacher call before it starts, then record the actual cost."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._reserved = Decimal("0")
        self._spent = Decimal("0")

    @property
    def requests(self) -> int:
        return self._requests

    @property
    def spent_usd(self) -> Decimal:
        return self._spent

    def try_reserve(self, *, max_requests: int, budget: Decimal | None, reserve: Decimal) -> Decimal:
        """Count one call and hold ``reserve`` dollars. Raise before any teacher I/O."""
        held = reserve if budget is not None else Decimal("0")
        with self._lock:
            if self._requests >= max_requests:
                raise TeacherBudgetError(
                    "The teacher request limit has been reached. No teacher request was sent."
                )
            if budget is not None and self._spent + self._reserved + held > budget:
                raise TeacherBudgetError(
                    "The teacher spending limit would be exceeded. No teacher request was sent."
                )
            self._requests += 1
            self._reserved += held
        return held

    def settle(self, reserved: Decimal, actual: Decimal | None) -> None:
        """Release the hold and add the measured cost. A missing cost adds nothing."""
        with self._lock:
            self._reserved -= reserved
            if actual is not None:
                self._spent += actual
