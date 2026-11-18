"""Small per-process rate limiter for a single-instance portfolio demo.

This is not a distributed rate limiter. Multi-instance deployment must use
a trusted edge proxy or shared store (Redis) instead.
"""
from __future__ import annotations

from collections import deque
from threading import Lock
from time import monotonic


class SlidingWindowRateLimiter:
    def __init__(self, *, limit: int = 12, window_seconds: float = 60.0) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit capacity and window must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._records: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._checks = 0

    def allow(self, client_id: str, *, now: float | None = None) -> bool:
        """Atomic admission. An untrusted forwarding header is never used."""
        timestamp = monotonic() if now is None else now
        with self._lock:
            self._checks += 1
            if self._checks % 128 == 0:
                # Evict stale clients to bound memory in a public demo.
                for key, seen in list(self._records.items()):
                    if not seen or timestamp - seen[-1] >= self.window_seconds:
                        self._records.pop(key, None)

            times = self._records.setdefault(client_id, deque())
            while times and timestamp - times[0] >= self.window_seconds:
                times.popleft()
            if len(times) >= self.limit:
                return False
            times.append(timestamp)
            return True
