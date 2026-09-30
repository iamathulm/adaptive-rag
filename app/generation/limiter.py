from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from threading import BoundedSemaphore, Lock
from time import monotonic


class GenerationRateLimitError(RuntimeError):
    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__("Generation request limit reached")
        self.retry_after_seconds = retry_after_seconds


class GenerationRateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        requests_per_day: int,
        max_concurrent_requests: int,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self._minute_requests: deque[float] = deque()
        self._day_started = datetime.now(UTC).date()
        self._day_requests = 0
        self._lock = Lock()
        self._concurrency = BoundedSemaphore(max_concurrent_requests)

    @contextmanager
    def slot(self) -> Iterator[None]:
        if not self._concurrency.acquire(blocking=False):
            raise GenerationRateLimitError(1.0)

        try:
            self._reserve()
            yield
        finally:
            self._concurrency.release()

    def _reserve(self) -> None:
        now = monotonic()
        with self._lock:
            while self._minute_requests and now - self._minute_requests[0] >= 60:
                self._minute_requests.popleft()

            today = datetime.now(UTC).date()
            if today != self._day_started:
                self._day_started = today
                self._day_requests = 0

            if self._day_requests >= self.requests_per_day:
                tomorrow = datetime.combine(
                    today + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=UTC,
                )
                retry_after = (tomorrow - datetime.now(UTC)).total_seconds()
                raise GenerationRateLimitError(max(1.0, retry_after))

            if len(self._minute_requests) >= self.requests_per_minute:
                retry_after = 60 - (now - self._minute_requests[0])
                raise GenerationRateLimitError(max(1.0, retry_after))

            self._minute_requests.append(now)
            self._day_requests += 1
