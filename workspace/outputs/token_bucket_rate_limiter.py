"""
GridFlow AI Verified Production Artifact: Distributed Token Bucket Rate Limiter
Prompt: Build a thread-safe Token Bucket Rate Limiter API service in Python
Verified Citations: DOC-CODE-001, MEM-session-, DOC-CUSTOM-01
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BucketState:
    capacity: float
    refill_rate: float
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class TokenBucketRateLimiter:
    """Thread-safe Token Bucket Rate Limiter compliant with GridFlow AST Security Policy."""

    def __init__(self, default_capacity: float = 100.0, default_refill_rate: float = 10.0) -> None:
        self.default_capacity = float(default_capacity)
        self.default_refill_rate = float(default_refill_rate)
        self._buckets: Dict[str, BucketState] = {}
        self._lock = threading.Lock()

    def allow_request(self, client_id: str, cost: float = 1.0) -> Tuple[bool, Dict[str, float]]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(client_id)
            if bucket is None:
                bucket = BucketState(
                    capacity=self.default_capacity,
                    refill_rate=self.default_refill_rate,
                    tokens=self.default_capacity,
                    last_refill=now,
                )
                self._buckets[client_id] = bucket

            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_rate)
            bucket.last_refill = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, {
                    "remaining_tokens": round(bucket.tokens, 3),
                    "retry_after_seconds": 0.0,
                }

            deficit = cost - bucket.tokens
            retry_after = deficit / bucket.refill_rate if bucket.refill_rate > 0 else 60.0
            return False, {
                "remaining_tokens": round(bucket.tokens, 3),
                "retry_after_seconds": round(retry_after, 3),
            }


if __name__ == "__main__":
    limiter = TokenBucketRateLimiter(default_capacity=5.0, default_refill_rate=2.0)
    for i in range(7):
        allowed, meta = limiter.allow_request("substation-node-01", cost=1.0)
        print(f"Request #{i + 1}: allowed={allowed}, meta={meta}")
