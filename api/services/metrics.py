from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict


@dataclass
class _Requests:
    total: int = 0
    single: int = 0
    batch_records: int = 0


@dataclass
class _Latency:
    average: float = 0.0
    last: float = 0.0
    _count: int = 0


@dataclass
class _MetricsStore:
    _lock: Lock = field(default_factory=Lock)

    _requests: _Requests = field(default_factory=_Requests)
    _latency: _Latency = field(default_factory=_Latency)
    _model_decisions: Dict[str, int] = field(default_factory=dict)

    def record_request(self, duration_ms: float) -> None:
        with self._lock:
            self._requests.total += 1
            self._latency.last = float(duration_ms)
            self._latency._count += 1
            n = self._latency._count
            self._latency.average = (
                (self._latency.average * (n - 1)) + duration_ms) / n

    def record_single(self) -> None:
        with self._lock:
            self._requests.single += 1

    def record_batch(self, batch_size: int) -> None:
        with self._lock:
            self._requests.batch_records += int(batch_size)

    def record_decision(self, decision: str) -> None:
        if not decision:
            return
        with self._lock:
            self._model_decisions[decision] = self._model_decisions.get(
                decision, 0) + 1

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "requests": {
                    "total": self._requests.total,
                    "single": self._requests.single,
                    "batch_records": self._requests.batch_records,
                },
                "latency_ms": {
                    "average": round(self._latency.average, 2),
                    "last": round(self._latency.last, 2),
                },
                "model_decisions": dict(self._model_decisions),
            }


metrics_store = _MetricsStore()
