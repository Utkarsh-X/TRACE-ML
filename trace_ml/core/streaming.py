"""Lightweight runtime event streaming primitives for future service integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Protocol

from trace_ml.core.models import utc_now_iso


@dataclass(slots=True)
class StreamEvent:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=utc_now_iso)


class EventStreamPublisher(Protocol):
    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a runtime event."""

    def subscribe(self, listener: Callable[[StreamEvent], None]) -> Callable[[], None]:
        """Register a listener and return an unsubscribe function."""

    def recent(self, limit: int = 100) -> list[StreamEvent]:
        """Return recent published events."""

    def latest(self, topic: str | None = None) -> StreamEvent | None:
        """Return the latest event, optionally filtered by topic."""

    def subscriber_count(self) -> int:
        """Return current subscriber count."""

    def last_published_at(self) -> str:
        """Return the timestamp of the latest publish operation."""


class NullEventStreamPublisher:
    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        return None

    def subscribe(self, listener: Callable[[StreamEvent], None]) -> Callable[[], None]:
        def _unsubscribe() -> None:
            return None

        return _unsubscribe

    def recent(self, limit: int = 100) -> list[StreamEvent]:
        return []

    def latest(self, topic: str | None = None) -> StreamEvent | None:
        return None

    def subscriber_count(self) -> int:
        return 0

    def last_published_at(self) -> str:
        return ""


class InMemoryEventStreamPublisher:
    def __init__(self, max_events: int = 512) -> None:
        self._events: deque[StreamEvent] = deque(maxlen=max_events)
        self._listeners: dict[int, Callable[[StreamEvent], None]] = {}
        self._next_id = 1
        self._last_published_at = ""
        self._lock = RLock()

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        event = StreamEvent(topic=topic, payload=payload)
        with self._lock:
            self._events.append(event)
            self._last_published_at = event.timestamp_utc
            listeners = list(self._listeners.values())
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                continue

    def subscribe(self, listener: Callable[[StreamEvent], None]) -> Callable[[], None]:
        with self._lock:
            token = self._next_id
            self._next_id += 1
            self._listeners[token] = listener

        def _unsubscribe() -> None:
            with self._lock:
                self._listeners.pop(token, None)

        return _unsubscribe

    def recent(self, limit: int = 100) -> list[StreamEvent]:
        with self._lock:
            items = list(self._events)
        if limit <= 0:
            return []
        return items[-limit:]

    def latest(self, topic: str | None = None) -> StreamEvent | None:
        with self._lock:
            items = list(self._events)
        if not items:
            return None
        if topic is None:
            return items[-1]
        for item in reversed(items):
            if item.topic == topic:
                return item
        return None

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._listeners)

    def last_published_at(self) -> str:
        with self._lock:
            return self._last_published_at
