"""Deterministic event rules engine for alert generation."""

from __future__ import annotations

import time

import numpy as np

from trace_aml.core.config import Settings
from trace_aml.core.ids import new_alert_id
from trace_aml.core.models import AlertRecord, AlertSeverity, AlertType, EventRecord
from trace_aml.store.vector_store import VectorStore


class RulesEngine:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.config = settings
        self.store = store
        self.cache: dict[tuple[str, str], float] = {}

    def process_event(self, event: EventRecord) -> list[AlertRecord]:
        alerts: list[AlertRecord] = []
        alerts.extend(self._check_reappearance(event))
        alerts.extend(self._check_unknown_recurrence(event))
        alerts.extend(self._check_instability(event))
        return alerts

    def _check_reappearance(self, event: EventRecord) -> list[AlertRecord]:
        cfg = self.config.rules.reappearance
        events = self.store.get_events(event.entity_id, cfg.window_sec)
        if len(events) >= cfg.min_events and self._cooldown(event.entity_id, AlertType.reappearance):
            return [self._build_alert(event, AlertType.reappearance, len(events))]
        return []

    def _check_unknown_recurrence(self, event: EventRecord) -> list[AlertRecord]:
        if not event.is_unknown:
            return []
        cfg = self.config.rules.unknown
        events = self.store.get_events(event.entity_id, cfg.window_sec)
        if len(events) >= cfg.min_events and self._cooldown(event.entity_id, AlertType.unknown_recurrence):
            return [self._build_alert(event, AlertType.unknown_recurrence, len(events))]
        return []

    def _check_instability(self, event: EventRecord) -> list[AlertRecord]:
        cfg = self.config.rules.instability
        events = self.store.get_events(event.entity_id, cfg.window_sec)
        confidences: list[float] = []
        for item in events:
            value = float(item.get("confidence", 0.0))
            if value > 1.0:
                value = value / 100.0
            confidences.append(value)

        if len(confidences) < 3:
            return []

        std = float(np.std(np.asarray(confidences, dtype=np.float32)))
        if std > cfg.std_threshold and self._cooldown(event.entity_id, AlertType.instability):
            return [self._build_alert(event, AlertType.instability, len(events))]
        return []

    def _build_alert(self, event: EventRecord, alert_type: AlertType, count: int) -> AlertRecord:
        severity = self._map_severity(event, alert_type, count)
        # We provide a clean reason without the type prefix here, 
        # as the incident manager or UI will add the type context.
        reason = f"Detected with {count} events"
        if alert_type == AlertType.instability:
            reason = f"Confidence instability across {count} events"
        
        return AlertRecord(
            alert_id=new_alert_id(),
            entity_id=event.entity_id,
            type=alert_type,
            severity=severity,
            reason=reason,
            timestamp_utc=event.timestamp_utc,
            first_seen_at=event.timestamp_utc,
            last_seen_at=event.timestamp_utc,
            event_count=count,
        )

    def _cooldown(self, entity_id: str, rule_type: AlertType) -> bool:
        key = (entity_id, rule_type.value)
        now = time.time()
        last = self.cache.get(key)
        if last and now - last < float(self.config.rules.cooldown_sec):
            return False
        self.cache[key] = now
        return True

    @staticmethod
    def _map_severity(event: EventRecord, alert_type: AlertType, count: int) -> AlertSeverity:
        if alert_type == AlertType.unknown_recurrence:
            return AlertSeverity.high
        if count >= 5:
            return AlertSeverity.medium
        return AlertSeverity.low
