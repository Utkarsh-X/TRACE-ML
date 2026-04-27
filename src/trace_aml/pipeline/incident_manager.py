"""Incident grouping and lifecycle management."""

from __future__ import annotations

import time
from typing import ClassVar

from trace_aml.core.ids import new_incident_id
from trace_aml.core.models import AlertRecord, AlertSeverity, IncidentRecord, IncidentStatus
from trace_aml.store.vector_store import VectorStore


class IncidentManager:
    # Re-notification gap: if an entity has been absent for longer than this,
    # the next sighting is treated as a fresh on_create so email/WA fire again.
    RENOTIFY_GAP_SEC: ClassVar[float] = 300.0  # 5 minutes

    def __init__(self, store: VectorStore) -> None:
        self.store = store
        # Maps entity_id → monotonic timestamp of last seen event
        self._last_seen: dict[str, float] = {}

    @staticmethod
    def _build_summary(alert: AlertRecord) -> str:
        # alert.type is like 'REAPPEARANCE'
        # alert.reason is now 'Detected with X events'
        # Result: 'REAPPEARANCE: Detected with 8 events'
        return f"{alert.type.value}: {alert.reason}"

    def mark_seen(self, entity_id: str) -> None:
        """Called each time an entity event is processed; tracks last-seen time."""
        self._last_seen[entity_id] = time.monotonic()

    def _was_recently_seen(self, entity_id: str) -> bool:
        """Return True if this entity was seen within RENOTIFY_GAP_SEC."""
        last = self._last_seen.get(entity_id)
        if last is None:
            return False  # Never seen this session → fresh
        return (time.monotonic() - last) < self.RENOTIFY_GAP_SEC

    def handle_alert(self, alert: AlertRecord) -> tuple[IncidentRecord, str]:
        # Capture whether entity was recently seen BEFORE marking it as seen now.
        # This ensures the 5-min re-notification gap is computed against the
        # PREVIOUS sighting, not the current one.
        was_recent = self._was_recently_seen(alert.entity_id)
        self.mark_seen(alert.entity_id)  # Update timestamp for next call

        active = self.store.get_active_incident(alert.entity_id)
        if active:
            alert_ids = [str(v) for v in active.get("alert_ids", [])]
            if alert.alert_id not in alert_ids:
                alert_ids.append(alert.alert_id)

            # ── Severity sync: always reflect the current alert's severity ──
            # This prevents a stale "low" from blocking high-severity actions
            # when the entity's enrolled priority was updated after incident creation.
            current_severity = alert.severity

            # ── Re-notification: clear last_action_at if entity was absent ──
            # If the entity disappeared for > RENOTIFY_GAP_SEC and just came back,
            # we reset last_action_at so the action engine treats this like a
            # fresh create — firing email/WA again without creating a new incident.
            existing_last_action = str(active.get("last_action_at", ""))
            if not was_recent:
                # Entity was absent long enough — force re-notification
                existing_last_action = ""

            updated = IncidentRecord(
                incident_id=str(active.get("incident_id", "")),
                entity_id=str(active.get("entity_id", "")),
                status=IncidentStatus.open,
                start_time=str(active.get("start_time", alert.timestamp_utc)),
                last_seen_time=alert.timestamp_utc,
                alert_ids=alert_ids,
                alert_count=len(alert_ids),
                severity=current_severity,
                summary=self._build_summary(alert),
                last_action_at=existing_last_action,
            )
            self.store.update_incident(updated)

            # Treat as on_create if:
            #  a) incident has never been actioned, OR
            #  b) entity was absent long enough (re-notification gap expired)
            if not existing_last_action:
                return updated, "on_create"
            return updated, "on_update"

        created = IncidentRecord(
            incident_id=new_incident_id(),
            entity_id=alert.entity_id,
            status=IncidentStatus.open,
            start_time=alert.timestamp_utc,
            last_seen_time=alert.timestamp_utc,
            alert_ids=[alert.alert_id],
            alert_count=1,
            severity=alert.severity,
            summary=self._build_summary(alert),
        )
        self.store.create_incident(created)
        return created, "on_create"
