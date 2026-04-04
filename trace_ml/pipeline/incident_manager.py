"""Incident grouping and lifecycle management."""

from __future__ import annotations

from trace_ml.core.ids import new_incident_id
from trace_ml.core.models import AlertRecord, IncidentRecord, IncidentStatus
from trace_ml.store.vector_store import VectorStore


class IncidentManager:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    @staticmethod
    def _build_summary(alert: AlertRecord) -> str:
        return f"{alert.type.value}: {alert.reason}"

    def handle_alert(self, alert: AlertRecord) -> tuple[IncidentRecord, str]:
        active = self.store.get_active_incident(alert.entity_id)
        if active:
            alert_ids = [str(v) for v in active.get("alert_ids", [])]
            if alert.alert_id not in alert_ids:
                alert_ids.append(alert.alert_id)
            updated = IncidentRecord(
                incident_id=str(active.get("incident_id", "")),
                entity_id=str(active.get("entity_id", "")),
                status=IncidentStatus.open,
                start_time=str(active.get("start_time", alert.timestamp_utc)),
                last_seen_time=alert.timestamp_utc,
                alert_ids=alert_ids,
                alert_count=len(alert_ids),
                severity=str(active.get("severity", "low")),
                summary=self._build_summary(alert),
                last_action_at=str(active.get("last_action_at", "")),
            )
            self.store.update_incident(updated)
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
