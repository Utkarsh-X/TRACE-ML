"""Synchronous action execution with cooldown and audit logging."""

from __future__ import annotations

from datetime import datetime, timezone

from trace_aml.core.config import Settings
from trace_aml.core.ids import new_action_id
from trace_aml.core.models import ActionRecord, ActionStatus, ActionTrigger, ActionType, IncidentRecord
from trace_aml.store.vector_store import VectorStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionEngine:
    def __init__(self, store: VectorStore, settings: Settings) -> None:
        self.store = store
        self.config = settings

    @staticmethod
    def _parse_iso(value: str) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _cooldown_allows(self, incident: IncidentRecord) -> bool:
        last = self._parse_iso(incident.last_action_at)
        if last is None:
            return True
        delta = (datetime.now(timezone.utc) - last).total_seconds()
        return delta >= float(self.config.actions.cooldown_sec)

    def _run(self, action_type: ActionType, incident: IncidentRecord, trigger: ActionTrigger) -> tuple[bool, str]:
        if action_type == ActionType.log:
            print(f"[ACTION] log incident {incident.incident_id} ({trigger.value})")
            return True, "logged"
        if action_type == ActionType.email:
            print(f"[ACTION] email sent for {incident.incident_id} ({trigger.value})")
            return True, "email_sent"
        if action_type == ActionType.alarm:
            print(f"[ACTION] alarm triggered for {incident.incident_id} ({trigger.value})")
            return True, "alarm_triggered"
        return False, "unsupported_action_type"

    @staticmethod
    def _context_for(incident: IncidentRecord, trigger: ActionTrigger, action_type: ActionType, reason: str) -> dict[str, str]:
        return {
            "incident_id": incident.incident_id,
            "entity_id": incident.entity_id,
            "incident_status": str(incident.status),
            "incident_severity": str(incident.severity),
            "incident_summary": str(incident.summary),
            "trigger": trigger.value,
            "action_type": action_type.value,
            "explanation": reason,
        }

    def execute(
        self,
        incident: IncidentRecord,
        actions: list[ActionType],
        trigger: ActionTrigger | str,
    ) -> list[ActionRecord]:
        if not actions:
            return []

        trigger_value = ActionTrigger(str(trigger))
        if not self._cooldown_allows(incident):
            return []

        emitted: list[ActionRecord] = []
        for action_type in actions:
            ok, reason = self._run(action_type, incident, trigger_value)
            record = ActionRecord(
                action_id=new_action_id(),
                incident_id=incident.incident_id,
                action_type=action_type,
                trigger=trigger_value,
                status=ActionStatus.success if ok else ActionStatus.failed,
                reason=reason,
                context=self._context_for(incident, trigger_value, action_type, reason),
            )
            self.store.insert_action(record)
            emitted.append(record)

        if emitted:
            self.store.set_incident_last_action(incident.incident_id, utc_now_iso())
        return emitted
