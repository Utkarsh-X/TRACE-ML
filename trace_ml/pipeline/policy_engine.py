"""Policy evaluation for incident-level actions."""

from __future__ import annotations

from trace_aml.core.config import Settings
from trace_aml.core.models import ActionTrigger, ActionType, IncidentRecord


class PolicyEngine:
    def __init__(self, settings: Settings) -> None:
        self.config = settings

    def evaluate(self, incident: IncidentRecord, trigger: ActionTrigger | str) -> list[ActionType]:
        if not self.config.actions.enabled:
            return []

        trigger_value = str(trigger)
        severity = str(incident.severity)
        if trigger_value == ActionTrigger.on_create.value:
            configured = getattr(self.config.actions.on_create, severity, [])
        else:
            configured = getattr(self.config.actions.on_update, severity, [])

        resolved: list[ActionType] = []
        for value in configured:
            try:
                resolved.append(ActionType(str(value)))
            except ValueError:
                continue
        return resolved
