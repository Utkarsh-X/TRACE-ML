"""LocalLogHandler — writes incident audit entries to the loguru logger."""

from __future__ import annotations

from typing import Any

from loguru import logger

from trace_aml.actions.base import BaseActionHandler
from trace_aml.core.models import ActionTrigger, IncidentRecord


class LocalLogHandler(BaseActionHandler):
    """Writes a structured audit log entry for every triggered action.

    This handler is synchronous and always succeeds (unless loguru itself
    is broken, which would be catastrophic anyway). It replaces the old
    ``print()`` statement in ActionEngine._run().

    The log entry includes:
      - Incident ID, entity ID
      - Severity and alert count
      - Trigger type (on_create / on_update)
      - Incident summary
    """

    def execute(
        self,
        incident: IncidentRecord,
        trigger: ActionTrigger,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        sev = str(incident.severity.value if hasattr(incident.severity, "value") else incident.severity).upper()
        logger.info(
            "[ACTION:LOG] incident={} entity={} severity={} alerts={} trigger={} | {}",
            incident.incident_id[-8:],
            incident.entity_id,
            sev,
            incident.alert_count,
            trigger.value,
            incident.summary or "(no summary)",
        )
        return True, "logged"
