"""Action handler registry — maps ActionType values to handler instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from trace_aml.core.models import ActionType

if TYPE_CHECKING:
    from trace_aml.core.config import Settings
    from trace_aml.core.models import ActionTrigger, IncidentRecord
    from trace_aml.store.vector_store import VectorStore


class ActionHandlerRegistry:
    """Instantiates and owns all action handlers.

    Handlers are imported lazily (inside ``__init__``) to keep import cost
    minimal during module load. The registry is created once per ``ActionEngine``
    instance and re-used for every ``dispatch()`` call.

    Adding a new channel:
      1. Create ``src/trace_aml/actions/<name>_handler.py``
      2. Import the class here and map a new ``ActionType`` value to it.
    """

    def __init__(self, settings: Settings, store: VectorStore) -> None:
        # Lazy imports so missing optional deps only fail at runtime when the
        # handler is actually used, not at import time.
        from trace_aml.actions.log_handler import LocalLogHandler
        from trace_aml.actions.pdf_handler import PdfReportHandler
        from trace_aml.actions.email_handler import EmailHandler
        from trace_aml.actions.whatsapp_handler import WhatsAppHandler

        wa_handler = WhatsAppHandler(settings, store)

        self._handlers: dict[ActionType, object] = {
            ActionType.log:        LocalLogHandler(settings, store),
            ActionType.pdf_report: PdfReportHandler(settings, store),
            ActionType.email:      EmailHandler(settings, store),
            ActionType.whatsapp:   wa_handler,
            ActionType.alarm:      wa_handler,  # Deprecated alias → WhatsApp
        }

    def dispatch(
        self,
        action_type: ActionType,
        incident: IncidentRecord,
        trigger: ActionTrigger,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Dispatch an action to its registered handler.

        Args:
            action_type: The type of action to execute.
            incident:    The triggering incident.
            trigger:     ``on_create`` or ``on_update``.
            context:     Mutable shared dict across handlers in the same execution batch.

        Returns:
            (success, reason) from the handler, or (False, "no_handler") if
            no handler is registered for this action type.
        """
        handler = self._handlers.get(action_type)
        if handler is None:
            logger.warning(
                "ActionHandlerRegistry: no handler registered for action type '{}'",
                action_type.value,
            )
            return False, f"no_handler_for_{action_type.value}"

        try:
            return handler.execute(incident, trigger, context)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover
            logger.error(
                "ActionHandlerRegistry: unhandled exception in {} handler: {}",
                action_type.value,
                exc,
            )
            return False, f"handler_exception:{exc}"
