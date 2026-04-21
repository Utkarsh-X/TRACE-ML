"""Abstract base class for all TRACE-AML action handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_aml.core.config import Settings
    from trace_aml.core.models import ActionTrigger, IncidentRecord
    from trace_aml.store.vector_store import VectorStore


class BaseActionHandler(ABC):
    """Base class for all action handlers.

    Each subclass implements a single delivery channel (email, WhatsApp, PDF, etc.).
    Handlers receive a shared *context* dict that is mutated in-place as actions
    execute sequentially. The pdf_report handler, for example, writes
    ``context["pdf_report_path"]`` so that downstream email/WhatsApp handlers
    can attach the generated file.

    Contract:
      - ``execute()`` must be safe to call even if the channel is disabled
        (return ``(False, "channel_disabled")`` cleanly).
      - ``execute()`` must never raise — catch all exceptions internally and
        return ``(False, error_reason)`` so the pipeline continues.
      - Heavy I/O (network calls) must run in a background daemon thread so
        the recognition pipeline is not blocked.
    """

    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings = settings
        self.store = store

    @abstractmethod
    def execute(
        self,
        incident: IncidentRecord,
        trigger: ActionTrigger,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Execute the action.

        Args:
            incident: The incident record that triggered this action.
            trigger:  ``on_create`` or ``on_update``.
            context:  Mutable shared dict passed between handlers in sequence.
                      Read keys from earlier handlers; write keys for later ones.

        Returns:
            (success: bool, reason: str) — reason is stored in ActionRecord.
        """
        ...
