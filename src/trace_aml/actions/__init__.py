"""Action handler package for TRACE-AML.

Provides a registry of pluggable action handlers that the ActionEngine
dispatches to. Add new channels by creating a handler module and
registering it in ActionHandlerRegistry.

Available handlers:
  - LocalLogHandler    (ActionType.log)
  - PdfReportHandler   (ActionType.pdf_report)
  - EmailHandler       (ActionType.email)
  - WhatsAppHandler    (ActionType.whatsapp / ActionType.alarm)
"""

from trace_aml.actions.base import BaseActionHandler
from trace_aml.actions.registry import ActionHandlerRegistry

__all__ = ["BaseActionHandler", "ActionHandlerRegistry"]
