"""EmailHandler — sends formatted HTML email with optional PDF attachment via SMTP.

Security:
  - `smtp_password` is read from env var TRACE_AML_SMTP_PASSWORD first.
  - The YAML config field is an insecure fallback only.
  - Passwords are never logged.

Threading:
  - `_build_message()` is synchronous and fast (string ops only).
  - `_send()` runs in a background daemon thread so the pipeline is never blocked.
  - On failure: logs the error, returns (False, reason) — never raises.
"""

from __future__ import annotations

import os
import smtplib
import threading
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from loguru import logger

from trace_aml.actions.base import BaseActionHandler
from trace_aml.core.models import ActionTrigger, IncidentRecord


def _fmt_ts(iso: str) -> str:
    if not iso:
        return "—"
    try:
        text = str(iso).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        from datetime import datetime
        dt = datetime.fromisoformat(text)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(iso)[:19]


class EmailHandler(BaseActionHandler):
    """Sends an HTML email with incident summary and optional PDF attachment."""

    def execute(
        self,
        incident: IncidentRecord,
        trigger: ActionTrigger,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        cfg = self.settings.notifications.email
        if not cfg.enabled:
            return False, "email_disabled"

        if not cfg.recipient_addresses:
            return False, "email_no_recipients"

        if not cfg.smtp_host:
            return False, "email_no_smtp_host"

        # PDF path may have been set by PdfReportHandler earlier in the batch
        pdf_path = context.get("pdf_report_path")

        try:
            msg = self._build_message(incident, cfg, pdf_path)
        except Exception as exc:
            logger.error("[ACTION:EMAIL] message build failed: {}", exc)
            return False, f"email_build_error:{exc}"

        # Fire-and-forget — SMTP delivery in a background thread
        threading.Thread(
            target=self._send,
            args=(msg, cfg),
            daemon=True,
            name=f"email-{incident.incident_id[-6:]}",
        ).start()
        logger.info(
            "[ACTION:EMAIL] queued for {} recipient(s) (incident {})",
            len(cfg.recipient_addresses),
            incident.incident_id[-8:],
        )
        return True, "email_queued"

    def execute_sync(
        self,
        incident: IncidentRecord,
        trigger: ActionTrigger,
        context: dict[str, Any],
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """Synchronous version that waits for email delivery (used for testing).

        Args:
            incident: The incident record that triggered this action.
            trigger:  ``on_create`` or ``on_update``.
            context:  Mutable shared dict passed between handlers in sequence.
            timeout:  Max seconds to wait for SMTP delivery.

        Returns:
            (success: bool, reason: str) — reason is stored in ActionRecord.
        """
        cfg = self.settings.notifications.email
        if not cfg.enabled:
            return False, "email_disabled"

        if not cfg.recipient_addresses:
            return False, "email_no_recipients"

        if not cfg.smtp_host:
            return False, "email_no_smtp_host"

        # PDF path may have been set by PdfReportHandler earlier in the batch
        pdf_path = context.get("pdf_report_path")

        try:
            msg = self._build_message(incident, cfg, pdf_path)
        except Exception as exc:
            logger.error("[ACTION:EMAIL] message build failed: {}", exc)
            return False, f"email_build_error:{exc}"

        # Use a queue to get the result from the background thread
        import queue
        result_queue: queue.Queue[tuple[bool, str | None]] = queue.Queue()

        # Start non-daemon thread and wait for it (for test mode)
        def send_with_result():
            self._send(msg, cfg, result_queue)

        t = threading.Thread(
            target=send_with_result,
            daemon=False,  # Non-daemon so we can wait for it
            name=f"email-sync-{incident.incident_id[-6:]}",
        )
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Thread is still running after timeout
            return False, "email_timeout"

        # Get result from queue (should be available since thread completed)
        try:
            success, error_reason = result_queue.get(block=False)
            if success:
                return True, "email_sent"
            else:
                return False, error_reason or "email_failed"
        except queue.Empty:
            return False, "email_no_result"

    # ── Message builder ────────────────────────────────────────────────────────

    def _build_message(
        self,
        incident: IncidentRecord,
        cfg: Any,
        pdf_path: str | None,
    ) -> MIMEMultipart:
        severity = str(
            incident.severity.value if hasattr(incident.severity, "value")
            else incident.severity
        ).lower()

        SEV_COLORS = {"high": "#dc2626", "medium": "#eab308", "low": "#4ade80"}
        sev_color = SEV_COLORS.get(severity, "#6b7280")

        subject = (
            f"🚨 TRACE-AML Alert — {severity.upper()} — "
            f"Incident {incident.incident_id[-8:]}"
        )

        sender_str = (
            f"{cfg.sender_name} <{cfg.sender_address}>"
            if cfg.sender_address else
            cfg.sender_name
        )

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = sender_str
        msg["To"]      = ", ".join(cfg.recipient_addresses)

        # ── HTML body ─────────────────────────────────────────────────────────
        html_body = self._render_html(incident, severity, sev_color, pdf_path)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # ── PDF attachment ────────────────────────────────────────────────────
        if pdf_path and cfg.attach_pdf:
            try:
                import os as _os
                with open(pdf_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{_os.path.basename(pdf_path)}"',
                )
                msg.attach(part)
            except Exception as exc:
                logger.warning("[ACTION:EMAIL] PDF attachment failed: {}", exc)

        return msg

    def _render_html(
        self,
        incident: IncidentRecord,
        severity: str,
        sev_color: str,
        pdf_path: str | None,
    ) -> str:
        inc_short = incident.incident_id[-8:]
        ent_id    = incident.entity_id
        status    = str(
            incident.status.value if hasattr(incident.status, "value")
            else incident.status
        ).upper()
        summary   = incident.summary or "—"[:120]
        start     = _fmt_ts(incident.start_time)
        last_seen = _fmt_ts(incident.last_seen_time)

        pdf_note = ""
        if pdf_path:
            import os as _os
            fname = _os.path.basename(pdf_path)
            pdf_note = (
                f'<tr><td class="k">Report</td>'
                f'<td><span style="color:#4ade80">Full PDF attached ({fname})</span></td></tr>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<style>
  body{{margin:0;padding:0;background:#0a0e1a;font-family:'Courier New',monospace;color:#e6ebf5}}
  .wrap{{max-width:600px;margin:24px auto;background:#141928;border:1px solid #28324a;
          border-top:3px solid {sev_color};padding:24px;border-radius:4px}}
  h1{{font-size:18px;letter-spacing:0.15em;margin:0 0 4px}}
  .sub{{color:#64748b;font-size:11px;margin-bottom:16px;letter-spacing:0.1em}}
  .badge{{display:inline-block;padding:2px 10px;background:{sev_color}22;color:{sev_color};
           border:1px solid {sev_color};font-size:11px;letter-spacing:0.15em;border-radius:2px}}
  table{{width:100%;border-collapse:collapse;margin:16px 0}}
  td{{padding:6px 0;border-bottom:1px solid #1a2236;font-size:12px;vertical-align:top}}
  .k{{color:#64748b;width:110px;padding-right:12px;white-space:nowrap}}
  .footer{{margin-top:20px;padding-top:12px;border-top:1px solid #1a2236;
            color:#3b4a6b;font-size:10px;text-align:center}}
</style>
</head>
<body>
<div class="wrap">
  <h1>TRACE-AML</h1>
  <div class="sub">SECURITY ALERT NOTIFICATION</div>
  <span class="badge">{severity.upper()} SEVERITY</span>

  <table>
    <tr><td class="k">Incident ID</td><td>{inc_short}</td></tr>
    <tr><td class="k">Entity</td><td>{ent_id}</td></tr>
    <tr><td class="k">Status</td><td>{status}</td></tr>
    <tr><td class="k">Severity</td><td style="color:{sev_color}">{severity.upper()}</td></tr>
    <tr><td class="k">Alert Count</td><td>{incident.alert_count}</td></tr>
    <tr><td class="k">Started</td><td>{start}</td></tr>
    <tr><td class="k">Last Seen</td><td>{last_seen}</td></tr>
    <tr><td class="k">Summary</td><td>{summary}</td></tr>
    {pdf_note}
  </table>

  <div class="footer">
    TRACE-AML v4 Security System &middot; Incident {inc_short} &middot; Do not reply
  </div>
</div>
</body>
</html>"""

    # ── SMTP delivery (runs in background thread) ──────────────────────────────

    def _send(self, msg: MIMEMultipart, cfg: Any, result_queue: Any = None) -> None:
        """Actually deliver the email. Always runs in a daemon thread.

        Connection strategy (auto-detected from port):
          Port 465  → SMTP_SSL  (implicit TLS from the start)
          Port 587  → SMTP + STARTTLS  (upgrade after hello)
          Other     → respects cfg.use_tls flag

        STARTTLS requires:
          1. ehlo()          — identify client, server advertises plain capabilities
          2. starttls()      — upgrade socket to TLS
          3. ehlo()          — re-identify; server now advertises AUTH, etc.
          4. login()         — only now can AUTH succeed
        """
        # Password priority: 1) env var, 2) config file, 3) empty
        password = os.getenv("TRACE_AML_SMTP_PASSWORD") or cfg.smtp_password or ""

        use_ssl      = cfg.smtp_port == 465
        use_starttls = (cfg.smtp_port == 587) or (cfg.use_tls and not use_ssl)

        success = False
        error_reason = None
        try:
            if use_ssl:
                # ── Mode 1: Implicit TLS (port 465) ──────────────────────
                import ssl
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port,
                                      context=ctx, timeout=15) as server:
                    server.ehlo()
                    if cfg.smtp_user and password:
                        server.login(cfg.smtp_user, password)
                    server.send_message(msg)

            elif use_starttls:
                # ── Mode 2: STARTTLS (port 587) ───────────────────────────
                # MANDATORY double-EHLO: server only advertises AUTH after TLS upgrade
                with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as server:
                    server.ehlo()            # Step 1 — identify before upgrade
                    server.starttls()        # Step 2 — upgrade to TLS
                    server.ehlo()            # Step 3 — re-identify over TLS so AUTH is visible
                    if cfg.smtp_user and password:
                        server.login(cfg.smtp_user, password)
                    server.send_message(msg)

            else:
                # ── Mode 3: Plain SMTP (local relay / no auth) ────────────
                with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as server:
                    server.ehlo()
                    if cfg.smtp_user and password:
                        server.login(cfg.smtp_user, password)
                    server.send_message(msg)

            success = True
            logger.info(
                "[ACTION:EMAIL] delivered to: {}",
                ", ".join(cfg.recipient_addresses),
            )

        except smtplib.SMTPAuthenticationError:
            error_reason = "smtp_auth_failed"
            logger.error(
                "[ACTION:EMAIL] SMTP auth failed — check smtp_user / password. "
                "For Gmail use an App Password (myaccount.google.com/apppasswords)."
            )
        except smtplib.SMTPException as exc:
            error_reason = f"smtp_error:{exc}"
            logger.error("[ACTION:EMAIL] SMTP error: {}", exc)
        except OSError as exc:
            error_reason = f"network_error:{exc}"
            logger.error("[ACTION:EMAIL] network error (host={} port={}): {}",
                         cfg.smtp_host, cfg.smtp_port, exc)
        except Exception as exc:
            error_reason = f"unexpected_error:{exc}"
            logger.error("[ACTION:EMAIL] unexpected error: {}", exc)

        # Report result if a queue was provided (for synchronous/blocking mode)
        if result_queue is not None:
            try:
                result_queue.put((success, error_reason))
            except Exception:
                pass

