"""PdfReportHandler — generates fpdf2-based PDF incident reports.

The generated PDF is saved to ``data/exports/{YYYY}/{entity_id}_{inc_id}_{ts}.pdf``
and the path is written to ``context["pdf_report_path"]`` so that the
EmailHandler and WhatsAppHandler can attach it.

A companion ``.html`` file is also generated at the same path for
browser-based viewing, served as a static file by FastAPI.

Report sections:
  1. Header   — TRACE-AML watermark + generation metadata
  2. Entity   — ID, type, person name (if known), portrait image, first/last seen
  3. Incident — severity badge, open duration, status, alert count
  4. Alert log table (capped at max_alert_rows)
  5. Detection log table (capped at max_detection_rows)
  6. Action history table
  7. Footer   — generation timestamp + confidentiality notice
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from trace_aml.actions.base import BaseActionHandler
from trace_aml.core.models import ActionTrigger, IncidentRecord


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_ts(iso: str) -> str:
    """Format ISO timestamp into human-readable string."""
    if not iso:
        return "—"
    try:
        text = str(iso).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(iso)[:19]


def _duration(start_iso: str, end_iso: str) -> str:
    """Calculate human-readable duration between two ISO timestamps."""
    try:
        def _parse(s: str) -> datetime:
            s = s.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        start = _parse(start_iso)
        end   = _parse(end_iso) if end_iso else _utc_now()
        secs  = int((end - start).total_seconds())
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        else:
            h = secs // 3600
            m = (secs % 3600) // 60
            return f"{h}h {m}m"
    except Exception:
        return "—"


class PdfReportHandler(BaseActionHandler):
    """Generates a PDF + HTML incident report using fpdf2."""

    def execute(
        self,
        incident: IncidentRecord,
        trigger: ActionTrigger,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        cfg = self.settings.notifications.pdf_report
        if not cfg.enabled:
            return False, "pdf_report_disabled"

        try:
            pdf_path, html_path = self._generate(incident, cfg)
            # Write path to context so email/WA handlers can attach the PDF
            context["pdf_report_path"] = str(pdf_path)
            context["html_report_path"] = str(html_path)
            reason = f"pdf_generated:{pdf_path.name}"
            logger.info("[ACTION:PDF] report generated → {}", pdf_path)
            return True, reason
        except Exception as exc:
            logger.error("[ACTION:PDF] report generation failed for {}: {}",
                         incident.incident_id, exc)
            return False, f"pdf_error:{exc}"

    # ── Internal generation logic ──────────────────────────────────────────────

    def _generate(self, incident: IncidentRecord, cfg: Any) -> tuple[Path, Path]:
        """Build both PDF and HTML reports. Returns (pdf_path, html_path)."""
        now = _utc_now()
        ts_str = now.strftime("%Y%m%d_%H%M%S")
        year_str = now.strftime("%Y")

        out_dir = Path(cfg.output_dir).resolve() / year_str
        out_dir.mkdir(parents=True, exist_ok=True)

        inc_short = incident.incident_id[-8:]
        stem = f"{incident.entity_id}_{inc_short}_{ts_str}"

        pdf_path  = out_dir / f"{stem}.pdf"
        html_path = out_dir / f"{stem}.html"

        # Fetch all data we need from the store
        entity_row   = self.store.get_entity(incident.entity_id) or {}
        alerts       = self.store.list_alerts(
            limit=cfg.max_alert_rows, entity_id=incident.entity_id
        )
        # Filter only alerts linked to this incident
        inc_alert_ids = set(incident.alert_ids)
        if inc_alert_ids:
            alerts = [a for a in alerts if a.get("alert_id") in inc_alert_ids]
        alerts = alerts[:cfg.max_alert_rows]

        detections   = self.store.list_events(
            limit=cfg.max_detection_rows, entity_id=incident.entity_id
        )[:cfg.max_detection_rows]
        actions_hist = self.store.get_actions(incident.incident_id, limit=30)

        # Resolve linked person name/category
        person_name = "Unknown"
        person_category = "—"
        src_pid = entity_row.get("source_person_id", "")
        if src_pid:
            try:
                person_row = self.store.get_person(src_pid)
                if person_row:
                    person_name = person_row.get("name", "Unknown")
                    person_category = person_row.get("category", "—")
            except Exception:
                pass

        # Portrait path
        portrait_path: str | None = None
        if cfg.include_entity_portrait and src_pid:
            portraits_dir = Path(self.settings.store.portraits_dir).resolve()
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = portraits_dir / f"{src_pid}{ext}"
                if candidate.exists():
                    portrait_path = str(candidate)
                    break

        self._build_pdf(
            pdf_path=pdf_path,
            incident=incident,
            entity_row=entity_row,
            person_name=person_name,
            person_category=person_category,
            portrait_path=portrait_path,
            alerts=alerts,
            detections=detections,
            actions_hist=actions_hist,
            generated_at=now,
        )
        self._build_html(
            html_path=html_path,
            incident=incident,
            entity_row=entity_row,
            person_name=person_name,
            person_category=person_category,
            portrait_path=portrait_path,
            alerts=alerts,
            detections=detections,
            actions_hist=actions_hist,
            generated_at=now,
        )
        return pdf_path, html_path

    # ── PDF builder (fpdf2) ──────────────────────────────────────────────────

    def _build_pdf(
        self,
        pdf_path: Path,
        incident: IncidentRecord,
        entity_row: dict,
        person_name: str,
        person_category: str,
        portrait_path: str | None,
        alerts: list[dict],
        detections: list[dict],
        actions_hist: list[dict],
        generated_at: datetime,
    ) -> None:
        from fpdf import FPDF

        SEV_COLORS = {
            "high":   (220, 38,  38),   # red
            "medium": (234, 179, 8),    # amber
            "low":    (74,  222, 128),  # green
        }
        BG      = (10, 14, 26)
        FG      = (230, 235, 245)
        GREY    = (100, 110, 130)
        BORDER  = (40, 50, 70)

        severity = str(
            incident.severity.value if hasattr(incident.severity, "value")
            else incident.severity
        ).lower()
        sev_rgb = SEV_COLORS.get(severity, (100, 110, 130))

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        W = pdf.w - pdf.l_margin - pdf.r_margin  # usable width

        # ── Header ──────────────────────────────────────────────────────────
        pdf.set_fill_color(*BG)
        pdf.rect(0, 0, pdf.w, 28, "F")

        pdf.set_xy(pdf.l_margin, 8)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*FG)
        pdf.cell(W // 2, 8, "TRACE-AML", ln=False)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*GREY)
        pdf.set_x(pdf.l_margin + W // 2)
        pdf.cell(W // 2, 8,
                 f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 align="R", ln=True)

        pdf.set_xy(pdf.l_margin, 18)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*sev_rgb)
        pdf.cell(W, 6, f"INCIDENT REPORT  ·  {severity.upper()} SEVERITY", ln=True)

        pdf.ln(8)

        # ── Incident Summary ─────────────────────────────────────────────────
        pdf.set_fill_color(*BORDER)
        pdf.rect(pdf.l_margin, pdf.get_y(), W, 0.3, "F")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(W, 5, "INCIDENT", ln=True)
        pdf.ln(1)

        inc_id_short = incident.incident_id[-8:]
        dur = _duration(incident.start_time, incident.last_seen_time)

        inc_fields = [
            ("Incident ID",  inc_id_short),
            ("Status",       str(incident.status.value if hasattr(incident.status, "value") else incident.status).upper()),
            ("Severity",     severity.upper()),
            ("Duration",     dur),
            ("Alert Count",  str(incident.alert_count)),
            ("Summary",      (incident.summary or "—")[:80]),
        ]
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*FG)
        col_w = W / 2
        for i, (label, val) in enumerate(inc_fields):
            if i % 2 == 0 and i > 0:
                pdf.ln(0)
            x = pdf.l_margin + (i % 2) * col_w
            pdf.set_xy(x, pdf.get_y())
            pdf.set_text_color(*GREY)
            pdf.cell(30, 5, f"{label}:", ln=False)
            pdf.set_text_color(*FG)
            pdf.cell(col_w - 30, 5, val, ln=(i % 2 == 1))

        pdf.ln(4)

        # ── Entity ──────────────────────────────────────────────────────────
        pdf.set_fill_color(*BORDER)
        pdf.rect(pdf.l_margin, pdf.get_y(), W, 0.3, "F")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GREY)

        # Portrait on the right if available
        portrait_w = 28
        text_w = W - portrait_w - 4 if portrait_path else W

        y_before_entity = pdf.get_y()
        pdf.cell(text_w, 5, "ENTITY", ln=True)
        pdf.ln(1)

        entity_type = str(entity_row.get("type", "unknown")).upper()
        status_str  = str(entity_row.get("status", "active")).upper()

        ent_fields = [
            ("Entity ID",    incident.entity_id),
            ("Type",         entity_type),
            ("Status",       status_str),
            ("Name",         person_name),
            ("Category",     person_category),
            ("First Seen",   _fmt_ts(str(entity_row.get("created_at", "")))),
            ("Last Seen",    _fmt_ts(str(entity_row.get("last_seen_at", "")))),
        ]
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*FG)
        for label, val in ent_fields:
            pdf.set_text_color(*GREY)
            pdf.cell(30, 5, f"{label}:", ln=False)
            pdf.set_text_color(*FG)
            pdf.cell(text_w - 30, 5, val, ln=True)

        if portrait_path and os.path.exists(portrait_path):
            try:
                px = pdf.l_margin + text_w + 4
                py = y_before_entity
                pdf.image(portrait_path, x=px, y=py, w=portrait_w)
            except Exception as exc:
                logger.debug("PDF portrait embed failed: {}", exc)

        pdf.ln(4)

        # ── Alert Table ──────────────────────────────────────────────────────
        self._pdf_section_header(pdf, "ALERT LOG", W, BORDER, GREY, FG)
        col_defs = [("Type", 50), ("Severity", 25), ("Time", 50), ("Events", 20), ("Reason", W - 145)]
        self._pdf_table(pdf, col_defs, alerts, lambda row: [
            str(row.get("type", "—")).upper(),
            str(row.get("severity", "—")).upper(),
            _fmt_ts(str(row.get("timestamp_utc", ""))),
            str(row.get("event_count", "1")),
            (str(row.get("reason", "—")))[:40],
        ], GREY, FG, BORDER)

        # ── Detection Table ──────────────────────────────────────────────────
        self._pdf_section_header(pdf, "RECENT DETECTIONS", W, BORDER, GREY, FG)
        det_cols = [("Time", 45), ("Confidence", 28), ("Decision", 35), ("Source", 30), ("Track", W - 138)]
        self._pdf_table(pdf, det_cols, detections, lambda row: [
            _fmt_ts(str(row.get("timestamp_utc", ""))),
            f"{float(row.get('confidence', 0)) * 100:.1f}%",
            str(row.get("decision_state", row.get("result", "—"))),
            str(row.get("source", "cam:0")),
            str(row.get("track_id", "—")),
        ], GREY, FG, BORDER)

        # ── Action History ───────────────────────────────────────────────────
        self._pdf_section_header(pdf, "ACTIONS TAKEN", W, BORDER, GREY, FG)
        act_cols = [("Type", 35), ("Status", 25), ("Time", 50), ("Reason", W - 110)]
        self._pdf_table(pdf, act_cols, actions_hist, lambda row: [
            str(row.get("action_type", "—")).upper(),
            str(row.get("status", "—")).upper(),
            _fmt_ts(str(row.get("timestamp_utc", ""))),
            str(row.get("reason", "—"))[:50],
        ], GREY, FG, BORDER)

        # ── Footer ───────────────────────────────────────────────────────────
        pdf.set_y(-18)
        pdf.set_fill_color(*BORDER)
        pdf.rect(0, pdf.get_y(), pdf.w, 0.3, "F")
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*GREY)
        pdf.cell(W / 2, 5,
                 f"TRACE-AML v4 · Incident {inc_id_short} · CONFIDENTIAL",
                 ln=False)
        pdf.cell(W / 2, 5,
                 f"Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 align="R", ln=True)

        pdf.output(str(pdf_path))

    @staticmethod
    def _pdf_section_header(pdf: Any, title: str, W: float,
                            BORDER: tuple, GREY: tuple, FG: tuple) -> None:
        pdf.set_fill_color(*BORDER)
        pdf.rect(pdf.l_margin, pdf.get_y(), W, 0.3, "F")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(W, 5, title, ln=True)
        pdf.ln(1)

    @staticmethod
    def _pdf_table(pdf: Any, col_defs: list[tuple[str, float]], rows: list[dict],
                   row_fn: Any, GREY: tuple, FG: tuple, BORDER: tuple) -> None:
        if not rows:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*GREY)
            pdf.cell(sum(w for _, w in col_defs), 5, "No records", ln=True)
            pdf.ln(2)
            return

        # Header row
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*GREY)
        for label, w in col_defs:
            pdf.cell(w, 5, label, ln=False)
        pdf.ln()

        # Data rows
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*FG)
        for row in rows:
            try:
                values = row_fn(row)
            except Exception:
                continue
            for (_, w), val in zip(col_defs, values):
                pdf.cell(w, 4, str(val)[:40], ln=False)
            pdf.ln()

        pdf.ln(3)

    # ── HTML builder (companion for browser viewing) ─────────────────────────

    def _build_html(
        self,
        html_path: Path,
        incident: IncidentRecord,
        entity_row: dict,
        person_name: str,
        person_category: str,
        portrait_path: str | None,
        alerts: list[dict],
        detections: list[dict],
        actions_hist: list[dict],
        generated_at: datetime,
    ) -> None:
        severity = str(
            incident.severity.value if hasattr(incident.severity, "value")
            else incident.severity
        ).lower()
        SEV_CSS = {"high": "#dc2626", "medium": "#eab308", "low": "#4ade80"}
        sev_color = SEV_CSS.get(severity, "#6b7280")
        inc_short = incident.incident_id[-8:]
        dur = _duration(incident.start_time, incident.last_seen_time)

        def _rows_html(cols: list[tuple[str, str]], rows: list[dict],
                       row_fn: Any, empty_msg: str = "No records") -> str:
            thead = "".join(f"<th>{c}</th>" for c, _ in cols)
            if not rows:
                return (f"<table><thead><tr>{thead}</tr></thead>"
                        f"<tbody><tr><td colspan='{len(cols)}' "
                        f"style='color:#6b7280;font-style:italic'>{empty_msg}</td>"
                        f"</tr></tbody></table>")
            tbody = ""
            for row in rows:
                try:
                    vals = row_fn(row)
                except Exception:
                    continue
                tbody += "<tr>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
            return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"

        portrait_img = ""
        if portrait_path and os.path.exists(portrait_path):
            import base64
            with open(portrait_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = Path(portrait_path).suffix.lstrip(".").lower()
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            portrait_img = (f'<img class="portrait" src="data:image/{mime};base64,{b64}" '
                            f'alt="Entity Portrait">')

        alerts_html = _rows_html(
            [("Type", ""), ("Severity", ""), ("Time", ""), ("Events", ""), ("Reason", "")],
            alerts,
            lambda r: [
                str(r.get("type", "—")).upper(),
                str(r.get("severity", "—")).upper(),
                _fmt_ts(str(r.get("timestamp_utc", ""))),
                str(r.get("event_count", "1")),
                str(r.get("reason", "—"))[:80],
            ],
        )
        det_html = _rows_html(
            [("Time", ""), ("Confidence", ""), ("Decision", ""), ("Source", ""), ("Track", "")],
            detections,
            lambda r: [
                _fmt_ts(str(r.get("timestamp_utc", ""))),
                f"{float(r.get('confidence', 0)) * 100:.1f}%",
                str(r.get("decision_state", r.get("result", "—"))),
                str(r.get("source", "cam:0")),
                str(r.get("track_id", "—")),
            ],
        )
        act_html = _rows_html(
            [("Type", ""), ("Status", ""), ("Time", ""), ("Reason", "")],
            actions_hist,
            lambda r: [
                str(r.get("action_type", "—")).upper(),
                str(r.get("status", "—")).upper(),
                _fmt_ts(str(r.get("timestamp_utc", ""))),
                str(r.get("reason", "—"))[:80],
            ],
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRACE-AML Incident Report · {inc_short}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0a0e1a;color:#e6ebf5;font-family:'Courier New',monospace;
        font-size:12px;padding:24px;line-height:1.5}}
  h1{{font-size:22px;letter-spacing:0.15em;color:#e6ebf5;margin-bottom:4px}}
  h2{{font-size:11px;letter-spacing:0.2em;color:#64748b;text-transform:uppercase;
        margin:20px 0 8px;border-bottom:1px solid #28324a;padding-bottom:4px}}
  .header{{display:flex;justify-content:space-between;align-items:flex-start;
            padding-bottom:16px;border-bottom:2px solid {sev_color};margin-bottom:20px}}
  .header .meta{{color:#64748b;font-size:11px;text-align:right}}
  .sev-badge{{display:inline-block;padding:2px 10px;border-radius:2px;
               background:{sev_color}22;color:{sev_color};border:1px solid {sev_color};
               font-size:11px;letter-spacing:0.15em}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}}
  .card{{background:#141928;border:1px solid #28324a;padding:14px;border-radius:4px}}
  .card-label{{color:#64748b;font-size:10px;text-transform:uppercase;
                letter-spacing:0.15em;margin-bottom:8px}}
  .field{{display:flex;gap:8px;margin-bottom:4px}}
  .field .k{{color:#64748b;min-width:90px;flex-shrink:0}}
  .field .v{{color:#e6ebf5}}
  .portrait{{float:right;margin:-4px 0 8px 16px;border:1px solid #28324a;
              border-radius:4px;max-width:80px;max-height:100px;object-fit:cover}}
  table{{width:100%;border-collapse:collapse;margin-bottom:12px}}
  th{{background:#141928;color:#64748b;text-align:left;padding:5px 8px;
       font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
       border-bottom:1px solid #28324a}}
  td{{padding:4px 8px;border-bottom:1px solid #1a2236;color:#e6ebf5;font-size:11px}}
  tr:last-child td{{border-bottom:none}}
  .footer{{margin-top:24px;padding-top:12px;border-top:1px solid #28324a;
            color:#64748b;font-size:10px;display:flex;justify-content:space-between}}
  @media print{{body{{background:#fff;color:#000}}
    .card{{border:1px solid #ccc;background:#f9f9f9}}
    th{{background:#eee;color:#333}}td{{color:#111}}
    .footer,.header .meta{{color:#666}}}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>TRACE-AML</h1>
    <div style="color:#64748b;font-size:11px;margin-top:4px">INCIDENT REPORT</div>
    <div style="margin-top:6px"><span class="sev-badge">{severity.upper()} SEVERITY</span></div>
  </div>
  <div class="meta">
    <div>Incident: <strong style="color:#e6ebf5">{inc_short}</strong></div>
    <div>Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    <div style="margin-top:4px;font-size:10px;color:#3b4a6b">CONFIDENTIAL</div>
  </div>
</div>

<h2>Entity</h2>
<div class="card">
  {portrait_img}
  <div class="card-label">Identity</div>
  <div class="field"><span class="k">Entity ID</span><span class="v">{incident.entity_id}</span></div>
  <div class="field"><span class="k">Type</span><span class="v">{str(entity_row.get('type','unknown')).upper()}</span></div>
  <div class="field"><span class="k">Status</span><span class="v">{str(entity_row.get('status','active')).upper()}</span></div>
  <div class="field"><span class="k">Name</span><span class="v">{person_name}</span></div>
  <div class="field"><span class="k">Category</span><span class="v">{person_category}</span></div>
  <div class="field"><span class="k">First Seen</span><span class="v">{_fmt_ts(str(entity_row.get('created_at','')))}</span></div>
  <div class="field"><span class="k">Last Seen</span><span class="v">{_fmt_ts(str(entity_row.get('last_seen_at','')))}</span></div>
  <div style="clear:both"></div>
</div>

<h2>Incident</h2>
<div class="grid-2">
  <div class="card">
    <div class="field"><span class="k">Incident ID</span><span class="v">{inc_short}</span></div>
    <div class="field"><span class="k">Status</span><span class="v">{str(incident.status.value if hasattr(incident.status,'value') else incident.status).upper()}</span></div>
    <div class="field"><span class="k">Severity</span><span class="v" style="color:{sev_color}">{severity.upper()}</span></div>
  </div>
  <div class="card">
    <div class="field"><span class="k">Duration</span><span class="v">{dur}</span></div>
    <div class="field"><span class="k">Alert Count</span><span class="v">{incident.alert_count}</span></div>
    <div class="field"><span class="k">Summary</span><span class="v">{(incident.summary or '—')[:80]}</span></div>
  </div>
</div>

<h2>Alert Log</h2>
{alerts_html}

<h2>Recent Detections</h2>
{det_html}

<h2>Actions Taken</h2>
{act_html}

<div class="footer">
  <span>TRACE-AML v4 · Incident {inc_short} · CONFIDENTIAL · Do not distribute</span>
  <span>Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
</div>
</body>
</html>"""

        html_path.write_text(html, encoding="utf-8")
