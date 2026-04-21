"""PdfReportHandler — HTML→PDF forensic incident report.

Rendering pipeline
------------------
1. execute()        → calls _generate()
2. _generate()      → _collect_data() → _render_html() → _write_html() + _write_pdf()
3. _render_html()   → assembles 4-section HTML string from sub-builders
4. _write_pdf()     → WeasyPrint HTML→PDF; fpdf2 cover page as fallback

Report layout (A4 portrait)
-----------------------------
Page 1  Entity Identity Dossier
        ├─ Severity header banner
        ├─ Entity profile (ID, type, name, category) + portrait photo
        ├─ Person biometrics (DoB, gender, city, country, notes)
        ├─ Enrollment data (score, embeddings, lifecycle)
        └─ Incident summary (status, duration, alerts, timeline)

Page 2  Alert Log & Automated Actions
        ├─ Alert log table (type, severity, time, events, ack status, reason)
        └─ Automated response actions table (type, status, time, reason)

Page 3  Detection Timeline
        └─ Chronological detection events (time, confidence, decision, source, flags)

Page 4  Visual Evidence Gallery
        └─ Up to 12 screenshots, 3-per-row, best-quality per 10-min bucket
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from trace_aml.actions.base import BaseActionHandler
from trace_aml.core.models import ActionTrigger, IncidentRecord

# ── Playwright Chromium (primary: perfect CSS rendering on Windows) ──────────
try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _PLAYWRIGHT_OK = True
    logger.debug("[PDF] Playwright available")
except Exception as _pw_err:
    sync_playwright = None  # type: ignore
    _PLAYWRIGHT_OK = False
    logger.warning("[PDF] Playwright not available ({}). Falling back to fpdf2 tables.", _pw_err)

# ── Shared CSS — NOT an f-string to avoid {{ }} brace-escaping hell ──────────
_REPORT_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #0a0e1a;
  color: #e6ebf5;
  font-family: 'Courier New', Courier, monospace;
  font-size: 10.5px;
  line-height: 1.55;
}

/* ── Page setup (WeasyPrint @page) ────────────────────────── */
@page {
  size: A4;
  margin: 16mm 13mm 18mm 13mm;
  background: #0a0e1a;
}

/* ── Typography ───────────────────────────────────────────── */
h2 {
  font-size: 8.5px;
  letter-spacing: 0.22em;
  color: #64748b;
  text-transform: uppercase;
  margin: 18px 0 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid #1e2a3d;
  break-after: avoid;
}

/* ── Report cover header ──────────────────────────────────── */
.rpt-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 13px 16px;
  margin-bottom: 20px;
  background: #050810;
  border-bottom: 3px solid #6b7280;
  break-inside: avoid;
}
.rpt-header.sev-high   { border-bottom-color: #dc2626; }
.rpt-header.sev-medium { border-bottom-color: #d97706; }
.rpt-header.sev-low    { border-bottom-color: #16a34a; }

.rpt-header__brand {
  font-size: 18px;
  font-weight: bold;
  letter-spacing: 0.22em;
  color: #e6ebf5;
}
.rpt-header__sub {
  font-size: 8px;
  color: #64748b;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-top: 3px;
}
.rpt-header__right { text-align: right; }

.sev-badge {
  display: inline-block;
  padding: 2px 10px;
  font-size: 8.5px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  border-radius: 2px;
  border: 1px solid;
}
.sev-badge.high   { background: rgba(220,38,38,0.15);  color: #dc2626; border-color: #dc2626; }
.sev-badge.medium { background: rgba(217,119,6,0.15);  color: #d97706; border-color: #d97706; }
.sev-badge.low    { background: rgba(22,163,74,0.15);  color: #16a34a; border-color: #16a34a; }

/* ── Cards ────────────────────────────────────────────────── */
.card {
  background: #141928;
  border: 1px solid #1e2a3d;
  padding: 13px 15px;
  border-radius: 3px;
  margin-bottom: 10px;
  break-inside: avoid;
}
.card__label {
  font-size: 8px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  margin-bottom: 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid #1e2a3d;
}

/* ── Two-column grid ─────────────────────────────────────── */
.cols-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 10px;
}

/* ── Entity cover card (portrait floated right) ──────────── */
.entity-wrap {
  display: flex;
  gap: 18px;
  break-inside: avoid;
}
.entity-wrap__fields { flex: 1; min-width: 0; }
.entity-wrap__portrait { width: 110px; flex-shrink: 0; }
.entity-wrap__portrait img {
  width: 110px;
  height: 140px;
  object-fit: cover;
  border: 1px solid #1e2a3d;
  border-radius: 3px;
  display: block;
}
.portrait-placeholder {
  width: 110px;
  height: 140px;
  background: #0d1220;
  border: 1px dashed #1e2a3d;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #3b4a6b;
  font-size: 9px;
  text-align: center;
  line-height: 1.6;
}

/* ── Field rows ──────────────────────────────────────────── */
.field { display: flex; margin-bottom: 3px; }
.field__k {
  color: #64748b;
  min-width: 130px;
  flex-shrink: 0;
  font-size: 9.5px;
}
.field__v {
  color: #e6ebf5;
  font-size: 9.5px;
  flex: 1;
  overflow: hidden;
}
.field__v--high   { color: #dc2626; font-weight: bold; }
.field__v--medium { color: #d97706; font-weight: bold; }
.field__v--low    { color: #16a34a; font-weight: bold; }

/* ── Tables ──────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; margin-bottom: 10px; }
thead tr { break-after: avoid; }
tr { break-inside: avoid; }
th {
  background: #0d1220;
  color: #64748b;
  text-align: left;
  padding: 5px 8px;
  font-size: 8px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  border-bottom: 1px solid #1e2a3d;
}
td {
  padding: 4px 8px;
  border-bottom: 1px solid #0d1220;
  color: #e6ebf5;
  font-size: 9.5px;
}
tr:last-child td { border-bottom: none; }
tr.alt td { background: rgba(20,25,40,0.5); }

.sev-high   { color: #dc2626; }
.sev-medium { color: #d97706; }
.sev-low    { color: #16a34a; }
.conf-ok    { color: #16a34a; }
.conf-mid   { color: #d97706; }
.conf-low   { color: #dc2626; }
.ok         { color: #16a34a; }
.fail       { color: #dc2626; }
.muted      { color: #64748b; }
.ack-yes    { color: #64748b; font-size: 8px; }

/* ── Page breaks ─────────────────────────────────────────── */
.page-break { break-before: page; }

/* ── Screenshot gallery ──────────────────────────────────── */
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 10px;
}
.gallery-item { break-inside: avoid; }
.gallery-item img {
  width: 100%;
  height: 138px;
  object-fit: cover;
  border: 1px solid #1e2a3d;
  border-radius: 3px;
  display: block;
}
.gallery-item figcaption {
  font-size: 8px;
  color: #64748b;
  text-align: center;
  margin-top: 4px;
  line-height: 1.5;
}

/* ── Report footer ───────────────────────────────────────── */
.rpt-footer {
  margin-top: 22px;
  padding-top: 8px;
  border-top: 1px solid #1e2a3d;
  display: flex;
  justify-content: space-between;
  color: #3b4a6b;
  font-size: 8.5px;
}

/* ── Print / light-mode override ────────────────────────── */
@media print {
  body { background: #fff; color: #111; }
  .card { background: #f8f9fa; border-color: #dee2e6; }
  th { background: #e9ecef; color: #495057; }
  td { color: #212529; }
  .field__k { color: #6c757d; }
  .field__v { color: #212529; }
  h2 { color: #495057; border-color: #dee2e6; }
  .portrait-placeholder { border-color: #dee2e6; color: #adb5bd; }
}
"""


# ── Standalone helpers ────────────────────────────────────────────────────────

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_ts(iso: str) -> str:
    """ISO → 'YYYY-MM-DD HH:MM:SS UTC'."""
    if not iso:
        return "—"
    try:
        t = str(iso).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(iso)[:19]


def _duration(start_iso: str, end_iso: str) -> str:
    """Return human-readable duration between two ISO timestamps."""
    try:
        def _p(s: str) -> datetime:
            s = s.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        start = _p(start_iso)
        end   = _p(end_iso) if end_iso else _utc_now()
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


def _b64_image(path: str) -> str | None:
    """Return inline base64 data URI for an image, or None if unavailable."""
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        with open(p, "rb") as f:
            data = f.read()
        ext  = p.suffix.lstrip(".").lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else (ext if ext in ("png", "gif", "webp") else "jpeg")
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


def _esc(s: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ── Handler ───────────────────────────────────────────────────────────────────

class PdfReportHandler(BaseActionHandler):
    """Generates a multi-page HTML + PDF forensic incident report."""

    # ── Public interface ────────────────────────────────────────────────────

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
            context["pdf_report_path"]  = str(pdf_path)
            context["html_report_path"] = str(html_path)
            logger.info("[ACTION:PDF] report generated → {}", pdf_path.name)
            return True, f"pdf_generated:{pdf_path.name}"
        except Exception as exc:
            logger.error("[ACTION:PDF] report generation failed for {}: {}", incident.incident_id, exc)
            return False, f"pdf_error:{exc}"

    # ── Orchestration ───────────────────────────────────────────────────────

    def _generate(self, incident: IncidentRecord, cfg: Any) -> tuple[Path, Path]:
        now     = _utc_now()
        ts_str  = now.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg.output_dir).resolve() / now.strftime("%Y")
        out_dir.mkdir(parents=True, exist_ok=True)

        stem      = f"{incident.entity_id}_{incident.incident_id[-8:]}_{ts_str}"
        pdf_path  = out_dir / f"{stem}.pdf"
        html_path = out_dir / f"{stem}.html"

        data     = self._collect_data(incident, cfg)
        html_str = self._render_html(data, now)

        self._write_html(html_str, html_path)
        self._write_pdf(html_str, pdf_path, data, now)
        return pdf_path, html_path

    # ── Data collection ─────────────────────────────────────────────────────

    def _collect_data(self, incident: IncidentRecord, cfg: Any) -> dict[str, Any]:
        entity_row = self.store.get_entity(incident.entity_id) or {}

        # Alerts linked to this incident
        alerts = self.store.list_alerts(
            limit=cfg.max_alert_rows + 200, entity_id=incident.entity_id
        )
        inc_alert_ids = set(incident.alert_ids)
        if inc_alert_ids:
            alerts = [a for a in alerts if a.get("alert_id") in inc_alert_ids]
        alerts = alerts[:cfg.max_alert_rows]

        # Detections for this entity
        detections: list[dict] = []
        try:
            detections = self.store.list_events(
                limit=cfg.max_detection_rows, entity_id=incident.entity_id
            )[:cfg.max_detection_rows]
        except Exception as exc:
            logger.debug("[ACTION:PDF] list_events failed: {}", exc)

        # Action history
        actions_hist: list[dict] = []
        try:
            actions_hist = self.store.get_actions(incident.incident_id, limit=50)
        except Exception as exc:
            logger.debug("[ACTION:PDF] get_actions failed: {}", exc)

        # Person record (for KNOWN entities)
        person_name     = "Unknown"
        person_category = "—"
        person_row: dict = {}
        src_pid = entity_row.get("source_person_id", "")
        if src_pid:
            try:
                pr = self.store.get_person(src_pid)
                if pr:
                    person_row     = pr
                    person_name    = pr.get("name", "Unknown")
                    person_category = pr.get("category", "—")
            except Exception:
                pass

        # Portrait as base64
        portrait_b64: str | None = None
        if cfg.include_entity_portrait and src_pid:
            portraits_dir = Path(self.settings.store.portraits_dir).resolve()
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = portraits_dir / f"{src_pid}{ext}"
                if candidate.exists():
                    portrait_b64 = _b64_image(str(candidate))
                    break

        # Gallery screenshots
        gallery_shots: list[dict] = []
        if getattr(cfg, "include_screenshots", True):
            gallery_shots = self._select_gallery_shots(detections)

        return {
            "incident":        incident,
            "entity_row":      entity_row,
            "person_row":      person_row,
            "person_name":     person_name,
            "person_category": person_category,
            "portrait_b64":    portrait_b64,
            "alerts":          alerts,
            "detections":      detections,
            "actions_hist":    actions_hist,
            "gallery_shots":   gallery_shots,
        }

    # ── Gallery shot selection ──────────────────────────────────────────────

    @staticmethod
    def _select_gallery_shots(
        detections: list[dict],
        max_shots: int = 12,
        bucket_minutes: int = 10,
    ) -> list[dict]:
        """
        Pick the highest-confidence screenshot per 10-minute bucket,
        returned in chronological order (max max_shots images).
        """
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: str(d.get("timestamp_utc", "")))
        buckets: dict[int, dict] = {}

        for det in sorted_dets:
            spath = str(det.get("screenshot_path", ""))
            if not spath or not Path(spath).exists():
                continue
            try:
                ts_str = str(det.get("timestamp_utc", "")).replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str)
                bucket_key = int(ts.timestamp()) // (bucket_minutes * 60)
            except Exception:
                continue
            conf     = float(det.get("confidence", 0))
            existing = buckets.get(bucket_key)
            if existing is None or conf > float(existing.get("confidence", 0)):
                buckets[bucket_key] = det

        selected = sorted(buckets.values(), key=lambda d: str(d.get("timestamp_utc", "")))
        return selected[:max_shots]

    # ── HTML rendering ──────────────────────────────────────────────────────

    def _render_html(self, data: dict, generated_at: datetime) -> str:
        incident   = data["incident"]
        inc_short  = incident.incident_id[-8:]
        severity   = str(
            incident.severity.value if hasattr(incident.severity, "value")
            else incident.severity
        ).lower()

        cover      = self._section_cover(data, severity, inc_short, generated_at)
        alerts_sec = self._section_alerts(data)
        dets_sec   = self._section_detections(data)
        gallery    = self._section_gallery(data)

        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>'
            f'<meta charset="UTF-8">'
            f'<title>TRACE-AML Incident Report \xb7 {_esc(inc_short)}</title>'
            f'<style>{_REPORT_CSS}</style>'
            f'</head>\n<body>\n'
            f'{cover}\n{alerts_sec}\n{dets_sec}\n{gallery}\n'
            f'</body>\n</html>'
        )

    # ── Page 1: Identity dossier & incident summary ─────────────────────────

    def _section_cover(
        self,
        data: dict,
        severity: str,
        inc_short: str,
        generated_at: datetime,
    ) -> str:
        incident    = data["incident"]
        entity_row  = data["entity_row"]
        person_row  = data["person_row"]
        person_name = data["person_name"]
        person_cat  = data["person_category"]
        portrait_b64 = data["portrait_b64"]

        entity_type   = _esc(str(entity_row.get("type",   "unknown")).upper())
        entity_status = _esc(str(entity_row.get("status", "active")).upper())
        first_seen    = _fmt_ts(str(entity_row.get("created_at",  "")))
        last_seen     = _fmt_ts(str(entity_row.get("last_seen_at","")))

        # Person biometrics
        dob     = _esc(str(person_row.get("dob", "—")               or "—"))
        gender  = _esc(str(person_row.get("gender", "—")            or "—"))
        city    = _esc(str(person_row.get("last_seen_city", "—")    or "—"))
        country = _esc(str(person_row.get("last_seen_country", "—") or "—"))
        notes   = _esc((str(person_row.get("notes", "—") or "—"))[:150])

        # Enrollment stats
        enr_score  = float(person_row.get("enrollment_score",  0) or 0)
        valid_emb  = int(person_row.get("valid_embeddings",    0) or 0)
        lifecycle  = _esc(str(person_row.get("lifecycle_state","—") or "—").upper())

        # Incident stats
        inc_status = str(
            incident.status.value if hasattr(incident.status, "value")
            else incident.status
        ).upper()
        dur = _duration(incident.start_time, incident.last_seen_time)
        summary_text = _esc((incident.summary or "—")[:250])
        sev_cls = f"field__v--{severity}"

        # Portrait HTML
        if portrait_b64:
            portrait_html = f'<img src="{portrait_b64}" alt="Entity Portrait">'
        else:
            portrait_html = (
                '<div class="portrait-placeholder">'
                'NO PORTRAIT<br>AVAILABLE</div>'
            )

        return f"""
<!-- ═══ PAGE 1: IDENTITY DOSSIER ═══ -->
<div class="rpt-header sev-{severity}">
  <div>
    <div class="rpt-header__brand">TRACE&#x2011;AML</div>
    <div class="rpt-header__sub">Forensic Incident Report &middot; Anti-Money Laundering Intelligence System</div>
  </div>
  <div class="rpt-header__right">
    <div class="muted" style="font-size:9px;">Generated</div>
    <div style="color:#e6ebf5;font-size:9.5px;">{generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    <div style="margin-top:6px;"><span class="sev-badge {severity}">{severity.upper()} SEVERITY</span></div>
    <div style="margin-top:5px;color:#3b4a6b;font-size:8px;">CONFIDENTIAL &mdash; DO NOT DISTRIBUTE</div>
  </div>
</div>

<h2>&#x1F464; Entity Identification</h2>
<div class="card">
  <div class="entity-wrap">
    <div class="entity-wrap__fields">
      <div class="card__label">Biometric Identity Profile</div>
      <div class="field"><span class="field__k">Entity ID</span><span class="field__v">{_esc(incident.entity_id)}</span></div>
      <div class="field"><span class="field__k">Type</span><span class="field__v">{entity_type}</span></div>
      <div class="field"><span class="field__k">Status</span><span class="field__v">{entity_status}</span></div>
      <div class="field"><span class="field__k">Name</span><span class="field__v"><strong>{_esc(person_name)}</strong></span></div>
      <div class="field"><span class="field__k">Category</span><span class="field__v {sev_cls}">{_esc(person_cat.upper())}</span></div>
      <div class="field"><span class="field__k">Date of Birth</span><span class="field__v">{dob}</span></div>
      <div class="field"><span class="field__k">Gender</span><span class="field__v">{gender}</span></div>
      <div class="field"><span class="field__k">Last Known City</span><span class="field__v">{city}</span></div>
      <div class="field"><span class="field__k">Last Known Country</span><span class="field__v">{country}</span></div>
    </div>
    <div class="entity-wrap__portrait">{portrait_html}</div>
  </div>
</div>

<div class="cols-2">
  <div class="card">
    <div class="card__label">Enrollment Data</div>
    <div class="field"><span class="field__k">Lifecycle State</span><span class="field__v">{lifecycle}</span></div>
    <div class="field"><span class="field__k">Enrollment Score</span><span class="field__v">{enr_score * 100:.1f}%</span></div>
    <div class="field"><span class="field__k">Valid Embeddings</span><span class="field__v">{valid_emb}</span></div>
    <div class="field"><span class="field__k">Notes</span><span class="field__v">{notes}</span></div>
  </div>
  <div class="card">
    <div class="card__label">Sighting Record</div>
    <div class="field"><span class="field__k">First Detected</span><span class="field__v">{first_seen}</span></div>
    <div class="field"><span class="field__k">Last Detected</span><span class="field__v">{last_seen}</span></div>
  </div>
</div>

<h2>&#x26A0; Incident Summary</h2>
<div class="cols-2">
  <div class="card">
    <div class="card__label">Incident Header</div>
    <div class="field"><span class="field__k">Incident ID</span><span class="field__v"><strong>{_esc(inc_short)}</strong></span></div>
    <div class="field"><span class="field__k">Status</span><span class="field__v">{inc_status}</span></div>
    <div class="field"><span class="field__k">Severity</span><span class="field__v {sev_cls}">{severity.upper()}</span></div>
    <div class="field"><span class="field__k">Duration</span><span class="field__v">{_esc(dur)}</span></div>
    <div class="field"><span class="field__k">Alert Count</span><span class="field__v">{incident.alert_count}</span></div>
  </div>
  <div class="card">
    <div class="card__label">Timeline</div>
    <div class="field"><span class="field__k">Opened</span><span class="field__v">{_fmt_ts(incident.start_time)}</span></div>
    <div class="field"><span class="field__k">Last Activity</span><span class="field__v">{_fmt_ts(incident.last_seen_time)}</span></div>
    <div class="field"><span class="field__k">Full ID</span><span class="field__v" style="font-size:8px;">{_esc(incident.incident_id)}</span></div>
  </div>
</div>

<div class="card">
  <div class="card__label">Summary Narrative</div>
  <div style="color:#e6ebf5;font-size:10px;line-height:1.7;">{summary_text}</div>
</div>

<div class="rpt-footer">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; CONFIDENTIAL</span>
  <span>Page 1 of 4</span>
</div>
"""

    # ── Page 2: Alert log & automated actions ──────────────────────────────

    def _section_alerts(self, data: dict) -> str:
        alerts      = data["alerts"]
        actions_hist = data["actions_hist"]
        incident     = data["incident"]
        inc_short    = incident.incident_id[-8:]

        # Alert rows
        if alerts:
            rows_html = ""
            for i, a in enumerate(alerts):
                sev   = str(a.get("severity", "low")).lower()
                atype = _esc(str(a.get("type", "—")).upper())
                ts    = _fmt_ts(str(a.get("timestamp_utc", "")))
                ev_ct = str(a.get("event_count", "1"))
                reason = _esc(str(a.get("reason", "—"))[:100])
                ack   = a.get("acknowledged", False)
                ack_td = '<span class="ack-yes">&#x2713; ACK</span>' if ack else '<span style="color:#e6ebf5;">OPEN</span>'
                alt   = ' class="alt"' if i % 2 else ""
                rows_html += (
                    f'<tr{alt}>'
                    f'<td class="sev-{sev}">{atype}</td>'
                    f'<td class="sev-{sev}">{sev.upper()}</td>'
                    f'<td class="muted">{ts}</td>'
                    f'<td style="text-align:center;">{ev_ct}</td>'
                    f'<td>{ack_td}</td>'
                    f'<td>{reason}</td>'
                    f'</tr>\n'
                )
            alerts_table = (
                '<table><thead><tr>'
                '<th>Type</th><th>Severity</th><th>Timestamp (UTC)</th>'
                '<th>Events</th><th>Status</th><th>Reason</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>'
            )
        else:
            alerts_table = '<div class="muted" style="padding:8px 4px;font-style:italic;">No alerts linked to this incident.</div>'

        # Action rows
        _ACTION_ICONS: dict[str, str] = {
            "LOG":        "&#x1F4CB;",
            "EMAIL":      "&#x1F4E7;",
            "WHATSAPP":   "&#x1F4AC;",
            "PDF_REPORT": "&#x1F4C4;",
            "ALARM":      "&#x26A0;",
        }
        if actions_hist:
            act_rows = ""
            for i, a in enumerate(actions_hist):
                atype_raw = str(a.get("action_type", "—")).upper()
                icon  = _ACTION_ICONS.get(atype_raw, "&#x2699;")
                stat  = str(a.get("status", "—")).upper()
                scls  = "ok" if stat == "SUCCESS" else "fail"
                ts    = _fmt_ts(str(a.get("timestamp_utc", "")))
                reason = _esc(str(a.get("reason", "—"))[:100])
                alt   = ' class="alt"' if i % 2 else ""
                act_rows += (
                    f'<tr{alt}>'
                    f'<td>{icon}&nbsp;{_esc(atype_raw)}</td>'
                    f'<td class="{scls}">{stat}</td>'
                    f'<td class="muted">{ts}</td>'
                    f'<td>{reason}</td>'
                    f'</tr>\n'
                )
            actions_table = (
                '<table><thead><tr>'
                '<th>Action Type</th><th>Status</th><th>Timestamp (UTC)</th><th>Reason / Detail</th>'
                f'</tr></thead><tbody>{act_rows}</tbody></table>'
            )
        else:
            actions_table = '<div class="muted" style="padding:8px 4px;font-style:italic;">No actions recorded for this incident.</div>'

        return f"""
<!-- ═══ PAGE 2: ALERT LOG & ACTIONS ═══ -->
<div class="page-break">
<h2>&#x1F514; Alert Log</h2>
{alerts_table}
<h2>&#x26A1; Automated Response Actions</h2>
{actions_table}
<div class="rpt-footer">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; CONFIDENTIAL</span>
  <span>Page 2 of 4</span>
</div>
</div>
"""

    # ── Page 3: Detection timeline ─────────────────────────────────────────

    def _section_detections(self, data: dict) -> str:
        detections = data["detections"]
        inc_short  = data["incident"].incident_id[-8:]

        if detections:
            rows_html = ""
            for i, d in enumerate(detections):
                ts       = _fmt_ts(str(d.get("timestamp_utc", "")))
                raw_conf = float(d.get("confidence", 0))
                # Guard: if stored as 0-1 fraction multiply by 100, otherwise use as-is
                conf = raw_conf * 100 if raw_conf <= 1.0 else raw_conf
                conf = min(conf, 100.0)  # cap at 100%
                decision = _esc(str(d.get("decision_state", d.get("result", "—"))))
                source   = _esc(str(d.get("source", "—")))
                track    = _esc(str(d.get("track_id", "—"))[:14])

                raw_flags = d.get("quality_flags", [])
                if isinstance(raw_flags, str):
                    try:
                        raw_flags = json.loads(raw_flags)
                    except Exception:
                        raw_flags = []
                flags_str = _esc(", ".join(str(f) for f in raw_flags) if raw_flags else "—")

                conf_cls = "conf-ok" if conf >= 75 else ("conf-mid" if conf >= 50 else "conf-low")
                alt = ' class="alt"' if i % 2 else ""

                rows_html += (
                    f'<tr{alt}>'
                    f'<td class="muted">{ts}</td>'
                    f'<td class="{conf_cls}">{conf:.1f}%</td>'
                    f'<td>{decision}</td>'
                    f'<td class="muted">{source}</td>'
                    f'<td class="muted">{track}</td>'
                    f'<td style="font-size:8px;color:#64748b;">{flags_str}</td>'
                    f'</tr>\n'
                )
            det_table = (
                '<table><thead><tr>'
                '<th>Timestamp (UTC)</th><th>Confidence</th><th>Decision</th>'
                '<th>Source</th><th>Track ID</th><th>Quality Flags</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>'
            )
        else:
            det_table = '<div class="muted" style="padding:8px 4px;font-style:italic;">No detection events recorded for this entity.</div>'

        return f"""
<!-- ═══ PAGE 3: DETECTION TIMELINE ═══ -->
<div class="page-break">
<h2>&#x1F4F9; Detection Timeline ({len(detections)} events)</h2>
{det_table}
<div class="rpt-footer">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; CONFIDENTIAL</span>
  <span>Page 3 of 4</span>
</div>
</div>
"""

    # ── Page 4: Visual evidence gallery ────────────────────────────────────

    def _section_gallery(self, data: dict) -> str:
        shots     = data["gallery_shots"]
        inc_short = data["incident"].incident_id[-8:]

        if not shots:
            body = (
                '<div class="card" style="text-align:center;padding:30px;">'
                '<div class="muted">No captured screenshots available for this entity.</div>'
                '<div class="muted" style="margin-top:8px;font-size:9px;">'
                'Screenshots are stored automatically when the entity is detected by the recognition camera.<br>'
                'Ensure the live-ops pipeline is running and the entity has been sighted at least once.'
                '</div></div>'
            )
        else:
            items = ""
            rendered = 0
            for shot in shots:
                spath = str(shot.get("screenshot_path", ""))
                b64   = _b64_image(spath)
                if not b64:
                    continue
                ts    = _fmt_ts(str(shot.get("timestamp_utc", "")))
                raw_c = float(shot.get("confidence", 0))
                conf  = raw_c * 100 if raw_c <= 1.0 else min(raw_c, 100.0)
                dec   = _esc(str(shot.get("decision_state", "—")))
                items += (
                    f'<figure class="gallery-item">'
                    f'<img src="{b64}" alt="Detection screenshot">'
                    f'<figcaption>{ts}<br>'
                    f'Conf: {conf:.1f}% &middot; {dec}'
                    f'</figcaption></figure>\n'
                )
                rendered += 1

            if not items:
                items = '<div class="muted" style="grid-column:1/-1;text-align:center;padding:20px;">Screenshot files not found on disk.</div>'

            info = f"Showing {rendered} best-quality captures (1 per 10-min window)"
            body = (
                f'<div class="card" style="padding:6px 10px;margin-bottom:10px;">'
                f'<span class="muted" style="font-size:9px;">{info}. '
                f'Images are face crops with context padding.</span></div>'
                f'<div class="gallery-grid">{items}</div>'
            )

        return f"""
<!-- ═══ PAGE 4: VISUAL EVIDENCE GALLERY ═══ -->
<div class="page-break">
<h2>&#x1F4F7; Visual Evidence Gallery</h2>
{body}
<div class="rpt-footer" style="margin-top:30px;">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; CONFIDENTIAL &middot; Do not distribute</span>
  <span>Page 4 of 4</span>
</div>
</div>
"""

    # ── PDF + HTML writers ──────────────────────────────────────────────────

    def _write_pdf(
        self,
        html_str: str,
        pdf_path: Path,
        data: dict,
        generated_at: datetime,
    ) -> None:
        if _PLAYWRIGHT_OK:
            try:
                self._write_pdf_playwright(html_str, pdf_path)
                logger.debug("[PDF] Playwright Chromium → {}", pdf_path.name)
                return
            except Exception as exc:
                logger.warning("[PDF] Playwright render failed ({}), using fpdf2 fallback", exc)

        # ── fpdf2 fallback ────────────────────────────────────────────────────
        self._write_pdf_fpdf2(pdf_path, data, generated_at)

    def _write_pdf_playwright(self, html_str: str, pdf_path: Path) -> None:
        """Render HTML to PDF via Playwright headless Chromium."""
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            # Load HTML directly as content; use a base URL for any relative assets
            page.set_content(html_str, wait_until="domcontentloaded")
            page.emulate_media(media="print")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "16mm", "bottom": "18mm", "left": "13mm", "right": "13mm"},
            )
            browser.close()

    def _write_pdf_fpdf2(
        self,
        pdf_path: Path,
        data: dict,
        generated_at: datetime,
    ) -> None:
        """fpdf2-based fallback — covers all 4 sections in simple table form."""
        from fpdf import FPDF

        incident    = data["incident"]
        entity_row  = data["entity_row"]
        person_name = data["person_name"]
        person_cat  = data["person_category"]
        alerts      = data["alerts"]
        detections  = data["detections"]
        actions_hist = data["actions_hist"]
        portrait_b64 = data["portrait_b64"]

        severity = str(
            incident.severity.value if hasattr(incident.severity, "value")
            else incident.severity
        ).lower()
        SEV_RGB = {"high": (220, 38, 38), "medium": (217, 119, 6), "low": (22, 163, 74)}
        sev_rgb = SEV_RGB.get(severity, (107, 114, 128))

        BG, FG, GREY, BORDER = (10,14,26), (230,235,245), (100,110,130), (40,50,70)
        inc_short = incident.incident_id[-8:]
        dur       = _duration(incident.start_time, incident.last_seen_time)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        W = lambda: pdf.w - pdf.l_margin - pdf.r_margin

        def header():
            pdf.set_fill_color(*BG)
            pdf.rect(0, 0, pdf.w, 28, "F")
            pdf.set_xy(pdf.l_margin, 8)
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(*FG)
            pdf.cell(W() / 2, 8, "TRACE-AML", ln=False)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*GREY)
            pdf.set_x(pdf.l_margin + W() / 2)
            pdf.cell(W() / 2, 8, f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}", align="R", ln=True)
            pdf.set_xy(pdf.l_margin, 18)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*sev_rgb)
            pdf.cell(W(), 6, f"INCIDENT REPORT  ·  {severity.upper()} SEVERITY  ·  {inc_short}", ln=True)
            pdf.ln(6)

        def section(title: str):
            pdf.set_fill_color(*BORDER)
            pdf.rect(pdf.l_margin, pdf.get_y(), W(), 0.3, "F")
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*GREY)
            pdf.cell(W(), 5, title, ln=True)
            pdf.ln(1)

        def kv(label: str, val: str):
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GREY)
            pdf.cell(35, 5, f"{label}:", ln=False)
            pdf.set_text_color(*FG)
            pdf.cell(W() - 35, 5, str(val)[:80], ln=True)

        def table(col_defs: list, rows: list, row_fn: Any):
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*GREY)
            for lbl, w in col_defs:
                pdf.cell(w, 5, lbl, ln=False)
            pdf.ln()
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*FG)
            for row in rows:
                try:
                    vals = row_fn(row)
                except Exception:
                    continue
                for (_, w), v in zip(col_defs, vals):
                    pdf.cell(w, 4, str(v)[:40], ln=False)
                pdf.ln()
            pdf.ln(3)

        # ── Page 1: Identity ────────────────────────────────────────────────
        pdf.add_page()
        header()

        portrait_path: str | None = None
        if portrait_b64 and portrait_b64.startswith("data:image"):
            import tempfile
            raw = base64.b64decode(portrait_b64.split(",", 1)[1])
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(raw)
                portrait_path = tmp.name

        section("ENTITY IDENTIFICATION")
        portrait_w = 28
        text_w = W() - portrait_w - 4 if portrait_path else W()
        y_before = pdf.get_y()
        for label, val in [
            ("Entity ID",  incident.entity_id),
            ("Type",       str(entity_row.get("type", "unknown")).upper()),
            ("Name",       person_name),
            ("Category",   person_cat),
            ("First Seen", _fmt_ts(str(entity_row.get("created_at", "")))),
            ("Last Seen",  _fmt_ts(str(entity_row.get("last_seen_at", "")))),
        ]:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GREY)
            pdf.cell(35, 5, f"{label}:", ln=False)
            pdf.set_text_color(*FG)
            pdf.cell(text_w - 35, 5, str(val)[:60], ln=True)

        if portrait_path and os.path.exists(portrait_path):
            try:
                pdf.image(portrait_path, x=pdf.l_margin + text_w + 4, y=y_before, w=portrait_w)
            except Exception:
                pass
            try:
                os.unlink(portrait_path)
            except Exception:
                pass

        pdf.ln(4)
        section("INCIDENT SUMMARY")
        for label, val in [
            ("Incident ID", inc_short),
            ("Status",     str(incident.status.value if hasattr(incident.status, "value") else incident.status).upper()),
            ("Severity",   severity.upper()),
            ("Duration",   dur),
            ("Alerts",     str(incident.alert_count)),
            ("Summary",    (incident.summary or "—")[:80]),
        ]:
            kv(label, val)

        # ── Page 2: Alerts & Actions ────────────────────────────────────────
        pdf.add_page()
        header()
        section("ALERT LOG")
        table(
            [("Type", 45), ("Sev", 22), ("Time", 50), ("Evt", 15), ("Reason", W() - 132)],
            alerts,
            lambda r: [
                str(r.get("type", "—")).upper(),
                str(r.get("severity", "—")).upper(),
                _fmt_ts(str(r.get("timestamp_utc", ""))),
                str(r.get("event_count", "1")),
                str(r.get("reason", "—"))[:45],
            ],
        )
        section("AUTOMATED ACTIONS")
        table(
            [("Type", 35), ("Status", 22), ("Time", 50), ("Reason", W() - 107)],
            actions_hist,
            lambda r: [
                str(r.get("action_type", "—")).upper(),
                str(r.get("status", "—")).upper(),
                _fmt_ts(str(r.get("timestamp_utc", ""))),
                str(r.get("reason", "—"))[:50],
            ],
        )

        # ── Page 3: Detections ──────────────────────────────────────────────
        pdf.add_page()
        header()
        section("DETECTION TIMELINE")
        table(
            [("Time", 45), ("Conf%", 20), ("Decision", 35), ("Source", 28), ("Track", W() - 128)],
            detections,
            lambda r: [
                _fmt_ts(str(r.get("timestamp_utc", ""))),
                f"{float(r.get('confidence', 0)) * 100:.1f}%",
                str(r.get("decision_state", r.get("result", "—"))),
                str(r.get("source", "—")),
                str(r.get("track_id", "—"))[:14],
            ],
        )

        # ── Footer ──────────────────────────────────────────────────────────
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*GREY)
        pdf.cell(W() / 2, 5, f"TRACE-AML v4  ·  Incident {inc_short}  ·  CONFIDENTIAL")
        pdf.cell(W() / 2, 5, f"Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}", align="R")

        pdf.output(str(pdf_path))

    def _write_html(self, html_str: str, html_path: Path) -> None:
        html_path.write_text(html_str, encoding="utf-8")
        logger.debug("[PDF] HTML companion → {}", html_path.name)
