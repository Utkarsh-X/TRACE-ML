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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #0e0e0e;
  color: #e2e2e2;
  font-family: 'Inter', -apple-system, sans-serif;
  font-size: 10px;
  line-height: 1.5;
  -webkit-print-color-adjust: exact;
}

/* ── Page setup ───────────────────────────────────────────── */
@page {
  size: A4;
  margin: 15mm 12mm 18mm 12mm;
  background: #0e0e0e;
}

/* ── Typography ───────────────────────────────────────────── */
.mono { font-family: 'JetBrains Mono', monospace; }

h2 {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.18em;
  color: #919191;
  text-transform: uppercase;
  margin: 24px 0 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid #1f1f1f;
  break-after: avoid;
  display: flex;
  align-items: center;
  gap: 8px;
}
h2::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  background: #ffffff;
}

/* ── Report cover header ──────────────────────────────────── */
.rpt-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 18px 20px;
  margin-bottom: 24px;
  background: #131313;
  border: 1px solid #1f1f1f;
  border-left: 3px solid #ffffff;
  break-inside: avoid;
}
.rpt-header.sev-critical { border-left-color: #d32f2f; }

.rpt-header__brand {
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.15em;
  color: #ffffff;
  margin-bottom: 4px;
}
.rpt-header__sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 7.5px;
  color: #474747;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.rpt-header__right { text-align: right; }

/* ── Badge System ────────────────────────────────────────── */
.badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 7.5px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 2px 8px;
  display: inline-flex;
  align-items: center;
  border-radius: 1px;
}
.badge--filled {
  background-color: #ffffff;
  color: #0e0e0e;
  border: 1px solid #ffffff;
}
.badge--ghost {
  background-color: transparent;
  border: 1px solid #474747;
  color: #c6c6c6;
}
.badge--neutral {
  background-color: transparent;
  border: 1px dashed rgba(145, 145, 145, 0.4);
  color: #919191;
}
.badge--critical {
  background-color: rgba(211, 47, 47, 0.15);
  color: #d32f2f;
  border: 1px solid rgba(211, 47, 47, 0.4);
}

/* ── Cards ────────────────────────────────────────────────── */
.card {
  background: #131313;
  border: 1px solid #1f1f1f;
  padding: 15px 18px;
  margin-bottom: 12px;
  break-inside: avoid;
}
.card__label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 7.5px;
  color: #474747;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-bottom: 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid #1f1f1f;
}

/* ── Two-column grid ─────────────────────────────────────── */
.cols-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 12px;
}

/* ── Entity cover card (portrait) ────────────────────────── */
.entity-wrap {
  display: flex;
  gap: 20px;
  break-inside: avoid;
}
.entity-wrap__fields { flex: 1; min-width: 0; }
.entity-wrap__portrait { width: 110px; flex-shrink: 0; }
.entity-wrap__portrait img {
  width: 110px;
  height: 140px;
  object-fit: cover;
  border: 1px solid #1f1f1f;
  display: block;
}
.portrait-placeholder {
  width: 110px;
  height: 140px;
  background: #0e0e0e;
  border: 1px dashed #1f1f1f;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #474747;
  font-family: 'JetBrains Mono', monospace;
  font-size: 8px;
  text-align: center;
  line-height: 1.6;
}

/* ── Field rows ──────────────────────────────────────────── */
.field { display: flex; margin-bottom: 4px; border-bottom: 1px solid rgba(31,31,31,0.5); padding-bottom: 4px; }
.field:last-child { border-bottom: none; }
.field__k {
  font-family: 'JetBrains Mono', monospace;
  color: #474747;
  min-width: 130px;
  flex-shrink: 0;
  font-size: 8.5px;
  text-transform: uppercase;
}
.field__v {
  color: #ffffff;
  font-size: 9.5px;
  font-weight: 500;
  flex: 1;
}

/* ── Tables ──────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
thead tr { break-after: avoid; }
tr { break-inside: avoid; }
th {
  background: #0e0e0e;
  color: #474747;
  text-align: left;
  padding: 8px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 7.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border-bottom: 1px solid #1f1f1f;
}
td {
  padding: 6px 10px;
  border-bottom: 1px solid #1f1f1f;
  color: #c6c6c6;
  font-size: 9px;
}
tr:last-child td { border-bottom: none; }
tr.alt td { background: rgba(31, 31, 31, 0.3); }

.text-white { color: #ffffff; }
.text-muted { color: #919191; }
.text-grey  { color: #474747; }
.text-critical { color: #d32f2f; }

/* ── Page breaks ─────────────────────────────────────────── */
.page-break { break-before: page; }

/* ── Screenshot gallery ──────────────────────────────────── */
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 15px;
  margin-top: 15px;
}
.gallery-item { break-inside: avoid; border: 1px solid #1f1f1f; background: #131313; padding: 4px; }
.gallery-item img {
  width: 100%;
  height: 130px;
  object-fit: cover;
  display: block;
}
.gallery-item figcaption {
  font-family: 'JetBrains Mono', monospace;
  font-size: 7.5px;
  color: #474747;
  text-align: left;
  margin-top: 6px;
  padding: 4px;
  line-height: 1.4;
}

/* ── Report footer ───────────────────────────────────────── */
.rpt-footer {
  margin-top: 30px;
  padding-top: 10px;
  border-top: 1px solid #1f1f1f;
  display: flex;
  justify-content: space-between;
  color: #474747;
  font-family: 'JetBrains Mono', monospace;
  font-size: 7.5px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* ── Print / light-mode override ────────────────────────── */
@media print {
  body { background: #ffffff; color: #1a1c1c; }
  .card, .rpt-header, .gallery-item { background: #ffffff; border-color: #e2e2e2; }
  th { background: #f5f5f5; color: #757575; border-bottom-color: #e2e2e2; }
  td { color: #1a1c1c; border-bottom-color: #f5f5f5; }
  .field__k { color: #757575; }
  .field__v { color: #1a1c1c; }
  .field { border-bottom-color: #f5f5f5; }
  h2 { color: #757575; border-bottom-color: #e2e2e2; }
  .portrait-placeholder { border-color: #e2e2e2; color: #bdbdbd; background: #fafafa; }
  .rpt-footer { border-top-color: #e2e2e2; color: #bdbdbd; }
  .mono, .field__k, .card__label, h2, th, .rpt-header__brand, .rpt-header__sub, .badge, .rpt-footer, figcaption {
    font-family: 'JetBrains Mono', monospace !important;
  }
}
"""


# ── Standalone helpers ────────────────────────────────────────────────────────

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_ts(iso: str) -> str:
    """ISO → 'YYYY-MM-DD HH:MM:SS'."""
    if not iso:
        return "—"
    try:
        t = str(iso).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
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


def _b64_image(path: str, vault: Any = None) -> str | None:
    """Return inline base64 data URI for an image, or None if unavailable.

    Handles two path formats:
    * ``vault:{sha256}``  — decrypt from DataVault (new format)
    * Any other string    — legacy absolute/relative filesystem path
    """
    try:
        if path and path.startswith("vault:") and vault is not None:
            # New format: vault:sha256hex  OR  vault:detection_id
            key = path[len("vault:"):]
            # Try evidence lookup by detection_id first (stored in evidence.json)
            # The vault key may be the detection_id OR the sha256 blob key.
            # evidence.get_evidence_bytes() accepts detection_id.
            # We need to find the detection_id from the vault's index.
            # Simplest: store detection_id as the lookup key in put_evidence,
            # so vault.get_evidence_bytes(detection_id) works directly.
            data = None
            # Try interpreting key as detection_id
            with_det = vault.get_evidence_bytes(key.split(":")[0] if ":" in key else key)
            if with_det is not None:
                data = with_det
            if data is None:
                return None
            return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        # Legacy filesystem path
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


# ── UI Mapping Helpers ───────────────────────────────────────────────────────

def _severity_badge(severity: str) -> str:
    s = str(severity or "").lower()
    label = s.upper()
    if s in ("critical", "extreme", "emergency"):
        return f'<span class="badge badge--critical">{label}</span>'
    if s == "high":
        return f'<span class="badge badge--neutral">{label}</span>'
    if s == "medium":
        return f'<span class="badge badge--ghost">{label}</span>'
    return f'<span class="badge badge--ghost">{label}</span>'

def _status_badge(status: str) -> str:
    s = str(status or "").lower()
    label = s.upper()
    if s in ("open", "active", "criminal"):
        return f'<span class="badge badge--filled">{label}</span>'
    if s in ("pending", "unknown"):
        return f'<span class="badge badge--neutral">{label}</span>'
    return f'<span class="badge badge--ghost">{label}</span>'


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

        # Portrait as base64 — try vault first, then legacy filesystem
        portrait_b64: str | None = None
        if cfg.include_entity_portrait:
            from trace_aml.store.portrait_store import PortraitStore
            _ps = PortraitStore(self.settings)
            # Try the entity_id directly (works for UNKNOWN entities)
            _portrait_entity_id = incident.entity_id
            jpeg_bytes = _ps.get_portrait_bytes(_portrait_entity_id)
            # For KNOWN entities also try src_pid (enrolled portrait)
            if jpeg_bytes is None and src_pid:
                jpeg_bytes = _ps.get_portrait_bytes(src_pid)
            if jpeg_bytes is not None:
                portrait_b64 = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode()}"
            # Final fallback: legacy flat-file portrait
            if portrait_b64 is None and src_pid:
                portraits_dir = Path(self.settings.store.portraits_dir).resolve()
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = portraits_dir / f"{src_pid}{ext}"
                    if candidate.exists():
                        portrait_b64 = _b64_image(str(candidate))
                        break

        # Gallery screenshots
        gallery_shots: list[dict] = []
        if getattr(cfg, "include_screenshots", True):
            from trace_aml.store.portrait_store import PortraitStore
            _vault = PortraitStore(self.settings).vault
            gallery_shots = self._select_gallery_shots(detections, vault=_vault)

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
            "_vault":          _vault if getattr(cfg, "include_screenshots", True) else None,
        }

    # ── Gallery shot selection ──────────────────────────────────────────────


    @staticmethod
    def _select_gallery_shots(
        detections: list[dict],
        max_shots: int = 12,
        bucket_minutes: int = 10,
        vault: Any = None,
    ) -> list[dict]:
        """
        Pick the best screenshot per 10-minute bucket, returned chronologically.

        Winner selection uses a **composite quality score**::

            composite = 0.6 × (smoothed_confidence / 100) + 0.4 × face_quality

        This ensures the gallery shows *sharp, well-lit* frames rather than just
        high-similarity-score detections that may be blurry.

        Supports both vault-keyed paths (``vault:{sha256}``) and legacy filesystem
        absolute paths for backward compatibility.
        """
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: str(d.get("timestamp_utc", "")))
        buckets: dict[int, dict] = {}

        for det in sorted_dets:
            spath = str(det.get("screenshot_path", ""))
            if not spath:
                continue

            # Resolve whether we have a valid image for this detection
            has_image = False
            if spath.startswith("vault:") and vault is not None:
                det_id = spath[len("vault:"):]
                has_image = vault.has_evidence(det_id)
            else:
                has_image = Path(spath).exists() if spath else False

            if not has_image:
                continue

            try:
                ts_str = str(det.get("timestamp_utc", "")).replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str)
                bucket_key = int(ts.timestamp()) // (bucket_minutes * 60)
            except Exception:
                continue

            # Composite score: weighted combination of match confidence + face quality
            smth_c   = float(det.get("smoothed_confidence", det.get("confidence", 0))) / 100.0
            face_q   = float((det.get("metadata") or {}).get("face_quality", 0.0))
            composite = 0.6 * smth_c + 0.4 * face_q

            existing = buckets.get(bucket_key)
            if existing is None:
                buckets[bucket_key] = {**det, "_composite": composite}
            else:
                if composite > existing.get("_composite", 0.0):
                    buckets[bucket_key] = {**det, "_composite": composite}

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
            f'<title>TRACE-AML Report \xb7 {_esc(inc_short)}</title>'
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

        # Badges
        sev_badge = _severity_badge(severity)
        cat_badge = _status_badge(person_cat)
        inc_status_badge = _status_badge(inc_status)

        # Portrait HTML
        if portrait_b64:
            portrait_html = f'<img src="{portrait_b64}" alt="Entity Portrait">'
        else:
            portrait_html = (
                '<div class="portrait-placeholder">'
                'NO PORTRAIT<br>DATA</div>'
            )

        return f"""
<!-- ═══ PAGE 1: IDENTITY DOSSIER ═══ -->
<div class="rpt-header sev-{severity}">
  <div>
    <div class="rpt-header__brand">TRACE&#x2011;AML</div>
    <div class="rpt-header__sub">Forensic Incident Report &middot; Anti-Money Laundering Intel</div>
  </div>
  <div class="rpt-header__right">
    <div class="text-grey mono" style="font-size:7.5px;">SYSTEM GENERATED</div>
    <div class="text-white mono" style="font-size:9px; margin-top:2px;">{generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    <div style="margin-top:8px;">{sev_badge}</div>
  </div>
</div>

<h2>Entity Identification</h2>
<div class="card">
  <div class="entity-wrap">
    <div class="entity-wrap__fields">
      <div class="card__label">Biometric Identity Profile</div>
      <div class="field"><span class="field__k">Entity ID</span><span class="field__v mono">{_esc(incident.entity_id)}</span></div>
      <div class="field"><span class="field__k">Type</span><span class="field__v">{entity_type}</span></div>
      <div class="field"><span class="field__k">Status</span><span class="field__v">{entity_status}</span></div>
      <div class="field"><span class="field__k">Name</span><span class="field__v text-white">{_esc(person_name)}</span></div>
      <div class="field"><span class="field__k">Category</span><span class="field__v">{cat_badge}</span></div>
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
    <div class="field"><span class="field__k">Lifecycle</span><span class="field__v">{lifecycle}</span></div>
    <div class="field"><span class="field__k">Enr. Score</span><span class="field__v">{enr_score * 100:.1f}%</span></div>
    <div class="field"><span class="field__k">Valid Emb.</span><span class="field__v">{valid_emb}</span></div>
    <div class="field"><span class="field__k">Notes</span><span class="field__v text-muted">{notes}</span></div>
  </div>
  <div class="card">
    <div class="card__label">Detection Record</div>
    <div class="field"><span class="field__k">First Seen</span><span class="field__v mono">{first_seen}</span></div>
    <div class="field"><span class="field__k">Last Seen</span><span class="field__v mono">{last_seen}</span></div>
  </div>
</div>

<h2>Incident Summary</h2>
<div class="cols-2">
  <div class="card">
    <div class="card__label">Incident Header</div>
    <div class="field"><span class="field__k">ID</span><span class="field__v mono text-white">{_esc(inc_short)}</span></div>
    <div class="field"><span class="field__k">Status</span><span class="field__v">{inc_status_badge}</span></div>
    <div class="field"><span class="field__k">Duration</span><span class="field__v">{_esc(dur)}</span></div>
    <div class="field"><span class="field__k">Alert Count</span><span class="field__v">{incident.alert_count}</span></div>
  </div>
  <div class="card">
    <div class="card__label">Timeline</div>
    <div class="field"><span class="field__k">Opened</span><span class="field__v mono">{_fmt_ts(incident.start_time)}</span></div>
    <div class="field"><span class="field__k">Last Activity</span><span class="field__v mono">{_fmt_ts(incident.last_seen_time)}</span></div>
  </div>
</div>

<div class="card">
  <div class="card__label">Summary Narrative</div>
  <div style="color:#ffffff; font-size:10px; line-height:1.7;">{summary_text}</div>
</div>

<div class="rpt-footer">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; FORENSIC USE ONLY</span>
  <span>Page 1 / 4</span>
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
                ack_td = '<span class="text-grey mono" style="font-size:7.5px;">[ACK]</span>' if ack else '<span class="text-white mono" style="font-size:7.5px;">OPEN</span>'
                alt   = ' class="alt"' if i % 2 else ""
                sev_b = _severity_badge(sev)
                rows_html += (
                    f'<tr{alt}>'
                    f'<td class="mono text-white" style="font-size:8px;">{atype}</td>'
                    f'<td>{sev_b}</td>'
                    f'<td class="mono text-grey">{ts}</td>'
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
            alerts_table = '<div class="text-grey mono" style="padding:15px; font-style:italic; border:1px solid #1f1f1f; background:#131313;">No alerts linked to this incident.</div>'

        # Action rows
        _ACTION_ICONS: dict[str, str] = {
            "LOG":        "LOG",
            "EMAIL":      "MAIL",
            "WHATSAPP":   "CHAT",
            "PDF_REPORT": "PDF",
            "ALARM":      "ALARM",
        }
        if actions_hist:
            act_rows = ""
            for i, a in enumerate(actions_hist):
                atype_raw = str(a.get("action_type", "—")).upper()
                icon  = _ACTION_ICONS.get(atype_raw, "ACT")
                stat  = str(a.get("status", "—")).upper()
                sbadge = '<span class="badge badge--filled">SUCCESS</span>' if stat == "SUCCESS" else f'<span class="badge badge--critical">{stat}</span>'
                ts    = _fmt_ts(str(a.get("timestamp_utc", "")))
                reason = _esc(str(a.get("reason", "—"))[:100])
                alt   = ' class="alt"' if i % 2 else ""
                act_rows += (
                    f'<tr{alt}>'
                    f'<td class="mono text-white" style="font-size:8px;"><span class="text-grey">[{icon}]</span> {atype_raw}</td>'
                    f'<td>{sbadge}</td>'
                    f'<td class="mono text-grey">{ts}</td>'
                    f'<td>{reason}</td>'
                    f'</tr>\n'
                )
            actions_table = (
                '<table><thead><tr>'
                '<th>Action Type</th><th>Status</th><th>Timestamp (UTC)</th><th>Reason / Detail</th>'
                f'</tr></thead><tbody>{act_rows}</tbody></table>'
            )
        else:
            actions_table = '<div class="text-grey mono" style="padding:15px; font-style:italic; border:1px solid #1f1f1f; background:#131313;">No actions recorded for this incident.</div>'

        return f"""
<!-- ═══ PAGE 2: ALERT LOG & ACTIONS ═══ -->
<div class="page-break">
<h2>Alert Log</h2>
{alerts_table}

<h2 style="margin-top:30px;">Automated Response Actions</h2>
{actions_table}

<div class="rpt-footer">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; FORENSIC USE ONLY</span>
  <span>Page 2 / 4</span>
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

                conf_badge = f'<span class="badge badge--filled" style="width:45px; justify-content:center;">{conf:.1f}%</span>' if conf >= 75 else f'<span class="badge badge--ghost" style="width:45px; justify-content:center;">{conf:.1f}%</span>'
                alt = ' class="alt"' if i % 2 else ""

                rows_html += (
                    f'<tr{alt}>'
                    f'<td class="mono text-grey">{ts}</td>'
                    f'<td>{conf_badge}</td>'
                    f'<td class="text-white">{decision}</td>'
                    f'<td class="mono text-grey">{source}</td>'
                    f'<td class="mono text-grey">{track}</td>'
                    f'<td class="mono" style="font-size:7.5px; color:#474747;">{flags_str}</td>'
                    f'</tr>\n'
                )
            det_table = (
                '<table><thead><tr>'
                '<th>Timestamp (UTC)</th><th>Conf.</th><th>Decision</th>'
                '<th>Source</th><th>Track ID</th><th>Quality Flags</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>'
            )
        else:
            det_table = '<div class="text-grey mono" style="padding:15px; font-style:italic; border:1px solid #1f1f1f; background:#131313;">No detection events recorded for this entity.</div>'

        return f"""
<!-- ═══ PAGE 3: DETECTION TIMELINE ═══ -->
<div class="page-break">
<h2>Detection Timeline ({len(detections)} events)</h2>
{det_table}

<div class="rpt-footer">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; FORENSIC USE ONLY</span>
  <span>Page 3 / 4</span>
</div>
</div>
"""

    # ── Page 4: Visual evidence gallery ────────────────────────────────────

    def _section_gallery(self, data: dict) -> str:
        shots     = data["gallery_shots"]
        inc_short = data["incident"].incident_id[-8:]

        if not shots:
            body = (
                '<div class="card" style="text-align:center; padding:40px; border-style:dashed;">'
                '<div class="text-grey mono">NO VISUAL EVIDENCE CAPTURED</div>'
                '<div class="text-grey mono" style="margin-top:10px; font-size:8px;">'
                'Check pipeline stream status and entity sighting history.'
                '</div></div>'
            )
        else:
            items = ""
            rendered = 0
            for shot in shots:
                spath = str(shot.get("screenshot_path", ""))
                vault = data.get("_vault")
                b64   = _b64_image(spath, vault=vault)
                if not b64:
                    continue
                ts    = _fmt_ts(str(shot.get("timestamp_utc", "")))
                raw_c = float(shot.get("confidence", 0))
                conf  = raw_c * 100 if raw_c <= 1.0 else min(raw_c, 100.0)
                dec   = _esc(str(shot.get("decision_state", "—")))
                items += (
                    f'<figure class="gallery-item">'
                    f'<img src="{b64}" alt="Capture">'
                    f'<figcaption>'
                    f'<div class="text-white mono">{ts}</div>'
                    f'<div style="margin-top:2px;">CONF: {conf:.1f}% &middot; {dec}</div>'
                    f'</figcaption></figure>\n'
                )
                rendered += 1

            if not items:
                items = '<div class="text-grey mono" style="grid-column:1/-1; text-align:center; padding:20px;">FILES_NOT_FOUND_ON_DISK</div>'

            info = f"SEQUENTIAL CAPTURE LOG: {rendered} SAMPLES"
            body = (
                f'<div class="mono text-grey" style="font-size:8px; margin-bottom:10px; letter-spacing:0.1em;">{info}</div>'
                f'<div class="gallery-grid">{items}</div>'
            )

        return f"""
<!-- ═══ PAGE 4: VISUAL EVIDENCE GALLERY ═══ -->
<div class="page-break">
<h2>Visual Evidence Gallery</h2>
{body}

<div class="rpt-footer" style="margin-top:40px;">
  <span>TRACE-AML v4 &middot; Incident {_esc(inc_short)} &middot; FORENSIC USE ONLY</span>
  <span>Page 4 / 4</span>
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
        """Attempt to render PDF. Fails gracefully if dependencies are missing."""
        try:
            if _PLAYWRIGHT_OK:
                try:
                    self._write_pdf_playwright(html_str, pdf_path)
                    logger.debug("[PDF] Playwright Chromium → {}", pdf_path.name)
                    return
                except Exception as exc:
                    logger.warning("[PDF] Playwright render failed: {}", exc)

            # ── fpdf2 fallback ────────────────────────────────────────────────────
            self._write_pdf_fpdf2(pdf_path, data, generated_at)
        except ImportError:
            logger.warning("[PDF] Skipping PDF generation: 'fpdf2' or 'playwright' not installed.")
        except Exception as exc:
            logger.error("[PDF] Generation failed: {}", exc)

    def _write_pdf_playwright(self, html_str: str, pdf_path: Path) -> None:
        """Render HTML to PDF via Playwright headless Chromium."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError("playwright not installed in this environment")

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
        try:
            from fpdf import FPDF
        except ImportError:
            raise ImportError("fpdf2 not installed")

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
        
        # Monochromatic severity logic for fallback
        if severity in ("critical", "extreme", "emergency"):
            sev_rgb = (211, 47, 47) # Red
        elif severity == "high":
            sev_rgb = (145, 145, 145) # Grey
        else:
            sev_rgb = (198, 198, 198) # Light Grey

        BG, FG, GREY, BORDER = (14, 14, 14), (255, 255, 255), (145, 145, 145), (31, 31, 31)
        inc_short = incident.incident_id[-8:]
        dur       = _duration(incident.start_time, incident.last_seen_time)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        W = lambda: pdf.w - pdf.l_margin - pdf.r_margin

        def header():
            pdf.set_fill_color(*BG)
            pdf.rect(0, 0, pdf.w, pdf.h, "F") # Fill entire page background
            
            pdf.set_fill_color(19, 19, 19)
            pdf.rect(pdf.l_margin, 8, W(), 22, "F")
            pdf.set_draw_color(*FG)
            pdf.line(pdf.l_margin, 8, pdf.l_margin, 30)
            
            pdf.set_xy(pdf.l_margin + 5, 12)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(*FG)
            pdf.cell(W() / 2, 6, "TRACE-AML", ln=False)
            
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*GREY)
            pdf.set_x(pdf.l_margin + W() / 2)
            pdf.cell(W() / 2 - 5, 6, f"GENERATED: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}", align="R", ln=True)
            
            pdf.set_xy(pdf.l_margin + 5, 20)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*sev_rgb)
            pdf.cell(W(), 6, f"FORENSIC REPORT  \xb7  {severity.upper()} SEVERITY  \xb7  {inc_short}", ln=True)
            pdf.ln(10)

        def section(title: str):
            pdf.set_fill_color(*BORDER)
            pdf.rect(pdf.l_margin, pdf.get_y(), W(), 0.3, "F")
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*GREY)
            pdf.cell(W(), 5, title.upper(), ln=True)
            pdf.ln(1)

        def kv(label: str, val: str):
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*GREY)
            pdf.cell(35, 5, f"{label.upper()}:", ln=False)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*FG)
            pdf.cell(W() - 35, 5, str(val)[:80], ln=True)

        def table(col_defs: list, rows: list, row_fn: Any):
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*GREY)
            for lbl, w in col_defs:
                pdf.cell(w, 5, lbl.upper(), ln=False)
            pdf.ln()
            pdf.set_draw_color(*BORDER)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W(), pdf.get_y())
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(198, 198, 198)
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
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*GREY)
            pdf.cell(35, 5, f"{label.upper()}:", ln=False)
            pdf.set_font("Helvetica", "", 8)
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
        pdf.set_font("Helvetica", "I", 6)
        pdf.set_text_color(*GREY)
        pdf.cell(W() / 2, 5, f"TRACE-AML V4  \xb7  INCIDENT {inc_short}  \xb7  FORENSIC USE ONLY")
        pdf.cell(W() / 2, 5, f"GENERATED {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}", align="R")

        pdf.output(str(pdf_path))

    def _write_html(self, html_str: str, html_path: Path) -> None:
        html_path.write_text(html_str, encoding="utf-8")
        logger.debug("[PDF] HTML companion → {}", html_path.name)
