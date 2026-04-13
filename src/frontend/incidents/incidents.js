/**
 * Incidents Page Controller
 *
 * Fixes applied vs v1:
 *  - Arrow nav buttons (prev/next) scroll the card strip; sentinel still lazy-loads more
 *  - All colors now use design-system tokens (badge--neutral, badge--ghost, #ffb4ab, etc.)
 *  - Timeline cards are fully rendered with border, typed dots, proper time + meta layout
 *  - Trigger alert rows use the app's tonal / border-left pattern
 *  - Timeline items windowed at 30, "Show older" reveals more from already-fetched data
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  /* ── Constants ──────────────────────────────────── */
  var PAGE_SIZE     = 20;
  var TL_PAGE       = 30;
  var SCROLL_STEP   = 440; // px per arrow click

  /* ── State ──────────────────────────────────────── */
  var _currentId  = null;
  var _offset     = 0;
  var _done       = false;
  var _tlAll      = [];
  var _tlShown    = 0;

  /* ════════════════════════════════════════════════
     CARD STRIP — lazy load + arrow nav
  ════════════════════════════════════════════════ */

  function loadCards(initial) {
    if (_done) return;
    TraceClient.incidents({ limit: PAGE_SIZE, skip: _offset }).then(function (list) {
      // Remove skeleton placeholders on first batch
      if (initial) {
        ["sk1","sk2","sk3"].forEach(function (id) {
          var el = $(id);
          if (el) el.parentNode.removeChild(el);
        });
      }

      if (!list || list.length === 0) {
        _done = true;
        if (initial) showEmpty();
        return;
      }

      if (list.length < PAGE_SIZE) _done = true;
      _offset += list.length;

      var strip    = $("card-strip");
      var sentinel = $("card-sentinel");
      list.forEach(function (inc) {
        strip.insertBefore(buildCard(inc), sentinel);
      });

      // Auto-select first card
      if (initial) {
        var first = strip.querySelector(".inc-card");
        if (first) first.click();
      }

      updateArrows();
    });
  }

  function buildCard(inc) {
    var div = document.createElement("div");
    div.className = "inc-card";
    div.setAttribute("data-id", inc.incident_id);

    var sev       = String(inc.severity || "low");
    var sevBadge  = sev === "high" ? "badge--neutral" : "badge--ghost";
    var status    = String(inc.status  || "open").toUpperCase();
    /* Show last 8 chars of incident ID so it stays compact */
    var shortId   = TraceClient.escapeHtml(String(inc.incident_id || "").slice(-8));
    var summary   = TraceClient.escapeHtml(inc.summary || "Incident");
    var entityId  = TraceClient.escapeHtml(inc.entity_id || "—");
    var timeStr   = TraceClient.escapeHtml(TraceClient.formatTime(inc.last_seen_time || inc.start_time));

    div.innerHTML =
      '<div class="inc-card__id">' + shortId + '</div>'
    + '<div class="inc-card__sum">' + summary + '</div>'
    + '<div class="inc-card__foot">'
    +   '<span class="inc-card__ent">' + entityId + '</span>'
    +   '<span class="badge ' + sevBadge + '" style="font-size:0.52rem;padding:1px 5px">' + sev.toUpperCase() + '</span>'
    + '</div>'
    + '<div class="inc-card__time mt-1">' + timeStr + ' · ' + status + '</div>';

    div.addEventListener("click", function () {
      activateCard(div);
      loadIncident(inc.incident_id);
    });
    return div;
  }

  function activateCard(card) {
    var strip = $("card-strip");
    if (strip) {
      var prev = strip.querySelector(".inc-card.active");
      if (prev) prev.classList.remove("active");
    }
    card.classList.add("active");
  }

  function showEmpty() {
    var el = $("no-incidents"); if (el) el.classList.remove("hidden");
    var b  = $("inc-body");    if (b)  b.classList.add("hidden");
  }

  /* ── Arrow navigation ───────────────────────────── */

  function updateArrows() {
    var strip = $("card-strip");
    var prev  = $("strip-prev");
    var next  = $("strip-next");
    if (!strip) return;
    if (prev) prev.disabled = (strip.scrollLeft <= 0);
    if (next) next.disabled = _done && (strip.scrollLeft + strip.clientWidth >= strip.scrollWidth - 4);
  }

  function initArrows() {
    var strip = $("card-strip");
    var prev  = $("strip-prev");
    var next  = $("strip-next");
    if (!strip || !prev || !next) return;

    prev.addEventListener("click", function () {
      strip.scrollBy({ left: -SCROLL_STEP, behavior: "smooth" });
      setTimeout(updateArrows, 350);
    });

    next.addEventListener("click", function () {
      strip.scrollBy({ left: SCROLL_STEP, behavior: "smooth" });
      setTimeout(updateArrows, 350);
    });

    strip.addEventListener("scroll", updateArrows);
  }

  /* ── IntersectionObserver for lazy card loading ── */

  function initSentinel() {
    var sentinel = $("card-sentinel");
    var strip    = $("card-strip");
    if (!sentinel || !strip || !("IntersectionObserver" in window)) return;

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting && !_done) loadCards(false);
      });
    }, { root: strip, threshold: 0.1 });

    observer.observe(sentinel);
  }

  /* ════════════════════════════════════════════════
     INCIDENT DETAIL
  ════════════════════════════════════════════════ */

  function loadIncident(id) {
    if (!id) return;
    _currentId = id;

    // Show skeleton in timeline while loading
    var tlEl = $("tl-events");
    if (tlEl) tlEl.innerHTML =
      '<div class="skeleton mb-2" style="height:72px;opacity:.35"></div>'
    + '<div class="skeleton mb-2" style="height:72px;opacity:.2"></div>'
    + '<div class="skeleton" style="height:72px;opacity:.1"></div>';

    var ph = $("tl-placeholder"); if (ph) ph.style.display = "none";
    var sm = $("tl-show-more");   if (sm) sm.classList.add("hidden");

    TraceClient.incident(id).then(function (detail) {
      if (!detail) return;
      renderHeader(detail.incident);
      renderEntity(detail.entity);
      renderAlerts(detail.alerts);
      renderTimeline(detail.timeline);
      renderActions(detail.actions);
      syncControls(detail.incident);
    });
  }

  /* ════════════════════════════════════════════════
     RENDER HELPERS
  ════════════════════════════════════════════════ */

  function renderHeader(inc) {
    if (!inc) return;
    var shortId = String(inc.incident_id || "").slice(-8);
    setText("inc-id-label",    "Case #" + shortId);
    setText("inc-title",       inc.summary || ("Incident " + inc.incident_id));
    var statusEl = $("inc-status");
    if (statusEl) {
      statusEl.textContent = String(inc.status || "open").toUpperCase();
      statusEl.className   = inc.status === "closed"
        ? "font-mono text-[0.72rem] text-outline"
        : "font-mono text-[0.72rem] text-primary";
    }
    setText("inc-alert-count", String(inc.alert_count || 0));
    setText("inc-start",       TraceClient.formatTime(inc.start_time));
  }

  function renderEntity(e) {
    if (!e) return;
    setText("ent-name",  e.name || e.entity_id || "—");
    var typeEl = $("ent-type");
    if (typeEl) typeEl.textContent = String(e.type || e.entity_type || "unknown").toUpperCase();
    setText("ent-first", TraceClient.formatDateTime(e.created_at)   || "—");
    setText("ent-last",  TraceClient.formatDateTime(e.last_seen_at) || "—");
    setText("ent-dets",  String(e.detection_count || e.recent_alert_count || 0));
  }

  function renderAlerts(alerts) {
    var root = $("inc-alerts-root");
    if (!root) return;
    if (!alerts || alerts.length === 0) {
      root.innerHTML = '<div class="text-outline font-mono text-[0.62rem] py-3 text-center">No alerts linked</div>';
      return;
    }
    root.innerHTML = alerts.slice(0, 8).map(function (a) {
      var sev    = String(a.severity || "low");
      var type   = TraceClient.escapeHtml(String(a.type || "ALERT").toUpperCase());
      var reason = TraceClient.escapeHtml(a.reason || "");
      var time   = TraceClient.escapeHtml(TraceClient.formatTime(a.timestamp_utc));
      var cnt    = a.event_count ? " · " + a.event_count + " events" : "";
      return '<div class="alert-row alert-row--' + sev + '">'
        + '<div class="flex items-center justify-between mb-0.5">'
        +   '<span class="font-mono text-[0.6rem] font-medium text-on-surface">' + type + '</span>'
        +   '<span class="font-mono text-[0.55rem] text-outline">' + time + TraceClient.escapeHtml(cnt) + '</span>'
        + '</div>'
        + '<p class="text-[0.68rem] text-on-surface-variant leading-snug">' + reason + '</p>'
        + '</div>';
    }).join("");
  }

  function renderTimeline(timeline) {
    _tlAll   = timeline ? timeline.slice().reverse() : [];
    _tlShown = 0;
    var root = $("tl-events");
    if (!root) return;
    root.innerHTML = "";

    if (_tlAll.length === 0) {
      root.innerHTML = '<div class="text-outline font-mono text-[0.7rem] py-12 text-center">No timeline events</div>';
      updateTlMore();
      return;
    }
    appendTlBatch(TL_PAGE);
    updateTlMore();
  }

  function appendTlBatch(count) {
    var root  = $("tl-events");
    if (!root) return;
    var batch = _tlAll.slice(_tlShown, _tlShown + count);
    batch.forEach(function (item) {
      var div = document.createElement("div");
      div.className = "tl-entry";
      div.innerHTML = buildTlCard(item);
      root.appendChild(div);
    });
    _tlShown += batch.length;
    var label = $("tl-count-label");
    if (label) label.textContent = _tlShown + " / " + _tlAll.length + " events";
  }

  function buildTlCard(item) {
    var kind      = String(item.kind || "event");
    /* Badge + dot use design-system classes only */
    var badgeHtml = "";
    if      (kind === "incident") badgeHtml = '<span class="badge badge--filled">INCIDENT</span>';
    else if (kind === "alert")    badgeHtml = '<span class="badge badge--neutral">ALERT</span>';
    else if (kind === "action")   badgeHtml = '<span class="badge badge--ghost">ACTION</span>';
    else                          badgeHtml = '<span class="badge badge--ghost">EVENT</span>';

    var title   = TraceClient.escapeHtml(item.title   || "");
    var summary = TraceClient.escapeHtml(item.summary || "");
    var time    = TraceClient.escapeHtml(TraceClient.formatTime(item.timestamp_utc));
    var date    = TraceClient.escapeHtml(TraceClient.formatDateTime(item.timestamp_utc).slice(0, 10));

    /* Build meta pills */
    var meta = [];
    if (item.entity_id) meta.push("Entity: " + TraceClient.escapeHtml(item.entity_id));
    if (item.source)    meta.push("Source: " + TraceClient.escapeHtml(item.source));
    if (item.metadata) {
      if (item.metadata.track_id)   meta.push("Track: "  + TraceClient.escapeHtml(item.metadata.track_id));
      if (item.metadata.event_count) meta.push("Events: " + item.metadata.event_count);
    }

    return '<div class="tl-dot tl-dot--' + kind + '"></div>'
      + '<div class="tl-card tl-card--' + kind + '">'
      /* Row 1: badge + title + time */
      + '<div class="flex items-center justify-between gap-2 mb-1.5">'
      +   '<div class="flex items-center gap-2 min-w-0">'
      +     badgeHtml
      +     '<span class="font-mono text-[0.7rem] font-medium text-on-surface truncate">' + title + '</span>'
      +   '</div>'
      +   '<div class="flex-shrink-0 text-right">'
      +     '<span class="font-mono text-[0.65rem] text-white block">' + time + '</span>'
      +     '<span class="font-mono text-[0.52rem] text-outline block">' + date + '</span>'
      +   '</div>'
      + '</div>'
      /* Row 2: summary */
      + (summary
        ? '<p class="text-[0.7rem] text-on-surface-variant leading-relaxed mb-1.5">' + summary + '</p>'
        : '')
      /* Row 3: meta chips */
      + (meta.length
        ? '<div class="flex flex-wrap gap-2 mt-1">'
          + meta.map(function(m){ return '<span class="font-mono text-[0.55rem] text-outline border border-outline-variant/20 px-1.5 py-0.5">' + m + '</span>'; }).join("")
          + '</div>'
        : '')
      + '</div>';
  }

  function updateTlMore() {
    var btn = $("tl-show-more");
    if (!btn) return;
    if (_tlShown < _tlAll.length) btn.classList.remove("hidden");
    else btn.classList.add("hidden");
  }

  function renderActions(actions) {
    var root = $("inc-actions-root");
    if (!root) return;
    if (!actions || actions.length === 0) {
      root.innerHTML = '<div class="text-outline font-mono text-[0.62rem] py-3 text-center">No actions recorded</div>';
      return;
    }
    root.innerHTML = actions.slice(0, 10).map(function (a) {
      var type  = TraceClient.escapeHtml(String(a.action_type || a.type || "LOG").toUpperCase());
      var trig  = TraceClient.escapeHtml(a.trigger || "");
      var time  = TraceClient.escapeHtml(TraceClient.formatTime(a.timestamp_utc));
      var id    = TraceClient.escapeHtml(String(a.action_id || "").slice(-6));
      return '<div class="bg-surface-lowest border-l border-outline-variant/20 pl-2.5 pr-2 py-2 mb-1.5">'
        + '<div class="flex items-center justify-between">'
        +   '<span class="font-mono text-[0.6rem] text-on-surface font-medium">' + type + '</span>'
        +   '<span class="font-mono text-[0.55rem] text-outline">' + time + '</span>'
        + '</div>'
        + (trig ? '<p class="font-mono text-[0.58rem] text-on-surface-variant mt-0.5">' + trig + '</p>' : '')
        + '<span class="font-mono text-[0.52rem] text-outline block mt-0.5">' + id + '</span>'
        + '</div>';
    }).join("");
  }

  function syncControls(inc) {
    var el = $("sev-select");
    if (el) el.value = inc.severity || "low";
    setText("ctrl-status", "");
  }

  /* ── Utility ──────────────────────────────────── */
  function setText(id, val) { var el = $(id); if (el) el.textContent = val; }

  function resetStrip() {
    var strip = $("card-strip");
    if (!strip) return;
    var cards = strip.querySelectorAll(".inc-card");
    cards.forEach(function (c) { c.parentNode.removeChild(c); });
    _offset = 0; _done = false;
  }

  /* ════════════════════════════════════════════════
     FORENSIC LOG PANEL
  ════════════════════════════════════════════════ */

  var _flBytes = 0;
  var _flCollapsed = false;

  function flTimestamp(raw) {
    if (!raw) { var n = new Date(); return '[' + n.toISOString().slice(11,23) + ']'; }
    try {
      var d = new Date(String(raw).replace('Z','') + (String(raw).endsWith('Z') ? '' : 'Z'));
      return '[' + d.toISOString().slice(11,23) + ']';
    } catch(e) { return '[' + String(raw).slice(11,23) + ']'; }
  }

  function flTopicClass(topic) {
    var t = String(topic || '').toLowerCase();
    if (t.indexOf('alert')    !== -1) return 'll-topic-alert';
    if (t.indexOf('incident') !== -1) return 'll-topic-incident';
    if (t.indexOf('action')   !== -1) return 'll-topic-action';
    return 'll-topic-default';
  }

  function flChipHtml(payload) {
    if (!payload) return '';
    var sev = String(payload.severity || '').toUpperCase();
    if (sev === 'HIGH' || sev === 'CRITICAL') return '<span class="ll-chip ll-chip--error">' + sev + '</span>';
    if (sev === 'MEDIUM' || sev === 'LOW')    return '<span class="ll-chip">' + sev + '</span>';
    var dec = String(payload.decision || '').toUpperCase();
    if (dec) return '<span class="ll-chip">' + dec + '</span>';
    var eid = payload.entity_id || payload.entity || '';
    if (eid) return '<span class="ll-chip">' + TraceClient.escapeHtml(String(eid)) + '</span>';
    return '';
  }

  function flAppendLine(event) {
    var body = document.getElementById('inc-fl-body');
    if (!body) return;
    var topic = String(event.topic || 'SESSION').toUpperCase();
    var payload = event.payload || {};
    var ts  = flTimestamp(event.timestamp_utc);
    var cls = flTopicClass(topic);
    var chip = flChipHtml(payload);
    var msg = payload.message || payload.reason || payload.summary || '';
    if (!msg) {
      var parts = [];
      Object.keys(payload).slice(0,4).forEach(function(k) {
        var v = payload[k];
        if (v !== null && v !== undefined && typeof v !== 'object') parts.push(k + '=' + String(v).slice(0,40));
      });
      msg = parts.join('  ') || topic.toLowerCase() + ' event';
    }
    msg = TraceClient.escapeHtml(String(msg).slice(0,200));
    var html = '<div class="log-line">'
      + '<span class="ll-ts">' + ts + '</span>'
      + '<span class="' + cls + '">' + TraceClient.escapeHtml(topic) + '</span>'
      + '<span class="ll-msg">' + msg + chip + '</span>'
      + '</div>';
    body.insertAdjacentHTML('afterbegin', html);
    _flBytes += html.length;
    var buf = document.getElementById('inc-fl-buffer');
    if (buf) buf.textContent = Math.round(_flBytes / 1024);
    var lines = body.querySelectorAll('.log-line');
    if (lines.length > 300) lines[lines.length - 1].remove();
  }

  function initForensicPanel() {
    var MINIMIZED_H = 33;       // header-only height (px)
    var MAX_H_RATIO = 0.30;     // max 30% of viewport
    var SNAP_THRESH = 48;       // snap to minimized if opened less than this many px

    var panel     = document.getElementById('inc-forensic-panel');
    var splitter  = document.getElementById('inc-forensic-splitter');
    var toggleBtn = document.getElementById('inc-fl-toggle-btn');
    var clearBtn  = document.getElementById('inc-fl-clear');
    var body      = document.getElementById('inc-fl-body');
    var bufEl     = document.getElementById('inc-fl-buffer');

    if (!panel || !splitter) return;

    /* ── Height helpers ──────────────────────────────── */
    function maxH() { return Math.floor(window.innerHeight * MAX_H_RATIO); }

    function applyHeight(h, animate) {
      if (animate) {
        panel.classList.add('animating');
        var onEnd = function() {
          panel.classList.remove('animating');
          panel.removeEventListener('transitionend', onEnd);
        };
        panel.addEventListener('transitionend', onEnd);
      } else {
        panel.classList.remove('animating');
      }
      panel.style.height = h + 'px';
      var expanded = h > MINIMIZED_H;
      panel.classList.toggle('is-expanded', expanded);
      if (toggleBtn) toggleBtn.innerHTML = expanded ? '&darr; COLLAPSE' : '&uarr; EXPAND';
    }

    /* Start minimized */
    applyHeight(MINIMIZED_H, false);

    /* ── Toggle button ───────────────────────────────── */
    if (toggleBtn) {
      toggleBtn.addEventListener('click', function() {
        var cur = parseFloat(panel.style.height) || MINIMIZED_H;
        applyHeight(cur <= MINIMIZED_H ? maxH() : MINIMIZED_H, true);
      });
    }

    /* ── Double-click splitter ───────────────────────── */
    splitter.addEventListener('dblclick', function(e) {
      e.preventDefault();
      var cur = parseFloat(panel.style.height) || MINIMIZED_H;
      applyHeight(cur <= MINIMIZED_H ? maxH() : MINIMIZED_H, true);
    });

    /* ── Drag to resize ──────────────────────────────── */
    var _drag = false, _startY = 0, _startH = 0;

    splitter.addEventListener('mousedown', function(e) {
      if (e.button !== 0) return;
      _drag = true;
      _startY = e.clientY;
      _startH = parseFloat(panel.style.height) || MINIMIZED_H;
      splitter.classList.add('is-dragging');
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
      if (!_drag) return;
      var newH = Math.max(MINIMIZED_H, Math.min(maxH(), _startH + (_startY - e.clientY)));
      panel.style.height = newH + 'px';
      var expanded = newH > MINIMIZED_H;
      panel.classList.toggle('is-expanded', expanded);
      if (toggleBtn) toggleBtn.innerHTML = expanded ? '&darr; COLLAPSE' : '&uarr; EXPAND';
    });

    document.addEventListener('mouseup', function() {
      if (!_drag) return;
      _drag = false;
      splitter.classList.remove('is-dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      var finalH = parseFloat(panel.style.height) || MINIMIZED_H;
      if (finalH > MINIMIZED_H && finalH < MINIMIZED_H + SNAP_THRESH) {
        applyHeight(MINIMIZED_H, true);
      }
    });

    /* ── Clear console ───────────────────────────────── */
    if (clearBtn) {
      clearBtn.addEventListener('click', function() {
        if (body) body.innerHTML = '';
        _flBytes = 0;
        if (bufEl) bufEl.textContent = '0';
      });
    }

    /* ── Connect SSE ─────────────────────────────────── */
    TraceClient.connectSSE(function(event) { 
      // Only display meaningful events; skip metrics (session.state)
      if (TraceClient.isMeaningfulEvent(event)) {
        flAppendLine(event);
      }
    });
  }

  /* ════════════════════════════════════════════════
     CONTROLS wiring
  ════════════════════════════════════════════════ */

  function wireControls() {
    var applyBtn = $("btn-apply-sev");
    if (applyBtn) applyBtn.addEventListener("click", function () {
      if (!_currentId) return;
      var sev = ($("sev-select") || {}).value || "low";
      setText("ctrl-status", "Updating…");
      TraceClient.setSeverity(_currentId, sev).then(function (result) {
        setText("ctrl-status", result ? ("Severity → " + sev) : "Failed — offline");
      });
    });

    var closeBtn = $("btn-close-inc");
    if (closeBtn) closeBtn.addEventListener("click", function () {
      if (!_currentId) return;
      setText("ctrl-status", "Closing…");
      TraceClient.closeIncident(_currentId).then(function (result) {
        if (result) {
          setText("ctrl-status", "Closed");
          resetStrip();
          loadCards(true);
        } else {
          setText("ctrl-status", "Failed — offline");
        }
      });
    });

    var moreBtn = $("btn-show-more-tl");
    if (moreBtn) moreBtn.addEventListener("click", function () {
      appendTlBatch(TL_PAGE);
      updateTlMore();
    });
  }

  /* ════════════════════════════════════════════════
     INIT
  ════════════════════════════════════════════════ */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    wireControls();
    initArrows();
    initSentinel();
    initForensicPanel();

    // Load cards immediately — no probe() gate
    loadCards(true);
    TraceClient.probe(); // fire-and-forget for connection badge
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
