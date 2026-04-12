/**
 * Live Ops Page Controller
 *
 * - Polls /api/v1/live/snapshot every 3s → entities, incidents, alerts, health
 * - Connects SSE → terminal output + live alert updates
 * - Polls /api/v1/live/overlay every 200ms (when pipeline active)
 * - Wires Refresh / Enable Camera / Timeline filter buttons
 * - Measures and displays API latency
 */
(function () {
  "use strict";

  var SNAPSHOT_INTERVAL = 5000;
  var TIMELINE_INTERVAL = 8000;
  var _snapshotTimer = null;
  var _timelineTimer = null;
  var _snapshotInFlight = false;
  var _timelineInFlight = false;
  var _cameraActive = false;
  var _lastSnapshotData = null;

  /* ─── DOM references ─── */
  function $(id) { return document.getElementById(id); }

  /* ─── Snapshot rendering ─── */

  function renderSnapshot(snap) {
    if (!snap) return;

    // ── Footer stats: set FIRST, unconditionally, with try/catch ──
    try {
      var h = snap.system_health || {};
      var hEnt = document.getElementById("health-entities");
      var hInc = document.getElementById("health-incidents");
      var hAlr = document.getElementById("health-alerts");
      var hDet = document.getElementById("health-detections");
      if (hEnt) hEnt.textContent = String(h.active_entity_count != null ? h.active_entity_count : (snap.active_entities || []).length);
      if (hInc) hInc.textContent = String(h.open_incident_count != null ? h.open_incident_count : (snap.active_incidents || []).length);
      if (hAlr) hAlr.textContent = String(h.recent_alert_count != null ? h.recent_alert_count : (snap.recent_alerts || []).length);
      if (hDet) hDet.textContent = String(h.total_detection_count || 0);
    } catch (e) { console.warn("[LiveOps] footer stats error:", e); }

    // Active entities
    var entRoot = $("active-entities-root");
    if (entRoot && snap.active_entities) {
      // Update entity count header
      var countEl = $("entity-count");
      if (countEl) countEl.textContent = String(snap.active_entities.length);

      if (snap.active_entities.length === 0) {
        entRoot.innerHTML = TraceRender.emptyState("No active entities");
      } else {
        entRoot.innerHTML = snap.active_entities.map(function (e) {
          return TraceRender.entityCard(e);
        }).join("");
      }
    }

    // Active incidents
    var incRoot = $("incident-updates-root");
    if (incRoot && snap.active_incidents) {
      if (snap.active_incidents.length === 0) {
        incRoot.innerHTML = TraceRender.emptyState("No open incidents");
      } else {
        incRoot.innerHTML = snap.active_incidents.map(function (inc) {
          return TraceRender.incidentCard(inc);
        }).join("");
      }
    }

    // Recent alerts
    var alertRoot = $("alert-stream-root");
    if (alertRoot && snap.recent_alerts) {
      // Update alert badge count
      var alertBadge = $("alert-badge");
      if (alertBadge) {
        var highCount = snap.recent_alerts.filter(function (a) {
          return String(a.severity || "").toLowerCase() === "high";
        }).length;
        alertBadge.textContent = highCount > 0 ? "Critical (" + highCount + ")" : "Active (" + snap.recent_alerts.length + ")";
        alertBadge.className = highCount > 0 ? "badge badge--error" : "badge badge--ghost";
      }

      if (snap.recent_alerts.length === 0) {
        alertRoot.innerHTML = TraceRender.emptyState("No recent alerts");
      } else {
        alertRoot.innerHTML = snap.recent_alerts.map(function (a) {
          return TraceRender.alertRow(a);
        }).join("");
      }
    }

    // FPS from health
    try {
      var fpsEl = document.getElementById("health-fps");
      if (fpsEl) {
        var rt = (snap.system_health && snap.system_health.runtime) || {};
        fpsEl.textContent = (typeof rt.fps === "number" && rt.fps > 0) ? rt.fps.toFixed(1) : "—";
      }
    } catch (e) { /* ignore */ }
  }

  /** Render the Recent Timeline center panel from global timeline data */
  function renderTimeline(items) {
    var root = $("recent-timeline-root");
    if (!root) return;

    // Apply filter
    var activeFilter = _activeTimelineFilter;
    if (activeFilter !== "all") {
      items = items.filter(function (item) {
        var kind = String(item.kind || "").toLowerCase();
        if (activeFilter === "events") return kind === "event" || kind === "detection" || kind === "entity_resolve";
        if (activeFilter === "alerts") return kind === "alert" || kind === "incident" || kind === "action";
        return true;
      });
    }

    if (items.length === 0) {
      root.innerHTML = TraceRender.emptyState("No " + activeFilter + " items");
    } else {
      root.innerHTML = items.map(function (item) {
        return TraceRender.timelineCard(item);
      }).join("");
    }
  }

  var _lastTimelineData = [];
  var _activeTimelineFilter = "all";

  function renderHealth(health) {
    if (!health) return;

    var el;
    el = $("health-entities");
    if (el) el.textContent = String(health.active_entity_count || 0);

    el = $("health-incidents");
    if (el) el.textContent = String(health.open_incident_count || 0);

    el = $("health-alerts");
    if (el) el.textContent = String(health.recent_alert_count || 0);

    el = $("health-detections");
    if (el) el.textContent = String(health.total_detection_count || 0);

    // FPS
    el = $("health-fps");
    if (el) {
      var fps = (health.runtime && health.runtime.fps);
      if (typeof fps === "number" && fps > 0) {
        el.textContent = fps.toFixed(1);
      } else {
        el.textContent = "—";
      }
    }
  }

  /* ─── Forensic log panel ─── */

  var _logBytes = 0;
  var _logCollapsed = false;

  function flTimestamp(raw) {
    // Format as [HH:mm:ss.mmm]
    if (!raw) {
      var n = new Date();
      return '[' + n.toISOString().slice(11, 23) + ']';
    }
    try {
      var d = new Date(String(raw).replace('Z','') + (String(raw).endsWith('Z') ? '' : 'Z'));
      return '[' + d.toISOString().slice(11, 23) + ']';
    } catch (e) {
      return '[' + String(raw).slice(11, 23) + ']';
    }
  }

  function flTopicClass(topic) {
    var t = String(topic || '').toLowerCase();
    if (t.indexOf('alert') !== -1)    return 'll-topic-alert';
    if (t.indexOf('incident') !== -1) return 'll-topic-incident';
    if (t.indexOf('action') !== -1)   return 'll-topic-action';
    return 'll-topic-default';
  }

  function flChipHtml(payload) {
    // Pull out severity or decision as a chip
    if (!payload) return '';
    var sev = String(payload.severity || '').toUpperCase();
    if (sev === 'HIGH' || sev === 'CRITICAL') {
      return '<span class="ll-chip ll-chip--error">' + sev + '</span>';
    }
    if (sev === 'MEDIUM' || sev === 'LOW') {
      return '<span class="ll-chip">' + sev + '</span>';
    }
    var dec = String(payload.decision || '').toUpperCase();
    if (dec) return '<span class="ll-chip">' + dec + '</span>';
    var entityId = payload.entity_id || payload.entity || '';
    if (entityId) return '<span class="ll-chip">' + TraceClient.escapeHtml(String(entityId)) + '</span>';
    return '';
  }

  function appendLogLine(event) {
    var body = document.getElementById('fl-body');
    if (!body) return;

    var topic   = String(event.topic || 'SESSION').toUpperCase();
    var payload = event.payload || {};
    var ts      = flTimestamp(event.timestamp_utc);
    var cls     = flTopicClass(topic);
    var chip    = flChipHtml(payload);

    // Build message text from payload
    var msg = '';
    if (payload.message)     msg = payload.message;
    else if (payload.reason) msg = payload.reason;
    else if (payload.summary)msg = payload.summary;
    else {
      // Compact key=value rendering
      var parts = [];
      Object.keys(payload).slice(0, 4).forEach(function (k) {
        var v = payload[k];
        if (v !== null && v !== undefined && typeof v !== 'object') {
          parts.push(k + '=' + String(v).slice(0, 40));
        }
      });
      msg = parts.join('  ') || topic.toLowerCase() + ' event';
    }
    msg = TraceClient.escapeHtml(String(msg).slice(0, 200));

    var lineHtml = '<div class="log-line">'
      + '<span class="ll-ts">' + ts + '</span>'
      + '<span class="' + cls + '">' + TraceClient.escapeHtml(topic) + '</span>'
      + '<span class="ll-msg">' + msg + chip + '</span>'
      + '</div>';

    // Prepend so newest is at top
    body.insertAdjacentHTML('afterbegin', lineHtml);

    // Update buffer counter
    _logBytes += lineHtml.length;
    var bufEl = document.getElementById('fl-buffer');
    if (bufEl) bufEl.textContent = Math.round(_logBytes / 1024);

    // Cap at 300 lines
    var lines = body.querySelectorAll('.log-line');
    if (lines.length > 300) lines[lines.length - 1].remove();
  }

  function initSSE() {
    TraceClient.connectSSE(function (event) {
      appendLogLine(event);
    });
  }

  function initForensicControls() {
    var MINIMIZED_H  = 33;                         // header-only height (px)
    var MAX_H_RATIO  = 0.30;                       // max 30% of viewport
    var SNAP_THRESH  = 48;                         // px — snap to minimized if within this

    var panel     = document.getElementById('forensic-panel');
    var splitter  = document.getElementById('forensic-splitter');
    var toggleBtn = document.getElementById('fl-toggle-btn');
    var clearBtn  = document.getElementById('fl-clear');
    var body      = document.getElementById('fl-body');
    var bufEl     = document.getElementById('fl-buffer');

    if (!panel || !splitter) return;

    /* ── Height utility ─────────────────────────────────── */
    function maxH() { return Math.floor(window.innerHeight * MAX_H_RATIO); }

    function applyHeight(h, animate) {
      if (animate) {
        panel.classList.add('animating');
        // Remove after transition ends
        var onEnd = function() {
          panel.classList.remove('animating');
          panel.removeEventListener('transitionend', onEnd);
        };
        panel.addEventListener('transitionend', onEnd);
      } else {
        panel.classList.remove('animating');
      }
      panel.style.height = h + 'px';

      var isExpanded = h > MINIMIZED_H;
      panel.classList.toggle('is-expanded', isExpanded);
      if (toggleBtn) {
        toggleBtn.innerHTML = isExpanded ? '&darr; COLLAPSE' : '&uarr; EXPAND';
      }
    }

    /* Start minimized */
    applyHeight(MINIMIZED_H, false);

    /* ── Toggle button (click to expand / collapse) ──────── */
    if (toggleBtn) {
      toggleBtn.addEventListener('click', function() {
        var current = parseFloat(panel.style.height) || MINIMIZED_H;
        if (current <= MINIMIZED_H) {
          // Expand to 30%
          applyHeight(maxH(), true);
        } else {
          // Collapse
          applyHeight(MINIMIZED_H, true);
        }
      });
    }

    /* ── Double-click splitter → expand / collapse ───────── */
    splitter.addEventListener('dblclick', function(e) {
      e.preventDefault();
      var current = parseFloat(panel.style.height) || MINIMIZED_H;
      if (current <= MINIMIZED_H) {
        applyHeight(maxH(), true);
      } else {
        applyHeight(MINIMIZED_H, true);
      }
    });

    /* ── Drag to resize ──────────────────────────────────── */
    var _dragging  = false;
    var _dragStartY = 0;
    var _dragStartH = 0;

    splitter.addEventListener('mousedown', function(e) {
      if (e.button !== 0) return;               // left-click only
      _dragging   = true;
      _dragStartY = e.clientY;
      _dragStartH = parseFloat(panel.style.height) || MINIMIZED_H;
      splitter.classList.add('is-dragging');
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
      if (!_dragging) return;
      var delta  = _dragStartY - e.clientY;     // drag up = positive delta = grow
      var newH   = Math.max(MINIMIZED_H, Math.min(maxH(), _dragStartH + delta));
      // Apply instantly, no animation during drag
      panel.style.height = newH + 'px';
      var isExpanded = newH > MINIMIZED_H;
      panel.classList.toggle('is-expanded', isExpanded);
      if (toggleBtn) {
        toggleBtn.innerHTML = isExpanded ? '&darr; COLLAPSE' : '&uarr; EXPAND';
      }
    });

    document.addEventListener('mouseup', function() {
      if (!_dragging) return;
      _dragging = false;
      splitter.classList.remove('is-dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';

      // Snap: if barely opened, collapse all the way to minimized
      var finalH = parseFloat(panel.style.height) || MINIMIZED_H;
      if (finalH < MINIMIZED_H + SNAP_THRESH && finalH > MINIMIZED_H) {
        applyHeight(MINIMIZED_H, true);
      }
    });

    /* ── Clear console ───────────────────────────────────── */
    if (clearBtn) {
      clearBtn.addEventListener('click', function() {
        if (body) body.innerHTML = '';
        _logBytes = 0;
        if (bufEl) bufEl.textContent = '0';
      });
    }
  }

  /* ─── Polling loops ─── */

  function pollSnapshot() {
    if (_snapshotInFlight) return; // prevent request pileup
    _snapshotInFlight = true;
    var t0 = performance.now();
    TraceClient.liveSnapshot({ entity_limit: 12, incident_limit: 6, alert_limit: 10 })
      .then(function (snap) {
        _snapshotInFlight = false;
        // Measure and display latency
        var latency = Math.round(performance.now() - t0);
        var latEl = $("health-latency");
        if (latEl) latEl.textContent = latency + " ms";

        _lastSnapshotData = snap;
        renderSnapshot(snap);

        // If snapshot's system_health was empty, fetch health directly
        if (snap && (!snap.system_health || !snap.system_health.active_entity_count)) {
          TraceClient.health().then(function (h) {
            if (h) renderHealth(h);
          });
        }
      })
      .catch(function () { _snapshotInFlight = false; });
  }

  function pollTimeline() {
    if (_timelineInFlight) return;
    _timelineInFlight = true;
    TraceClient.globalTimeline({ limit: 20 }).then(function (items) {
      _timelineInFlight = false;
      if (!items) return;
      _lastTimelineData = items;
      renderTimeline(items);
    }).catch(function () { _timelineInFlight = false; });
  }

  /* ─── Camera feed + Detection overlay ─── */

  var _overlayTimer = null;

  var OVERLAY_COLORS = {
    accept:  "rgba(0, 255, 120, 0.85)",
    review:  "rgba(255, 200, 0, 0.85)",
    reject:  "rgba(255, 60, 60, 0.85)"
  };

  function toggleCamera() {
    var feedImg = $("camera-feed");
    var placeholder = $("camera-placeholder");
    var btn = $("btn-enable-camera");
    var canvas = $("detection-overlay");

    if (_cameraActive) {
      // Stop: clear the MJPEG src and overlay
      if (feedImg) {
        feedImg.src = "";
        feedImg.style.display = "none";
      }
      if (placeholder) placeholder.style.display = "";
      if (btn) btn.textContent = "Enable Camera";
      stopOverlayPoll();
      if (canvas) {
        canvas.classList.add("hidden");
        var ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      _cameraActive = false;
    } else {
      // Start: set MJPEG src
      var mjpegUrl = TraceClient.baseUrl + "/api/v1/live/mjpeg";
      if (feedImg) {
        feedImg.onerror = function () {
          // Stream unavailable (404/503/etc). Revert UI cleanly.
          feedImg.style.display = "none";
          feedImg.src = "";
          if (placeholder) placeholder.style.display = "";
          if (btn) btn.textContent = "Enable Camera";
          stopOverlayPoll();
          _cameraActive = false;
        };
        feedImg.src = mjpegUrl;
        feedImg.style.display = "block";
      }
      if (placeholder) placeholder.style.display = "none";
      if (btn) btn.textContent = "Disable Camera";
      if (canvas) canvas.classList.remove("hidden");
      _cameraActive = true;
      startOverlayPoll();
    }
  }

  function startOverlayPoll() {
    if (_overlayTimer) return;
    pollOverlay();
    _overlayTimer = setInterval(pollOverlay, 200); // ~5fps
  }

  function stopOverlayPoll() {
    if (_overlayTimer) {
      clearInterval(_overlayTimer);
      _overlayTimer = null;
    }
  }

  function pollOverlay() {
    TraceClient.liveOverlay().then(function (data) {
      drawOverlay(data);
    }).catch(function () {
      // silently ignore — overlay not available
    });
  }

  function drawOverlay(data) {
    var canvas = $("detection-overlay");
    var feedImg = $("camera-feed");
    if (!canvas || !feedImg) return;

    // Match canvas size to rendered image size
    var rect = feedImg.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    canvas.width = rect.width;
    canvas.height = rect.height;

    var ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data || !data.active || !data.boxes || data.boxes.length === 0) {
      // Update FPS display from overlay when available
      var fpsEl = $("camera-fps");
      if (fpsEl && data && data.fps) {
        fpsEl.textContent = data.fps.toFixed(1) + " FPS";
      }
      return;
    }

    // Update FPS from pipeline
    var fpsEl2 = $("camera-fps");
    if (fpsEl2) fpsEl2.textContent = data.fps.toFixed(1) + " FPS";

    var cw = canvas.width;
    var ch = canvas.height;

    for (var i = 0; i < data.boxes.length; i++) {
      var box = data.boxes[i];
      var x = box.x * cw;
      var y = box.y * ch;
      var w = box.w * cw;
      var h = box.h * ch;
      var decision = String(box.decision || "reject").toLowerCase();
      var color = OVERLAY_COLORS[decision] || OVERLAY_COLORS.reject;

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Draw corner brackets (tactical look)
      var bracketLen = Math.max(8, Math.min(w, h) * 0.15);
      ctx.lineWidth = 3;
      // Top-left
      ctx.beginPath(); ctx.moveTo(x, y + bracketLen); ctx.lineTo(x, y); ctx.lineTo(x + bracketLen, y); ctx.stroke();
      // Top-right
      ctx.beginPath(); ctx.moveTo(x + w - bracketLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + bracketLen); ctx.stroke();
      // Bottom-left
      ctx.beginPath(); ctx.moveTo(x, y + h - bracketLen); ctx.lineTo(x, y + h); ctx.lineTo(x + bracketLen, y + h); ctx.stroke();
      // Bottom-right
      ctx.beginPath(); ctx.moveTo(x + w - bracketLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - bracketLen); ctx.stroke();

      // Label: decision + name + confidence
      var label = decision.toUpperCase() + " " + (box.label || "Unknown") + " " + (box.confidence || 0).toFixed(1) + "%";
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      var textW = ctx.measureText(label).width;
      // Label background
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(x, Math.max(0, y - 20), textW + 8, 18);
      // Label text
      ctx.fillStyle = color;
      ctx.fillText(label, x + 4, Math.max(13, y - 6));

      // Track ID below box
      if (box.track_id) {
        var trackLabel = box.track_id;
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.fillStyle = "rgba(200, 200, 200, 0.7)";
        ctx.fillText(trackLabel, x, y + h + 14);
      }
    }
  }

  /* ─── Timeline filter tabs ─── */

  function initFilterTabs() {
    var tabs = document.querySelectorAll("[data-timeline-filter]");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        // Update active state
        tabs.forEach(function (t) {
          t.classList.remove("text-white", "border-b-2", "border-white");
          t.classList.add("text-outline");
        });
        tab.classList.add("text-white", "border-b-2", "border-white");
        tab.classList.remove("text-outline");

        // Apply filter
        _activeTimelineFilter = tab.getAttribute("data-timeline-filter");
        if (_lastTimelineData.length) renderTimeline(_lastTimelineData);
      });
    });
  }

  /* ─── Init ─── */

  function init() {
    // Setup offline UI
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    // Clock ticker — respects the selected timezone preference
    var clockEl = $("utc-clock");
    if (clockEl) {
      function updateClock() {
        var tz = TraceClient.getTZ();
        var now = new Date();
        var offsetMin = tz === "IST" ? 330 : 0;
        var shifted = new Date(now.getTime() + offsetMin * 60000);
        clockEl.textContent = tz + " " + shifted.toISOString().slice(11, 19);
      }
      updateClock();
      setInterval(updateClock, 1000);
      // Re-render immediately when user switches timezone in Settings
      window.addEventListener("trace:tz-change", updateClock);
    }
    // Wire button handlers
    var refreshBtn = $("btn-refresh-all");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", function () {
        refreshBtn.textContent = "Loading...";
        pollSnapshot();
        pollTimeline();
        setTimeout(function () { refreshBtn.textContent = "Refresh"; }, 500);
      });
    }

    var cameraBtn = $("btn-enable-camera");
    if (cameraBtn) {
      cameraBtn.addEventListener("click", toggleCamera);
    }

    // Wire notification bell → navigate to alerts
    var notifBtn = $("nav-btn-notifications");
    if (notifBtn) {
      notifBtn.addEventListener("click", function () {
        // Scroll to alert stream on same page
        var alertSection = $("alert-stream-root");
        if (alertSection) alertSection.scrollIntoView({ behavior: "smooth" });
      });
    }

    // Wire settings gear → navigate to settings
    var settingsBtn = $("nav-btn-settings");
    if (settingsBtn) {
      settingsBtn.addEventListener("click", function () {
        window.location.href = "../settings/index.html";
      });
    }

    // Init timeline filter tabs
    initFilterTabs();

    // Initial probe
    TraceClient.probe().then(function (info) {
      if (!info) return;

      // Load initial data
      pollSnapshot();
      pollTimeline();

      // Also load health directly as a fallback
      TraceClient.health().then(function (h) {
        if (h) renderHealth(h);
      });

      // Start polling (just 2 loops — snapshot and timeline)
      _snapshotTimer = setInterval(pollSnapshot, SNAPSHOT_INTERVAL);
      _timelineTimer = setInterval(pollTimeline, TIMELINE_INTERVAL);

      // Connect SSE + wire forensic panel controls
      initForensicControls();
      initSSE();
    });

    // On state change, start/stop polling
    TraceClient.onStateChange(function (state) {
      if (state === "online" && !_snapshotTimer) {
        pollSnapshot();
        _snapshotTimer = setInterval(pollSnapshot, SNAPSHOT_INTERVAL);
        _timelineTimer = setInterval(pollTimeline, TIMELINE_INTERVAL);
      }
      if (state === "offline") {
        if (_snapshotTimer) { clearInterval(_snapshotTimer); _snapshotTimer = null; }
        if (_timelineTimer) { clearInterval(_timelineTimer); _timelineTimer = null; }
      }
    });
  }

  // Wait for DOM
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Cleanup on page unload
  window.addEventListener("beforeunload", function () {
    TraceClient.disconnectSSE();
    if (_snapshotTimer) clearInterval(_snapshotTimer);
    if (_timelineTimer) clearInterval(_timelineTimer);
  });
})();
