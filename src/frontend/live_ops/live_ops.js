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
        // Wire click-to-profile navigation
        entRoot.querySelectorAll("[data-entity-id]").forEach(function (card) {
          card.addEventListener("click", function () {
            var eid = card.getAttribute("data-entity-id");
            if (eid) window.location.href = "../entities/index.html?id=" + encodeURIComponent(eid);
          });
        });
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
        alertBadge.className = highCount > 0 ? "badge badge--neutral" : "badge badge--ghost";
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

  function appendLogLine(event) {
    var body = document.getElementById('fl-body');
    if (!body) return;

    var lineHtml = TraceRender.terminalLine(
      event.topic || 'SESSION',
      event.payload || {},
      event.timestamp_utc
    );

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
      // Only display meaningful events; skip metrics (session.state)
      if (TraceClient.isMeaningfulEvent(event)) {
        appendLogLine(event);
      }
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

  // ── 6-type entity style definitions ──────────────────────────────────────
  // Each entry defines the bracket/label colour, short label prefix,
  // whether to render confidence, and the intel-panel status line.
  var ENTITY_TYPE_STYLES = {
    criminal:    { color: "#ff3366", badge: "CRIM",  showConf: true,
                   status: "IDENTIFIED \u00b7 CRIMINAL / POI" },
    missing:     { color: "#ffaa33", badge: "MISS",  showConf: true,
                   status: "IDENTIFIED \u00b7 MISSING PERSON" },
    employee:    { color: "#33ff77", badge: "STAFF", showConf: true,
                   status: "IDENTIFIED \u00b7 AUTH STAFF" },
    vip:         { color: "#ffdd44", badge: "VIP",   showConf: true,
                   status: "IDENTIFIED \u00b7 PROTECTED ENTITY" },
    unknown:     { color: "#bb44ff", badge: "UNK",   showConf: false,
                   status: "NEW UNKNOWN \u00b7 No DB match" },
    reappearing: { color: "#ff44cc", badge: "RPT",   showConf: true,
                   status: "REAPPEARING \u00b7 Seen before" },
  };

  // Pulse map: track_id -> timestamp when committed (for 800ms on-appear glow).
  var _pulseMap = {};

  // ── Resolve which of the 6 types a box belongs to ────────────────────────
  function _resolveEntityType(box) {
    if (box.is_unknown) {
      return box.is_repeated ? "reappearing" : "unknown";
    }
    var cat = (box.person_category || "unknown").toLowerCase();
    return ENTITY_TYPE_STYLES[cat] ? cat : "employee"; // fallback: treat as staff
  }

  // ── Truncate name to first-name, max N chars ──────────────────────────────
  function _shortName(label, maxLen) {
    if (!label || label === "Unknown") return null;
    var first = label.split(" ")[0].toLowerCase();
    return first.length > maxLen ? first.slice(0, maxLen) : first;
  }

  // ── Camera Control State Management ────────────────────────────────
  var _cameraUISync = false;  // Track if UI matches backend state
  var _recognitionUISync = false;  // Track if recognition UI matches backend state

  function _updateCameraUI(enabled) {
    // Update UI to reflect actual camera state from backend
    var feedImg = $("camera-feed");
    var placeholder = $("camera-placeholder");
    var btn = $("btn-enable-camera");
    var recognitionBtn = $("btn-enable-recognition");
    var canvas = $("detection-overlay");

    if (enabled) {
      // Camera is enabled on backend
      if (feedImg) {
        feedImg.src = TraceClient.baseUrl + "/api/v1/live/mjpeg?quality=90&fps=15";
        feedImg.style.display = "block";
        feedImg.style.imageRendering = "auto";  // let browser use best upscale algo
      }
      if (placeholder) placeholder.style.display = "none";
      if (btn) btn.textContent = "Disable Camera";
      if (canvas) canvas.classList.remove("hidden");
      _cameraActive = true;
      
      // Enable recognition button when camera is on
      if (recognitionBtn) {
        recognitionBtn.disabled = false;
      }
    } else {
      // Camera is disabled on backend
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
      
      // Disable recognition button when camera is off
      if (recognitionBtn) {
        recognitionBtn.disabled = true;
        recognitionBtn.textContent = "⚙ START RECOGNITION";
      }
    }
    _cameraUISync = true;
  }

  function _updateRecognitionUI(enabled) {
    // Update UI to reflect actual recognition state from backend
    var btn = $("btn-enable-recognition");
    if (!btn) return;
    
    if (enabled) {
      btn.textContent = "⏹ STOP RECOGNITION";
      startOverlayPoll();
    } else {
      btn.textContent = "⚙ START RECOGNITION";
      stopOverlayPoll();
    }
    _recognitionUISync = true;
  }

  function toggleCamera() {
    // Called when user clicks Enable/Disable Camera button
    var btn = $("btn-enable-camera");
    if (btn) btn.disabled = true;

    if (_cameraActive) {
      // Request disable from backend
      TraceClient.cameraDisable().then(function (response) {
        if (response && response.status) {
          if (response.status === "disabled" || response.status === "already_disabled") {
            _updateCameraUI(false);
            if (btn) btn.disabled = false;
          } else {
            console.warn("[Camera] Disable failed:", response.message);
            if (btn) btn.disabled = false;
          }
        } else {
          console.warn("[Camera] No response from disable endpoint");
          if (btn) btn.disabled = false;
        }
      });
    } else {
      // Request enable from backend
      TraceClient.cameraEnable().then(function (response) {
        if (response && response.status) {
          if (response.status === "enabled" || response.status === "already_enabled") {
            _updateCameraUI(true);
            if (btn) btn.disabled = false;
          } else {
            console.warn("[Camera] Enable failed:", response.message);
            if (btn) btn.disabled = false;
          }
        } else {
          console.warn("[Camera] No response from enable endpoint");
          if (btn) btn.disabled = false;
        }
      });
    }
  }

  function toggleRecognition() {
    // Called when user clicks Start/Stop Recognition button
    var btn = $("btn-enable-recognition");
    if (btn) btn.disabled = true;

    // Check current state from backend
    TraceClient.recognitionStatus().then(function (status) {
      if (!status) {
        console.warn("[Recognition] No status response");
        if (btn) btn.disabled = false;
        return;
      }

      var isEnabled = status.enabled;
      
      if (isEnabled) {
        // Request disable
        TraceClient.recognitionDisable().then(function (response) {
          if (response && (response.status === "disabled" || response.status === "already_disabled")) {
            _updateRecognitionUI(false);
            if (btn) btn.disabled = false;
          } else {
            console.warn("[Recognition] Disable failed:", response ? response.message : "no response");
            if (btn) btn.disabled = false;
          }
        });
      } else {
        // Request enable
        TraceClient.recognitionEnable().then(function (response) {
          if (response && (response.status === "enabled" || response.status === "already_enabled")) {
            _updateRecognitionUI(true);
            if (btn) btn.disabled = false;
          } else {
            console.warn("[Recognition] Enable failed:", response ? response.message : "no response");
            if (btn) btn.disabled = false;
          }
        });
      }
    });
  }

  function checkCameraStatus() {
    // Poll backend for actual camera status on page load.
    // Returns the promise so callers can chain (e.g. auto-resume logic).
    return TraceClient.cameraStatus().then(function (status) {
      if (status) {
        _updateCameraUI(status.enabled);
      }
    });
  }

  function checkRecognitionStatus() {
    // Poll backend for actual recognition status on page load
    TraceClient.recognitionStatus().then(function (status) {
      if (status) {
        _updateRecognitionUI(status.enabled);
      }
    });
  }
  // ──────────────────────────────────────────────────────────────────

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
    // Clear the intel panel when recognition stops
    _intelHistory = [];
    updateEntityIntelPanel([]);
  }

  // ── Graceful shutdown: stop recognition (if active) then stop camera ──────────
  // Chains async API calls in the correct order and invokes callback when done.
  // A 3-second safety timeout ensures navigation always proceeds even if APIs fail.
  function gracefulShutdown(callback) {
    var _done = false;
    var _timer = setTimeout(function () {
      if (!_done) { _done = true; callback(); }
    }, 3000);

    function _finish() {
      if (!_done) {
        _done = true;
        clearTimeout(_timer);
        callback();
      }
    }

    function _stopCamera(cb) {
      if (!_cameraActive) { cb(); return; }
      TraceClient.cameraDisable().then(function (r) {
        if (r) _updateCameraUI(false);
        cb();
      }).catch(cb);
    }

    // Check recognition first; if enabled, disable it before stopping camera
    TraceClient.recognitionStatus().then(function (status) {
      if (status && status.enabled) {
        TraceClient.recognitionDisable().then(function () {
          _updateRecognitionUI(false);
          _stopCamera(_finish);
        }).catch(function () {
          _stopCamera(_finish);
        });
      } else {
        _stopCamera(_finish);
      }
    }).catch(function () {
      // Cannot reach backend — just stop camera UI and proceed
      _stopCamera(_finish);
    });
  }

  // ── Nav-link interception: graceful shutdown before leaving live ops ──────────
  // Intercepts clicks on all sidebar/nav links that lead away from this page.
  // Saves camera state to localStorage (for auto-resume on return), triggers
  // gracefulShutdown(), then navigates once shutdown completes.
  function initNavInterception() {
    var links = document.querySelectorAll('a[href]');
    links.forEach(function (link) {
      var href = (link.getAttribute('href') || '').trim();
      // Only intercept relative links to other pages — skip anchors, external URLs,
      // and links that stay on the live_ops page itself.
      if (!href || href === '#' || href.startsWith('http') || href.indexOf('live_ops') !== -1) return;

      link.addEventListener('click', function (e) {
        if (!_cameraActive) return; // nothing to shut down, let the browser navigate

        e.preventDefault();
        var target = href; // capture before async completes
        gracefulShutdown(function () {
          window.location.href = target;
        });
      });
    });
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

    // Update FPS display
    var fpsEl = $("camera-fps");
    if (fpsEl && data && data.fps) fpsEl.textContent = data.fps.toFixed(1) + " FPS";

    if (!data || !data.active || !data.boxes || data.boxes.length === 0) {
      updateEntityIntelPanel([]);
      return;
    }

    var cw  = canvas.width;
    var ch  = canvas.height;
    var now = Date.now();

    for (var i = 0; i < data.boxes.length; i++) {
      var box  = data.boxes[i];
      var x    = box.x * cw;
      var y    = box.y * ch;
      var w    = box.w * cw;
      var h    = box.h * ch;
      var tid  = box.track_id || ("t" + i);

      // Determine entity type + colour
      var typeName = _resolveEntityType(box);
      var style    = ENTITY_TYPE_STYLES[typeName];
      var color    = style.color;

      // Pulse animation: 800ms on first appearance
      if (!(tid in _pulseMap)) _pulseMap[tid] = now;
      var age       = now - _pulseMap[tid];
      var glowAlpha = 1.0;
      if (age < 800) {
        glowAlpha = 1.0 + Math.sin((1 - age / 800) * Math.PI) * 0.8;
      }

      // Draw L-shaped corner brackets (no full rectangle)
      _drawCornerBrackets(ctx, x, y, w, h, color, glowAlpha);

      // Build label text
      // smoothed_confidence is already 0-100, do NOT multiply by 100.
      var conf = Math.round(box.confidence);
      var name = _shortName(box.label, 9);
      var labelText;
      if (typeName === "unknown") {
        labelText = style.badge + " " + (box.entity_id || "--");
      } else if (typeName === "reappearing") {
        labelText = style.badge + " " + (box.entity_id || "--") + " " + conf + "%";
      } else if (style.showConf) {
        labelText = style.badge + (name ? " " + name : "") + " " + conf + "%";
      } else {
        labelText = style.badge + (name ? " " + name : "");
      }
      _drawLabel(ctx, x, y, labelText, color);
    }

    // Clean up pulse map entries older than 5s
    var cutoff = now - 5000;
    Object.keys(_pulseMap).forEach(function (k) {
      if (_pulseMap[k] < cutoff) delete _pulseMap[k];
    });

    updateEntityIntelPanel(data.boxes);
  }

  // ── Entity Intelligence panel renderer ── HISTORY BUFFER mode ───────────────────
  //
  // Design: cards accumulate (up to MAX_HISTORY) so operators see every entity
  // that appeared, not just the one currently on screen. Live entities are shown
  // at the top with a pulsing dot; past ones fade to 50% with "Last seen Xs ago".
  // History is keyed by entity_id so one person = one card even across tracks.
  // History is cleared when recognition stops.

  var MAX_HISTORY = 15;
  // Array of {entity_id, box, typeName, firstSeen, lastSeen, isLive}
  var _intelHistory = [];

  function updateEntityIntelPanel(boxes) {
    var root    = $("entity-intel-root");
    var emptyEl = $("intel-empty-state");
    var countEl = $("intel-live-count");
    if (!root) return;

    var now = Date.now();

    // Build current live entity_id set
    var liveMap = {};
    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i];
      if (!box.entity_id) continue; // skip uncommitted warmup tracks
      liveMap[box.entity_id] = box;
    }

    // Update existing history entries
    _intelHistory.forEach(function (entry) {
      if (liveMap[entry.entity_id]) {
        entry.box      = liveMap[entry.entity_id]; // latest data
        entry.lastSeen = now;
        entry.isLive   = true;
        entry.typeName = _resolveEntityType(entry.box);
      } else {
        entry.isLive = false;
      }
    });

    // Add brand-new entities to the FRONT of history
    Object.keys(liveMap).forEach(function (eid) {
      var already = _intelHistory.some(function (e) { return e.entity_id === eid; });
      if (!already) {
        _intelHistory.unshift({
          entity_id: eid,
          box:       liveMap[eid],
          typeName:  _resolveEntityType(liveMap[eid]),
          firstSeen: now,
          lastSeen:  now,
          isLive:    true,
        });
      }
    });

    // Cap history size — drop oldest non-live entries first
    if (_intelHistory.length > MAX_HISTORY) {
      // Remove trailing non-live entries
      _intelHistory = _intelHistory.filter(function (e, idx) {
        return e.isLive || idx < MAX_HISTORY;
      }).slice(0, MAX_HISTORY);
    }

    // ── Render all history entries ───────────────────────────────────────────────
    // Clear existing cards (but keep the empty-state element)
    var oldCards = root.querySelectorAll(".intel-card");
    oldCards.forEach(function (c) { c.parentNode.removeChild(c); });

    _intelHistory.forEach(function (entry) {
      var box      = entry.box;
      var style    = ENTITY_TYPE_STYLES[entry.typeName];
      var color    = style.color;
      // smoothed_confidence is already 0-100
      var conf     = Math.round(box.confidence);
      var isLive   = entry.isLive;

      // Display name
      var displayName = (box.is_unknown || !box.label || box.label === "Unknown")
        ? "Unknown Entity"
        : box.label;

      // Time line
      var elapsedSec  = Math.floor((now - entry.firstSeen) / 1000);
      var lastSecAgo  = Math.floor((now - entry.lastSeen)  / 1000);
      var timeStr = isLive
        ? (elapsedSec + "s on screen")
        : ("Last seen " + lastSecAgo + "s ago");

      // Confidence string
      var confStr = (entry.typeName === "reappearing" || style.showConf) ? (conf + "%") : "";

      var card = document.createElement("div");
      // Live = full opacity + left glow border, past = dimmed
      card.className = "intel-card" + (isLive ? " intel-card--live" : " intel-card--past");
      card.style.borderLeftColor = color;
      if (!isLive) card.style.opacity = "0.45";

      // Live status dot (small pulsing element)
      var liveDot = isLive
        ? "<span style='width:5px;height:5px;border-radius:50%;background:" + color +
          ";display:inline-block;margin-right:4px;flex-shrink:0;animation:intel-dot-pulse 1.2s infinite;'></span>"
        : "";

      card.innerHTML =
        "<div class='intel-card__row1'>" +
          liveDot +
          "<span class='intel-type-badge' style='color:" + color +
            ";border-left:2px solid " + color + ";'>" +
            TraceClient.escapeHtml(style.badge) +
          "</span>" +
          "<span class='intel-entity-id'>" + TraceClient.escapeHtml(box.entity_id || "") + "</span>" +
          "<span class='intel-name'>" + TraceClient.escapeHtml(displayName) + "</span>" +
          (confStr
            ? "<span class='intel-conf' style='color:" + color + ";'>" + confStr + "</span>"
            : "") +
        "</div>" +
        "<div class='intel-card__row2'>" +
          "<span class='intel-status'>" + TraceClient.escapeHtml(style.status) + "</span>" +
          " &nbsp;\u00b7&nbsp; " + TraceClient.escapeHtml(timeStr) +
        "</div>";

      // Insert before empty-state placeholder
      root.insertBefore(card, emptyEl);
    });

    // Empty state + live count
    var liveCount   = _intelHistory.filter(function (e) { return e.isLive; }).length;
    var totalSeen   = _intelHistory.length;
    if (emptyEl)  emptyEl.style.display = totalSeen > 0 ? "none" : "block";
    if (countEl) {
      if (liveCount > 0) {
        countEl.textContent = liveCount + " LIVE";
        countEl.style.color = "rgba(255,255,255,0.7)";
      } else if (totalSeen > 0) {
        countEl.textContent = totalSeen + " SEEN";
        countEl.style.color = "rgba(255,255,255,0.25)";
      } else {
        countEl.textContent = "\u2014";
        countEl.style.color = "rgba(255,255,255,0.2)";
      }
    }
  }

  /* ─── Helper: drawing functions (used by drawOverlay) ─── */

  function _drawCornerBrackets(ctx, x, y, w, h, color, glowAlpha) {
    var arm = Math.max(10, Math.min(w, h) * 0.18);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.lineCap     = "square";
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8 * glowAlpha;
    ctx.beginPath();
    ctx.moveTo(x, y + arm);         ctx.lineTo(x, y);         ctx.lineTo(x + arm, y);
    ctx.moveTo(x + w - arm, y);     ctx.lineTo(x + w, y);     ctx.lineTo(x + w, y + arm);
    ctx.moveTo(x, y + h - arm);     ctx.lineTo(x, y + h);     ctx.lineTo(x + arm, y + h);
    ctx.moveTo(x + w - arm, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - arm);
    ctx.stroke();
    ctx.restore();
  }

  function _drawLabel(ctx, x, y, text, color) {
    ctx.save();
    ctx.font = "bold 11px 'JetBrains Mono', monospace";
    var tw   = ctx.measureText(text).width;
    var ph   = 18;    // pill height
    var ppad = 5;     // horizontal text padding
    var strip = 2;    // left colour strip width
    var lx   = x;
    var ly   = Math.max(0, y - ph - 3);
    // Dark pill
    ctx.fillStyle = "rgba(0,0,0,0.84)";
    ctx.fillRect(lx, ly, tw + ppad * 2 + strip, ph);
    // Left colour strip
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, strip, ph);
    // Text with very subtle glow
    ctx.fillStyle   = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 3;
    ctx.fillText(text, lx + strip + ppad, ly + ph - 4);
    ctx.restore();
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

    var recognitionBtn = $("btn-enable-recognition");
    if (recognitionBtn) {
      recognitionBtn.addEventListener("click", toggleRecognition);
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

    // Init nav-link interception for graceful camera/recognition shutdown
    initNavInterception();

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

      // CAMERA STATE SYNC: Check backend camera status and sync UI accordingly.
      // Camera is NOT auto-started; user must click "Enable Camera" button.
      // This ensures the backend and frontend are in sync on page load.
      checkCameraStatus();
      checkRecognitionStatus();

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

  // Sync camera/recognition UI immediately (fallback if probe is delayed)
  // This ensures UI matches backend state even if probe().then() takes time
  setTimeout(function () {
    checkCameraStatus();
    checkRecognitionStatus();
    
    // Also wire button event listeners as fallback
    var cameraBtn = $("btn-enable-camera");
    if (cameraBtn && !cameraBtn._hasListener) {
      cameraBtn.addEventListener("click", toggleCamera);
      cameraBtn._hasListener = true;
    }
    
    var recognitionBtn = $("btn-enable-recognition");
    if (recognitionBtn && !recognitionBtn._hasListener) {
      recognitionBtn.addEventListener("click", toggleRecognition);
      recognitionBtn._hasListener = true;
    }
  }, 100);

  // Cleanup on page unload
  // Also covers browser back/forward/close-tab cases that nav-link interception
  // cannot catch. We use navigator.sendBeacon for fire-and-forget API calls since
  // async fetch() won't complete during beforeunload.
  window.addEventListener('beforeunload', function () {
    // Best-effort: fire-and-forget shutdown via sendBeacon
    // (nav-link interception already handles graceful shutdown for sidebar navigation;
    //  this is a safety net for back button / tab close / keyboard shortcuts)
    if (_cameraActive && navigator.sendBeacon) {
      try {
        var base = TraceClient.baseUrl;
        navigator.sendBeacon(base + '/api/v1/recognition/disable');
        navigator.sendBeacon(base + '/api/v1/camera/disable');
      } catch (e) { /* sendBeacon unavailable */ }
    }

    TraceClient.disconnectSSE();
    if (_snapshotTimer) clearInterval(_snapshotTimer);
    if (_timelineTimer) clearInterval(_timelineTimer);
  });
})();
