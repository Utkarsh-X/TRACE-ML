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
  var OVERLAY_INTERVAL = 2000;
  var _snapshotTimer = null;
  var _overlayTimer = null;
  var _cameraActive = false;

  /* ─── DOM references ─── */
  function $(id) { return document.getElementById(id); }

  /* ─── Snapshot rendering ─── */

  function renderSnapshot(snap) {
    if (!snap) return;

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

    // Recent timeline (center panel) — fetched separately
    // renderTimeline handled via pollTimeline()

    // System health
    renderHealth(snap.system_health);
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

  /* ─── SSE terminal ─── */

  function initSSE() {
    var terminal = $("terminal-output");
    if (!terminal) return;

    TraceClient.connectSSE(function (event) {
      // Append to terminal
      var line = TraceRender.terminalLine(event.topic, event.payload, event.timestamp_utc);
      terminal.innerHTML = line + terminal.innerHTML;

      // Cap terminal lines (~200 lines max)
      var children = terminal.querySelectorAll("span.log-time");
      if (children.length > 200) {
        // Remove excess from end
        while (terminal.childNodes.length > 600) {
          terminal.removeChild(terminal.lastChild);
        }
      }
    });
  }

  /* ─── Polling loops ─── */

  function pollSnapshot() {
    var t0 = performance.now();
    TraceClient.liveSnapshot({ entity_limit: 12, incident_limit: 6, alert_limit: 10 })
      .then(function (snap) {
        // Measure and display latency
        var latency = Math.round(performance.now() - t0);
        var latEl = $("health-latency");
        if (latEl) latEl.textContent = latency + " ms";

        _lastSnapshotData = snap;
        renderSnapshot(snap);
      });
  }

  function pollOverlay() {
    TraceClient.liveOverlay().then(function (data) {
      // Update FPS if available
      if (data && data.fps) {
        var el = $("health-fps");
        if (el) el.textContent = data.fps.toFixed(1);
      }
    });
  }

  function pollTimeline() {
    TraceClient.globalTimeline({ limit: 20 }).then(function (items) {
      if (!items) return;
      _lastTimelineData = items;
      renderTimeline(items);
    });
  }

  /* ─── Camera feed ─── */

  function toggleCamera() {
    var feedImg = $("camera-feed");
    var placeholder = $("camera-placeholder");
    var btn = $("btn-enable-camera");

    if (_cameraActive) {
      // Stop: clear the MJPEG src
      if (feedImg) {
        feedImg.src = "";
        feedImg.style.display = "none";
      }
      if (placeholder) placeholder.style.display = "";
      if (btn) btn.textContent = "Enable Camera";
      _cameraActive = false;
    } else {
      // Start: set MJPEG src
      var mjpegUrl = TraceClient.baseUrl + "/api/v1/live/mjpeg";
      if (feedImg) {
        feedImg.src = mjpegUrl;
        feedImg.style.display = "block";
      }
      if (placeholder) placeholder.style.display = "none";
      if (btn) btn.textContent = "Disable Camera";
      _cameraActive = true;
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

    // UTC clock ticker
    var clockEl = $("utc-clock");
    if (clockEl) {
      function updateClock() {
        var now = new Date();
        clockEl.textContent = "UTC " + now.toISOString().slice(11, 19);
      }
      updateClock();
      setInterval(updateClock, 1000);
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

      // Start polling
      _snapshotTimer = setInterval(pollSnapshot, SNAPSHOT_INTERVAL);
      _overlayTimer = setInterval(pollOverlay, OVERLAY_INTERVAL);
      setInterval(pollTimeline, 5000); // Timeline every 5s

      // Connect SSE
      initSSE();
    });

    // On state change, start/stop polling
    TraceClient.onStateChange(function (state) {
      if (state === "online" && !_snapshotTimer) {
        pollSnapshot();
        _snapshotTimer = setInterval(pollSnapshot, SNAPSHOT_INTERVAL);
        _overlayTimer = setInterval(pollOverlay, OVERLAY_INTERVAL);
      }
      if (state === "offline") {
        if (_snapshotTimer) { clearInterval(_snapshotTimer); _snapshotTimer = null; }
        if (_overlayTimer) { clearInterval(_overlayTimer); _overlayTimer = null; }
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
    if (_overlayTimer) clearInterval(_overlayTimer);
  });
})();
