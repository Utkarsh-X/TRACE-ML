/**
 * Settings Page Controller
 *
 * - Loads system info (GET /) and health (GET /health)
 * - Fills version, health checks, config display
 * - Optionally connects SSE for engine log panel
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  function loadSystemInfo() {
    TraceClient.probe().then(function (info) {
      if (!info) return;

      // Update version info in settings nav
      var versionEls = document.querySelectorAll("section:first-of-type .font-mono");
      versionEls.forEach(function (el) {
        var text = el.textContent.trim();
        if (text === "3.0.0" || text.match(/^\d+\.\d+\.\d+$/)) {
          el.textContent = info.version || "—";
        }
        if (text === "demo") {
          el.textContent = info.environment || "—";
        }
      });
    });
  }

  function loadHealth() {
    TraceClient.health().then(function (health) {
      if (!health) return;

      // Update health check items
      var checksRoot = document.querySelector(".space-y-1");
      if (!checksRoot) return;

      // Build health checks from real data
      var checks = [];
      checks.push({
        name: "LanceDB Vector Store",
        detail: (health.total_detection_count || 0) + " detections stored",
        ok: health.status === "ok",
      });
      checks.push({
        name: "Active Entities",
        detail: (health.active_entity_count || 0) + " entities tracked",
        ok: health.active_entity_count >= 0,
      });
      checks.push({
        name: "Open Incidents",
        detail: (health.open_incident_count || 0) + " active",
        ok: true,
      });
      checks.push({
        name: "Event Stream",
        detail: (health.publisher_subscribers || 0) + " subscribers",
        ok: true,
      });
      checks.push({
        name: "Latest Event",
        detail: health.latest_event_at ? TraceClient.formatDateTime(health.latest_event_at) : "none",
        ok: !!health.latest_event_at,
      });

      checksRoot.innerHTML = checks.map(function (c) {
        return TraceRender.healthCheck(c.name, c.detail, c.ok);
      }).join("");
    });
  }

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    TraceClient.probe().then(function (info) {
      if (info) {
        loadSystemInfo();
        loadHealth();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.addEventListener("beforeunload", function () {
    TraceClient.disconnectSSE();
  });
})();
