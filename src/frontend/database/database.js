/**
 * Database Page Controller
 *
 * - Lists entities → renders table
 * - Loads health stats → sidebar
 * - Client-side search filter on entity table
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var _allEntities = [];

  function loadEntities() {
    var limitEl = $("db-limit");
    var val = limitEl ? limitEl.value : "";
    var limit = parseInt(val, 10);
    
    // Default fallback if not a valid number or <= 0
    if (isNaN(limit) || limit <= 0) {
      limit = 100;
      if (limitEl) limitEl.value = "100";
    }

    TraceClient.entities({ limit: limit }).then(function (list) {
      if (!list) return;
      _allEntities = list;
      renderTable(list);
      updateFooter(list.length, list.length);
      // Update entity count stat immediately — avoids race with loadHealth()
      var el = document.getElementById('db-stat-entities');
      if (el) el.textContent = String(list.length);
    });
  }

  function renderTable(entities) {
    var tbody = document.querySelector(".db-table tbody");
    if (!tbody) return;
    if (entities.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-center py-8">'
        + TraceRender.emptyState("No entities in database")
        + '</td></tr>';
      return;
    }
    tbody.innerHTML = entities.map(function (ent, idx) {
      return TraceRender.tableRow(ent, idx);
    }).join("");
  }

  function updateFooter(showing, total) {
    var el = document.querySelector(".db-table + div span, [class*='font-mono'][class*='text-outline']");
    // Find the "Showing X of Y" text
    var footerSpans = document.querySelectorAll("main span.font-mono");
    footerSpans.forEach(function (span) {
      if (span.textContent.indexOf("Showing") >= 0) {
        span.textContent = "Showing " + showing + " of " + total + " entities";
      }
    });
  }

  function loadHealth() {
    TraceClient.health().then(function (health) {
      if (!health) return;
      var el;
      // Entity count: use active_entity_count from health (reliable, no race condition)
      // loadEntities() will also update this independently with the exact loaded count
      el = document.getElementById('db-stat-entities');
      if (el && el.textContent === '—') el.textContent = String(health.active_entity_count || 0);

      el = document.getElementById('db-stat-detections');
      if (el) el.textContent = String(health.total_detection_count || 0);

      el = document.getElementById('db-stat-alerts');
      if (el) el.textContent = String(health.recent_alert_count || 0);

      el = document.getElementById('db-stat-incidents');
      if (el) el.textContent = String(health.open_incident_count || 0);
    });
  }

  function wireSearch() {
    var searchInput = $("db-search");
    if (!searchInput) return;
    searchInput.addEventListener("input", function () {
      var query = searchInput.value.trim().toLowerCase();
      if (!query) {
        renderTable(_allEntities);
        updateFooter(_allEntities.length, _allEntities.length);
        return;
      }
      var filtered = _allEntities.filter(function (ent) {
        return (ent.entity_id || "").toLowerCase().indexOf(query) >= 0
          || (ent.name || "").toLowerCase().indexOf(query) >= 0
          || (ent.category || "").toLowerCase().indexOf(query) >= 0
          || (ent.status || "").toLowerCase().indexOf(query) >= 0;
      });
      renderTable(filtered);
      updateFooter(filtered.length, _allEntities.length);
    });
  }

  function wireRefresh() {
    var btn = $("db-refresh");
    if (btn) {
      btn.addEventListener("click", function () {
        loadEntities();
        loadHealth();
      });
    }
    
    var executeBtn = $("db-execute");
    if (executeBtn) {
      executeBtn.addEventListener("click", function () {
        loadEntities();
        loadHealth();
        appendLog("SYSTEM", "Manual index synchronization triggered.");
      });
    }

    var limitEl = $("db-limit");
    if (limitEl) {
      limitEl.addEventListener("change", function () {
        loadEntities();
        appendLog("SYSTEM", "Query limit updated to " + limitEl.value + " records.");
      });
    }
  }

  function appendLog(type, message) {
    var consoleEl = $("query-console");
    if (!consoleEl) return;
    
    var lineHtml = TraceRender.terminalLine(type, { message: message });
    consoleEl.insertAdjacentHTML('afterbegin', lineHtml);
    
    // Cap at 100 lines
    var lines = consoleEl.querySelectorAll('.log-line');
    if (lines.length > 100) lines[lines.length - 1].remove();
  }

  function initConsole() {
    var consoleEl = $("query-console");
    // Clear mock logs on load
    if (consoleEl) consoleEl.innerHTML = "";

    var clearBtn = document.querySelector("[class*='cursor-pointer'][class*='hover:text-white']");
    if (clearBtn && clearBtn.textContent.indexOf("CLEAR") >= 0) {
      clearBtn.addEventListener("click", function() {
        if (consoleEl) consoleEl.innerHTML = "";
      });
    }

    TraceClient.connectSSE(function (event) {
      if (!TraceClient.isMeaningfulEvent(event)) return;
      var lineHtml = TraceRender.terminalLine(event.topic, event.payload, event.timestamp_utc);
      if (consoleEl) {
        consoleEl.insertAdjacentHTML('afterbegin', lineHtml);
        // Cap at 100 lines
        var lines = consoleEl.querySelectorAll('.log-line');
        if (lines.length > 100) lines[lines.length - 1].remove();
      }
    });
  }

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);
    wireSearch();
    wireRefresh();
    initConsole();

    TraceClient.probe().then(function (info) {
      if (info) {
        loadEntities();
        loadHealth();
        appendLog("SYSTEM", "Connection established to " + info.name + " v" + info.version);
      }
    });

    // Nav buttons
    var settingsBtn = $("nav-btn-settings");
    if (settingsBtn) settingsBtn.addEventListener("click", function () { window.location.href = "../settings/index.html"; });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
