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
    TraceClient.entities({ limit: 500 }).then(function (list) {
      if (!list) return;
      _allEntities = list;
      renderTable(list);
      updateFooter(list.length, list.length);
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
      // Update health stats in sidebar by finding the stat elements
      var statsEls = document.querySelectorAll("#db-health-root .font-headline, section:last-child .font-headline");

      // We'll update via known structure
      var mapping = {
        "active_entity_count": null,
        "total_detection_count": null,
        "recent_alert_count": null,
        "open_incident_count": null,
      };

      // Also update with entity count from loaded data
      var entityCountEl = document.querySelector("section:last-child");
      if (!entityCountEl) return;

      var statItems = entityCountEl.querySelectorAll(".bg-surface-container");
      statItems.forEach(function (item) {
        var label = item.querySelector(".stat-label");
        var value = item.querySelector(".font-headline, .font-mono");
        if (!label || !value) return;
        var labelText = label.textContent.trim().toLowerCase();

        if (labelText === "entities") value.textContent = String(_allEntities.length);
        else if (labelText === "detections") value.textContent = String(health.total_detection_count || 0);
        else if (labelText === "alerts") value.textContent = String(health.recent_alert_count || 0);
        else if (labelText === "incidents") value.textContent = String(health.open_incident_count || 0);
      });
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
    if (!btn) return;
    btn.addEventListener("click", function () {
      loadEntities();
      loadHealth();
    });
  }

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);
    wireSearch();
    wireRefresh();

    TraceClient.probe().then(function (info) {
      if (info) {
        loadEntities();
        loadHealth();
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
