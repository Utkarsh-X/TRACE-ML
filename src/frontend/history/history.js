/**
 * History Page Controller
 *
 * - Loads global timeline on init
 * - Wire filter controls (entity_id, kind) + Query / Refresh buttons
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  function getFilters() {
    var entityId = ($("filter-entity") || {}).value || "";
    var kind = ($("filter-kind") || {}).value || "";
    return {
      entity_id: entityId || undefined,
      kinds: kind ? [kind] : undefined,
      limit: 200,
    };
  }

  function loadTimeline() {
    var filters = getFilters();
    TraceClient.globalTimeline(filters).then(function (items) {
      renderTimeline(items);
    });
  }

  function renderTimeline(items) {
    var root = $("timeline-results");
    if (!root) return;
    if (!items || items.length === 0) {
      root.innerHTML = TraceRender.emptyState("No timeline events");
      return;
    }
    var sorted = items.slice().reverse();
    root.innerHTML = sorted.map(function (item) {
      return TraceRender.timelineCard(item);
    }).join("");
  }

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    var goBtn = $("filter-go");
    if (goBtn) goBtn.addEventListener("click", loadTimeline);

    var refreshBtn = $("filter-refresh");
    if (refreshBtn) refreshBtn.addEventListener("click", loadTimeline);

    // Enter key in filter input
    var filterInput = $("filter-entity");
    if (filterInput) {
      filterInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") loadTimeline();
      });
    }

    TraceClient.probe().then(function (info) {
      if (info) loadTimeline();
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
