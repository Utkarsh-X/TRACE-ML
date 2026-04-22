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
      limit: 1000,
    };
  }

  function getSuppressedTimeline(rawEvents, windowMinutes) {
    if (windowMinutes === 0) return rawEvents;
    
    var windowMs = windowMinutes * 60000;
    var result = [];
    if (rawEvents.length === 0) return result;

    var currentGroup = null;

    rawEvents.forEach(function(ev) {
      var evTime = new Date(ev.timestamp_utc).getTime();
      
      if (!currentGroup) {
        currentGroup = {
          ev: ev,
          count: 1,
          startTime: evTime,
          endTime: evTime
        };
      } else {
        var diff = Math.abs(evTime - currentGroup.startTime);
        var sameKind = (ev.kind === currentGroup.ev.kind);
        var sameTitle = (ev.title === currentGroup.ev.title);

        if (diff <= windowMs && sameKind && sameTitle) {
          currentGroup.count++;
          currentGroup.endTime = evTime;
        } else {
          result.push(currentGroup);
          currentGroup = {
            ev: ev,
            count: 1,
            startTime: evTime,
            endTime: evTime
          };
        }
      }
    });

    if (currentGroup) result.push(currentGroup);
    return result;
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
    
    var win = parseInt(($("filter-suppress") || {}).value || "1", 10);
    // Sort chronological before grouping
    items.sort(function(a, b) {
      return new Date(a.timestamp_utc).getTime() - new Date(b.timestamp_utc).getTime();
    });
    
    var suppressed = getSuppressedTimeline(items, win);
    // Reverse to show newest first
    var sorted = suppressed.slice().reverse();
    
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
    
    var suppressSelect = $("filter-suppress");
    if (suppressSelect) {
      suppressSelect.addEventListener("change", loadTimeline);
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
