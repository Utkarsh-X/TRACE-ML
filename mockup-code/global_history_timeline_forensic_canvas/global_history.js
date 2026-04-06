(function () {
  "use strict";
  var TA = window.TraceApi;
  if (!TA) {
    return;
  }
  var api = TA.buildApiBase();
  var root = document.getElementById("history-timeline-root");
  var statusEl = document.getElementById("history-status");
  var entityFilter = document.getElementById("history-filter-entity");
  var kindFilter = document.getElementById("history-filter-kind");
  var refreshBtn = document.getElementById("history-refresh");

  function buildQuery() {
    var params = new URLSearchParams();
    params.set("limit", "150");
    var eid = (entityFilter && entityFilter.value.trim()) || "";
    if (eid) {
      params.set("entity_id", eid);
    }
    var kind = (kindFilter && kindFilter.value) || "";
    if (kind) {
      params.append("kinds", kind);
    }
    return params.toString();
  }

  function render(items) {
    if (!root) {
      return;
    }
    if (!items || !items.length) {
      root.innerHTML = '<p class="text-sm text-outline font-mono">No timeline items</p>';
      return;
    }
    root.innerHTML = items
      .map(function (item) {
        var kind = typeof item.kind === "object" && item.kind ? item.kind.value || item.kind : item.kind;
        var loc = item.location || {};
        var lat = loc.lat != null ? loc.lat : "";
        var lon = loc.lon != null ? loc.lon : "";
        var mapHint =
          lat !== "" && lon !== "" ? "lat " + String(lat) + " lon " + String(lon) : "no coordinates";
        return (
          '<div class="bg-surface-container-low p-5 rounded-lg border border-outline-variant/20">' +
          '<div class="flex justify-between items-start gap-4 mb-2">' +
          '<span class="font-label text-xs text-primary uppercase">' +
          TA.escapeHtml(String(kind || "")) +
          "</span>" +
          '<span class="font-mono text-xs text-on-surface-variant">' +
          TA.escapeHtml(TA.formatDateTime(item.timestamp_utc)) +
          "</span>" +
          "</div>" +
          '<h3 class="font-bold text-white mb-1">' +
          TA.escapeHtml(item.title || "") +
          "</h3>" +
          '<p class="text-on-surface-variant text-sm mb-3">' +
          TA.escapeHtml(item.summary || "") +
          "</p>" +
          '<div class="flex flex-wrap gap-3 text-[0.65rem] font-mono text-outline">' +
          "<span>entity: " +
          TA.escapeHtml(item.entity_id || "—") +
          "</span>" +
          "<span>incident: " +
          TA.escapeHtml(item.incident_id || "—") +
          "</span>" +
          "<span>" +
          TA.escapeHtml(mapHint) +
          "</span>" +
          "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function load() {
    if (statusEl) {
      statusEl.textContent = "loading…";
    }
    var url = api + "/api/v1/timeline?" + buildQuery();
    TA.fetchJson(url)
      .then(function (rows) {
        render(Array.isArray(rows) ? rows : []);
        if (statusEl) {
          statusEl.textContent = "ok (" + (Array.isArray(rows) ? rows.length : 0) + ")";
        }
      })
      .catch(function (e) {
        if (statusEl) {
          statusEl.textContent = e.message || "error";
        }
        if (root) {
          root.innerHTML = '<p class="text-sm text-error font-mono">Failed to load timeline</p>';
        }
      });
  }

  var debounceTimer = null;
  function debouncedLoad() {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(function () {
      debounceTimer = null;
      load();
    }, 600);
  }

  function connectStream() {
    if (!("EventSource" in window)) {
      return;
    }
    var streamUrl = api + "/api/v1/events/stream?backfill=5";
    var es = new EventSource(streamUrl);
    ["event", "alert", "incident", "action", "detection"].forEach(function (topic) {
      es.addEventListener(topic, function () {
        debouncedLoad();
      });
    });
  }

  function init() {
    var params = new URLSearchParams(window.location.search);
    if (entityFilter && params.get("entity_id")) {
      entityFilter.value = params.get("entity_id");
    }
    if (kindFilter && params.get("kind")) {
      kindFilter.value = params.get("kind");
    }
    load();
    connectStream();
    if (refreshBtn) {
      refreshBtn.addEventListener("click", load);
    }
  }

  init();
})();
