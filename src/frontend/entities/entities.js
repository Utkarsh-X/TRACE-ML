/**
 * Entities Page Controller
 *
 * - Lists entities → populates <select>
 * - On select → loads entity profile → fills header, stats, timeline, linked incidents
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  /* ─── Entity list ─── */

  function loadEntityList() {
    TraceClient.entities({ limit: 200 }).then(function (list) {
      if (!list) return;
      var sel = $("entity-select");
      if (!sel) return;

      sel.innerHTML = "";
      if (list.length === 0) {
        sel.innerHTML = '<option value="">No entities</option>';
        $("entity-load-status").textContent = "0 entities";
        return;
      }

      list.forEach(function (ent) {
        var opt = document.createElement("option");
        opt.value = ent.entity_id;
        opt.textContent = ent.entity_id + " — " + (ent.name || "Unknown") + " (" + ent.category + ")";
        sel.appendChild(opt);
      });

      $("entity-load-status").textContent = list.length + " entities loaded";
      loadEntityProfile(list[0].entity_id);
    });
  }

  /* ─── Entity profile ─── */

  function loadEntityProfile(entityId) {
    if (!entityId) return;

    TraceClient.entityProfile(entityId).then(function (profile) {
      if (!profile) return;
      renderHeader(profile.entity);
      renderStats(profile);
      renderTimeline(profile.timeline);
      renderIncidents(profile.incidents);
    });
  }

  function renderHeader(entity) {
    var el;
    el = $("entity-profile-label");
    if (el) el.textContent = "Entity Profile // ID: " + entity.entity_id;

    el = $("entity-display-name");
    if (el) el.textContent = entity.name || entity.entity_id;

    el = $("entity-status");
    if (el) el.textContent = String(entity.status || "active").toUpperCase();

    el = $("entity-type");
    if (el) el.textContent = String(entity.category || "unknown").toUpperCase();

    el = $("entity-severity");
    if (el) {
      var sev = entity.open_incident_count > 0 ? "HIGH" : "LOW";
      el.textContent = sev;
      el.className = sev === "HIGH"
        ? "text-[0.875rem] text-error uppercase font-medium"
        : "text-[0.875rem] text-on-surface-variant uppercase font-medium";
    }

    el = $("entity-clock");
    if (el) el.textContent = TraceClient.formatDateTime(entity.last_seen_at) || "\u2014";
  }

  function renderStats(profile) {
    var el;
    var stats = profile.stats || {};

    el = $("stat-appearances");
    if (el) el.textContent = String(stats.detection_count || 0);

    el = $("stat-incident-count");
    if (el) el.textContent = String(stats.incident_count || 0);

    el = $("stat-avg-conf");
    if (el) {
      var conf = stats.avg_confidence || stats.confidence;
      el.textContent = typeof conf === "number" ? conf.toFixed(2) : "\u2014";
    }

    el = $("entity-confidence");
    if (el) {
      var c = stats.avg_confidence || stats.confidence;
      el.textContent = typeof c === "number" ? c.toFixed(2) : "\u2014";
    }
  }

  function renderTimeline(timeline) {
    var root = $("entity-timeline-root");
    if (!root) return;
    if (!timeline || timeline.length === 0) {
      root.innerHTML = TraceRender.emptyState("No timeline events");
      return;
    }
    var sorted = timeline.slice().reverse().slice(0, 20);
    root.innerHTML = sorted.map(function (item) {
      var kindLabel = String(item.kind || "event").toUpperCase();
      var badgeKind = item.kind === "incident" ? "filled" : (item.kind === "alert" ? "error" : "ghost");
      var badgeHtml = TraceRender.badge(badgeKind, kindLabel);
      var time = TraceClient.formatTime(item.timestamp_utc);
      var summary = TraceClient.escapeHtml(item.summary || item.title || "");

      return '<div class="flex items-start gap-4 p-3 hover:bg-surface-high transition-colors">'
        + '<span class="font-mono text-[0.6rem] text-outline whitespace-nowrap mt-0.5">' + TraceClient.escapeHtml(time) + '</span>'
        + badgeHtml
        + '<div><span class="text-[0.75rem] text-on-surface">' + summary + '</span></div>'
        + '</div>';
    }).join("");
  }

  function renderIncidents(incidents) {
    var root = $("entity-incidents-root");
    if (!root) return;
    if (!incidents || incidents.length === 0) {
      root.innerHTML = TraceRender.emptyState("No linked incidents");
      return;
    }
    root.innerHTML = incidents.map(function (inc) {
      return TraceRender.incidentCard(inc);
    }).join("");
  }

  /* ─── Init ─── */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    var sel = $("entity-select");
    if (sel) {
      sel.addEventListener("change", function () {
        loadEntityProfile(sel.value);
      });
    }

    TraceClient.probe().then(function (info) {
      if (info) loadEntityList();
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
