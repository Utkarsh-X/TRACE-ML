/**
 * Incidents Page Controller
 *
 * - Lists incidents → populates <select>
 * - On select → loads incident detail → renders timeline, entity, alerts, actions
 * - Wires severity change + close incident buttons
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var _currentIncidentId = null;

  /* ─── Incident list ─── */

  function loadIncidentList() {
    TraceClient.incidents({ limit: 100 }).then(function (list) {
      if (!list) return;
      var sel = $("incident-select");
      if (!sel) return;

      sel.innerHTML = "";
      if (list.length === 0) {
        sel.innerHTML = '<option value="">No incidents</option>';
        $("incident-load-status").textContent = "0 incidents";
        return;
      }

      list.forEach(function (inc) {
        var opt = document.createElement("option");
        opt.value = inc.incident_id;
        // Show readable summary in dropdown
        var shortSummary = (inc.summary || "No summary");
        if (shortSummary.length > 50) shortSummary = shortSummary.substring(0, 47) + "...";
        opt.textContent = inc.incident_id + " \u2014 " + shortSummary + " (" + (inc.entity_id || "") + ")";
        sel.appendChild(opt);
      });

      $("incident-load-status").textContent = "Loaded " + list.length + " incidents";

      // Load first incident
      loadIncident(list[0].incident_id);
    });
  }

  /* ─── Incident detail ─── */

  function loadIncident(incidentId) {
    if (!incidentId) return;
    _currentIncidentId = incidentId;

    TraceClient.incident(incidentId).then(function (detail) {
      if (!detail) return;
      renderIncidentHeader(detail.incident);
      renderEntityProfile(detail.entity);
      renderAlerts(detail.alerts);
      renderTimeline(detail.timeline);
      renderActions(detail.actions);
      renderMetadata(detail.incident);
    });
  }

  function renderIncidentHeader(inc) {
    var el;
    el = $("incident-id-label");
    if (el) el.textContent = "Incident ID: " + inc.incident_id;

    el = $("incident-title");
    if (el) el.textContent = inc.summary || "Incident " + inc.incident_id;

    el = $("incident-status");
    if (el) el.textContent = String(inc.status || "open").toUpperCase();
  }

  function renderEntityProfile(entity) {
    if (!entity) return;
    var el;
    el = $("entity-name");
    if (el) el.textContent = entity.name || entity.entity_id;

    el = $("entity-id");
    if (el) el.textContent = "entity_id: " + entity.entity_id;

    el = $("entity-type");
    if (el) el.textContent = String(entity.type || "unknown").charAt(0).toUpperCase() + String(entity.type || "unknown").slice(1);

    el = $("entity-status");
    if (el) {
      el.textContent = String(entity.status || "active").charAt(0).toUpperCase() + String(entity.status || "active").slice(1);
      el.className = entity.status === "active" ? "font-mono text-[0.8rem] text-primary" : "font-mono text-[0.8rem] text-error";
    }

    el = $("entity-first-seen");
    if (el) el.textContent = TraceClient.formatDateTime(entity.created_at) || "\u2014";

    el = $("entity-last-seen");
    if (el) el.textContent = TraceClient.formatDateTime(entity.last_seen_at) || "\u2014";

    el = $("entity-detections");
    if (el) el.textContent = String(entity.detection_count || entity.recent_alert_count || 0);
  }

  function renderAlerts(alerts) {
    var root = $("incident-alerts-root");
    if (!root) return;
    if (!alerts || alerts.length === 0) {
      root.innerHTML = TraceRender.emptyState("No linked alerts");
      return;
    }
    root.innerHTML = alerts.map(function (a) {
      return TraceRender.alertRow(a);
    }).join("");
  }

  function renderTimeline(timeline) {
    var root = $("incident-timeline-root");
    if (!root) return;
    if (!timeline || timeline.length === 0) {
      root.innerHTML = TraceRender.emptyState("No timeline events");
      return;
    }
    // Reverse so newest first
    var sorted = timeline.slice().reverse();
    root.innerHTML = sorted.map(function (item) {
      // Build timeline node entry
      var kindLabel = String(item.kind || "event").toUpperCase();
      var nodeClass = "timeline-node--event";
      if (item.kind === "incident") nodeClass = "timeline-node--incident";
      else if (item.kind === "alert") nodeClass = "timeline-node--alert";
      else if (item.kind === "action") nodeClass = "timeline-node--action";

      var badgeKind = item.kind === "incident" ? "filled" : (item.kind === "alert" ? "error" : "ghost");
      var badgeHtml = TraceRender.badge(badgeKind, kindLabel);
      var title = TraceClient.escapeHtml(item.title || "");
      var time = TraceClient.formatTime(item.timestamp_utc);
      var summary = TraceClient.escapeHtml(item.summary || "");

      var meta = [];
      if (item.entity_id) meta.push("Entity: " + TraceClient.escapeHtml(item.entity_id));
      if (item.source) meta.push("Source: " + TraceClient.escapeHtml(item.source));
      if (item.metadata && item.metadata.track_id) meta.push("Track: " + TraceClient.escapeHtml(item.metadata.track_id));

      return '<div class="relative pl-10">'
        + '<div class="timeline-node ' + nodeClass + '" style="left: 32px; top: 4px;"></div>'
        + '<div class="bg-surface-container p-4 hover:bg-surface-high transition-colors">'
        + '<div class="flex items-center justify-between mb-2">'
        + '<div class="flex items-center gap-2">' + badgeHtml
        + '<span class="font-headline font-semibold text-[0.8rem] text-primary">' + title + '</span>'
        + '</div>'
        + '<span class="font-mono text-[0.6rem] text-outline">' + TraceClient.escapeHtml(time) + '</span>'
        + '</div>'
        + '<p class="text-[0.75rem] text-on-surface-variant leading-relaxed">' + summary + '</p>'
        + (meta.length
          ? '<div class="mt-2 flex items-center gap-3">'
            + meta.map(function (m) { return '<span class="font-mono text-[0.6rem] text-outline">' + m + '</span>'; }).join("")
            + '</div>'
          : '')
        + '</div></div>';
    }).join("");
  }

  function renderActions(actions) {
    var root = $("actions-log");
    if (!root) return;
    if (!actions || actions.length === 0) {
      root.innerHTML = TraceRender.emptyState("No actions recorded");
      return;
    }
    root.innerHTML = actions.map(function (a) {
      return TraceRender.actionRow(a);
    }).join("");
  }

  function renderMetadata(inc) {
    var el;
    el = $("meta-severity");
    if (el) {
      el.textContent = String(inc.severity || "low").toUpperCase();
      el.className = inc.severity === "high" ? "font-mono text-[0.8rem] text-error" : "font-mono text-[0.8rem] text-primary";
    }
    el = $("meta-alert-count");
    if (el) el.textContent = String(inc.alert_count || 0);

    el = $("meta-start");
    if (el) el.textContent = TraceClient.formatTime(inc.start_time);

    el = $("meta-updated");
    if (el) el.textContent = TraceClient.formatTime(inc.last_seen_time);

    // Set severity select to current value
    el = $("incident-severity-select");
    if (el) el.value = inc.severity || "low";
  }

  /* ─── Actions ─── */

  function wireControls() {
    var sel = $("incident-select");
    if (sel) {
      sel.addEventListener("change", function () {
        loadIncident(sel.value);
      });
    }

    var applyBtn = $("incident-apply-severity");
    if (applyBtn) {
      applyBtn.addEventListener("click", function () {
        if (!_currentIncidentId) return;
        var sevSelect = $("incident-severity-select");
        var severity = sevSelect ? sevSelect.value : "low";
        var statusEl = $("incident-action-status");
        if (statusEl) statusEl.textContent = "Updating severity...";

        TraceClient.setSeverity(_currentIncidentId, severity).then(function (result) {
          if (result) {
            if (statusEl) statusEl.textContent = "Severity updated to " + severity;
            loadIncident(_currentIncidentId);
          } else {
            if (statusEl) statusEl.textContent = "Failed — backend offline";
          }
        });
      });
    }

    var closeBtn = $("incident-close-btn");
    if (closeBtn) {
      closeBtn.addEventListener("click", function () {
        if (!_currentIncidentId) return;
        var statusEl = $("incident-action-status");
        if (statusEl) statusEl.textContent = "Closing incident...";

        TraceClient.closeIncident(_currentIncidentId).then(function (result) {
          if (result) {
            if (statusEl) statusEl.textContent = "Incident closed";
            loadIncidentList(); // Refresh list
          } else {
            if (statusEl) statusEl.textContent = "Failed — backend offline";
          }
        });
      });
    }
  }

  /* ─── Init ─── */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);
    wireControls();

    TraceClient.probe().then(function (info) {
      if (info) loadIncidentList();
    });

    // Nav button wiring
    var settingsBtn = $("nav-btn-settings");
    if (settingsBtn) settingsBtn.addEventListener("click", function () { window.location.href = "../settings/index.html"; });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
