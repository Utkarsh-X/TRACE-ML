/**
 * TRACE-AML DOM Rendering Helpers & Offline UI
 *
 * Shared functions that convert API response objects into HTML strings.
 * Also manages connection badge, offline banner, and content dimming.
 *
 * @fileoverview Depends on TraceClient being loaded first.
 */
(function (global) {
  "use strict";

  var esc = global.TraceClient ? global.TraceClient.escapeHtml : function (v) { return String(v || ""); };
  var fmtTime = global.TraceClient ? global.TraceClient.formatTime : function (v) { return v || ""; };
  var fmtDateTime = global.TraceClient ? global.TraceClient.formatDateTime : function (v) { return v || ""; };

  /**
   * Truncate long IDs (UUID-style) to a readable short form.
   * "INC-20260406T141045Z-8c110c15" → "INC-8c110c15"
   * "ALT-20260406T141051Z-8c2d5b36" → "ALT-8c2d5b36"
   * Short IDs like "INC-018" pass through unchanged.
   */
  function shortId(id) {
    if (!id) return "—";
    var s = String(id);
    // If the ID has a UUID-ish tail after a timestamp segment, keep prefix + last segment
    var parts = s.split("-");
    if (parts.length >= 3 && s.length > 20) {
      return parts[0] + "-" + parts[parts.length - 1].substring(0, 8);
    }
    return s;
  }

  /* ───────────────────── Badge / Label Helpers ───────────────────── */

  /**
   * Forensic badge.
   * @param {"filled"|"ghost"|"error"} kind
   * @param {string} label
   * @returns {string} HTML
   */
  function badge(kind, label) {
    var cls = "badge";
    if (kind === "filled") cls += " badge--filled";
    else if (kind === "error") cls += " badge--error";
    else cls += " badge--ghost";
    return '<span class="' + cls + '">' + esc(label) + "</span>";
  }

  /**
   * Severity → badge kind mapping.
   * @param {string} severity
   * @returns {"filled"|"ghost"|"error"}
   */
  function _severityKind(severity) {
    var s = String(severity || "").toLowerCase();
    if (s === "high") return "error";
    if (s === "medium") return "ghost";
    return "ghost";
  }

  /**
   * Timeline item kind → badge kind.
   * @param {string} kind
   * @returns {"filled"|"ghost"|"error"}
   */
  function _timelineKind(kind) {
    var k = String(kind || "").toLowerCase();
    if (k === "incident") return "filled";
    if (k === "alert") return "error";
    if (k === "action") return "filled";
    return "ghost";
  }

  /* ───────────────────── Component Renderers ─────────────────────── */

  /**
   * Render a timeline card (used on Incidents, History, Entity pages).
   * @param {Object} item  TimelineItem from API
   * @returns {string} HTML
   */
  function timelineCard(item) {
    var kindLabel = String(item.kind || "event").toUpperCase();
    var badgeHtml = badge(_timelineKind(item.kind), kindLabel);
    var title = esc(item.title || "");
    var time = esc(fmtTime(item.timestamp_utc));
    var summary = esc(item.summary || "");
    var meta = [];
    if (item.entity_id) meta.push("Entity: " + esc(item.entity_id));
    if (item.source) meta.push("Source: " + esc(item.source));
    if (item.incident_id) meta.push("Incident: " + esc(item.incident_id));
    if (item.metadata) {
      if (item.metadata.track_id) meta.push("Track: " + esc(item.metadata.track_id));
      if (item.metadata.event_count) meta.push("Events: " + esc(item.metadata.event_count));
    }

    return '<div class="bg-surface-container p-4 hover:bg-surface-high transition-colors">'
      + '<div class="flex items-center justify-between mb-2">'
      + '<div class="flex items-center gap-2">' + badgeHtml
      + '<span class="font-headline font-semibold text-[0.8rem] text-primary">' + title + "</span>"
      + "</div>"
      + '<span class="font-mono text-[0.6rem] text-outline">' + time + "</span>"
      + "</div>"
      + '<p class="text-[0.75rem] text-on-surface-variant leading-relaxed">' + summary + "</p>"
      + (meta.length
        ? '<div class="mt-2 flex items-center gap-3 flex-wrap">'
          + meta.map(function (m) { return '<span class="font-mono text-[0.6rem] text-outline">' + m + "</span>"; }).join("")
          + "</div>"
        : "")
      + "</div>";
  }

  /**
   * Render an entity summary card (sidebar-style).
   * @param {Object} entity  EntitySummary
   * @returns {string} HTML
   */
  function entityCard(entity) {
    var name = esc(entity.name || entity.entity_id || "—");
    var cat = String(entity.category || entity.type || "unknown").toUpperCase();
    var catBadge;
    if (cat === "CRIMINAL") catBadge = badge("filled", cat);
    else if (cat === "UNKNOWN") catBadge = badge("error", cat);
    else catBadge = badge("ghost", cat);

    // Build info line: confidence + last-seen
    var info = [];
    if (entity.confidence !== undefined && entity.confidence !== null) {
      info.push("Conf: " + Number(entity.confidence).toFixed(2));
    }
    if (entity.last_seen_at) {
      info.push("Last: " + fmtTime(entity.last_seen_at));
    }
    var infoLine = info.length
      ? '<span class="font-mono text-[0.6rem] text-outline">' + esc(info.join("  ")) + "</span>"
      : '';

    return '<div class="bg-surface-high p-3 hover:bg-surface-bright transition-colors cursor-pointer">'
      + '<div class="flex items-center justify-between mb-1">'
      + '<span class="font-headline font-semibold text-[0.8rem] text-primary truncate max-w-[160px]">' + name + "</span>"
      + catBadge
      + "</div>"
      + infoLine
      + "</div>";
  }

  /**
   * Render an alert row.
   * @param {Object} alert  AlertRecord
   * @returns {string} HTML
   */
  function alertRow(alert) {
    var sevBadge = badge(_severityKind(alert.severity), String(alert.severity || "").toUpperCase());
    var alertType = esc(String(alert.type || "").replace(/_/g, " "));
    var reason = esc(alert.reason || "");
    if (reason.length > 60) reason = reason.substring(0, 57) + "...";
    var displayAlertId = shortId(alert.alert_id);

    return '<div class="bg-surface-high p-3 hover:bg-surface-bright transition-colors cursor-pointer">'
      + '<div class="flex items-center justify-between mb-1">'
      + '<span class="font-mono text-[0.6rem] text-error font-medium uppercase truncate">' + alertType + "</span>"
      + sevBadge
      + "</div>"
      + '<p class="text-[0.7rem] text-on-surface-variant leading-snug line-clamp-2">' + reason + "</p>"
      + '<span class="font-mono text-[0.6rem] text-outline block mt-1">'
      + fmtTime(alert.timestamp_utc) + " · " + esc(displayAlertId)
      + "</span></div>";
  }

  /**
   * Render an incident card.
   * @param {Object} inc  IncidentSummary
   * @returns {string} HTML
   */
  function incidentCard(inc) {
    var status = String(inc.status || "open").toUpperCase();
    var statusBadge = status === "OPEN" ? badge("error", status) : badge("ghost", status);
    var displayId = shortId(inc.incident_id);
    var summary = esc(inc.summary || inc.reason || "");
    // Truncate summary to avoid overflow
    if (summary.length > 80) summary = summary.substring(0, 77) + "...";

    return '<div class="bg-surface-high p-4 hover:bg-surface-bright transition-colors cursor-pointer">'
      + '<div class="flex items-center justify-between mb-1 gap-2">'
      + '<span class="font-headline font-semibold text-[0.8rem] text-primary truncate">' + esc(displayId) + "</span>"
      + statusBadge
      + "</div>"
      + '<p class="text-[0.7rem] text-on-surface-variant leading-snug line-clamp-2">' + summary + "</p>"
      + '<span class="font-mono text-[0.55rem] text-outline mt-1 block">'
      + "Started: " + fmtTime(inc.start_time) + " · Alerts: " + esc(inc.alert_count)
      + "</span></div>";
  }

  /**
   * Render an action row.
   * @param {Object} action  ActionRecord
   * @returns {string} HTML
   */
  function actionRow(action) {
    var typeLabel = esc(String(action.action_type || "").toUpperCase());
    return '<div class="bg-surface-container p-3">'
      + '<div class="flex items-center justify-between mb-1">'
      + '<span class="font-mono text-[0.6rem] text-primary font-medium uppercase">' + typeLabel + "</span>"
      + '<span class="font-mono text-[0.6rem] text-outline">' + fmtTime(action.timestamp_utc) + "</span>"
      + "</div>"
      + '<p class="text-[0.7rem] text-on-surface-variant leading-snug">'
      + "Action type: " + esc(action.action_type)
      + ". Trigger: " + esc(action.trigger)
      + ". Status: " + esc(action.status) + "."
      + "</p>"
      + '<span class="font-mono text-[0.55rem] text-outline block mt-1">' + esc(action.action_id) + "</span>"
      + "</div>";
  }

  /**
   * Render a terminal log line from SSE event.
   * @param {string} topic
   * @param {Object} payload
   * @param {string} timestamp
   * @returns {string} HTML
   */
  function terminalLine(topic, payload, timestamp) {
    var time = fmtTime(timestamp);
    var topicUpper = String(topic || "").split(".").pop().toUpperCase();
    var isError = topic.indexOf("alert") >= 0 || topic.indexOf("error") >= 0;
    var cls = isError ? "log-error" : "log-type";
    var detail = "";
    if (payload) {
      var parts = [];
      Object.keys(payload).forEach(function (key) {
        var val = payload[key];
        if (typeof val === "object") return;
        parts.push(key + "=" + val);
      });
      detail = parts.slice(0, 5).join(" ");
    }
    return '<span class="log-time">[' + esc(time) + ']</span> '
      + '<span class="' + cls + '">' + esc(topicUpper) + "</span> "
      + esc(detail) + "\n";
  }

  /**
   * Render a table row for database entity listing.
   * @param {Object} entity  EntitySummary
   * @param {number} idx
   * @returns {string} HTML <tr>
   */
  function tableRow(entity, idx) {
    var bgClass = idx % 2 === 0 ? "bg-surface" : "bg-surface-container";
    var idClass = entity.type === "unknown" ? "text-error" : "text-primary";
    var catLabel = esc(String(entity.category || "unknown").toUpperCase());
    var catBadge = entity.category === "criminal" ? badge("filled", catLabel) : (entity.type === "unknown" ? badge("error", "UNKNOWN") : badge("ghost", catLabel));
    var statusClass = entity.status === "active" ? "text-primary" : "text-on-surface-variant";
    return '<tr class="' + bgClass + ' cursor-pointer">'
      + '<td class="font-mono ' + idClass + '">' + esc(entity.entity_id) + "</td>"
      + '<td class="text-on-surface">' + esc(entity.name || "—") + "</td>"
      + "<td>" + catBadge + "</td>"
      + '<td class="' + statusClass + '">' + esc(entity.status) + "</td>"
      + '<td class="font-mono text-on-surface-variant">' + esc(entity.person_id ? "—" : "0") + "</td>"
      + '<td class="font-mono text-on-surface-variant">—</td>'
      + '<td class="font-mono text-outline text-[0.65rem]">' + esc(entity.last_seen_at || "—") + "</td>"
      + "</tr>";
  }

  /**
   * Render a health check row.
   * @param {string} name
   * @param {string} detail
   * @param {boolean} ok
   * @returns {string} HTML
   */
  function healthCheck(name, detail, ok) {
    var dotCls = ok ? "status-dot--active" : "status-dot--idle";
    var badgeHtml = ok ? badge("filled", "OK") : badge("ghost", "SKIP");
    return '<div class="bg-surface-container p-4 flex items-center justify-between">'
      + '<div class="flex items-center gap-3">'
      + '<span class="status-dot ' + dotCls + '"></span>'
      + '<span class="text-[0.8rem] text-on-surface">' + esc(name) + "</span>"
      + "</div>"
      + '<div class="flex items-center gap-3">'
      + '<span class="font-mono text-[0.65rem] text-on-surface-variant">' + esc(detail) + "</span>"
      + badgeHtml
      + "</div></div>";
  }

  /**
   * Empty state placeholder.
   * @param {string} message
   * @returns {string} HTML
   */
  function emptyState(message) {
    return '<div class="flex flex-col items-center justify-center py-12 text-center">'
      + '<span class="material-symbols-outlined text-outline-variant text-[36px] mb-2">info</span>'
      + '<p class="font-mono text-[0.75rem] text-outline">' + esc(message) + "</p>"
      + "</div>";
  }

  /* ───────────────────── Offline UI Management ──────────────────── */

  /**
   * Update the connection badge in the top nav.
   * Expects an element with id="connection-badge" in the DOM.
   * @param {"online"|"offline"|"connecting"} state
   */
  function updateConnectionBadge(state) {
    var el = document.getElementById("connection-badge");
    if (!el) return;
    var dotColor, label;
    if (state === "online") {
      dotColor = "bg-emerald-400";
      label = "System Active";
    } else if (state === "connecting") {
      dotColor = "bg-yellow-400 animate-pulse";
      label = "Connecting...";
    } else {
      dotColor = "bg-red-400";
      label = "Offline";
    }
    el.innerHTML =
      '<span class="w-1.5 h-1.5 rounded-full ' + dotColor + '"></span>'
      + '<span class="text-[0.6875rem] font-mono text-outline uppercase tracking-widest">' + esc(label) + "</span>";
  }

  /**
   * Show or hide the offline banner.
   */
  function updateOfflineBanner(state) {
    var bannerId = "offline-banner";
    var existing = document.getElementById(bannerId);
    if (state === "online") {
      if (existing) existing.remove();
      return;
    }
    if (existing) return; // already showing
    if (state !== "offline") return;

    var banner = document.createElement("div");
    banner.id = bannerId;
    banner.className = "fixed top-14 left-20 right-0 z-30 bg-error-container/90 backdrop-blur-sm px-6 py-2 flex items-center gap-3";
    banner.innerHTML =
      '<span class="material-symbols-outlined text-error text-[16px]">cloud_off</span>'
      + '<span class="font-mono text-[0.7rem] text-error">Backend disconnected — showing layout only</span>';
    document.body.appendChild(banner);
  }

  /**
   * Dim content area to indicate offline state.
   * @param {HTMLElement} rootEl
   */
  function dimContent(rootEl) {
    if (!rootEl) return;
    rootEl.style.opacity = "0.4";
    rootEl.style.pointerEvents = "none";
  }

  /**
   * Restore content area.
   * @param {HTMLElement} rootEl
   */
  function enableContent(rootEl) {
    if (!rootEl) return;
    rootEl.style.opacity = "";
    rootEl.style.pointerEvents = "";
  }

  /**
   * Auto-wire connection state changes to UI.
   * Call once per page after DOM is ready.
   * @param {HTMLElement} [contentRoot]  Main content area to dim/enable
   */
  function initOfflineUI(contentRoot) {
    function handleState(state) {
      updateConnectionBadge(state);
      updateOfflineBanner(state);
      if (state === "online") {
        enableContent(contentRoot);
      } else if (state === "offline") {
        dimContent(contentRoot);
      }
    }
    // Set initial state
    handleState(global.TraceClient ? global.TraceClient.state : "connecting");
    // Listen for changes
    if (global.TraceClient) {
      global.TraceClient.onStateChange(handleState);
    }
  }

  /* ───────────────────────── Public API ──────────────────────────── */

  global.TraceRender = {
    badge: badge,
    timelineCard: timelineCard,
    entityCard: entityCard,
    alertRow: alertRow,
    incidentCard: incidentCard,
    actionRow: actionRow,
    terminalLine: terminalLine,
    tableRow: tableRow,
    healthCheck: healthCheck,
    emptyState: emptyState,

    updateConnectionBadge: updateConnectionBadge,
    updateOfflineBanner: updateOfflineBanner,
    dimContent: dimContent,
    enableContent: enableContent,
    initOfflineUI: initOfflineUI,
  };
})(typeof window !== "undefined" ? window : globalThis);
