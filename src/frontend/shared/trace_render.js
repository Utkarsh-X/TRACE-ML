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
   * Forensic badge. Monochromatic system:
   * - "filled": white background (primary states)
   * - "ghost": transparent with gray border (secondary states)
   * - "neutral": dashed border (informational states like UNKNOWN, PENDING)
   * - "critical": red background (ONLY for system emergencies)
   * @param {"filled"|"ghost"|"neutral"|"critical"} kind
   * @param {string} label
   * @returns {string} HTML
   */
  function badge(kind, label) {
    var cls = "badge";
    if (kind === "filled") cls += " badge--filled";
    else if (kind === "critical") cls += " badge--critical";
    else if (kind === "neutral") cls += " badge--neutral";
    else cls += " badge--ghost";
    return '<span class="' + cls + '">' + esc(label) + "</span>";
  }

  /**
   * Severity → badge kind mapping (monochromatic).
   * Only 'critical' severity uses red. All others use neutral/ghost.
   * @param {string} severity
   * @returns {"filled"|"ghost"|"neutral"|"critical"}
   */
  function _severityKind(severity) {
    var s = String(severity || "").toLowerCase();
    /* CRITICAL: Only for actual system emergencies */
    if (s === "critical" || s === "extreme") return "critical";
    /* All other severities: neutral grayscale */
    if (s === "high") return "neutral";
    if (s === "medium") return "ghost";
    return "ghost";
  }

  /**
   * Timeline item kind → badge kind (monochromatic).
   * @param {string} kind
   * @returns {"filled"|"ghost"|"neutral"|"critical"}
   */
  function _timelineKind(kind) {
    var k = String(kind || "").toLowerCase();
    if (k === "incident") return "filled";
    /* Alerts: use neutral (gray) instead of red error badge */
    if (k === "alert") return "neutral";
    if (k === "action") return "ghost";
    if (k === "critical") return "critical";
    return "ghost";
  }

  /* ───────────────────── Component Renderers ─────────────────────── */

  function timelineCard(item) {
    var ev = item.ev || item;
    var count = item.count || 1;

    var kindLabel = String(ev.kind || "event").toUpperCase();
    var badgeHtml = badge(_timelineKind(ev.kind), kindLabel);
    
    var title = esc(ev.title || "");
    if (count > 1) {
      title = '<span class="text-primary mr-1">[' + count + 'x]</span>' + title;
    }

    var timeStr;
    if (count > 1) {
      var earliest = Math.min(item.startTime, item.endTime);
      var latest = Math.max(item.startTime, item.endTime);
      timeStr = esc(fmtTime(new Date(earliest).toISOString())) + " \u2014 " + esc(fmtTime(new Date(latest).toISOString()));
    } else {
      timeStr = esc(fmtTime(ev.timestamp_utc));
    }

    var summary = esc(ev.summary || "");
    var meta = [];
    if (count === 1) {
      if (ev.entity_id) meta.push("Entity: " + esc(ev.entity_id));
      if (ev.source) meta.push("Source: " + esc(ev.source));
      if (ev.incident_id) meta.push("Incident: " + esc(ev.incident_id));
      if (ev.metadata) {
        if (ev.metadata.track_id) meta.push("Track: " + esc(ev.metadata.track_id));
        if (ev.metadata.event_count) meta.push("Events: " + esc(ev.metadata.event_count));
      }
    }

    var payloadStr = esc(encodeURIComponent(JSON.stringify(item)));

    return '<div class="bg-surface-container p-4 hover:bg-surface-high transition-colors cursor-pointer timeline-card-item" data-payload="' + payloadStr + '">'
      + '<div class="flex items-center justify-between mb-2">'
      + '<div class="flex items-center gap-2">' + badgeHtml
      + '<span class="font-headline font-semibold text-[0.8rem] text-primary">' + title + "</span>"
      + "</div>"
      + '<span class="font-mono text-[0.6rem] text-outline">' + timeStr + "</span>"
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
    else if (cat === "UNKNOWN") catBadge = badge("neutral", cat);
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

    return '<div class="bg-surface-high p-3 hover:bg-surface-bright transition-colors cursor-pointer" data-entity-id="' + esc(entity.entity_id || "") + '">'
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
    var statusBadge = status === "OPEN" ? badge("neutral", status) : badge("ghost", status);
    var displayId = shortId(inc.incident_id);
    var summary = esc(inc.summary || inc.reason || "");
    // Truncate summary to avoid overflow
    if (summary.length > 80) summary = summary.substring(0, 77) + "...";

    return '<div class="bg-surface-high p-4 hover:bg-surface-bright transition-colors cursor-pointer" data-incident-id="' + esc(inc.incident_id || "") + '">'
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
   * Consistently used across Live Ops, Incidents and Database views.
   * @param {string} topic
   * @param {Object} payload
   * @param {string} timestamp
   * @returns {string} HTML
   */
  function terminalLine(topic, payload, timestamp) {
    var time = fmtTime(timestamp);
    var topicBase = String(topic || "SYSTEM").toUpperCase();
    var topicUpper = topicBase.split(".").pop();
    
    // Topic classification for color
    var t = topicBase.toLowerCase();
    var topicCls = "ll-topic-default";
    if      (t.indexOf("alert")    >= 0) topicCls = "ll-topic-alert";
    else if (t.indexOf("incident") >= 0) topicCls = "ll-topic-incident";
    else if (t.indexOf("action")   >= 0) topicCls = "ll-topic-action";

    // Chip logic (severity, decision, entity)
    var chipHtml = "";
    if (payload) {
      var sev = String(payload.severity || "").toUpperCase();
      if (sev === "HIGH" || sev === "CRITICAL") {
        chipHtml = '<span class="ll-chip ll-chip--error">' + esc(sev) + '</span>';
      } else if (sev) {
        chipHtml = '<span class="ll-chip">' + esc(sev) + '</span>';
      } else {
        var dec = String(payload.decision || "").toUpperCase();
        if (dec) {
          chipHtml = '<span class="ll-chip">' + esc(dec) + '</span>';
        } else {
          var eid = payload.entity_id || payload.entity || "";
          if (eid) chipHtml = '<span class="ll-chip">' + esc(String(eid)) + '</span>';
        }
      }
    }

    // Message logic
    var msg = (payload && (payload.message || payload.reason || payload.summary)) || "";
    if (!msg && payload) {
      var parts = [];
      Object.keys(payload).forEach(function (k) {
        var v = payload[k];
        if (v !== null && v !== undefined && typeof v !== "object") {
          parts.push(k + "=" + String(v).substring(0, 40));
        }
      });
      msg = parts.slice(0, 4).join("  ") || topicUpper.toLowerCase() + " event";
    }
    msg = esc(String(msg).substring(0, 200));

    return '<div class="log-line">'
      + '<span class="ll-ts">[' + esc(time) + ']</span>'
      + '<span class="' + topicCls + '">' + esc(topicUpper) + '</span>'
      + '<span class="ll-msg">' + msg + chipHtml + '</span>'
      + '</div>\n';
  }

  /**
   * Render a table row for database entity listing.
   * @param {Object} entity  EntitySummary
   * @param {number} idx
   * @returns {string} HTML <tr>
   */
  function tableRow(entity, idx) {
    var bgClass = idx % 2 === 0 ? "bg-surface" : "bg-surface-container";
    var cat = String(entity.category || "unknown").toLowerCase();
    var idClass = cat === "unknown" ? "text-error" : "text-primary";
    var catLabel = cat.toUpperCase();
    var catBadge = cat === "criminal" ? badge("filled", catLabel) : (cat === "unknown" ? badge("neutral", catLabel) : badge("ghost", catLabel));
    var statusClass = entity.status === "active" ? "text-primary" : "text-on-surface-variant";
    var typeLabel = String(entity.type || "unknown").toUpperCase();
    var typeBadge = entity.type === "known" ? badge("filled", typeLabel) : badge("neutral", typeLabel);
    var created = entity.created_at ? fmtDateTime(entity.created_at) : "\u2014";
    var alerts = entity.recent_alert_count != null ? String(entity.recent_alert_count) : "0";
    var incidents = entity.open_incident_count != null ? String(entity.open_incident_count) : "0";
    var updated = entity.last_seen_at ? fmtDateTime(entity.last_seen_at) : "\u2014";
    var url = "../entities/index.html?id=" + encodeURIComponent(entity.entity_id || "");
    return '<tr class="' + bgClass + ' hover:bg-surface-high transition-colors cursor-pointer" onclick="window.location.href=\'' + url + '\'">'
      + '<td class="font-mono ' + idClass + '">' + esc(entity.entity_id) + "</td>"
      + "<td>" + typeBadge + "</td>"
      + '<td class="text-on-surface">' + esc(entity.name || "\u2014") + "</td>"
      + "<td>" + catBadge + "</td>"
      + '<td class="' + statusClass + '">' + esc(entity.status) + "</td>"
      + '<td class="font-mono text-outline text-[0.65rem]">' + created + "</td>"
      + '<td class="font-mono text-on-surface-variant">' + alerts + "</td>"
      + '<td class="font-mono text-on-surface-variant">' + incidents + "</td>"
      + '<td class="font-mono text-outline text-[0.65rem]">' + updated + "</td>"
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
    /* ── New sidenav: #connection-badge is the dot <span>,
       sibling .status-label is the text.               ── */
    var dot   = document.getElementById("connection-badge");
    var label = dot && dot.parentElement
                  ? dot.parentElement.querySelector(".status-label")
                  : null;

    if (!dot) return;

    var dotColor, labelText;
    if (state === "online") {
      dotColor  = "#ffffff";   /* green */
      labelText = "System Active";
    } else if (state === "connecting") {
      dotColor  = "#bdbdbd";   /* amber */
      labelText = "Connecting\u2026";
    } else {
      dotColor  = "#757575";   /* red */
      labelText = "Offline";
    }

    /* Update dot colour and pulse animation */
    dot.style.background = dotColor;

    /* Update label text if present */
    if (label) label.textContent = labelText;
  }


  /**
   * Show or hide the offline showcase overlay.
   * Inspired by the reference style: centered status + retry action.
   */
  function updateOfflineBanner(state) {
    var bannerId = "offline-banner";
    var existing = document.getElementById(bannerId);
    if (state === "online") {
      if (existing) existing.remove();
      return;
    }
    if (state !== "offline") return;

    if (!existing) {
      var overlay = document.createElement("div");
      overlay.id = bannerId;
      overlay.style.position = "fixed";
      overlay.style.inset = "0";
      overlay.style.zIndex = "120";
      overlay.style.display = "flex";
      overlay.style.alignItems = "center";
      overlay.style.justifyContent = "center";
      overlay.style.background = "linear-gradient(to top, rgba(0,0,0,0.62), rgba(0,0,0,0.32) 45%, rgba(0,0,0,0.22))";
      overlay.style.backdropFilter = "blur(2px)";
      overlay.style.webkitBackdropFilter = "blur(2px)";
      overlay.innerHTML =
        '<div style="display:flex;flex-direction:column;align-items:center;gap:18px;text-align:center;padding:24px 28px;max-width:560px;">'
        + '<span class="material-symbols-outlined" style="font-size:38px;color:#d8d8d8;opacity:0.88;">cloud_off</span>'
        + '<div style="font-family:\'Inter\',sans-serif;font-size:1.25rem;letter-spacing:0.16em;text-transform:uppercase;color:#f2f2f2;font-weight:300;">Connection Lost</div>'
        + '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;letter-spacing:0.08em;color:#a0a0a0;">Backend disconnected — showing layout only</div>'
        + '<button id="offline-retry-btn" type="button" style="margin-top:6px;padding:10px 22px;border:1px solid rgba(255,255,255,0.3);background:#fff;color:#0f0f0f;font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;letter-spacing:0.16em;text-transform:uppercase;cursor:pointer;border-radius:2px;">Retry</button>'
        + "</div>";
      document.body.appendChild(overlay);

      var retryBtn = document.getElementById("offline-retry-btn");
      if (retryBtn) {
        retryBtn.addEventListener("click", function () {
          retryBtn.textContent = "Retrying...";
          retryBtn.setAttribute("disabled", "disabled");
          retryBtn.style.opacity = "0.7";
          retryBtn.style.cursor = "wait";
          if (global.TraceClient && typeof global.TraceClient.probe === "function") {
            global.TraceClient.probe().finally(function () {
              retryBtn.textContent = "Retry";
              retryBtn.removeAttribute("disabled");
              retryBtn.style.opacity = "";
              retryBtn.style.cursor = "pointer";
            });
          }
        });
      }
    }
  }

  /**
   * Dim content area to indicate offline state.
   * @param {HTMLElement} rootEl
   */
  function dimContent(rootEl) {
    if (!rootEl) return;
    rootEl.style.opacity = "0.33";
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
