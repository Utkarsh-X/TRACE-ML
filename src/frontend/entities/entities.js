/**
 * Entities / Subjects Page Controller
 *
 * Two-view pattern:
 *   Overview → stat tiles + searchable card grid (default landing)
 *   Detail   → profile + timeline + incidents (only after user clicks a card)
 *
 * No entity is auto-selected on load.
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var _allEntities = [];

  /* ─────────────────────────── Overview ─────────────────────────── */

  function showOverview() {
    $("view-overview").style.display = "block";
    $("view-detail").style.display   = "none";
  }

  function showDetail() {
    $("view-overview").style.display = "none";
    $("view-detail").style.display   = "block";
  }

  function loadEntityList() {
    TraceClient.entities({ limit: 300 }).then(function (list) {
      _allEntities = list || [];
      renderOverviewStats(_allEntities);
      renderGrid(_allEntities);
    });
  }

  function renderOverviewStats(list) {
    var known   = list.filter(function (e) { return String(e.type || e.entity_type || "") === "known"; }).length;
    var unknown = list.filter(function (e) { return String(e.type || e.entity_type || "") !== "known"; }).length;
    var withInc = list.reduce(function (acc, e) {
      return acc + (parseInt(e.open_incident_count, 10) || 0);
    }, 0);

    var t = $("ov-total");     if (t) t.textContent = String(list.length);
    var k = $("ov-known");     if (k) k.textContent = String(known);
    var u = $("ov-unknown");   if (u) u.textContent = String(unknown);
    var i = $("ov-incidents"); if (i) i.textContent = String(withInc);
  }

  function renderGrid(list) {
    var grid  = $("entity-grid");
    var label = $("entity-count-label");
    if (!grid) return;
    if (label) label.textContent = list.length + " entit" + (list.length !== 1 ? "ies" : "y");


    if (list.length === 0) {
      grid.innerHTML = '<div class="col-span-full flex flex-col items-center justify-center py-20 text-outline font-mono text-[0.75rem]">'
        + '<span class="material-symbols-outlined text-[36px] mb-3">person_off</span>'
        + 'No entities found</div>';
      return;
    }

    grid.innerHTML = list.map(function (ent) {
      var isKnown   = String(ent.type || ent.entity_type || "") === "known";
      var name      = TraceClient.escapeHtml(ent.name || ent.entity_id);
      var shortId   = TraceClient.escapeHtml(String(ent.entity_id || ""));
      var cat       = String(ent.category || (isKnown ? "known" : "unknown")).toLowerCase();
      var lastSeen  = ent.last_seen_at ? TraceClient.formatTime(ent.last_seen_at) : "—";
      var openInc   = parseInt(ent.open_incident_count, 10) || 0;

      var typeKey   = cat === "criminal" ? "criminal" : cat === "vip" ? "vip" : isKnown ? "known" : "unknown";
      var badgeText = isKnown ? cat.toUpperCase() : "UNKNOWN";
      var cardType  = isKnown ? "known" : "unknown";

      // Build portrait thumbnail for the avatar cell.
      // If no portrait exists yet, the browser onerror falls back to the icon.
      var portraitTs = Math.floor(Date.now() / 30000); // 30 s cache window
      var avatarHtml = ent.entity_id
        ? '<img src="' + TraceClient.entityPortraitUrl(ent.entity_id) + '?t=' + portraitTs + '"'
          + ' alt="" loading="lazy"'
          + ' onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'"'
          + ' style="display:block" />'
          + '<span class="material-symbols-outlined" style="font-size:18px;color:#919191;display:none">'
            + (isKnown ? 'person' : 'person_off')
          + '</span>'
        : '<span class="material-symbols-outlined" style="font-size:18px;color:#919191">'
            + (isKnown ? 'person' : 'person_off')
          + '</span>';

      return '<div class="entity-card entity-card--' + cardType + '" data-entity-id="' + TraceClient.escapeHtml(ent.entity_id) + '">'
        /* top row: badge + arrow */
        + '<div class="flex items-center justify-between mb-2">'
        +   '<span class="entity-card__badge entity-card__badge--' + typeKey + '">' + badgeText + '</span>'
        +   '<span class="ec-arrow material-symbols-outlined" style="font-size:14px;color:#919191">arrow_forward</span>'
        + '</div>'
        /* avatar + name block */
        + '<div class="flex items-center gap-3">'
        +   '<div class="entity-card__avatar">' + avatarHtml + '</div>'
        +   '<div class="flex-1 min-w-0">'
        +     '<div style="font-family:Inter,sans-serif;font-weight:600;font-size:0.9rem;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.2">' + name + '</div>'
        +     '<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#666;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + shortId + '</div>'
        +   '</div>'
        + '</div>'
        /* footer */
        + '<div class="entity-card__footer">'
        +   '<span style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#666">' + TraceClient.escapeHtml(lastSeen) + '</span>'
        +   (openInc > 0
              ? '<span style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#ffb4ab">' + openInc + ' open case' + (openInc > 1 ? 's' : '') + '</span>'
              : '<span style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#474747">No open cases</span>')
        + '</div>'
        + '</div>';
    }).join("");

    // Wire click handlers
    var cards = grid.querySelectorAll(".entity-card");
    cards.forEach(function (card) {
      card.addEventListener("click", function () {
        var eid = card.getAttribute("data-entity-id");
        if (eid) loadEntityProfile(eid);
      });
    });
  }

  /* ─────────────────────────── Filtering ────────────────────────── */

  function applyFilters() {
    var search = ($("entity-search") || {}).value || "";
    var type   = ($("entity-filter") || {}).value || "";
    var q      = search.toLowerCase();

    var filtered = _allEntities.filter(function (e) {
      var matchText = (e.name || "").toLowerCase().includes(q)
        || (e.entity_id || "").toLowerCase().includes(q)
        || (e.category || "").toLowerCase().includes(q);
      // API may return 'type' or 'entity_type' depending on serialiser version
      var entityType = String(e.type || e.entity_type || "");
      var matchType = !type || entityType === type;
      return matchText && matchType;
    });

    renderGrid(filtered);
  }

  /* ─────────────────────────── Detail view ──────────────────────── */

  function loadEntityProfile(entityId) {
    if (!entityId) return;
    showDetail();

    TraceClient.entityProfile(entityId).then(function (profile) {
      if (!profile) {
        $("entity-display-name").textContent = "Failed to load";
        return;
      }
      renderHeader(profile.entity || {});
      renderStats(profile);
      renderTimeline(profile.timeline || []);
      renderIncidents(profile.incidents || []);
    });
  }

  function renderHeader(entity) {
    var label = $("entity-profile-label");
    if (label) label.textContent = "Entity // " + entity.entity_id;

    var nameEl = $("entity-display-name");
    if (nameEl) nameEl.textContent = entity.name || entity.entity_id || "—";

    var statusEl = $("entity-status");
    if (statusEl) statusEl.textContent = String(entity.status || "active").toUpperCase();

    var typeEl = $("entity-type");
    if (typeEl) typeEl.textContent = String(entity.category || entity.entity_type || "—").toUpperCase();

    var sevEl = $("entity-severity");
    if (sevEl) {
      var open = parseInt(entity.open_incident_count, 10) || 0;
      sevEl.textContent = open > 0 ? open + " OPEN" : "NONE";
      sevEl.className   = open > 0
        ? "text-[0.875rem] text-error uppercase font-medium"
        : "text-[0.875rem] text-on-surface-variant uppercase font-medium";
    }

    var clockEl = $("entity-clock");
    if (clockEl) clockEl.textContent = TraceClient.formatDateTime(entity.last_seen_at) || "—";

    // ── Best-match portrait ──────────────────────────────────────────
    //
    // Reset first so stale portrait from a previous entity doesn’t flash.
    var portraitImg = $("entity-portrait");
    var portraitPlaceholder = $("entity-portrait-placeholder");
    if (portraitImg) {
      portraitImg.classList.remove("loaded");
      portraitImg.removeAttribute("src"); // stop any pending request
    }
    if (portraitPlaceholder) portraitPlaceholder.classList.remove("hidden");

    if (entity.entity_id && portraitImg) {
      // Add a cache-busting timestamp so the browser always fetches fresh after
      // a recognition event (portraits improve over time).
      var ts = Math.floor(Date.now() / 10000); // invalidates every 10 s
      portraitImg.src = TraceClient.entityPortraitUrl(entity.entity_id) + "?t=" + ts;
    }
    // ─────────────────────────────────────────────────────────────────
  }

  function renderStats(profile) {
    var stats = profile.stats || {};

    var a = $("stat-appearances");    if (a) a.textContent = String(stats.detection_count || 0);
    var i = $("stat-incident-count"); if (i) i.textContent = String(stats.incident_count || 0);
    var r = $("stat-avg-conf");       if (r) r.textContent = String(stats.recent_alert_count || 0);

    var score = (profile.linked_person && typeof profile.linked_person.enrollment_score === "number")
      ? profile.linked_person.enrollment_score : null;
    var confEl = $("entity-confidence");
    if (confEl) confEl.textContent = score !== null ? (score * 100).toFixed(1) + "%" : "—";
    var bar = $("entity-confidence-bar");
    if (bar) bar.style.width = score !== null ? (score * 100).toFixed(1) + "%" : "0%";
  }

  function renderTimeline(timeline) {
    var root = $("entity-timeline-root");
    if (!root) return;
    if (!timeline || timeline.length === 0) {
      root.innerHTML = TraceRender.emptyState("No timeline events");
      return;
    }
    var sorted = timeline.slice().reverse().slice(0, 25);
    root.innerHTML = sorted.map(function (item) {
      var kindLabel = String(item.kind || "event").toUpperCase();
      var badgeKind = item.kind === "incident" ? "filled" : (item.kind === "alert" ? "error" : "ghost");
      var badgeHtml = TraceRender.badge(badgeKind, kindLabel);
      var time      = TraceClient.formatTime(item.timestamp_utc);
      var summary   = TraceClient.escapeHtml(item.summary || item.title || "");
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
      root.innerHTML = TraceRender.emptyState("No linked cases");
      return;
    }
    root.innerHTML = incidents.map(function (inc) {
      return TraceRender.incidentCard(inc);
    }).join("");
  }

  /* ─────────────────────────── Init ─────────────────────────────── */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    showOverview();

    // Search + filter wiring
    var searchEl = $("entity-search");
    if (searchEl) searchEl.addEventListener("input",  applyFilters);
    var filterEl = $("entity-filter");
    if (filterEl) filterEl.addEventListener("change", applyFilters);

    // Back button
    var backBtn = $("btn-back");
    if (backBtn) backBtn.addEventListener("click", function () { showOverview(); });

    // Load immediately — no probe() gate so the grid renders without delay.
    // probe() is still used for connection-status badge but doesn't block data.
    loadEntityList();
    TraceClient.probe(); // fire-and-forget for connection badge only

    // Auto-refresh every 10 seconds so new entities appear without manual navigation.
    // Only refreshes if the overview (grid) is visible to avoid stomping detail view.
    setInterval(function () {
      var overview = $("view-overview");
      if (overview && overview.style.display !== "none") {
        loadEntityList();
      }
    }, 10000);

    // Re-render timestamps when timezone changes
    window.addEventListener("trace:tz-change", function () {
      if (_allEntities.length) renderGrid(_allEntities);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
