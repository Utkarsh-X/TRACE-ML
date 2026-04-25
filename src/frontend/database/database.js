/**
 * Database Page Controller
 *
 * Unified filter pipeline:
 *   applyFilters({ query, type, status })
 *
 * Features:
 *  - Debounced search with match highlighting
 *  - Filter chips (type / status)
 *  - Row click → entity profile
 *  - Row-level [OPEN] action button
 *  - Live sync indicator (last synced X ago)
 *  - Stat strip population
 *  - Contextual loading + empty state with next actions
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  /* ──────────── State ────────────────────────────────── */
  var _allEntities  = [];
  var _lastSyncedAt = null;      // Date of last successful sync
  var _searchTimer  = null;      // debounce handle
  var _liveTimer    = null;      // interval for "last updated X ago"

  /* Active filter state */
  var _filters = {
    query:  "",
    type:   "",
    status: ""
  };

  /* ──────────── Utility ──────────────────────────────── */

  /**
   * Escape HTML special characters.
   */
  function esc(v) {
    return String(v || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /**
   * Wrap occurrences of `query` in <mark> tags for highlighting.
   * Returns safe HTML string.
   */
  function highlight(text, query) {
    if (!query || !text) return esc(text);
    var safe = esc(text);
    var safeQ = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return safe.replace(new RegExp(safeQ, "gi"), function (m) {
      return '<mark class="db-mark">' + m + "</mark>";
    });
  }

  /**
   * Human-readable "X ago" label.
   */
  function timeAgo(date) {
    if (!date) return "—";
    var diff = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diff < 5)  return "just now";
    if (diff < 60) return diff + "s ago";
    if (diff < 3600) return Math.floor(diff / 60) + "m ago";
    return Math.floor(diff / 3600) + "h ago";
  }

  /* ──────────── Live Indicator ───────────────────────── */

  function setLiveIndicator(state, extraLabel) {
    var dot   = $("db-live-dot");
    var label = $("db-live-label");
    if (!dot || !label) return;

    if (state === "loading") {
      dot.className   = "db-live-dot";
      label.textContent = "Syncing…";
    } else if (state === "live") {
      dot.className   = "db-live-dot active";
      label.textContent = extraLabel || "Live";
    } else {
      dot.className   = "db-live-dot";
      label.textContent = extraLabel || "—";
    }
  }

  function startLiveClock() {
    if (_liveTimer) clearInterval(_liveTimer);
    _liveTimer = setInterval(function () {
      if (_lastSyncedAt) {
        setLiveIndicator("live", "Updated " + timeAgo(_lastSyncedAt));
      }
    }, 5000);
  }

  /* ──────────── Data Loading ─────────────────────────── */

  function loadEntities() {
    var limitEl = $("db-limit");
    var limit   = parseInt(limitEl ? limitEl.value : "100", 10);
    if (isNaN(limit) || limit <= 0) {
      limit = 100;
      if (limitEl) limitEl.value = "100";
    }

    setLiveIndicator("loading");
    setSyncButtonLoading(true);

    TraceClient.entities({ limit: limit }).then(function (list) {
      if (!list) return;
      _allEntities  = list;
      _lastSyncedAt = new Date();
      setLiveIndicator("live", "Updated just now");
      startLiveClock();

      // Update entity count stat
      var el = $("db-stat-entities");
      if (el) el.textContent = String(list.length);

      applyFilters();
    }).catch(function (err) {
      console.error("Error loading entities:", err);
      setLiveIndicator("live", "Sync error");
    }).finally(function () {
      setSyncButtonLoading(false);
    });
  }

  function loadHealth() {
    TraceClient.health().then(function (health) {
      if (!health) return;
      var el;

      el = $("db-stat-entities");
      if (el && el.textContent === "—") el.textContent = String(health.active_entity_count || 0);

      el = $("db-stat-detections");
      if (el) el.textContent = String(health.total_detection_count || 0);

      el = $("db-stat-alerts");
      if (el) el.textContent = String(health.recent_alert_count || 0);

      el = $("db-stat-incidents");
      if (el) el.textContent = String(health.open_incident_count || 0);
    });
  }

  /* ──────────── Unified Filter Pipeline ─────────────── */

  /**
   * Master filter function. Reads from _filters state object.
   * Applies type, status, and freetext query together.
   * Handles special syntax: type:known, type:unknown
   */
  function applyFilters() {
    var rawQuery  = _filters.query.toLowerCase().trim();
    var typeFilter   = _filters.type;
    var statusFilter = _filters.status;

    // Parse inline "type:xxx" tokens from the search query
    var searchQuery = rawQuery;
    var typeMatch = rawQuery.match(/\btype:(\w+)/);
    if (typeMatch) {
      if (!typeFilter) typeFilter = typeMatch[1];   // search token wins if no chip
      searchQuery = rawQuery.replace(/\btype:\w+\s*/g, "").trim();
    }

    var filtered = _allEntities.filter(function (ent) {
      if (typeFilter   && (ent.type   || "").toLowerCase() !== typeFilter)   return false;
      if (statusFilter && (ent.status || "").toLowerCase() !== statusFilter) return false;
      if (!searchQuery) return true;
      return (ent.entity_id || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.name     || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.category || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.status   || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.type     || "").toLowerCase().indexOf(searchQuery) >= 0;
    });

    renderTable(filtered, searchQuery);
    updateFooter(filtered.length, _allEntities.length);
  }

  /* ──────────── Table Rendering ──────────────────────── */

  function renderTable(entities, highlightQuery) {
    var tbody = $("db-tbody");
    if (!tbody) return;

    if (entities.length === 0) {
      tbody.innerHTML = buildEmptyState();
      return;
    }

    var fmtDateTime = TraceClient.formatDateTime || function (v) { return v || "—"; };
    var q = highlightQuery || "";

    tbody.innerHTML = entities.map(function (ent, idx) {
      return buildRow(ent, idx, q, fmtDateTime);
    }).join("");
  }

  function buildRow(ent, idx, q, fmtDT) {
    var bgClass = idx % 2 === 0 ? "bg-surface" : "bg-surface-container";
    var entityUrl = "../entities/index.html?id=" + encodeURIComponent(ent.entity_id || "");

    /* Entity cell: name (bright) + ID (dim mono) */
    var nameHtml = highlight(ent.name || "—", q);
    var idHtml   = highlight(ent.entity_id || "—", q);
    var entityCell = '<div class="db-cell-entity">'
      + '<span class="db-entity-name">' + nameHtml + '</span>'
      + '<span class="db-entity-id">' + idHtml + '</span>'
      + '</div>';

    /* Type badge */
    var typeLabel  = String(ent.type || "unknown").toUpperCase();
    var typeBadge  = ent.type === "known"
      ? TraceRender.badge("filled", typeLabel)
      : TraceRender.badge("neutral", typeLabel);

    /* Category badge */
    var cat      = String(ent.category || "unknown").toLowerCase();
    var catLabel = cat.toUpperCase();
    var catKind  = cat === "criminal" ? "filled" : (cat === "unknown" ? "neutral" : "ghost");
    var catBadge = TraceRender.badge(catKind, catLabel);

    /* Status */
    var statusLabel = String(ent.status || "unknown").toUpperCase();
    var statusColor = ent.status === "active" ? "#fff" : "#555";
    var statusHtml  = '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;font-weight:600;color:' + statusColor + ';">' + esc(statusLabel) + '</span>';

    /* Dates */
    var created  = ent.created_at  ? fmtDT(ent.created_at)  : "—";
    var activity = ent.last_seen_at ? fmtDT(ent.last_seen_at) : "—";

    /* Alert / incident counts */
    var alertCnt    = ent.recent_alert_count   != null ? Number(ent.recent_alert_count)   : 0;
    var incidentCnt = ent.open_incident_count  != null ? Number(ent.open_incident_count)  : 0;
    var alertCls    = alertCnt > 0    ? "db-num db-num--alert" : "db-num db-num--zero";
    var incCls      = incidentCnt > 0 ? "db-num"               : "db-num db-num--zero";

    /* Row action buttons */
    var actions = '<div class="db-row-actions">'
      + '<button class="db-row-btn db-row-btn--primary" data-open-entity="' + esc(ent.entity_id || "") + '">OPEN</button>'
      + '</div>';

    return '<tr class="' + bgClass + '" data-entity-id="' + esc(ent.entity_id || "") + '">'
      + '<td>' + entityCell + '</td>'
      + '<td>' + typeBadge + '</td>'
      + '<td>' + catBadge + '</td>'
      + '<td>' + statusHtml + '</td>'
      + '<td class="font-mono text-outline" style="font-size:0.62rem;">' + esc(created) + '</td>'
      + '<td><span class="' + alertCls + '">' + alertCnt + '</span></td>'
      + '<td><span class="' + incCls + '">' + incidentCnt + '</span></td>'
      + '<td class="font-mono text-outline" style="font-size:0.62rem;">' + esc(activity) + '</td>'
      + '<td>' + actions + '</td>'
      + '</tr>';
  }

  function buildEmptyState() {
    var isFiltered = _filters.query || _filters.type || _filters.status;
    var msg = isFiltered
      ? "No entities match the current filters."
      : "No entities in database.";

    return '<tr><td colspan="9" style="padding:0;">'
      + '<div class="db-empty-state">'
      + '<span class="material-symbols-outlined db-empty-icon">manage_search</span>'
      + '<span class="db-empty-msg">' + esc(msg) + '</span>'
      + '<div class="db-empty-actions">'
      + (isFiltered
          ? '<button class="db-empty-btn" id="empty-clear-filters">Clear Filters</button>'
          : '<a href="../enrollment/index.html" class="db-empty-btn">Enroll Entity</a>'
            + '<a href="../live_ops/index.html" class="db-empty-btn">Go to Live Ops</a>')
      + '</div>'
      + '</div>'
      + '</td></tr>';
  }

  /* ──────────── Footer ───────────────────────────────── */

  function updateFooter(showing, total) {
    var el = $("db-footer-count");
    if (el) {
      el.textContent = "Showing " + showing + " of " + total + " entities"
        + (showing !== total ? " (filtered)" : "");
    }
  }

  /* ──────────── Sync Button ──────────────────────────── */

  function setSyncButtonLoading(isLoading) {
    var btn = $("db-execute");
    if (!btn) return;
    btn.disabled = isLoading;
    btn.innerHTML = isLoading
      ? '<span class="material-symbols-outlined" style="font-size:15px;animation:spin 0.8s linear infinite;">sync</span> SYNCING…'
      : '<span class="material-symbols-outlined" style="font-size:15px;">sync</span> SYNC';
  }

  /* ──────────── Wire: Search (debounced) ─────────────── */

  function wireSearch() {
    var inp = $("db-search");
    if (!inp) return;
    inp.addEventListener("input", function () {
      clearTimeout(_searchTimer);
      _searchTimer = setTimeout(function () {
        _filters.query = inp.value;
        applyFilters();
      }, 200);
    });
  }

  /* ──────────── Wire: Filter Chips ───────────────────── */

  function wireFilterChips() {
    // Delegate on both chip containers
    ["chips-type", "chips-status"].forEach(function (containerId) {
      var container = $(containerId);
      if (!container) return;
      container.addEventListener("click", function (e) {
        var chip = e.target.closest(".db-chip[data-filter]");
        if (!chip) return;

        var filterKey = chip.dataset.filter;   // "type" | "status"
        var filterVal = chip.dataset.value;    // "" | "known" | etc.

        // Update visual state
        var siblings = container.querySelectorAll(".db-chip[data-filter]");
        siblings.forEach(function (s) { s.classList.remove("active"); });
        chip.classList.add("active");

        // Update filter state + hidden selects (for backward compat)
        _filters[filterKey] = filterVal;
        var sel = $(filterKey === "type" ? "filter-type" : "filter-status");
        if (sel) sel.value = filterVal;

        applyFilters();
      });
    });

    // Hidden "Clear" button (kept for JS compat with wireInlineFilters)
    var clearBtn = $("filter-clear");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        _filters.type = "";
        _filters.status = "";
        resetChips();
        applyFilters();
      });
    }
  }

  function resetChips() {
    ["chips-type", "chips-status"].forEach(function (cid) {
      var c = $(cid);
      if (!c) return;
      c.querySelectorAll(".db-chip").forEach(function (ch) {
        ch.classList.toggle("active", ch.dataset.value === "");
      });
    });
  }

  /* ──────────── Wire: Sync button + Limit ────────────── */

  function wireRefresh() {
    var btn = $("db-execute");
    if (btn) {
      btn.addEventListener("click", function () {
        loadEntities();
        loadHealth();
      });
    }
    var limitEl = $("db-limit");
    if (limitEl) {
      limitEl.addEventListener("change", function () {
        loadEntities();
      });
    }
  }

  /* ──────────── Wire: Row clicks + Row actions ────────── */

  function wireTableClicks() {
    var tbody = $("db-tbody");
    if (!tbody) return;

    tbody.addEventListener("click", function (e) {
      // Row action: [OPEN] button
      var openBtn = e.target.closest("[data-open-entity]");
      if (openBtn) {
        e.stopPropagation();
        var eid = openBtn.dataset.openEntity;
        if (eid) window.location.href = "../entities/index.html?id=" + encodeURIComponent(eid);
        return;
      }

      // Empty state: clear filters button
      var clearBtn = e.target.closest("#empty-clear-filters");
      if (clearBtn) {
        _filters.type   = "";
        _filters.status = "";
        _filters.query  = "";
        var inp = $("db-search");
        if (inp) inp.value = "";
        resetChips();
        applyFilters();
        return;
      }

      // Row click → open entity profile
      var row = e.target.closest("tr[data-entity-id]");
      if (row && row.dataset.entityId) {
        window.location.href = "../entities/index.html?id=" + encodeURIComponent(row.dataset.entityId);
      }
    });
  }

  /* ──────────── Init ─────────────────────────────────── */

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    wireSearch();
    wireRefresh();
    wireFilterChips();
    wireTableClicks();

    TraceClient.probe().then(function (info) {
      if (info) {
        loadEntities();
        loadHealth();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
