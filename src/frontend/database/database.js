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
    var limitEl = $("db-limit");
    var val = limitEl ? limitEl.value : "";
    var limit = parseInt(val, 10);
    
    // Default fallback if not a valid number or <= 0
    if (isNaN(limit) || limit <= 0) {
      limit = 100;
      if (limitEl) limitEl.value = "100";
    }

    TraceClient.entities({ limit: limit }).then(function (list) {
      if (!list) return;
      _allEntities = list;
      applyCurrentFilters();
      updateFooter(list.length, list.length);
      // Update entity count stat immediately
      var el = document.getElementById('db-stat-entities');
      if (el) el.textContent = String(list.length);
    }).catch(function (err) {
      console.error("Error loading entities:", err);
    }).finally(function () {
      setRefreshButtonLoading(false);
    });
  }

  function renderTable(entities) {
    var tbody = document.querySelector(".db-table tbody");
    if (!tbody) return;
    if (entities.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" class="text-center py-8">'
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
      var el;
      // Entity count: use active_entity_count from health (reliable, no race condition)
      // loadEntities() will also update this independently with the exact loaded count
      el = document.getElementById('db-stat-entities');
      if (el && el.textContent === '—') el.textContent = String(health.active_entity_count || 0);

      el = document.getElementById('db-stat-detections');
      if (el) el.textContent = String(health.total_detection_count || 0);

      el = document.getElementById('db-stat-alerts');
      if (el) el.textContent = String(health.recent_alert_count || 0);

      el = document.getElementById('db-stat-incidents');
      if (el) el.textContent = String(health.open_incident_count || 0);
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
      
      // Support filter syntax: type:known, type:unknown
      var typeFilter = "";
      var searchQuery = query;
      var typeMatch = query.match(/type:(\w+)/);
      if (typeMatch) {
        typeFilter = typeMatch[1];
        searchQuery = query.replace(/type:\w+\s*/g, "").trim();
      }
      
      var filtered = _allEntities.filter(function (ent) {
        // Apply type filter if specified
        if (typeFilter && (ent.type || "").toLowerCase() !== typeFilter) {
          return false;
        }
        // If no search query remains, just return the type filter match
        if (!searchQuery) {
          return true;
        }
        // Search all text fields
        return (ent.entity_id || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.name || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.category || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.status || "").toLowerCase().indexOf(searchQuery) >= 0
          || (ent.type || "").toLowerCase().indexOf(searchQuery) >= 0;
      });
      renderTable(filtered);
      updateFooter(filtered.length, _allEntities.length);
    });
  }

  function setRefreshButtonLoading(isLoading) {
    var btn = $("db-execute");
    if (!btn) return;
    if (isLoading) {
      btn.disabled = true;
      btn.style.opacity = "0.6";
      btn.style.cursor = "not-allowed";
      btn.textContent = "⏳ LOADING...";
    } else {
      btn.disabled = false;
      btn.style.opacity = "1";
      btn.style.cursor = "pointer";
      btn.textContent = "🔄 REFRESH";
    }
  }

  function applyCurrentFilters() {
    var typeSelect = $("filter-type");
    var statusSelect = $("filter-status");
    
    var typeFilter = typeSelect ? typeSelect.value : "";
    var statusFilter = statusSelect ? statusSelect.value : "";
    
    applyFilters(typeFilter, statusFilter);
  }

  function wireQuickDelete() {
    var tbody = document.querySelector(".db-table tbody");
    if (!tbody) return;
    tbody.addEventListener("click", function (e) {
      var btn = e.target.closest(".db-quick-delete");
      if (!btn) return;
      e.stopPropagation();
      var entityId = btn.getAttribute("data-entity-id");
      if (!entityId) return;

      // Two-step confirm using a temporary overlay on the button
      if (btn.getAttribute("data-confirm") !== "1") {
        btn.setAttribute("data-confirm", "1");
        btn.style.color = "var(--color-error, #f28b82)";
        btn.style.borderColor = "rgba(242,139,130,0.4)";
        btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:13px;pointer-events:none">warning</span>';
        btn.title = "Click again to confirm deletion";
        // Auto-reset after 3 seconds
        setTimeout(function () {
          if (btn.getAttribute("data-confirm") === "1") {
            btn.removeAttribute("data-confirm");
            btn.style.color = "";
            btn.style.borderColor = "";
            btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:13px;pointer-events:none">delete</span>';
            btn.title = "Delete entity";
          }
        }, 3000);
        return;
      }

      // Second click — confirmed
      btn.removeAttribute("data-confirm");
      btn.disabled = true;
      btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:13px;pointer-events:none;animation:spin 0.6s linear infinite">progress_activity</span>';

      TraceClient.deleteEntity(entityId).then(function (res) {
        if (res && res.status === "deleted") {
          if (window.TraceToast) TraceToast.show("Entity " + entityId + " deleted.");
          // Remove from local array and re-render
          _allEntities = _allEntities.filter(function (e) { return e.entity_id !== entityId; });
          applyCurrentFilters();
          updateFooter(_allEntities.length, _allEntities.length);
          var statEl = document.getElementById("db-stat-entities");
          if (statEl) statEl.textContent = String(_allEntities.length);
        } else {
          if (window.TraceToast) TraceToast.show("Failed to delete entity.", "error");
          btn.disabled = false;
          btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:13px;pointer-events:none">delete</span>';
        }
      }).catch(function () {
        if (window.TraceToast) TraceToast.show("Network error during delete.", "error");
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:13px;pointer-events:none">delete</span>';
      });
    });
  }

  function wireRefresh() {
    var executeBtn = $("db-execute");
    if (executeBtn) {
      executeBtn.addEventListener("click", function () {
        setRefreshButtonLoading(true);
        loadEntities();
        loadHealth();
      });
    }

    var limitEl = $("db-limit");
    if (limitEl) {
      limitEl.addEventListener("change", function () {
        setRefreshButtonLoading(true);
        loadEntities();
      });
    }
  }

  function wireInlineFilters() {
    var typeSelect = $("filter-type");
    var statusSelect = $("filter-status");
    var clearBtn = $("filter-clear");

    if (typeSelect) {
      typeSelect.addEventListener("change", function () {
        applyCurrentFilters();
      });
    }

    if (statusSelect) {
      statusSelect.addEventListener("change", function () {
        applyCurrentFilters();
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        if (typeSelect) typeSelect.value = "";
        if (statusSelect) statusSelect.value = "";
        applyCurrentFilters();
      });
    }
  }

  function applyFilters(typeFilter, statusFilter) {
    var filtered = _allEntities.filter(function (ent) {
      if (typeFilter && (ent.type || "").toLowerCase() !== typeFilter) {
        return false;
      }
      if (statusFilter && (ent.status || "").toLowerCase() !== statusFilter) {
        return false;
      }
      return true;
    });
    renderTable(filtered);
    updateFooter(filtered.length, _allEntities.length);
  }

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);
    wireSearch();
    wireRefresh();
    wireInlineFilters();
    wireQuickDelete();

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
