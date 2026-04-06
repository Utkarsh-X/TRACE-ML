(function () {
  "use strict";
  var TA = window.TraceApi;
  if (!TA) {
    return;
  }
  var api = TA.buildApiBase();
  var sel = document.getElementById("incident-select");
  var statusEl = document.getElementById("incident-load-status");
  var idLabel = document.getElementById("incident-id-label");
  var titleEl = document.getElementById("incident-title");
  var statusLine = document.getElementById("incident-status");
  var entityName = document.getElementById("incident-entity-name");
  var entityId = document.getElementById("incident-entity-id");
  var alertsRoot = document.getElementById("incident-alerts-root");
  var timelineRoot = document.getElementById("incident-timeline-root");
  var sevSel = document.getElementById("incident-severity-select");
  var applyBtn = document.getElementById("incident-apply-severity");
  var closeBtn = document.getElementById("incident-close-btn");
  var actionStatus = document.getElementById("incident-action-status");

  var currentId = "";

  function renderAlerts(rows) {
    if (!alertsRoot) {
      return;
    }
    if (!rows || !rows.length) {
      alertsRoot.innerHTML = '<p class="text-[0.72rem] text-outline font-mono">No alerts</p>';
      return;
    }
    alertsRoot.innerHTML = rows
      .slice(0, 12)
      .map(function (a) {
        var typ = typeof a.type === "object" && a.type ? a.type.value || a.type : a.type;
        return (
          '<div class="flex justify-between items-center text-[0.6875rem] font-mono p-2 hover:bg-surface-container-high transition-colors">' +
          '<span class="text-outline">' +
          TA.escapeHtml(TA.formatTime(a.timestamp_utc)) +
          "</span>" +
          '<span class="text-white">' +
          TA.escapeHtml(String(typ || "")) +
          "</span>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderTimeline(items) {
    if (!timelineRoot) {
      return;
    }
    if (!items || !items.length) {
      timelineRoot.innerHTML = '<p class="text-[0.72rem] text-outline font-mono text-center">No timeline items</p>';
      return;
    }
    timelineRoot.innerHTML = items
      .map(function (item) {
        var kind = typeof item.kind === "object" && item.kind ? item.kind.value || item.kind : item.kind;
        return (
          '<div class="flex flex-col items-center">' +
          '<div class="z-10 bg-surface border border-outline-variant p-2 mb-4">' +
          '<span class="font-mono text-[0.65rem] text-primary">' +
          TA.escapeHtml(String(kind || "?")) +
          "</span>" +
          "</div>" +
          '<div class="text-center max-w-xs">' +
          '<span class="font-label text-[0.65rem] text-outline block mb-2">' +
          TA.escapeHtml(TA.formatDateTime(item.timestamp_utc)) +
          "</span>" +
          '<p class="font-mono text-sm text-on-surface-variant">' +
          TA.escapeHtml(item.title || item.summary || "") +
          "</p>" +
          "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function applyDetail(d) {
    var inc = d.incident || {};
    var ent = d.entity || {};
    currentId = inc.incident_id || "";
    var st = typeof inc.status === "object" && inc.status ? inc.status.value || inc.status : inc.status;
    var sev = typeof inc.severity === "object" && inc.severity ? inc.severity.value || inc.severity : inc.severity;
    if (idLabel) {
      idLabel.textContent = "Incident ID: " + (currentId || "—");
    }
    if (titleEl) {
      titleEl.textContent = inc.summary || currentId || "—";
    }
    if (statusLine) {
      statusLine.textContent = String(st || "—") + " // sev=" + String(sev || "");
    }
    if (entityName) {
      entityName.textContent = ent.name || "—";
    }
    if (entityId) {
      entityId.textContent = ent.entity_id || "";
    }
    if (sevSel && sev) {
      sevSel.value = String(sev).toLowerCase();
    }
    renderAlerts(d.alerts || []);
    renderTimeline(d.timeline || []);
  }

  function loadDetail(id) {
    if (!id) {
      return;
    }
    if (statusEl) {
      statusEl.textContent = "loading…";
    }
    TA.fetchJson(api + "/api/v1/incidents/" + encodeURIComponent(id))
      .then(function (d) {
        applyDetail(d);
        if (statusEl) {
          statusEl.textContent = "synced";
        }
        if (actionStatus) {
          actionStatus.textContent = "";
        }
      })
      .catch(function (e) {
        if (statusEl) {
          statusEl.textContent = e.message || "error";
        }
      });
  }

  function init() {
    TA.fetchJson(api + "/api/v1/incidents?limit=200&status=open")
      .then(function (rows) {
        if (!sel) {
          return;
        }
        sel.innerHTML = '<option value="">— open incidents —</option>';
        rows.forEach(function (row) {
          var o = document.createElement("option");
          o.value = row.incident_id;
          o.textContent = row.incident_id + " — " + (row.summary || "").slice(0, 48);
          sel.appendChild(o);
        });
        var params = new URLSearchParams(window.location.search);
        var pre = params.get("incident_id") || "";
        if (pre) {
          sel.value = pre;
          loadDetail(pre);
        }
        sel.addEventListener("change", function () {
          var id = sel.value;
          if (id) {
            history.replaceState(null, "", "?incident_id=" + encodeURIComponent(id));
            loadDetail(id);
          }
        });
      })
      .catch(function (e) {
        if (statusEl) {
          statusEl.textContent = e.message || "list failed";
        }
      });

    if (applyBtn) {
      applyBtn.addEventListener("click", function () {
        if (!currentId || !sevSel) {
          return;
        }
        var sev = sevSel.value;
        if (actionStatus) {
          actionStatus.textContent = "applying…";
        }
        TA.fetchJsonMethod(
          api + "/api/v1/incidents/" + encodeURIComponent(currentId) + "/severity",
          "PATCH",
          { severity: sev }
        )
          .then(function (d) {
            applyDetail(d);
            if (actionStatus) {
              actionStatus.textContent = "severity updated";
            }
          })
          .catch(function (e) {
            if (actionStatus) {
              actionStatus.textContent = e.message || "failed";
            }
          });
      });
    }

    if (closeBtn) {
      closeBtn.addEventListener("click", function () {
        if (!currentId) {
          return;
        }
        if (!window.confirm("Close incident " + currentId + "?")) {
          return;
        }
        if (actionStatus) {
          actionStatus.textContent = "closing…";
        }
        TA.fetchJsonMethod(api + "/api/v1/incidents/" + encodeURIComponent(currentId) + "/close", "POST")
          .then(function (d) {
            applyDetail(d);
            if (actionStatus) {
              actionStatus.textContent = "closed";
            }
            return TA.fetchJson(api + "/api/v1/incidents?limit=200&status=open");
          })
          .then(function (rows) {
            if (!sel) {
              return;
            }
            sel.innerHTML = '<option value="">— open incidents —</option>';
            rows.forEach(function (row) {
              var o = document.createElement("option");
              o.value = row.incident_id;
              o.textContent = row.incident_id + " — " + (row.summary || "").slice(0, 48);
              sel.appendChild(o);
            });
          })
          .catch(function (e) {
            if (actionStatus) {
              actionStatus.textContent = e.message || "failed";
            }
          });
      });
    }
  }

  init();
})();
