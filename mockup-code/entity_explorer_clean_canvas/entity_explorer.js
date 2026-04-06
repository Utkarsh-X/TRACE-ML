(function () {
  "use strict";
  var TA = window.TraceApi;
  if (!TA) {
    return;
  }
  var api = TA.buildApiBase();
  var sel = document.getElementById("entity-select");
  var statusEl = document.getElementById("entity-load-status");
  var labelEl = document.getElementById("entity-profile-label");
  var nameEl = document.getElementById("entity-display-name");
  var stEl = document.getElementById("entity-status");
  var typeEl = document.getElementById("entity-type");
  var clockEl = document.getElementById("entity-clock");
  var timelineRoot = document.getElementById("entity-timeline-root");
  var incidentsRoot = document.getElementById("entity-incidents-root");
  var incidentsLink = document.getElementById("entity-open-incidents-link");

  function tick() {
    if (clockEl) {
      clockEl.textContent = TA.formatDateTime(new Date().toISOString());
    }
  }

  function renderTimeline(items) {
    if (!timelineRoot) {
      return;
    }
    if (!items || !items.length) {
      timelineRoot.innerHTML = '<p class="text-[0.72rem] text-outline font-mono">No timeline items</p>';
      return;
    }
    timelineRoot.innerHTML = items
      .slice(-20)
      .reverse()
      .map(function (item) {
        var kind = typeof item.kind === "object" && item.kind ? item.kind.value || item.kind : item.kind;
        return (
          '<div class="flex group cursor-pointer hover:bg-surface-container-low transition-all duration-150 p-4 -mx-4">' +
          '<div class="w-24 shrink-0">' +
          '<span class="font-label text-[0.6rem] text-neutral-500">' +
          TA.escapeHtml(TA.formatTime(item.timestamp_utc)) +
          "</span>" +
          "</div>" +
          '<div class="flex-1 flex flex-col">' +
          '<div class="flex justify-between items-start mb-2">' +
          '<span class="font-body text-sm text-white uppercase font-medium">' +
          TA.escapeHtml(item.title || kind || "event") +
          "</span>" +
          '<span class="font-label text-[0.6rem] text-white tracking-widest uppercase">' +
          TA.escapeHtml(String(kind || "")) +
          "</span>" +
          "</div>" +
          '<p class="font-body text-xs text-neutral-400 max-w-lg">' +
          TA.escapeHtml(item.summary || "") +
          "</p>" +
          "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderIncidents(list) {
    if (!incidentsRoot) {
      return;
    }
    if (!list || !list.length) {
      incidentsRoot.innerHTML = '<p class="text-[0.72rem] text-outline font-mono">No linked incidents</p>';
      return;
    }
    incidentsRoot.innerHTML = list
      .map(function (inc) {
        var sev =
          typeof inc.severity === "object" && inc.severity ? inc.severity.value || inc.severity : inc.severity;
        return (
          '<div>' +
          '<span class="font-label text-[0.55rem] text-neutral-500 uppercase block mb-2">' +
          TA.escapeHtml(inc.incident_id || "") +
          "</span>" +
          '<span class="font-body text-sm text-white block mb-1">' +
          TA.escapeHtml(inc.summary || "") +
          "</span>" +
          '<span class="font-label text-[0.6rem] text-outline uppercase tracking-widest">Severity: ' +
          TA.escapeHtml(String(sev || "")) +
          "</span>" +
          "</div>"
        );
      })
      .join("");
  }

  function loadProfile(entityId) {
    if (!entityId) {
      return;
    }
    if (statusEl) {
      statusEl.textContent = "loading…";
    }
    TA.fetchJson(api + "/api/v1/entities/" + encodeURIComponent(entityId) + "/profile")
      .then(function (profile) {
        var ent = profile.entity || {};
        var typ = typeof ent.type === "object" && ent.type ? ent.type.value || ent.type : ent.type;
        if (labelEl) {
          labelEl.textContent = "Entity Profile // ID: " + (ent.entity_id || "—");
        }
        if (nameEl) {
          nameEl.textContent = (ent.name || "Unknown").replace(/\s/g, "_").toUpperCase();
        }
        if (stEl) {
          stEl.textContent = String(ent.status || "—");
        }
        if (typeEl) {
          typeEl.textContent = String(typ || "—");
        }
        renderTimeline(profile.timeline || []);
        renderIncidents(profile.incidents || []);
        if (statusEl) {
          statusEl.textContent = "synced";
        }
        if (incidentsLink) {
          if (profile.incidents && profile.incidents[0]) {
            incidentsLink.href =
              "../incidents_forensic_detail/code.html?incident_id=" +
              encodeURIComponent(profile.incidents[0].incident_id || "");
          } else {
            incidentsLink.href = "../incidents_forensic_detail/code.html";
          }
        }
      })
      .catch(function (e) {
        if (statusEl) {
          statusEl.textContent = e.message || "error";
        }
      });
  }

  function init() {
    tick();
    setInterval(tick, 1000);
    TA.fetchJson(api + "/api/v1/entities?limit=200")
      .then(function (rows) {
        if (!sel) {
          return;
        }
        sel.innerHTML = '<option value="">— select entity —</option>';
        rows.forEach(function (row) {
          var o = document.createElement("option");
          o.value = row.entity_id;
          o.textContent = row.entity_id + " — " + (row.name || "");
          sel.appendChild(o);
        });
        var params = new URLSearchParams(window.location.search);
        var pre = params.get("entity_id") || "";
        if (pre) {
          sel.value = pre;
          loadProfile(pre);
        }
        sel.addEventListener("change", function () {
          var id = sel.value;
          if (id) {
            history.replaceState(null, "", "?entity_id=" + encodeURIComponent(id));
            loadProfile(id);
          }
        });
      })
      .catch(function (e) {
        if (statusEl) {
          statusEl.textContent = e.message || "entities fetch failed";
        }
      });
  }

  init();
})();
