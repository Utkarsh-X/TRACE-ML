(function () {
  "use strict";

  var TA = window.TraceApi;
  if (!TA) {
    console.error("trace_api.js must load before live_ops.js");
    return;
  }

  var escapeHtml = TA.escapeHtml;
  var formatTime = TA.formatTime;
  var fetchJson = TA.fetchJson;
  var resolveMockupUrl = TA.resolveMockupUrl;

  function byId(id) {
    return document.getElementById(id);
  }

  function severityClass(level) {
    var text = String(level || "").toLowerCase();
    if (text === "high") {
      return "text-error";
    }
    if (text === "medium") {
      return "text-primary";
    }
    return "text-outline";
  }

  var apiBase = TA.buildApiBase();
  var endpoints = {
    snapshot: apiBase + "/api/v1/live/snapshot",
    timeline: apiBase + "/api/v1/timeline?limit=200",
    stream: apiBase + "/api/v1/events/stream?backfill=20",
    overlay: apiBase + "/api/v1/live/overlay",
  };

  var dom = {
    clock: byId("utc-clock"),
    protocol: byId("system-protocol-status"),
    activeEntities: byId("active-entities-list"),
    timelineList: byId("timeline-list"),
    timelineStatus: byId("timeline-status"),
    alertsCount: byId("alerts-count"),
    alertsList: byId("alerts-list"),
    incidentsList: byId("incidents-list"),
    streamLog: byId("stream-log"),
    systemHealth: byId("system-health"),
    mainCameraFeed: byId("main-camera-feed"),
    mainCameraPlaceholder: byId("main-camera-placeholder"),
    cameraEnableBtn: byId("camera-enable-btn"),
    cameraStatus: byId("camera-status"),
    overlayCanvas: byId("detection-overlay-canvas"),
  };

  if (!dom.mainCameraFeed) {
    dom.mainCameraFeed = document.querySelector("div.aspect-video.relative.bg-surface-container-lowest.group");
  }
  if (!dom.mainCameraPlaceholder && dom.mainCameraFeed) {
    dom.mainCameraPlaceholder = dom.mainCameraFeed.querySelector("img");
  }

  var state = {
    snapshot: null,
    timeline: [],
    timelineAll: [],
    timelineWindow: "7d",
    streamLines: [],
    eventSource: null,
    refreshTimer: null,
    streamConnected: false,
    cameraStream: null,
    overlayTimer: null,
  };

  function entityExplorerUrl(entityId) {
    var base = resolveMockupUrl("../entity_explorer_clean_canvas/code.html");
    if (!entityId) {
      return base;
    }
    return base + (base.indexOf("?") >= 0 ? "&" : "?") + "entity_id=" + encodeURIComponent(entityId);
  }

  function incidentsUrl(incidentId) {
    var base = resolveMockupUrl("../incidents_forensic_detail/code.html");
    if (!incidentId) {
      return base;
    }
    return base + (base.indexOf("?") >= 0 ? "&" : "?") + "incident_id=" + encodeURIComponent(incidentId);
  }

  function historyUrl(entityId) {
    var base = resolveMockupUrl("../global_history_timeline_forensic_canvas/code.html");
    if (!entityId) {
      return base;
    }
    return base + (base.indexOf("?") >= 0 ? "&" : "?") + "entity_id=" + encodeURIComponent(entityId);
  }

  function wireCrossLinks() {
    document.querySelectorAll("[data-mockup]").forEach(function (el) {
      var rel = el.getAttribute("data-mockup");
      if (rel) {
        el.setAttribute("href", resolveMockupUrl(rel));
      }
    });
  }

  function filterTimelineByWindow(items, win) {
    if (!items || !items.length) {
      return [];
    }
    if (win === "all") {
      return items.slice();
    }
    var now = Date.now();
    var ms = win === "24h" ? 864e5 : win === "7d" ? 7 * 864e5 : 30 * 864e5;
    return items.filter(function (item) {
      var t = new Date(item.timestamp_utc).getTime();
      return !Number.isNaN(t) && now - t <= ms;
    });
  }

  function applyTimelineFilter() {
    state.timeline = filterTimelineByWindow(state.timelineAll, state.timelineWindow);
    document.querySelectorAll(".timeline-filter").forEach(function (btn) {
      var w = btn.getAttribute("data-window");
      var on = w === state.timelineWindow;
      btn.className =
        "timeline-filter px-2 py-0.5 text-[0.6rem] font-mono border " +
        (on ? "border-white text-white" : "border-transparent text-neutral-500 hover:text-white");
    });
    renderTimeline();
  }

  function openDrawer(title, innerHtml) {
    var overlay = byId("panel-overlay");
    var backdrop = byId("panel-backdrop");
    var drawer = byId("panel-drawer");
    var tEl = byId("panel-title");
    var bEl = byId("panel-body");
    if (!overlay || !backdrop || !drawer || !tEl || !bEl) {
      return;
    }
    tEl.textContent = title;
    bEl.innerHTML = innerHtml;
    backdrop.classList.remove("hidden", "opacity-0");
    backdrop.classList.add("opacity-100");
    drawer.classList.remove("translate-x-full");
    overlay.classList.remove("pointer-events-none");
    overlay.setAttribute("aria-hidden", "false");
  }

  function closeDrawer() {
    var overlay = byId("panel-overlay");
    var backdrop = byId("panel-backdrop");
    var drawer = byId("panel-drawer");
    if (!overlay || !backdrop || !drawer) {
      return;
    }
    backdrop.classList.add("hidden", "opacity-0");
    backdrop.classList.remove("opacity-100");
    drawer.classList.add("translate-x-full");
    overlay.classList.add("pointer-events-none");
    overlay.setAttribute("aria-hidden", "true");
  }

  function wirePanels() {
    var notif = byId("nav-btn-notifications");
    var sett = byId("nav-btn-settings");
    var help = byId("nav-btn-help");
    var logout = byId("nav-btn-logout");
    var backdrop = byId("panel-backdrop");
    var closeBtn = byId("panel-close");

    function alertsPanelHtml() {
      var list = (state.snapshot && state.snapshot.recent_alerts) || [];
      if (!list.length) {
        return "<p>No alerts in current snapshot.</p>";
      }
      return (
        "<ul class='space-y-3'>" +
        list
          .slice(0, 20)
          .map(function (a) {
            var typ = typeof a.type === "object" && a.type ? a.type.value || a.type : a.type;
            return (
              "<li class='border-b border-outline-variant/30 pb-2'>" +
              "<span class='text-primary'>" +
              escapeHtml(String(typ || "")) +
              "</span> " +
              "<span class='text-outline'>" +
              escapeHtml(formatTime(a.timestamp_utc)) +
              "</span>" +
              "<p class='mt-1'>" +
              escapeHtml(a.reason || "") +
              "</p>" +
              "<p class='text-[0.65rem] mt-1'><a class='text-primary underline' href='" +
              entityExplorerUrl(a.entity_id) +
              "'>entity " +
              escapeHtml(a.entity_id || "") +
              "</a></p>" +
              "</li>"
            );
          })
          .join("") +
        "</ul>"
      );
    }

    function settingsPanelHtml() {
      return (
        "<p class='mb-2'>API base in use:</p>" +
        "<p class='text-primary break-all mb-4'>" +
        escapeHtml(apiBase) +
        "</p>" +
        "<p class='mb-2'>Override with query param:</p>" +
        "<code class='text-xs block bg-surface-container-high p-2 border border-outline-variant'>?api=http://127.0.0.1:8080</code>" +
        "<p class='mt-4 text-outline text-xs'>Serve from repo <code class='text-primary'>mockup-code</code> so navigation links resolve.</p>"
      );
    }

    function helpPanelHtml() {
      return (
        "<ul class='list-disc pl-4 space-y-2 text-xs'>" +
        "<li><strong>Refresh data</strong> — reloads snapshot and timeline from the API.</li>" +
        "<li><strong>Timeline</strong> — All / 24H / 7D / 30D filters client-side.</li>" +
        "<li><strong>Entities / Incidents</strong> — click a row to open the detail workspace.</li>" +
        "<li><strong>Notifications</strong> — quick view of recent alerts.</li>" +
        "<li><strong>Disconnect</strong> — closes the SSE stream (use Refresh to reconnect).</li>" +
        "<li><strong>Live recognition</strong> — run the service with <code class='text-primary'>--live</code> for webcam pipeline + SSE.</li>" +
        "</ul>"
      );
    }

    if (notif) {
      notif.addEventListener("click", function () {
        openDrawer("Recent alerts", alertsPanelHtml());
      });
    }
    if (sett) {
      sett.addEventListener("click", function () {
        openDrawer("Connection", settingsPanelHtml());
      });
    }
    if (help) {
      help.addEventListener("click", function () {
        openDrawer("Live Ops help", helpPanelHtml());
      });
    }
    if (logout) {
      logout.addEventListener("click", function () {
        if (state.eventSource) {
          state.eventSource.close();
          state.eventSource = null;
          state.streamConnected = false;
          pushStreamLine("stream disconnected (manual)", "normal");
          setProtocolStatus("System Protocol: Stream Off", true);
        }
      });
    }
    if (backdrop) {
      backdrop.addEventListener("click", closeDrawer);
    }
    if (closeBtn) {
      closeBtn.addEventListener("click", closeDrawer);
    }
  }

  function wireToolbar() {
    var refresh = byId("btn-refresh-all");
    if (refresh) {
      refresh.addEventListener("click", function () {
        pushStreamLine("manual refresh", "success");
        if (!state.eventSource) {
          connectStream();
        }
        fetchSnapshot();
        fetchTimeline();
      });
    }
    var clearTerm = byId("btn-terminal-clear");
    if (clearTerm) {
      clearTerm.addEventListener("click", function () {
        state.streamLines = [];
        renderStreamLog();
      });
    }
    document.querySelectorAll(".timeline-filter").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var w = btn.getAttribute("data-window") || "all";
        state.timelineWindow = w;
        applyTimelineFilter();
      });
    });
  }

  function setProtocolStatus(text, isError) {
    if (!dom.protocol) {
      return;
    }
    dom.protocol.textContent = text;
    dom.protocol.className =
      "text-[0.6875rem] font-mono uppercase tracking-widest " + (isError ? "text-error" : "text-outline");
  }

  function pushStreamLine(text, level) {
    state.streamLines.unshift({
      text: String(text || ""),
      level: level || "normal",
    });
    if (state.streamLines.length > 40) {
      state.streamLines.length = 40;
    }
    renderStreamLog();
  }

  function renderStreamLog() {
    if (!dom.streamLog) {
      return;
    }
    if (!state.streamLines.length) {
      dom.streamLog.innerHTML = '<p><span class="text-primary">&gt;</span> waiting for stream...</p>';
      return;
    }
    var html = state.streamLines
      .map(function (line) {
        var color = "text-on-surface-variant/80";
        if (line.level === "error") {
          color = "text-error";
        }
        if (line.level === "success") {
          color = "text-primary";
        }
        return '<p class="' + color + '"><span class="text-primary">&gt;</span> ' + escapeHtml(line.text) + "</p>";
      })
      .join("");
    dom.streamLog.innerHTML = html;
  }

  function renderActiveEntities() {
    if (!dom.activeEntities) {
      return;
    }
    var list = (state.snapshot && state.snapshot.active_entities) || [];
    if (!list.length) {
      dom.activeEntities.innerHTML = '<p class="text-[0.72rem] text-outline font-mono">No active entities</p>';
      return;
    }
    dom.activeEntities.innerHTML = list
      .slice(0, 8)
      .map(function (entity) {
        var typ = typeof entity.type === "object" && entity.type !== null ? String(entity.type.value || entity.type) : String(entity.type || "");
        var eid = entity.entity_id || "";
        var href = entityExplorerUrl(eid);
        return (
          '<a class="block bg-surface-container-high px-3 py-2 hover:bg-surface-container-highest no-underline text-inherit border border-transparent hover:border-outline-variant transition-colors" href="' +
          href +
          '">' +
          '<div class="flex items-center justify-between gap-2">' +
          '<p class="text-[0.72rem] text-white font-mono">' +
          escapeHtml(eid || "N/A") +
          "</p>" +
          '<p class="text-[0.62rem] text-outline font-mono uppercase">' +
          escapeHtml(typ) +
          "</p>" +
          "</div>" +
          '<p class="text-[0.68rem] text-on-surface-variant">' +
          escapeHtml(entity.name || "Unknown") +
          "</p>" +
          '<p class="text-[0.6rem] text-outline font-mono">alerts:' +
          Number(entity.recent_alert_count || 0) +
          " incidents:" +
          Number(entity.open_incident_count || 0) +
          " · open profile →</p>" +
          "</a>"
        );
      })
      .join("");
  }

  function renderAlerts() {
    if (!dom.alertsList || !dom.alertsCount) {
      return;
    }
    var list = (state.snapshot && state.snapshot.recent_alerts) || [];
    var critical = list.filter(function (item) {
      return String(item.severity || "").toLowerCase() === "high";
    }).length;
    dom.alertsCount.textContent = "CRITICAL (" + critical + ")";

    if (!list.length) {
      dom.alertsList.innerHTML = '<p class="text-[0.72rem] text-outline font-mono">No recent alerts</p>';
      return;
    }

    dom.alertsList.innerHTML = list
      .slice(0, 6)
      .map(function (alert) {
        var sev = typeof alert.severity === "object" && alert.severity !== null ? alert.severity.value || alert.severity : alert.severity;
        var tone = severityClass(sev);
        var typ = typeof alert.type === "object" && alert.type !== null ? alert.type.value || alert.type : alert.type;
        var entL = entityExplorerUrl(alert.entity_id);
        return (
          '<div class="bg-surface-container-high p-4 relative group">' +
          '<div class="absolute left-0 top-0 bottom-0 w-0.5 bg-outline-variant"></div>' +
          '<div class="flex justify-between items-start mb-1">' +
          '<span class="text-[0.65rem] font-mono uppercase ' +
          tone +
          '">' +
          escapeHtml(String(typ || "ALERT")) +
          "</span>" +
          '<span class="text-[0.55rem] font-mono text-outline">' +
          escapeHtml(formatTime(alert.timestamp_utc)) +
          "</span>" +
          "</div>" +
          '<p class="text-[0.75rem] font-body text-on-surface-variant leading-relaxed">' +
          escapeHtml(alert.reason || "") +
          "</p>" +
          '<p class="text-[0.6rem] font-mono text-outline mt-2">' +
          "<a class='text-primary underline' href='" +
          entL +
          "'>entity:" +
          escapeHtml(alert.entity_id || "") +
          "</a> events:" +
          Number(alert.event_count || 0) +
          "</p>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderIncidents() {
    if (!dom.incidentsList) {
      return;
    }
    var list = (state.snapshot && state.snapshot.active_incidents) || [];
    if (!list.length) {
      dom.incidentsList.innerHTML = '<p class="text-[0.72rem] text-outline font-mono">No open incidents</p>';
      return;
    }
    dom.incidentsList.innerHTML = list
      .slice(0, 8)
      .map(function (incident) {
        var sev = typeof incident.severity === "object" && incident.severity !== null ? incident.severity.value || incident.severity : incident.severity;
        var tone = severityClass(sev);
        var iid = incident.incident_id || "";
        var href = incidentsUrl(iid);
        return (
          '<a class="flex gap-4 no-underline text-inherit hover:bg-surface-container-high/50 p-1 -m-1 rounded border border-transparent hover:border-outline-variant/40" href="' +
          href +
          '">' +
          '<span class="w-1.5 h-1.5 mt-1.5 flex-shrink-0 bg-primary"></span>' +
          "<div>" +
          '<p class="text-[0.8rem] font-medium text-white mb-1">' +
          escapeHtml(iid) +
          " →</p>" +
          '<p class="text-[0.7rem] leading-snug ' +
          tone +
          '">severity=' +
          escapeHtml(String(sev || "")) +
          " alerts=" +
          Number(incident.alert_count || 0) +
          "</p>" +
          '<p class="text-[0.7rem] text-outline leading-snug">' +
          escapeHtml(incident.summary || "No summary") +
          "</p>" +
          '<span class="text-[0.6rem] font-mono text-outline-variant mt-2 block uppercase">' +
          escapeHtml(formatTime(incident.last_seen_time)) +
          " // " +
          escapeHtml(incident.entity_id || "") +
          "</span>" +
          "</div>" +
          "</a>"
        );
      })
      .join("");
  }

  function renderSystemHealth() {
    if (!dom.systemHealth) {
      return;
    }
    var health = (state.snapshot && state.snapshot.system_health) || {};
    dom.systemHealth.innerHTML =
      "<p>entities: " +
      Number(health.active_entity_count || 0) +
      "</p>" +
      "<p>incidents: " +
      Number(health.open_incident_count || 0) +
      "</p>" +
      "<p>alerts: " +
      Number(health.recent_alert_count || 0) +
      "</p>" +
      "<p>detections: " +
      Number(health.total_detection_count || 0) +
      "</p>" +
      "<p>subscribers: " +
      Number(health.publisher_subscribers || 0) +
      "</p>";
  }

  function renderTimeline() {
    if (!dom.timelineList || !dom.timelineStatus) {
      return;
    }
    if (!state.timeline.length) {
      dom.timelineStatus.textContent = "empty";
      dom.timelineList.innerHTML = '<p class="text-[0.72rem] text-outline font-mono">No timeline items</p>';
      return;
    }
    dom.timelineStatus.textContent =
      state.timelineWindow.toUpperCase() + " · " + formatTime(new Date().toISOString());
    var items = state.timeline.slice(-40).reverse();
    dom.timelineList.innerHTML = items
      .map(function (item) {
        var kind = typeof item.kind === "object" && item.kind !== null ? item.kind.value || item.kind : item.kind;
        var entL = item.entity_id ? historyUrl(item.entity_id) : historyUrl();
        var incL = item.incident_id ? incidentsUrl(item.incident_id) : incidentsUrl();
        return (
          '<div class="bg-surface-container-high px-3 py-2">' +
          '<div class="flex items-center justify-between gap-2">' +
          '<p class="text-[0.66rem] font-mono text-outline uppercase">' +
          escapeHtml(String(kind || "item")) +
          "</p>" +
          '<p class="text-[0.62rem] font-mono text-outline">' +
          escapeHtml(formatTime(item.timestamp_utc)) +
          "</p>" +
          "</div>" +
          '<p class="text-[0.72rem] text-white">' +
          escapeHtml(item.title || "") +
          "</p>" +
          '<p class="text-[0.65rem] text-on-surface-variant">' +
          escapeHtml(item.summary || "") +
          "</p>" +
          '<p class="text-[0.6rem] font-mono text-outline mt-1">' +
          (item.entity_id
            ? "<a class='text-primary underline' href='" + entL + "'>entity:" + escapeHtml(item.entity_id) + "</a>"
            : "entity:-") +
          " " +
          (item.incident_id
            ? "<a class='text-primary underline' href='" + incL + "'>inc:" + escapeHtml(item.incident_id) + "</a>"
            : "incident:-") +
          "</p>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderSnapshot() {
    renderActiveEntities();
    renderAlerts();
    renderIncidents();
    renderSystemHealth();
  }

  function updateClock() {
    if (!dom.clock) {
      return;
    }
    dom.clock.textContent = "UTC " + formatTime(new Date().toISOString());
  }

  function scheduleRefresh(delayMs) {
    if (state.refreshTimer) {
      clearTimeout(state.refreshTimer);
    }
    state.refreshTimer = setTimeout(function () {
      state.refreshTimer = null;
      fetchSnapshot();
      fetchTimeline();
    }, delayMs);
  }

  function setCameraStatus(message, isError) {
    if (!dom.cameraStatus) {
      return;
    }
    dom.cameraStatus.textContent = "camera: " + String(message || "");
    dom.cameraStatus.className = "text-[0.58rem] font-mono " + (isError ? "text-error" : "text-outline");
  }

  function syncOverlayCanvasSize() {
    var canvas = dom.overlayCanvas;
    var feed = dom.mainCameraFeed;
    if (!canvas || !feed) {
      return;
    }
    var video = feed.querySelector("video");
    var rect = feed.getBoundingClientRect();
    var w = Math.max(1, Math.floor(rect.width));
    var h = Math.max(1, Math.floor(rect.height));
    if (video && video.videoWidth) {
      var ar = video.videoWidth / Math.max(1, video.videoHeight);
      var fw = w;
      var fh = Math.round(fw / ar);
      if (fh > h) {
        fh = h;
        fw = Math.round(fh * ar);
      }
      canvas.width = fw;
      canvas.height = fh;
    } else {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function drawOverlay(data) {
    var canvas = dom.overlayCanvas;
    if (!canvas) {
      return;
    }
    var ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    syncOverlayCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!data || !data.active || !data.boxes || !data.boxes.length) {
      return;
    }
    var i;
    for (i = 0; i < data.boxes.length; i++) {
      var b = data.boxes[i];
      var x = b.x * canvas.width;
      var y = b.y * canvas.height;
      var bw = b.w * canvas.width;
      var bh = b.h * canvas.height;
      var dec = String(b.decision || "");
      ctx.strokeStyle = dec === "accept" ? "#00e676" : dec === "review" ? "#ffca28" : "#9e9e9e";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, bw, bh);
      var label = String(b.label || "") + " " + Math.round(Number(b.confidence || 0)) + "%";
      ctx.font = "11px ui-monospace, JetBrains Mono, monospace";
      var tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(x, Math.max(0, y - 16), Math.min(canvas.width - x, tw + 8), 16);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x + 4, Math.max(12, y - 4));
    }
  }

  function pollOverlay() {
    fetchJson(endpoints.overlay)
      .then(drawOverlay)
      .catch(function () {
        if (dom.overlayCanvas) {
          var ctx = dom.overlayCanvas.getContext("2d");
          if (ctx) {
            syncOverlayCanvasSize();
            ctx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
          }
        }
      });
  }

  function startOverlayPoll() {
    if (state.overlayTimer) {
      clearInterval(state.overlayTimer);
    }
    state.overlayTimer = setInterval(pollOverlay, 250);
    pollOverlay();
  }

  async function requestCameraStream() {
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
    } catch (primaryError) {
      try {
        return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (fallbackError) {
        throw fallbackError || primaryError;
      }
    }
  }

  async function initLocalCamera() {
    pushStreamLine("camera init start", "normal");
    if (!dom.mainCameraFeed) {
      setCameraStatus("container not found", true);
      pushStreamLine("camera container missing in DOM", "error");
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraStatus("unsupported", true);
      pushStreamLine("camera unavailable: getUserMedia not supported", "error");
      return;
    }
    if (!window.isSecureContext) {
      setCameraStatus("blocked (non-secure origin)", true);
      pushStreamLine("camera blocked: open with http://localhost:8081 (secure context required)", "error");
      return;
    }
    try {
      setCameraStatus("requesting permission", false);
      var stream = await requestCameraStream();
      state.cameraStream = stream;
      var video = document.createElement("video");
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.className =
        "w-full h-full object-cover opacity-70 grayscale brightness-90 group-hover:opacity-80 transition-opacity";
      video.srcObject = stream;
      if (dom.mainCameraPlaceholder) {
        dom.mainCameraPlaceholder.style.display = "none";
      }
      dom.mainCameraFeed.prepend(video);
      video.addEventListener("loadedmetadata", function () {
        syncOverlayCanvasSize();
      });
      try {
        await video.play();
      } catch (playError) {
        /* keep stream */
      }
      setCameraStatus("connected", false);
      pushStreamLine("camera connected (local preview active)", "success");
      syncOverlayCanvasSize();
    } catch (error) {
      setCameraStatus(error.name || "failed", true);
      pushStreamLine("camera start failed: " + error.message, "error");
    }
  }

  async function fetchSnapshot() {
    try {
      var data = await fetchJson(endpoints.snapshot);
      state.snapshot = data;
      renderSnapshot();
      setProtocolStatus("System Protocol: Active", false);
    } catch (error) {
      setProtocolStatus("System Protocol: Snapshot Error", true);
      pushStreamLine("snapshot fetch failed: " + error.message, "error");
    }
  }

  async function fetchTimeline() {
    try {
      var data = await fetchJson(endpoints.timeline);
      state.timelineAll = Array.isArray(data) ? data : [];
      applyTimelineFilter();
    } catch (error) {
      if (dom.timelineStatus) {
        dom.timelineStatus.textContent = "error";
      }
      pushStreamLine("timeline fetch failed: " + error.message, "error");
    }
  }

  function describeEvent(topic, payload) {
    if (topic === "event") {
      return "entity=" + (payload.entity_id || "N/A") + " decision=" + (payload.decision || "N/A");
    }
    if (topic === "alert") {
      return "type=" + (payload.type || "N/A") + " severity=" + (payload.severity || "N/A");
    }
    if (topic === "incident") {
      return "incident=" + (payload.incident_id || "N/A") + " status=" + (payload.status || "N/A");
    }
    if (topic === "action") {
      return "action=" + (payload.action_type || "N/A") + " status=" + (payload.status || "N/A");
    }
    if (topic === "detection") {
      return "name=" + (payload.name || "Unknown") + " conf=" + Number(payload.confidence || 0).toFixed(2);
    }
    if (topic === "session.state") {
      return "fps=" + Number(payload.fps || 0).toFixed(1) + " tracks=" + Number(payload.active_tracks || 0);
    }
    return "topic update";
  }

  function handleStreamPayload(rawText, fallbackTopic) {
    var parsed;
    try {
      parsed = JSON.parse(rawText);
    } catch (error) {
      pushStreamLine("stream parse error: " + error.message, "error");
      return;
    }
    var topic = parsed.topic || fallbackTopic || "stream";
    var payload = parsed.payload || {};
    var timestamp = parsed.timestamp_utc || new Date().toISOString();
    pushStreamLine("[" + formatTime(timestamp) + "] " + topic + " " + describeEvent(topic, payload), "success");
    if (topic !== "session.state") {
      scheduleRefresh(400);
    } else {
      pollOverlay();
    }
  }

  function connectStream() {
    if (!("EventSource" in window)) {
      pushStreamLine("EventSource unsupported in this browser", "error");
      return;
    }
    if (state.eventSource) {
      state.eventSource.close();
    }
    var es = new EventSource(endpoints.stream);
    state.eventSource = es;

    es.onopen = function () {
      state.streamConnected = true;
      setProtocolStatus("System Protocol: Active", false);
      pushStreamLine("stream connected -> " + endpoints.stream, "success");
    };

    es.onerror = function () {
      if (state.streamConnected) {
        pushStreamLine("stream disconnected, retrying...", "error");
      }
      state.streamConnected = false;
      setProtocolStatus("System Protocol: Stream Reconnecting", true);
    };

    ["event", "alert", "incident", "action", "detection", "session.state"].forEach(function (topic) {
      es.addEventListener(topic, function (evt) {
        handleStreamPayload(evt.data, topic);
      });
    });

    es.onmessage = function (evt) {
      handleStreamPayload(evt.data, "stream");
    };
  }

  var pollIntervalId = null;

  function start() {
    wireCrossLinks();
    wirePanels();
    wireToolbar();
    updateClock();
    setInterval(updateClock, 1000);
    pushStreamLine("api base: " + apiBase, "success");
    fetchSnapshot();
    fetchTimeline();
    connectStream();
    startOverlayPoll();
    window.addEventListener("resize", function () {
      syncOverlayCanvasSize();
      pollOverlay();
    });
    if (dom.cameraEnableBtn) {
      dom.cameraEnableBtn.addEventListener("click", function () {
        initLocalCamera();
      });
    }
    pollIntervalId = setInterval(function () {
      fetchSnapshot();
      fetchTimeline();
    }, 15000);
  }

  window.addEventListener("beforeunload", function () {
    if (state.overlayTimer) {
      clearInterval(state.overlayTimer);
    }
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
    }
    if (!state.cameraStream) {
      return;
    }
    state.cameraStream.getTracks().forEach(function (track) {
      try {
        track.stop();
      } catch (error) {
        return;
      }
    });
  });

  start();
})();
