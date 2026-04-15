/**
 * Settings Page Controller
 *
 * - Loads system info (GET /) and health (GET /health)
 * - Loads and manages live configuration (GET/PATCH /api/v1/config)
 * - Handles sidebar navigation and section switching
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var currentConfig = null;

  function loadSystemInfo() {
    TraceClient.probe().then(function (info) {
      if (!info) return;

      // Update version info in settings nav
      var versionEls = document.querySelectorAll("section:first-of-type .font-mono");
      versionEls.forEach(function (el) {
        var text = el.textContent.trim();
        if (text === "3.0.0" || text.match(/^\d+\.\d+\.\d+$/)) {
          el.textContent = info.version || "—";
        }
        if (text === "demo") {
          el.textContent = info.environment || "—";
        }
      });
    });
  }

  function loadHealth() {
    TraceClient.health().then(function (health) {
      if (!health) return;

      // Update health check items
      var headers = document.querySelectorAll("h3.section-header");
      var healthHeader = Array.from(headers).find(function(h) { 
        return h.textContent.trim() === "Health Checks"; 
      });
      var checksRoot = healthHeader ? healthHeader.nextElementSibling : null;
      
      if (!checksRoot) return;

      // Build health checks from real data
      var checks = [];
      checks.push({
        name: "LanceDB Vector Store",
        detail: (health.total_detection_count || 0) + " detections stored",
        ok: health.status === "ok",
      });
      checks.push({
        name: "Active Entities",
        detail: (health.active_entity_count || 0) + " entities tracked",
        ok: health.active_entity_count >= 0,
      });
      checks.push({
        name: "Open Incidents",
        detail: (health.open_incident_count || 0) + " active",
        ok: true,
      });
      checks.push({
        name: "Event Stream",
        detail: (health.publisher_subscribers || 0) + " subscribers",
        ok: true,
      });
      checks.push({
        name: "Latest Event",
        detail: health.latest_event_at ? TraceClient.formatDateTime(health.latest_event_at) : "none",
        ok: !!health.latest_event_at,
      });

      checksRoot.innerHTML = checks.map(function (c) {
        return TraceRender.healthCheck(c.name, c.detail, c.ok);
      }).join("");
    });
  }

  function renderConfig() {
    if (!currentConfig) return;

    // Find the config container
    var headers = document.querySelectorAll("h3.section-header");
    var configHeader = Array.from(headers).find(function(h) { 
      return h.textContent.trim() === "Active Configuration"; 
    });
    var configRoot = configHeader ? configHeader.nextElementSibling.nextElementSibling : null;
    if (!configRoot) return;

    var html = "";

    // 1. Recognition
    var rec = currentConfig.recognition || {};
    html += `
      <div id="section-recognition" class="bg-surface-container p-4 mb-3">
        <h4 class="font-mono text-[0.65rem] text-primary uppercase tracking-widest mb-3">Recognition Engine</h4>
        <div class="grid grid-cols-2 gap-x-8 gap-y-4">
          ${renderSlider("recognition.similarity_threshold", "Similarity Threshold", rec.similarity_threshold, 0, 1, 0.05)}
          ${renderSlider("recognition.accept_threshold", "Accept Threshold", rec.accept_threshold, 0, 1, 0.05)}
          ${renderSlider("recognition.review_threshold", "Review Threshold", rec.review_threshold, 0, 1, 0.05)}
          ${renderSlider("recognition.top_k", "Top K Results", rec.top_k, 1, 20, 1)}
          <div class="flex justify-between items-center">
            <span class="font-mono text-[0.65rem] text-outline">Model</span>
            <span class="font-mono text-[0.7rem] text-on-surface-variant">${rec.model_name || 'buffalo_sc'}</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="font-mono text-[0.65rem] text-outline">Provider</span>
            <span class="font-mono text-[0.7rem] text-on-surface-variant">${rec.provider || 'CPU'}</span>
          </div>
        </div>
      </div>
    `;

    // 2. Rules Engine
    var rules = currentConfig.rules || {};
    html += `
      <div id="section-rules" class="bg-surface-container p-4 mb-3">
        <h4 class="font-mono text-[0.65rem] text-primary uppercase tracking-widest mb-3">Intelligence Rules</h4>
        <div class="grid grid-cols-2 gap-x-8 gap-y-4">
          ${renderSlider("rules.cooldown_sec", "Global Cooldown (s)", rules.cooldown_sec, 0, 60, 1)}
          ${renderSlider("rules.reappearance.window_sec", "Reappearance Window (s)", rules.reappearance.window_sec, 1, 300, 5)}
          ${renderSlider("rules.reappearance.min_events", "Min Events (Reapp)", rules.reappearance.min_events, 1, 10, 1)}
          ${renderSlider("rules.unknown.window_sec", "Unknown Window (s)", rules.unknown.window_sec, 1, 300, 5)}
          ${renderSlider("rules.unknown.min_events", "Min Events (Unk)", rules.unknown.min_events, 1, 10, 1)}
          ${renderSlider("rules.instability.std_threshold", "Instability Threshold", rules.instability.std_threshold, 0, 0.5, 0.01)}
        </div>
      </div>
    `;

    // 3. Action Policy
    var actions = currentConfig.actions || {};
    html += `
      <div id="section-actions" class="bg-surface-container p-4 mb-3">
        <h4 class="font-mono text-[0.65rem] text-primary uppercase tracking-widest mb-3">Action Policy</h4>
        <div class="space-y-4">
          <div class="flex justify-between items-center">
            <span class="font-mono text-[0.65rem] text-outline uppercase">Actions Enabled</span>
            ${renderToggle("actions.enabled", actions.enabled)}
          </div>
          ${renderSlider("actions.cooldown_sec", "Action Cooldown (s)", actions.cooldown_sec, 0, 120, 5)}
          
          <div class="grid grid-cols-3 gap-4 text-center mt-4">
            <div>
              <span class="stat-label block mb-2">On Create — Low</span>
              <span class="font-mono text-[0.7rem] text-outline">${(actions.on_create.low || []).join(", ") || "—"}</span>
            </div>
            <div>
              <span class="stat-label block mb-2">On Create — Med</span>
              <span class="font-mono text-[0.7rem] text-on-surface-variant">${(actions.on_create.medium || []).join(", ") || "—"}</span>
            </div>
            <div>
              <span class="stat-label block mb-2">On Create — High</span>
              <span class="font-mono text-[0.7rem] text-primary">${(actions.on_create.high || []).join(", ") || "—"}</span>
            </div>
          </div>
        </div>
      </div>
    `;

    // 4. Camera & Storage (Read only)
    html += `
      <div id="section-camera" class="bg-surface-container p-4 mb-3">
        <h4 class="font-mono text-[0.65rem] text-primary uppercase tracking-widest mb-3">Hardware & Storage</h4>
        <div class="grid grid-cols-2 gap-x-8 gap-y-2">
          <div class="flex justify-between">
            <span class="font-mono text-[0.65rem] text-outline">Camera</span>
            <span class="font-mono text-[0.7rem] text-on-surface-variant">Device ${currentConfig.camera.device_index}</span>
          </div>
          <div class="flex justify-between">
            <span class="font-mono text-[0.65rem] text-outline">Resolution</span>
            <span class="font-mono text-[0.7rem] text-on-surface-variant">${currentConfig.camera.width}x${currentConfig.camera.height} @ ${currentConfig.camera.fps}fps</span>
          </div>
          <div id="section-storage" class="flex justify-between col-span-2 border-t border-outline-variant/10 mt-2 pt-2">
            <span class="font-mono text-[0.65rem] text-outline">Storage Root</span>
            <span class="font-mono text-[0.7rem] text-on-surface-variant truncate ml-4">${currentConfig.store.root}</span>
          </div>
        </div>
      </div>
    `;

    configRoot.innerHTML = html;

    // Attach event listeners
    configRoot.querySelectorAll("input[type=range], input[type=checkbox]").forEach(function(input) {
      input.addEventListener("change", handleConfigChange);
    });
  }

  function renderSlider(key, label, value, min, max, step) {
    return `
      <div>
        <div class="flex justify-between mb-1">
          <span class="font-mono text-[0.65rem] text-outline">${label}</span>
          <span class="font-mono text-[0.7rem] text-primary" id="val-${key}">${value}</span>
        </div>
        <input type="range" data-key="${key}" min="${min}" max="${max}" step="${step}" value="${value}" 
               class="w-full h-1 bg-surface-high rounded-lg appearance-none cursor-pointer accent-primary"
               oninput="document.getElementById('val-${key}').textContent = this.value">
      </div>
    `;
  }

  function renderToggle(key, checked) {
    return `
      <label class="relative inline-flex items-center cursor-pointer">
        <input type="checkbox" data-key="${key}" class="sr-only peer" ${checked ? 'checked' : ''}>
        <div class="w-9 h-5 bg-surface-high peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
      </label>
    `;
  }

  function handleConfigChange(e) {
    var key = e.target.getAttribute("data-key");
    var value = e.target.type === "checkbox" ? e.target.checked : parseFloat(e.target.value);
    
    // Build update payload
    var parts = key.split(".");
    var payload = {};
    var current = payload;
    for (var i = 0; i < parts.length - 1; i++) {
      current[parts[i]] = {};
      current = current[parts[i]];
    }
    current[parts[parts.length - 1]] = value;

    TraceClient.updateConfig(payload).then(function(newCfg) {
      if (newCfg) {
        currentConfig = newCfg;
        // Optional: show toast/notification
        console.log("Config updated:", key, value);
      }
    });
  }

  function initSidebar() {
    var navItems = document.querySelectorAll("section:first-of-type .space-y-1 div");
    var sections = {
      "System Health": "section-health",
      "Configuration": "section-recognition",
      "Camera": "section-camera",
      "Recognition": "section-recognition",
      "Rules Engine": "section-rules",
      "Actions Policy": "section-actions",
      "Storage": "section-storage"
    };

    navItems.forEach(function(item) {
      item.addEventListener("click", function() {
        // Update active state in sidebar
        navItems.forEach(function(i) {
          i.classList.remove("text-primary", "border-l-2", "border-white", "bg-surface-high");
          i.classList.add("text-on-surface-variant");
        });
        item.classList.add("text-primary", "border-l-2", "border-white", "bg-surface-high");
        item.classList.remove("text-on-surface-variant");

        // Scroll to section
        var label = item.textContent.trim();
        var targetId = sections[label];
        if (targetId) {
          var target = $(targetId);
          if (target) {
            target.scrollIntoView({ behavior: "smooth", block: "start" });
          } else if (label === "System Health") {
             document.querySelector("section.bg-surface").scrollTo({ top: 0, behavior: "smooth" });
          }
        }
      });
    });
  }

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    TraceClient.probe().then(function (info) {
      if (info) {
        loadSystemInfo();
        loadHealth();
        TraceClient.getConfig().then(function(cfg) {
          currentConfig = cfg;
          renderConfig();
          initSidebar();
        });
      }
    });
    
    // Polling for health
    setInterval(loadHealth, 5000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.addEventListener("beforeunload", function () {
    TraceClient.disconnectSSE();
  });
})();
