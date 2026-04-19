/**
 * Settings Page Controller - Modernized Version
 *
 * - Manages diagnostics, runtime config, and global preferences
 * - Handles section navigation and interaction feedback
 */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var currentConfig = null;
  var isRefreshing = false;
  var _trainPollTimer = null;

  function loadSystemInfo() {
    TraceClient.probe().then(function (info) {
      if (!info) return;
      if ($("core-version")) $("core-version").textContent = info.version || "—";
      if ($("core-env")) $("core-env").textContent = info.environment || "—";
      // We'll get config path from config object later
    });
  }

  function loadHealth() {
    if (isRefreshing) return;
    
    var grid = $("health-grid");
    if (!grid) return;

    TraceClient.health().then(function (health) {
      if (!health) {
        $("overall-status").textContent = "Connection Lost";
        $("overall-status").className = "px-2 py-0.5 bg-error/10 text-error text-[0.65rem] font-mono uppercase tracking-wider border border-error/20";
        return;
      }

      // Update overall status
      $("overall-status").textContent = health.status === "ok" ? "Operational" : "Attention Required";
      $("overall-status").className = health.status === "ok" 
        ? "px-2 py-0.5 bg-success/10 text-success text-[0.65rem] font-mono uppercase tracking-wider border border-success/20"
        : "px-2 py-0.5 bg-warn/10 text-warn text-[0.65rem] font-mono uppercase tracking-wider border border-warn/20";

      // Build health cards
      var checks = [
        { name: "Vector Store", detail: (health.total_detection_count || 0) + " detections", ok: health.status === "ok", icon: "database" },
        { name: "Recognition", detail: "ArcFace buffalo_sc", ok: health.status === "ok", icon: "face" },
        { name: "Active Entities", detail: (health.active_entity_count || 0) + " tracked", ok: health.active_entity_count >= 0, icon: "hub" },
        { name: "Open Incidents", detail: (health.open_incident_count || 0) + " active", ok: true, icon: "emergency" },
        { name: "Event Stream", detail: (health.publisher_subscribers || 0) + " subscribers", ok: true, icon: "sensors" },
        { name: "Last Signal", detail: health.latest_event_at ? TraceClient.formatTime(health.latest_event_at) : "Never", ok: !!health.latest_event_at, icon: "schedule" }
      ];

      grid.innerHTML = checks.map(function (c) {
        return `
          <div class="bg-surface-container/50 border border-outline-variant/10 p-4 flex items-start gap-4 hover:border-outline-variant/30 transition-colors">
            <div class="w-10 h-10 flex-shrink-0 bg-surface-high flex items-center justify-center rounded-sm">
              <span class="material-symbols-outlined text-primary text-[20px]">${c.icon}</span>
            </div>
            <div class="flex-grow min-w-0">
              <div class="flex justify-between items-start mb-1">
                <span class="font-mono text-[0.7rem] text-on-surface truncate">${c.name}</span>
                <span class="w-1.5 h-1.5 rounded-full ${c.ok ? 'bg-success shadow-[0_0_6px_rgba(var(--success-rgb),0.5)]' : 'bg-warn'}"></span>
              </div>
              <p class="font-mono text-[0.6rem] text-outline truncate">${c.detail}</p>
            </div>
          </div>
        `;
      }).join("");
    });
  }

  function renderConfig() {
    if (!currentConfig) return;

    if ($("core-config-path")) $("core-config-path").textContent = currentConfig.runtime_config_path || "config/config.yaml";

    var root = $("config-sections-root");
    if (!root) return;

    var html = "";

    // 1. Recognition
    var rec = currentConfig.recognition || {};
    html += `
      <div id="section-recognition" class="mb-12 scroll-mt-8">
        <h3 class="font-mono text-[0.8rem] text-primary uppercase tracking-widest mb-6 border-b border-outline-variant/10 pb-2">Recognition Tuning</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8 bg-surface-container/20 p-6 border border-outline-variant/5">
          ${renderSlider("recognition.similarity_threshold", "Similarity Threshold", rec.similarity_threshold, 0, 1, 0.01, "Minimum confidence to consider any match.")}
          ${renderSlider("recognition.accept_threshold", "Auto-Accept Threshold", rec.accept_threshold, 0, 1, 0.01, "Confidence required for immediate identification.")}
          ${renderSlider("recognition.review_threshold", "Review Threshold", rec.review_threshold, 0, 1, 0.01, "Confidence required to flag for manual review.")}
          ${renderSlider("recognition.top_k", "Search Depth (Top K)", rec.top_k, 1, 20, 1, "Number of candidate matches to aggregate.")}
        </div>
      </div>
    `;

    // 2. Rules Engine
    var rules = currentConfig.rules || {};
    html += `
      <div id="section-rules" class="mb-12 scroll-mt-8">
        <h3 class="font-mono text-[0.8rem] text-primary uppercase tracking-widest mb-6 border-b border-outline-variant/10 pb-2">Intelligence Rules</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8 bg-surface-container/20 p-6 border border-outline-variant/5">
          ${renderSlider("rules.cooldown_sec", "Incident Cooldown (s)", rules.cooldown_sec, 0, 60, 1, "Suppression window between identical alerts.")}
          ${renderSlider("rules.reappearance.window_sec", "Reappearance Window (s)", rules.reappearance.window_sec, 1, 300, 5, "Timeframe to group recurring detections.")}
          ${renderSlider("rules.reappearance.min_events", "Min Events (Reappearance)", rules.reappearance.min_events, 1, 10, 1, "Events needed to trigger a reappearance alert.")}
          ${renderSlider("rules.unknown.window_sec", "Unknown Window (s)", rules.unknown.window_sec, 1, 300, 5, "Timeframe to promote unknowns to entities.")}
          ${renderSlider("rules.unknown.min_events", "Min Events (Unknown)", rules.unknown.min_events, 1, 10, 1, "Events needed to create an unknown entity.")}
          ${renderSlider("rules.instability.std_threshold", "Instability Deviation", rules.instability.std_threshold, 0, 0.5, 0.01, "Sensitivity to biometric fluctuation.")}
        </div>
      </div>
    `;

    // 3. Action Policy
    var actions = currentConfig.actions || {};
    html += `
      <div id="section-actions" class="mb-12 scroll-mt-8">
        <h3 class="font-mono text-[0.8rem] text-primary uppercase tracking-widest mb-6 border-b border-outline-variant/10 pb-2">Execution Policy</h3>
        <div class="bg-surface-container/20 p-6 border border-outline-variant/5">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-12 mb-8">
             <div class="flex items-center justify-between p-4 bg-surface-high/30 border border-outline-variant/10">
                <div>
                  <span class="block font-mono text-[0.7rem] text-on-surface">External Actions</span>
                  <span class="block font-mono text-[0.55rem] text-outline uppercase">Master Kill Switch</span>
                </div>
                ${renderToggle("actions.enabled", actions.enabled)}
             </div>
             ${renderSlider("actions.cooldown_sec", "Global Action Cooldown (s)", actions.cooldown_sec, 0, 120, 5, "Delay between automated responses.")}
          </div>
          
          <div class="grid grid-cols-3 gap-6">
            ${renderPolicyCard("Low Severity", actions.on_create.low, "outline")}
            ${renderPolicyCard("Medium Severity", actions.on_create.medium, "on-surface-variant")}
            ${renderPolicyCard("High Severity", actions.on_create.high, "primary")}
          </div>
        </div>
      </div>
    `;

    // 4. Hardware (Read only)
    html += `
      <div id="section-camera" class="mb-12 scroll-mt-8">
        <h3 class="font-mono text-[0.8rem] text-primary uppercase tracking-widest mb-6 border-b border-outline-variant/10 pb-2">Hardware Environment</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="bg-surface-container/30 p-4 border border-outline-variant/10">
            <span class="block font-mono text-[0.55rem] text-outline uppercase mb-2">Capture Device</span>
            <span class="block font-mono text-[0.8rem] text-on-surface">Index ${currentConfig.camera.device_index}</span>
          </div>
          <div class="bg-surface-container/30 p-4 border border-outline-variant/10">
            <span class="block font-mono text-[0.55rem] text-outline uppercase mb-2">Sensor Resolution</span>
            <span class="block font-mono text-[0.8rem] text-on-surface">${currentConfig.camera.width} \u00D7 ${currentConfig.camera.height}</span>
          </div>
          <div class="bg-surface-container/30 p-4 border border-outline-variant/10">
            <span class="block font-mono text-[0.55rem] text-outline uppercase mb-2">Capture Frequency</span>
            <span class="block font-mono text-[0.8rem] text-on-surface">${currentConfig.camera.fps} FPS</span>
          </div>
        </div>
      </div>
    `;

    root.innerHTML = html;

    // Attach event listeners
    root.querySelectorAll("input[type=range], input[type=checkbox]").forEach(function(input) {
      input.addEventListener("change", handleConfigChange);
    });
  }

  function renderSlider(key, label, value, min, max, step, hint) {
    return `
      <div>
        <div class="flex justify-between items-center mb-3">
          <label class="font-mono text-[0.65rem] text-on-surface uppercase tracking-wider">${label}</label>
          <span class="font-mono text-[0.8rem] text-primary bg-primary/10 px-2 py-0.5 rounded-sm border border-primary/20" id="val-${key}">${value}</span>
        </div>
        <input type="range" data-key="${key}" min="${min}" max="${max}" step="${step}" value="${value}" 
               class="w-full h-1 bg-surface-high rounded-full appearance-none cursor-pointer accent-primary mb-2"
               oninput="document.getElementById('val-${key}').textContent = this.value">
        <p class="font-mono text-[0.55rem] text-outline italic opacity-70">${hint}</p>
      </div>
    `;
  }

  function renderToggle(key, checked) {
    return `
      <label class="relative inline-flex items-center cursor-pointer">
        <input type="checkbox" data-key="${key}" class="sr-only peer" ${checked ? 'checked' : ''}>
        <div class="w-10 h-5 bg-surface-high peer-focus:outline-none rounded-full peer peer-checked:bg-primary after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-5 shadow-inner"></div>
      </label>
    `;
  }

  function renderPolicyCard(label, actions, colorClass) {
    var actList = (actions || []);
    return `
      <div class="bg-surface-high/40 border border-outline-variant/10 p-4">
        <span class="block font-mono text-[0.55rem] text-outline uppercase mb-3 text-center border-b border-outline-variant/10 pb-2">${label}</span>
        <div class="flex flex-wrap gap-2 justify-center">
          ${actList.length > 0 ? actList.map(a => `<span class="px-2 py-1 bg-surface-container text-${colorClass} font-mono text-[0.6rem] uppercase border border-${colorClass}/20">${a}</span>`).join("") : '<span class="text-outline font-mono text-[0.6rem] uppercase opacity-40 italic">No Actions</span>'}
        </div>
      </div>
    `;
  }

  function handleConfigChange(e) {
    var key = e.target.getAttribute("data-key");
    var value = e.target.type === "checkbox" ? e.target.checked : parseFloat(e.target.value);
    
    // Optimistic UI update
    var valDisplay = $("val-" + key);
    if (valDisplay) valDisplay.textContent = value;

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
        console.log("[Settings] Hot-tune applied:", key, "=", value);
      }
    });
  }

  function initSidebar() {
    var navItems = document.querySelectorAll("#settings-nav .nav-pill");
    var scrollContainer = $("settings-scroll-container");

    navItems.forEach(function(item) {
      item.addEventListener("click", function() {
        var sectionId = item.getAttribute("data-section");
        var target = $(sectionId);
        
        if (target) {
          // Update visual state
          navItems.forEach(i => {
            i.classList.remove("active", "bg-surface-high", "text-primary", "border-l-2", "border-white");
            i.classList.add("text-on-surface-variant");
          });
          item.classList.add("active", "bg-surface-high", "text-primary", "border-l-2", "border-white");
          item.classList.remove("text-on-surface-variant");

          // Smooth scroll
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });

    // Scroll listener for active state
    scrollContainer.addEventListener("scroll", function() {
      // Throttle or just simple check
      var sections = ["section-health", "section-recognition", "section-rules", "section-actions", "section-neural", "section-camera"];
      var current = "";
      
      for (var id of sections) {
        var el = $(id);
        if (el && el.offsetTop - scrollContainer.offsetTop <= scrollContainer.scrollTop + 100) {
          current = id;
        }
      }

      if (current) {
        navItems.forEach(i => {
          if (i.getAttribute("data-section") === current) {
            i.classList.add("active", "bg-surface-high", "text-primary", "border-l-2", "border-white");
            i.classList.remove("text-on-surface-variant");
          } else {
            i.classList.remove("active", "bg-surface-high", "text-primary", "border-l-2", "border-white");
            i.classList.add("text-on-surface-variant");
          }
        });
      }
    }, { passive: true });
  }

  /* ─── Neural Index (Gallery Rebuild) ─── */

  function startTrainPoll() {
    if (_trainPollTimer) return;
    pollTrainStatus();
    _trainPollTimer = setInterval(pollTrainStatus, 2000);
  }

  function pollTrainStatus() {
    TraceClient.trainStatus().then(function (status) {
      if (!status) return;
      var runningEl = $("train-running");
      var activeEl = $("train-active");
      var readyEl = $("train-ready");
      var blockedEl = $("train-blocked");
      var dot = $("train-status-dot");

      if (status.running) {
        if (runningEl) runningEl.textContent = "REBUILDING...";
        if (dot) dot.className = "status-dot status-dot--active";
      } else {
        if (runningEl) runningEl.textContent = "IDLE";
        if (dot) dot.className = "status-dot status-dot--idle";
        if (_trainPollTimer) {
          clearInterval(_trainPollTimer);
          _trainPollTimer = null;
        }
      }

      var r = status.last_result;
      if (r && !r.error) {
        if (activeEl) activeEl.textContent = String(r.active_persons || 0);
        if (readyEl) readyEl.textContent = String(r.ready_persons || 0);
        if (blockedEl) blockedEl.textContent = String(r.blocked_persons || 0);
      }
    });

    // Also poll per-person enrollment queue to update status badge if needed
    if (typeof TraceClient.enrollStatus === "function") {
      TraceClient.enrollStatus().then(function (info) {
        if (!info) return;
        var runningEl = $("train-running");
        var dot = $("train-status-dot");
        var q = info.queue_depth || 0;
        var persons = info.persons || {};

        // Only update if not already being handled by rebuild status
        if (runningEl && (runningEl.textContent === "IDLE" || runningEl.textContent === "")) {
          var processing = Object.values(persons).filter(function(s) { return s === "processing"; }).length;
          var queued = Object.values(persons).filter(function(s) { return s === "queued"; }).length;

          if (processing > 0) {
            if (dot) dot.className = "status-dot status-dot--active";
            if (runningEl) runningEl.textContent = "ENROLLING...";
          } else if (queued > 0 || q > 0) {
            if (dot) dot.className = "status-dot status-dot--pending";
            if (runningEl) runningEl.textContent = q + " IN QUEUE";
          }
        }
      });
    }
  }

  function init() {
    var mainContent = document.querySelector("main");
    TraceRender.initOfflineUI(mainContent);

    // Refresh Health button
    var btnRefresh = $("btn-refresh-health");
    if (btnRefresh) {
      btnRefresh.addEventListener("click", function() {
        btnRefresh.classList.add("animate-spin");
        isRefreshing = true;
        loadHealth();
        setTimeout(function() {
          btnRefresh.classList.remove("animate-spin");
          isRefreshing = false;
        }, 1000);
      });
    }

    // Auto-refresh toggle
    var autoToggle = $("auto-refresh-toggle");
    var healthInterval = null;
    
    function startPolling() {
      if (healthInterval) clearInterval(healthInterval);
      healthInterval = setInterval(loadHealth, 5000);
    }
    
    if (autoToggle) {
      autoToggle.addEventListener("change", function() {
        if (this.checked) startPolling();
        else if (healthInterval) {
          clearInterval(healthInterval);
          healthInterval = null;
        }
      });
    }

    // Deduplicate database button
    var btnDeduplicate = $("btn-deduplicate-db");
    if (btnDeduplicate) {
      btnDeduplicate.addEventListener("click", function () {
        TraceDialog.confirm(
          "Confirm Deduplication",
          "Scan and remove duplicate incident entries from the database?\n\nThis may take several seconds.",
          { confirmText: "Start Scan" }
        ).then(function(ok) {
          if (!ok) return;
          btnDeduplicate.disabled = true;
          btnDeduplicate.textContent = "...";
          TraceClient.deduplicateIncidents().then(function (result) {
            btnDeduplicate.disabled = false;
            btnDeduplicate.innerHTML = '<span class="material-symbols-outlined" style="font-size:16px;">delete_sweep</span><span>Deduplicate Database</span>';
            if (result) {
              var count = result.removed_duplicates || 0;
              var msg = count > 0 ? ("Removed " + count + " duplicate(s) from database.") : "No duplicates found.";
              TraceToast.success("Deduplication Complete", msg);
            } else {
              TraceToast.error("Deduplication Failed", "Check backend logs for details.");
            }
          });
        });
      });
    }

    // ── Force Full System Reset ────────────────────────────────────────
    var btnFactoryReset   = $("btn-factory-reset");
    var progressEl        = $("factory-reset-progress");
    var statusEl          = $("factory-reset-status");

    if (btnFactoryReset) {
      btnFactoryReset.addEventListener("click", function() {
        TraceDialog.confirm(
          "Full System Reset",
          "This will permanently delete all persons, embeddings, entities, detections, incidents, portraits, and screenshots.",
          { type: "error", confirmText: "Wipe Everything", verifyText: "RESET" }
        ).then(function(ok) {
          if (!ok) return;

          // Show progress
          if (btnFactoryReset) { btnFactoryReset.disabled = true; }
          if (progressEl) { progressEl.style.display = "block"; }
          if (statusEl)   { statusEl.textContent = "Wiping all data — please wait..."; }

          TraceClient.factoryReset().then(function (result) {
            if (result && result.status === "success") {
              if (statusEl) { statusEl.textContent = "✓ Reset complete — redirecting to enrollment..."; }
              setTimeout(function () {
                window.location.href = "../enrollment/index.html";
              }, 1800);
            } else {
              if (statusEl) {
                var detail = (result && result.detail) ? JSON.stringify(result.detail) : "Unknown error. Check backend logs.";
                statusEl.textContent = "✗ Reset failed: " + detail;
              }
              if (btnFactoryReset) { btnFactoryReset.disabled = false; }
            }
          });
        });
      });
    }
    // ──────────────────────────────────────────────────────────────────

    TraceClient.probe().then(function (info) {
      if (info) {
        loadSystemInfo();
        loadHealth();
        startPolling();
        pollTrainStatus(); // Initial poll
        TraceClient.getConfig().then(function(cfg) {
          currentConfig = cfg;
          renderConfig();
          initSidebar();
        });
      }
    });
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
