/**
 * TRACE-AML API Client
 *
 * Single source of truth for all backend communication.
 * Every method returns a Promise that resolves to JSON or null (never throws).
 * Connection state is tracked via TraceClient.online.
 *
 * @fileoverview Plain JS + JSDoc types — no build step required.
 */
(function (global) {
  "use strict";

  /* ───────────────────────────── Config ──────────────────────────── */

  /** @type {"online"|"offline"|"connecting"} */
  var connectionState = "connecting";
  /** @type {number} */
  var _connectionFailureStreak = 0;
  /** @type {number} */
  var OFFLINE_FAILURE_THRESHOLD = 3;

  /** @type {EventSource|null} */
  var _sse = null;

  /** @type {number|null} */
  var _sseReconnectTimer = null;

  /** @type {number} */
  var _sseReconnectDelay = 1000;

  /** @type {number} Max SSE reconnect backoff (30s) */
  var SSE_MAX_BACKOFF = 30000;

  /**
   * Resolve API base URL.
   * Priority: ?api= query param > same origin > localhost:8080
   * @returns {string}
   */
  function _buildBaseUrl() {
    try {
      var params = new URLSearchParams(global.location.search);
      var fromQuery = (params.get("api") || "").trim();
      if (fromQuery) return fromQuery.replace(/\/$/, "");
    } catch (_) { /* ignore */ }
    try {
      if (global.location && global.location.origin && /^https?:/i.test(global.location.origin)) {
        return String(global.location.origin).replace(/\/$/, "");
      }
    } catch (_) { /* ignore */ }
    return "http://127.0.0.1:8080";
  }

  var BASE_URL = _buildBaseUrl();

  /* ───────────────────────────── Utilities ───────────────────────── */

  /**
   * Escape HTML entities to prevent XSS.
   * @param {*} value
   * @returns {string}
   */
  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /* ─────────────────── Timezone preference ─────────────────── */
  /* Supported: 'UTC' (default), 'IST' (+5:30 = +330 min)        */
  var TZ_OFFSETS = { UTC: 0, IST: 330 };

  /** Get the active timezone preference from localStorage. */
  function getTZ() {
    try { return localStorage.getItem("trace_tz") || "UTC"; } catch (e) { return "UTC"; }
  }

  /**
   * Set the timezone preference and broadcast to the page so all
   * live-rendered timestamps update without a full reload.
   * @param {string} tz  'UTC' | 'IST'
   */
  function setTZ(tz) {
    var key = String(tz).toUpperCase();
    if (!(key in TZ_OFFSETS)) return;
    try { localStorage.setItem("trace_tz", key); } catch (e) { }
    // Dispatch a custom event so any page can refresh its clocks.
    try { window.dispatchEvent(new CustomEvent("trace:tz-change", { detail: key })); } catch (e) { }
  }

  /** Apply offset minutes to a Date and return the shifted Date. */
  function _applyTZ(d) {
    var offsetMin = TZ_OFFSETS[getTZ()] || 0;
    return new Date(d.getTime() + offsetMin * 60000);
  }

  /* ─────────────────────────────────────────────────────────── */

  /**
   * Format ISO timestamp to HH:MM:SS in the selected timezone.
   * @param {string} iso
   * @returns {string}
   */
  function formatTime(iso) {
    if (!iso) return "--:--:--";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return "--:--:--";
    var shifted = _applyTZ(d);
    return shifted.toISOString().slice(11, 19);
  }

  /**
   * Format ISO timestamp to DD-MM-YYYY HH:MM:SS in the selected timezone.
   * @param {string} iso
   * @returns {string}
   */
  function formatDateTime(iso) {
    if (!iso) return "\u2014";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    var shifted = _applyTZ(d);
    return shifted.toISOString().replace("T", " ").slice(0, 19);
  }

  /* ───────────────────────── HTTP helpers ────────────────────────── */

  /**
   * Build URL with optional query parameters.
   * @param {string} path
   * @param {Object<string, string|number|string[]>} [params]
   * @returns {string}
   */
  function _url(path, params) {
    var url = BASE_URL + path;
    if (!params) return url;
    var parts = [];
    Object.keys(params).forEach(function (key) {
      var val = params[key];
      if (val === "" || val === null || val === undefined) return;
      if (Array.isArray(val)) {
        val.forEach(function (v) {
          parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(v));
        });
      } else {
        parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(val));
      }
    });
    return parts.length ? url + "?" + parts.join("&") : url;
  }

  /**
   * Fetch JSON from backend. Returns null on any error.
   * Sets connectionState accordingly.
   * @param {string} url
   * @param {RequestInit} [init]
   * @returns {Promise<*>}
   */
  function _fetchJson(url, init, options) {
    var opts = options || {};
    var affectConnectionState = opts.affectConnectionState !== false;
    var markOnlineOnSuccess = opts.markOnlineOnSuccess !== false;
    var controller = new AbortController();
    var timeoutId = setTimeout(function () { controller.abort(); }, 10000);
    var merged = Object.assign(
      { method: "GET", headers: { Accept: "application/json" }, cache: "no-store", signal: controller.signal },
      init || {}
    );
    return fetch(url, merged)
      .then(function (res) {
        clearTimeout(timeoutId);
        if (!res.ok) {
          if (affectConnectionState) {
            _recordConnectionFailure();
          }
          return res.text().then(function (body) {
            console.warn("[TraceClient] HTTP " + res.status + " " + url, body.slice(0, 200));
            return null;
          });
        }
        if (affectConnectionState && markOnlineOnSuccess) {
          _recordConnectionSuccess();
        }
        return res.json();
      })
      .catch(function (err) {
        clearTimeout(timeoutId);
        if (err.name === "AbortError") {
          console.warn("[TraceClient] Request timeout: " + url);
        }
        if (affectConnectionState) {
          _recordConnectionFailure();
        }
        return null;
      });
  }

  /**
   * Fetch with JSON body (PATCH/POST).
   * @param {string} url
   * @param {string} method
   * @param {*} [body]
   * @returns {Promise<*>}
   */
  function _fetchJsonMethod(url, method, body) {
    return _fetchJson(url, {
      method: method,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  }

  /* ─────────────────── Connection state events ──────────────────── */

  /** @type {Array<function(string):void>} */
  var _stateListeners = [];

  /** @param {function(string):void} fn */
  function onStateChange(fn) {
    _stateListeners.push(fn);
    return function () {
      _stateListeners = _stateListeners.filter(function (f) { return f !== fn; });
    };
  }

  function _dispatchStateChange() {
    var state = connectionState;
    _stateListeners.forEach(function (fn) {
      try { fn(state); } catch (_) { /* ignore */ }
    });
  }

  function _setConnectionState(nextState) {
    if (connectionState === nextState) return;
    connectionState = nextState;
    _dispatchStateChange();
  }

  function _recordConnectionSuccess() {
    _connectionFailureStreak = 0;
    _setConnectionState("online");
  }

  function _recordConnectionFailure() {
    _connectionFailureStreak += 1;
    if (_connectionFailureStreak >= OFFLINE_FAILURE_THRESHOLD) {
      _setConnectionState("offline");
    }
  }

  /* ─────────────────────── API Methods ───────────────────────────── */

  /**
   * Probe backend connectivity. GET /
   * @returns {Promise<{name:string, environment:string, version:string, status:string}|null>}
   */
  function probe() {
    _connectionFailureStreak = 0;
    _setConnectionState("connecting");
    return _fetchJson(_url("/"));
  }

  /**
   * System health. GET /health
   * @returns {Promise<Object|null>}
   */
  function health(options) {
    return _fetchJson(_url("/health"), undefined, options);
  }

  /* ── Live Ops ── */

  /**
   * GET /api/v1/live/snapshot
   * @param {{entity_limit?:number, incident_limit?:number, alert_limit?:number}} [opts]
   * @returns {Promise<Object|null>}
   */
  function liveSnapshot(opts) {
    return _fetchJson(_url("/api/v1/live/snapshot", opts));
  }

  /**
   * GET /api/v1/live/overlay
   * @returns {Promise<Object|null>}
   */
  function liveOverlay() {
    return _fetchJson(_url("/api/v1/live/overlay"), undefined, { affectConnectionState: false });
  }

  /**
   * Returns the MJPEG stream URL for <img src="...">.
   * @returns {string}
   */
  function mjpegUrl() {
    return BASE_URL + "/api/v1/live/mjpeg";
  }

  /* ── Camera Control ── */

  /**
   * GET /api/v1/camera/status
   * Check current camera status (enabled/disabled)
   * @returns {Promise<{enabled:boolean, camera_index:number, resolution:string, fps:number}|null>}
   */
  function cameraStatus() {
    return _fetchJson(_url("/api/v1/camera/status"), undefined, { affectConnectionState: false });
  }

  /**
   * POST /api/v1/camera/enable
   * Enable camera capture on backend
   * @returns {Promise<{status:string, message:string}|null>}
   */
  function cameraEnable() {
    return _fetchJsonMethod(_url("/api/v1/camera/enable"), "POST");
  }

  /**
   * POST /api/v1/camera/disable
   * Disable camera capture on backend
   * @returns {Promise<{status:string, message:string}|null>}
   */
  function cameraDisable() {
    return _fetchJsonMethod(_url("/api/v1/camera/disable"), "POST");
  }

  /**
   * GET /api/v1/recognition/status
   * Check current recognition status (inference/processing)
   * @returns {Promise<{enabled:boolean, camera_enabled:boolean}|null>}
   */
  function recognitionStatus() {
    return _fetchJson(_url("/api/v1/recognition/status"), undefined, { affectConnectionState: false });
  }

  /**
   * POST /api/v1/recognition/enable
   * Enable face recognition inference (requires camera to be enabled)
   * @returns {Promise<{status:string, message:string}|null>}
   */
  function recognitionEnable() {
    return _fetchJsonMethod(_url("/api/v1/recognition/enable"), "POST");
  }

  /**
   * POST /api/v1/recognition/disable
   * Disable face recognition inference (camera keeps running)
   * @returns {Promise<{status:string, message:string}|null>}
   */
  function recognitionDisable() {
    return _fetchJsonMethod(_url("/api/v1/recognition/disable"), "POST");
  }

  /* ── Entities ── */

  /**
   * GET /api/v1/entities
   * @param {{limit?:number, type_filter?:string, status?:string}} [opts]
   * @returns {Promise<Array|null>}
   */
  function entities(opts) {
    return _fetchJson(_url("/api/v1/entities", opts));
  }

  /**
   * GET /api/v1/entities/{id}
   * @param {string} entityId
   * @returns {Promise<Object|null>}
   */
  function entity(entityId) {
    return _fetchJson(_url("/api/v1/entities/" + encodeURIComponent(entityId)));
  }

  /**
   * GET /api/v1/entities/{id}/profile
   * @param {string} entityId
   * @returns {Promise<Object|null>}
   */
  function entityProfile(entityId) {
    return _fetchJson(_url("/api/v1/entities/" + encodeURIComponent(entityId) + "/profile"));
  }

  /**
   * Return the URL for the best-match portrait JPEG of an entity.
   * Set this directly as <img src="..."> — the browser will show a 404 if
   * no portrait exists yet (handle via img.onerror).
   * @param {string} entityId
   * @returns {string}
   */
  function entityPortraitUrl(entityId) {
    return BASE_URL + "/api/v1/entities/" + encodeURIComponent(entityId) + "/portrait";
  }

  /**
   * GET /api/v1/entities/{id}/timeline
   * @param {string} entityId
   * @param {{start?:string, end?:string, limit?:number}} [opts]
   * @returns {Promise<Array|null>}
   */
  function entityTimeline(entityId, opts) {
    return _fetchJson(_url("/api/v1/entities/" + encodeURIComponent(entityId) + "/timeline", opts));
  }

  /**
   * GET /api/v1/entities/{id}/incidents
   * @param {string} entityId
   * @returns {Promise<Array|null>}
   */
  function entityIncidents(entityId) {
    return _fetchJson(_url("/api/v1/entities/" + encodeURIComponent(entityId) + "/incidents"));
  }

  /**
   * GET /api/v1/entities/{id}/suggestions
   * @param {string} entityId
   * @param {number} [threshold]
   * @returns {Promise<Array|null>}
   */
  function entitySuggestions(entityId, threshold) {
    return _fetchJson(_url("/api/v1/entities/" + encodeURIComponent(entityId) + "/suggestions", threshold ? { threshold: threshold } : undefined));
  }

  /**
   * POST /api/v1/entities/{id}/merge
   * @param {string} entityId
   * @param {string} targetEntityId
   * @returns {Promise<Object|null>}
   */
  function entityMerge(entityId, targetEntityId) {
    return _fetchJsonMethod(_url("/api/v1/entities/" + encodeURIComponent(entityId) + "/merge"), "POST", { target_entity_id: targetEntityId });
  }

  /* ── Incidents ── */
  /**
   * GET /api/v1/incidents
   * @param {{limit?:number, skip?:number, status?:string, entity_id?:string}} [opts]
   * @returns {Promise<Array|null>}
   */
  function incidents(opts) {
    return _fetchJson(_url("/api/v1/incidents", opts));
  }

  /**
   * GET /api/v1/incidents/{id}
   * @param {string} incidentId
   * @returns {Promise<Object|null>}
   */
  function incident(incidentId) {
    return _fetchJson(_url("/api/v1/incidents/" + encodeURIComponent(incidentId)));
  }

  /**
   * PATCH /api/v1/incidents/{id}/severity
   * @param {string} incidentId
   * @param {string} severity  "low"|"medium"|"high"
   * @returns {Promise<Object|null>}
   */
  function setSeverity(incidentId, severity) {
    return _fetchJsonMethod(
      _url("/api/v1/incidents/" + encodeURIComponent(incidentId) + "/severity"),
      "PATCH",
      { severity: severity }
    );
  }

  /**
   * POST /api/v1/incidents/{id}/close
   * @param {string} incidentId
   * @returns {Promise<Object|null>}
   */
  function closeIncident(incidentId) {
    return _fetchJsonMethod(
      _url("/api/v1/incidents/" + encodeURIComponent(incidentId) + "/close"),
      "POST"
    );
  }

  /**
   * POST /api/v1/incidents/deduplicate
   * Removes duplicate incident records from the database.
   * @returns {Promise<Object|null>}
   */
  function deduplicateIncidents() {
    return _fetchJsonMethod(
      _url("/api/v1/incidents/deduplicate"),
      "POST"
    );
  }

  /**
   * POST /api/v1/system/factory-reset
   * Wipe ALL data (tables, portraits, screenshots, person images) for a clean start.
   * Camera must be disabled before calling — returns 409 if active.
   * @returns {Promise<{status:string, detail:Object}|null>}
   */
  function factoryReset() {
    return _fetchJsonMethod(_url("/api/v1/system/factory-reset"), "POST");
  }

  /* ── Timeline ── */

  /**
   * GET /api/v1/timeline
   * @param {{start?:string, end?:string, limit?:number, entity_id?:string, incident_id?:string, kinds?:string[]}} [opts]
   * @returns {Promise<Array|null>}
   */
  function globalTimeline(opts, options) {
    return _fetchJson(_url("/api/v1/timeline", opts), undefined, options);
  }

  /* ── Alerts & Actions ── */

  /**
   * GET /api/v1/alerts/recent
   * @param {number} [limit]
   * @returns {Promise<Array|null>}
   */
  function recentAlerts(limit) {
    return _fetchJson(_url("/api/v1/alerts/recent", limit ? { limit: limit } : undefined));
  }

  /**
   * GET /api/v1/actions
   * @param {{incident_id?:string, limit?:number}} [opts]
   * @returns {Promise<Array|null>}
   */
  function actions(opts) {
    return _fetchJson(_url("/api/v1/actions", opts));
  }

  /* ── SSE ── */

  /**
   * Connect to the Server-Sent Events stream.
   * @param {function({topic:string, timestamp_utc:string, payload:Object}):void} onEvent
   * @param {{backfill?:number, heartbeat_sec?:number}} [opts]
   */
  function connectSSE(onEvent, opts) {
    disconnectSSE();
    var params = Object.assign({ backfill: 50 }, opts || {});
    var url = _url("/api/v1/events/stream", params);

    try {
      _sse = new EventSource(url);
    } catch (err) {
      console.warn("[TraceClient] SSE not supported or blocked", err);
      return;
    }

    _sseReconnectDelay = 1000;

    _sse.onopen = function () {
      _sseReconnectDelay = 1000;
      connectionState = "online";
      _dispatchStateChange();
    };

    _sse.onerror = function () {
      if (_sse) { _sse.close(); _sse = null; }
      console.warn("[TraceClient] SSE connection lost; retrying background stream");
      // Reconnect with exponential backoff
      _sseReconnectTimer = setTimeout(function () {
        connectSSE(onEvent, opts);
      }, _sseReconnectDelay);
      _sseReconnectDelay = Math.min(_sseReconnectDelay * 2, SSE_MAX_BACKOFF);
    };

    // SSE events come as named events matching the topic
    _sse.onmessage = function (msg) {
      try {
        var parsed = JSON.parse(msg.data);
        onEvent(parsed);
      } catch (_) { /* ignore malformed */ }
    };

    // Also listen for named event types the backend publishes
    var knownTopics = [
      "detection.new", "entity.resolved", "entity.updated",
      "alert.created", "incident.created", "incident.updated",
      "action.executed", "session.state", "health.check",
    ];
    knownTopics.forEach(function (topic) {
      _sse.addEventListener(topic, function (msg) {
        try {
          var parsed = JSON.parse(msg.data);
          onEvent(parsed);
        } catch (_) { /* ignore */ }
      });
    });
  }

  /** Disconnect SSE and cancel any pending reconnect. */
  function disconnectSSE() {
    if (_sseReconnectTimer) {
      clearTimeout(_sseReconnectTimer);
      _sseReconnectTimer = null;
    }
    if (_sse) {
      _sse.close();
      _sse = null;
    }
  }

  /* ── Persons ── */

  /**
   * GET /api/v1/persons
   * @returns {Promise<Array|null>}
   */
  function listPersons() {
    return _fetchJson(_url("/api/v1/persons"));
  }

  /**
   * POST /api/v1/persons
   * @param {{name:string, category?:string, severity?:string, dob?:string, gender?:string, city?:string, country?:string, notes?:string}} data
   * @returns {Promise<Object|null>}
   */
  function createPerson(data) {
    return _fetchJsonMethod(_url("/api/v1/persons"), "POST", data);
  }

  /**
   * GET /api/v1/persons/{id}
   * @param {string} personId
   * @returns {Promise<Object|null>}
   */
  function getPerson(personId) {
    return _fetchJson(_url("/api/v1/persons/" + encodeURIComponent(personId)));
  }

  /**
   * PATCH /api/v1/persons/{id}
   * @param {string} personId
   * @param {Object} data
   * @returns {Promise<Object|null>}
   */
  function updatePerson(personId, data) {
    return _fetchJsonMethod(
      _url("/api/v1/persons/" + encodeURIComponent(personId)),
      "PATCH",
      data
    );
  }

  /**
   * DELETE /api/v1/persons/{id}
   * @param {string} personId
   * @returns {Promise<Object|null>}
   */
  function deletePerson(personId) {
    return _fetchJsonMethod(
      _url("/api/v1/persons/" + encodeURIComponent(personId)),
      "DELETE"
    );
  }

  /**
   * POST /api/v1/persons/{id}/images  (multipart/form-data)
   * @param {string} personId
   * @param {FileList|File[]} files
   * @returns {Promise<Object|null>}
   */
  function uploadPersonImages(personId, files) {
    var formData = new FormData();
    for (var i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    var url = _url("/api/v1/persons/" + encodeURIComponent(personId) + "/images");
    return fetch(url, { method: "POST", body: formData })
      .then(function (res) {
        if (!res.ok) return null;
        return res.json();
      })
      .catch(function () { return null; });
  }

  /* ── Training ── */

  /**
   * POST /api/v1/train/rebuild
   * @param {{scope?:string}} [data]
   * @returns {Promise<Object|null>}
   */
  function trainRebuild(data) {
    return _fetchJsonMethod(_url("/api/v1/train/rebuild"), "POST", data || { scope: "all" });
  }

  /**
   * GET /api/v1/train/status
   * @returns {Promise<Object|null>}
   */
  function trainStatus() {
    return _fetchJson(_url("/api/v1/train/status"));
  }

  /**
   * GET /api/v1/enroll/status
   * Returns per-person enrollment queue state {queue_depth, persons:{pid: status}}.
   * @returns {Promise<{queue_depth:number, persons:Object}|null>}
   */
  function enrollStatus() {
    return _fetchJson(_url("/api/v1/enroll/status"));
  }

  /**
   * GET /api/v1/enroll/status/{personId}
   * @param {string} personId
   * @returns {Promise<{person_id:string, status:string}|null>}
   */
  function enrollStatusPerson(personId) {
    return _fetchJson(_url("/api/v1/enroll/status/" + encodeURIComponent(personId)));
  }

  /**
   * Filter: only show meaningful business events, exclude metrics.
   * @param {Object} event - Event object with {topic, payload, timestamp_utc}
   * @returns {boolean} true if event should be displayed, false if it's a metric
   */
  function isMeaningfulEvent(event) {
    var topic = String(event.topic || '').toLowerCase();
    // Explicitly REJECT metrics
    if (topic === 'session.state') return false;
    // ACCEPT all other meaningful events
    return true;
  }

  /* ── Entity CRUD ── */

  /**
   * PATCH /api/v1/entities/{id}
   * Updates metadata for a known entity, or promotes an unknown entity to known.
   * @param {string} entityId
   * @param {{name?:string, category?:string, severity?:string, notes?:string}} payload
   * @returns {Promise<{status:string, entity_id?:string, new_entity_id?:string}|null>}
   */
  function updateEntity(entityId, payload) {
    return _fetchJsonMethod(
      _url("/api/v1/entities/" + encodeURIComponent(entityId)),
      "PATCH",
      payload
    );
  }

  /**
   * DELETE /api/v1/entities/{id}
   * Deletes any entity (known or unknown) and all linked data.
   * @param {string} entityId
   * @returns {Promise<{status:string, entity_id:string}|null>}
   */
  function deleteEntity(entityId) {
    return _fetchJsonMethod(
      _url("/api/v1/entities/" + encodeURIComponent(entityId)),
      "DELETE"
    );
  }

  /**
   * POST /api/v1/entities/{id}/portrait  (multipart/form-data)
   * Upload a new portrait image for an entity, replacing the auto-captured one.
   * @param {string} entityId
   * @param {File} file
   * @returns {Promise<{status:string, entity_id:string}|null>}
   */
  function uploadPortrait(entityId, file) {
    var formData = new FormData();
    formData.append("file", file);
    var url = _url("/api/v1/entities/" + encodeURIComponent(entityId) + "/portrait");
    return fetch(url, { method: "POST", body: formData })
      .then(function (res) {
        if (!res.ok) return null;
        return res.json();
      })
      .catch(function () { return null; });
  }

  /* ───────────────────────── Public API ──────────────────────────── */

  global.TraceClient = {
    /** @type {string} Resolved API base URL */
    get baseUrl() { return BASE_URL; },
    /** @type {string} Current connection state */
    get state() { return connectionState; },
    /** @type {boolean} Shorthand for state === "online" */
    get online() { return connectionState === "online"; },

    // Connection
    probe: probe,
    onStateChange: onStateChange,

    // Live Ops
    liveSnapshot: liveSnapshot,
    liveOverlay: liveOverlay,
    mjpegUrl: mjpegUrl,
    cameraStatus: cameraStatus,
    cameraEnable: cameraEnable,
    cameraDisable: cameraDisable,
    recognitionStatus: recognitionStatus,
    recognitionEnable: recognitionEnable,
    recognitionDisable: recognitionDisable,

    // Entities
    entities: entities,
    entity: entity,
    entityProfile: entityProfile,
    entityPortraitUrl: entityPortraitUrl,
    entityTimeline: entityTimeline,
    entityIncidents: entityIncidents,
    entitySuggestions: entitySuggestions,
    entityMerge: entityMerge,
    updateEntity: updateEntity,
    deleteEntity: deleteEntity,
    uploadPortrait: uploadPortrait,

    // Incidents
    incidents: incidents,
    incident: incident,
    setSeverity: setSeverity,
    closeIncident: closeIncident,
    deduplicateIncidents: deduplicateIncidents,
    factoryReset: factoryReset,

    // Timeline
    globalTimeline: globalTimeline,

    // Alerts & Actions
    recentAlerts: recentAlerts,
    actions: actions,

    // SSE
    connectSSE: connectSSE,
    disconnectSSE: disconnectSSE,

    // Health
    health: health,

    // Config
    getConfig: function() {
      return _fetchJson(_url("/api/v1/config"));
    },
    updateConfig: function(data) {
      return _fetchJsonMethod(_url("/api/v1/config"), "PATCH", data);
    },

    // Persons
    listPersons: listPersons,
    createPerson: createPerson,
    getPerson: getPerson,
    updatePerson: updatePerson,
    deletePerson: deletePerson,
    uploadPersonImages: uploadPersonImages,

    // Training
    rebuildGallery: function(data) {
      return _fetchJsonMethod(_url("/api/v1/train/rebuild"), "POST", data || { scope: "all" });
    },
    trainRebuild: trainRebuild,
    trainStatus: trainStatus,
    enrollStatus: enrollStatus,
    enrollStatusPerson: enrollStatusPerson,

    // Utilities
    escapeHtml: escapeHtml,
    formatTime: formatTime,
    formatDateTime: formatDateTime,
    getTZ: getTZ,
    setTZ: setTZ,
    isMeaningfulEvent: isMeaningfulEvent,

  };
})(typeof window !== "undefined" ? window : globalThis);
