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

  /**
   * Format ISO timestamp to HH:MM:SS.
   * @param {string} iso
   * @returns {string}
   */
  function formatTime(iso) {
    if (!iso) return "--:--:--";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return "--:--:--";
    return d.toISOString().slice(11, 19);
  }

  /**
   * Format ISO timestamp to YYYY-MM-DD HH:MM:SS.
   * @param {string} iso
   * @returns {string}
   */
  function formatDateTime(iso) {
    if (!iso) return "—";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    return d.toISOString().replace("T", " ").slice(0, 19);
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
  function _fetchJson(url, init) {
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
          return res.text().then(function (body) {
            console.warn("[TraceClient] HTTP " + res.status + " " + url, body.slice(0, 200));
            return null;
          });
        }
        connectionState = "online";
        _dispatchStateChange();
        return res.json();
      })
      .catch(function (err) {
        clearTimeout(timeoutId);
        if (err.name === "AbortError") {
          console.warn("[TraceClient] Request timeout: " + url);
        }
        connectionState = "offline";
        _dispatchStateChange();
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

  /* ─────────────────────── API Methods ───────────────────────────── */

  /**
   * Probe backend connectivity. GET /
   * @returns {Promise<{name:string, environment:string, version:string, status:string}|null>}
   */
  function probe() {
    connectionState = "connecting";
    _dispatchStateChange();
    return _fetchJson(_url("/"));
  }

  /**
   * System health. GET /health
   * @returns {Promise<Object|null>}
   */
  function health() {
    return _fetchJson(_url("/health"));
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
    return _fetchJson(_url("/api/v1/live/overlay"));
  }

  /**
   * Returns the MJPEG stream URL for <img src="...">.
   * @returns {string}
   */
  function mjpegUrl() {
    return BASE_URL + "/api/v1/live/mjpeg";
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

  /* ── Incidents ── */

  /**
   * GET /api/v1/incidents
   * @param {{limit?:number, status?:string, entity_id?:string}} [opts]
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

  /* ── Timeline ── */

  /**
   * GET /api/v1/timeline
   * @param {{start?:string, end?:string, limit?:number, entity_id?:string, incident_id?:string, kinds?:string[]}} [opts]
   * @returns {Promise<Array|null>}
   */
  function globalTimeline(opts) {
    return _fetchJson(_url("/api/v1/timeline", opts));
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
      connectionState = "offline";
      _dispatchStateChange();
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

    // Entities
    entities: entities,
    entity: entity,
    entityProfile: entityProfile,
    entityTimeline: entityTimeline,
    entityIncidents: entityIncidents,

    // Incidents
    incidents: incidents,
    incident: incident,
    setSeverity: setSeverity,
    closeIncident: closeIncident,

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

    // Persons
    listPersons: listPersons,
    createPerson: createPerson,
    getPerson: getPerson,
    updatePerson: updatePerson,
    deletePerson: deletePerson,
    uploadPersonImages: uploadPersonImages,

    // Training
    trainRebuild: trainRebuild,
    trainStatus: trainStatus,

    // Utilities
    escapeHtml: escapeHtml,
    formatTime: formatTime,
    formatDateTime: formatDateTime,
  };
})(typeof window !== "undefined" ? window : globalThis);
