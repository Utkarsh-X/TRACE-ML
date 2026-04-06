(function (global) {
  "use strict";

  function buildApiBase() {
    var search = new URLSearchParams(window.location.search);
    var fromQuery = (search.get("api") || "").trim();
    if (fromQuery) {
      return fromQuery.replace(/\/$/, "");
    }
    if (global.location.protocol === "http:" || global.location.protocol === "https:") {
      return global.location.origin.replace(/\/$/, "");
    }
    return "http://127.0.0.1:8080";
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatTime(isoText) {
    if (!isoText) {
      return "--:--:--";
    }
    var d = new Date(isoText);
    if (Number.isNaN(d.getTime())) {
      return "--:--:--";
    }
    return d.toISOString().slice(11, 19);
  }

  function formatDateTime(isoText) {
    if (!isoText) {
      return "—";
    }
    var d = new Date(isoText);
    if (Number.isNaN(d.getTime())) {
      return String(isoText);
    }
    return d.toISOString().replace("T", " ").slice(0, 19) + " UTC";
  }

  function fetchJson(url, init) {
    var merged = Object.assign(
      {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
      },
      init || {}
    );
    return fetch(url, merged).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (t) {
          throw new Error("HTTP " + response.status + " " + (t || "").slice(0, 120));
        });
      }
      return response.json();
    });
  }

  function fetchJsonMethod(url, method, body) {
    return fetchJson(url, {
      method: method,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  /**
   * Resolve a path relative to the current page (e.g. "../entity_explorer_clean_canvas/code.html")
   * so navigation works when the app is served from the mockup-code root.
   */
  function resolveMockupUrl(relativePath) {
    try {
      return new URL(relativePath, global.location.href).href;
    } catch (e) {
      return String(relativePath || "");
    }
  }

  global.TraceApi = {
    buildApiBase: buildApiBase,
    escapeHtml: escapeHtml,
    formatTime: formatTime,
    formatDateTime: formatDateTime,
    fetchJson: fetchJson,
    fetchJsonMethod: fetchJsonMethod,
    resolveMockupUrl: resolveMockupUrl,
  };
})(typeof window !== "undefined" ? window : globalThis);
