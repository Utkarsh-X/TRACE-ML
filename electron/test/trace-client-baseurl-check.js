const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const clientSource = fs.readFileSync(
  path.resolve(__dirname, "..", "..", "src", "frontend", "shared", "trace_client.js"),
  "utf-8",
);

function evaluateBaseUrl(href) {
  const location = new URL(href);
  const sandbox = {
    window: null,
    globalThis: null,
    location,
    URL,
    URLSearchParams,
    localStorage: { getItem() { return null; }, setItem() {} },
    CustomEvent: function CustomEvent(type, init) {
      this.type = type;
      this.detail = init && init.detail;
    },
    EventSource: function EventSource() {},
    AbortController,
    fetch() {
      throw new Error("fetch should not be called while resolving base URL");
    },
    setTimeout,
    clearTimeout,
    console,
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;

  vm.runInNewContext(clientSource, sandbox, { filename: "trace_client.js" });
  return sandbox.TraceClient.baseUrl;
}

assert.equal(
  evaluateBaseUrl("http://127.0.0.1:18080/ui/live_ops/index.html"),
  "http://127.0.0.1:18080",
);

assert.equal(
  evaluateBaseUrl("http://127.0.0.1:18080/ui/live_ops/index.html?api=http://127.0.0.1:19090"),
  "http://127.0.0.1:19090",
);

console.log("trace-client base URL checks passed");
