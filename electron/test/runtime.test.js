const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  DATA_SEED_MARKER,
  SEEDED_DATA_SUBDIRS,
  buildBackendEnv,
  buildBackendLaunchSpec,
  buildFrontendUrl,
  buildServiceBaseUrl,
  buildWelcomeModel,
  ensureSeedData,
  hasRecognitionSeedData,
  parseDotEnvText,
} = require("../runtime");

function run() {
  assert.equal(buildServiceBaseUrl({ host: "127.0.0.1", port: 8080 }), "http://127.0.0.1:8080");

  assert.equal(
    buildFrontendUrl({ host: "127.0.0.1", port: 9090 }),
    "http://127.0.0.1:9090/ui/live_ops/index.html",
  );

  const env = buildBackendEnv({
    baseEnv: { PATH: "C:\\Python;C:\\Windows", TRACE_DATA_ROOT: "old-root" },
    dataRoot: "C:\\Users\\Tester\\AppData\\Roaming\\TRACE-AML",
    appMode: "electron",
  });
  assert.equal(env.PATH, "C:\\Python;C:\\Windows");
  assert.equal(env.TRACE_DATA_ROOT, "C:\\Users\\Tester\\AppData\\Roaming\\TRACE-AML");
  assert.equal(env.TRACE_APP_MODE, "electron");

  const parsed = parseDotEnvText(`
# comment
TRACE_VAULT_KEY=abc123
TRACE_DATA_ROOT=C:\\Demo\\TRACE-AML
EMPTY_VALUE=

IGNORED LINE
  `);
  assert.deepEqual(parsed, {
    TRACE_VAULT_KEY: "abc123",
    TRACE_DATA_ROOT: "C:\\Demo\\TRACE-AML",
    EMPTY_VALUE: "",
  });

  const model = buildWelcomeModel({
    username: "Utkarsh",
    hostname: "DEMO-BOX",
    platform: "win32",
    appName: "TRACE-AML",
    version: "3.0.0",
  });
  assert.equal(model.greeting, "Welcome, Utkarsh");
  assert.match(model.subtitle, /Windows/i);
  assert.match(model.subtitle, /DEMO-BOX/);
  assert.equal(model.ctaLabel, "Launch Workspace");

  const packagedSpec = buildBackendLaunchSpec({
    host: "127.0.0.1",
    port: 8080,
    bundledBackendPath: "C:\\bundle\\backend\\trace-aml-service.exe",
    projectRoot: "D:\\github FORK\\TRACE-ML",
    pythonCommand: "python",
    configPath: "config/config.desktop.yaml",
  });
  assert.equal(packagedSpec.command, "C:\\bundle\\backend\\trace-aml-service.exe");
  assert.deepEqual(packagedSpec.args, [
    "--config",
    "config/config.desktop.yaml",
    "service",
    "run",
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
  ]);
  assert.equal(packagedSpec.cwd, "C:\\bundle\\backend");

  const devSpec = buildBackendLaunchSpec({
    host: "127.0.0.1",
    port: 8080,
    bundledBackendPath: "",
    projectRoot: "D:\\github FORK\\TRACE-ML",
    pythonCommand: "D:\\github FORK\\TRACE-ML\\.venv\\Scripts\\python.exe",
    configPath: "config/config.demo.yaml",
  });
  assert.equal(devSpec.command, "D:\\github FORK\\TRACE-ML\\.venv\\Scripts\\python.exe");
  assert.deepEqual(devSpec.args, [
    "-m",
    "trace_aml",
    "--config",
    "config/config.demo.yaml",
    "service",
    "run",
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
  ]);
  assert.equal(devSpec.cwd, "D:\\github FORK\\TRACE-ML");

  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "trace-aml-runtime-"));
  const seedRoot = path.join(tmpRoot, "seed");
  const dataRoot = path.join(tmpRoot, "appdata");
  for (const subdir of SEEDED_DATA_SUBDIRS) {
    fs.mkdirSync(path.join(seedRoot, subdir), { recursive: true });
  }
  fs.mkdirSync(path.join(seedRoot, "vectors", "persons.lance", "data"), { recursive: true });
  fs.mkdirSync(path.join(seedRoot, "vectors", "person_embeddings.lance", "data"), { recursive: true });
  fs.writeFileSync(path.join(seedRoot, "vectors", "persons.lance", "data", "a.lance"), "person");
  fs.writeFileSync(path.join(seedRoot, "vectors", "person_embeddings.lance", "data", "b.lance"), "embedding");
  fs.writeFileSync(path.join(seedRoot, "index", "portraits.json"), "{}");
  fs.writeFileSync(path.join(seedRoot, "vault", "dummy.bin"), "vault");

  assert.equal(hasRecognitionSeedData(seedRoot), true);
  assert.equal(hasRecognitionSeedData(dataRoot), false);

  const seedResult = ensureSeedData({ dataRoot, seedRoot, log: () => {} });
  assert.deepEqual(seedResult, { seeded: true, reason: "seed-copied" });
  assert.equal(hasRecognitionSeedData(dataRoot), true);
  assert.equal(fs.existsSync(path.join(dataRoot, DATA_SEED_MARKER)), true);

  const secondResult = ensureSeedData({ dataRoot, seedRoot, log: () => {} });
  assert.deepEqual(secondResult, { seeded: false, reason: "already-seeded" });

  fs.rmSync(tmpRoot, { recursive: true, force: true });

  console.log("runtime tests passed");
}

run();
