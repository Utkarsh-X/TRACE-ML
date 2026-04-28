const path = require("node:path");

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8080;
const DEFAULT_PAGE = "/ui/live_ops/index.html";
const DATA_SEED_MARKER = ".trace-aml-seeded";
const SEEDED_DATA_SUBDIRS = ["index", "vectors", "vault", "person_images", "portraits"];

function buildServiceBaseUrl({ host = DEFAULT_HOST, port = DEFAULT_PORT } = {}) {
  return `http://${host}:${port}`;
}

function buildFrontendUrl({
  host = DEFAULT_HOST,
  port = DEFAULT_PORT,
  page = DEFAULT_PAGE,
} = {}) {
  const normalizedPage = String(page || DEFAULT_PAGE).startsWith("/")
    ? String(page || DEFAULT_PAGE)
    : `/${page}`;
  return `${buildServiceBaseUrl({ host, port })}${normalizedPage}`;
}

function buildBackendEnv({ baseEnv = process.env, dataRoot = "", appMode = "electron" } = {}) {
  const env = { ...baseEnv };
  if (dataRoot) {
    env.TRACE_DATA_ROOT = dataRoot;
  }
  env.TRACE_APP_MODE = appMode;
  return env;
}

function parseDotEnvText(text = "") {
  const result = {};
  String(text)
    .split(/\r?\n/)
    .forEach((rawLine) => {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) {
        return;
      }
      const separator = line.indexOf("=");
      if (separator <= 0) {
        return;
      }
      const key = line.slice(0, separator).trim();
      const value = line.slice(separator + 1).trim();
      if (!key) {
        return;
      }
      result[key] = value;
    });
  return result;
}

function hasRecognitionSeedData(root = "") {
  if (!root) {
    return false;
  }

  const requiredDirs = [
    path.join(root, "vectors", "persons.lance", "data"),
    path.join(root, "vectors", "person_embeddings.lance", "data"),
  ];

  return requiredDirs.every((dirPath) => {
    try {
      return require("node:fs")
        .readdirSync(dirPath, { withFileTypes: true })
        .some((entry) => entry.isFile());
    } catch (_error) {
      return false;
    }
  });
}

function ensureSeedData({ fsImpl = require("node:fs"), dataRoot = "", seedRoot = "", log = () => {} } = {}) {
  if (!dataRoot) {
    return { seeded: false, reason: "missing-data-root" };
  }

  fsImpl.mkdirSync(dataRoot, { recursive: true });
  const markerPath = path.join(dataRoot, DATA_SEED_MARKER);

  if (fsImpl.existsSync(markerPath)) {
    return { seeded: false, reason: "already-seeded" };
  }

  if (hasRecognitionSeedData(dataRoot)) {
    fsImpl.writeFileSync(markerPath, `${new Date().toISOString()}\n`, "utf-8");
    return { seeded: false, reason: "data-present" };
  }

  if (!seedRoot || !hasRecognitionSeedData(seedRoot)) {
    return { seeded: false, reason: "seed-missing" };
  }

  for (const subdir of SEEDED_DATA_SUBDIRS) {
    const sourcePath = path.join(seedRoot, subdir);
    if (!fsImpl.existsSync(sourcePath)) {
      continue;
    }
    const destinationPath = path.join(dataRoot, subdir);
    fsImpl.rmSync(destinationPath, { recursive: true, force: true });
    fsImpl.cpSync(sourcePath, destinationPath, { recursive: true, force: true });
  }

  fsImpl.writeFileSync(markerPath, `${new Date().toISOString()}\n`, "utf-8");
  log(`Seeded desktop data root from ${seedRoot}`);
  return { seeded: true, reason: "seed-copied" };
}

function buildWelcomeModel({
  username = "",
  hostname = "",
  platform = process.platform,
  appName = "TRACE-AML",
  version = "dev",
} = {}) {
  const cleanName = String(username || "").trim();
  const name = cleanName || "Operator";
  const platformLabel = prettyPlatform(platform);
  const safeHost = String(hostname || "").trim() || "LOCALHOST";

  return {
    appName,
    version,
    greeting: cleanName ? `Welcome, ${name}` : "Welcome",
    subtitle: `${appName} desktop session is ready on ${platformLabel} · ${safeHost}`,
    detail: "The local service will stay isolated to this machine while the operator workspace loads.",
    ctaLabel: "Launch Workspace",
    skipLabel: "Skip Intro",
    systemLabel: `${platformLabel} · ${safeHost}`,
  };
}

function buildBackendLaunchSpec({
  host = DEFAULT_HOST,
  port = DEFAULT_PORT,
  bundledBackendPath = "",
  projectRoot = process.cwd(),
  pythonCommand = "python",
  configPath = "config/config.demo.yaml",
} = {}) {
  if (bundledBackendPath) {
    return {
      command: bundledBackendPath,
      args: [
        "--config",
        configPath,
        "service",
        "run",
        "--host",
        String(host),
        "--port",
        String(port),
      ],
      cwd: path.dirname(bundledBackendPath),
    };
  }

  return {
    command: pythonCommand,
    args: [
      "-m",
      "trace_aml",
      "--config",
      configPath,
      "service",
      "run",
      "--host",
      String(host),
      "--port",
      String(port),
    ],
    cwd: projectRoot,
  };
}

function prettyPlatform(platform) {
  const label = String(platform || "").toLowerCase();
  if (label === "win32") return "Windows";
  if (label === "darwin") return "macOS";
  if (label === "linux") return "Linux";
  return platform || "Desktop";
}

module.exports = {
  DEFAULT_HOST,
  DEFAULT_PAGE,
  DEFAULT_PORT,
  buildBackendEnv,
  buildBackendLaunchSpec,
  buildFrontendUrl,
  buildServiceBaseUrl,
  buildWelcomeModel,
  ensureSeedData,
  hasRecognitionSeedData,
  parseDotEnvText,
  prettyPlatform,
  DATA_SEED_MARKER,
  SEEDED_DATA_SUBDIRS,
};
