const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");

const { app, BrowserWindow, ipcMain, session } = require("electron");

const {
  buildBackendEnv,
  buildBackendLaunchSpec,
  buildFrontendUrl,
  buildServiceBaseUrl,
  buildWelcomeModel,
  ensureSeedData,
  parseDotEnvText,
} = require("./runtime");

const SERVICE_HOST = process.env.TRACE_ELECTRON_HOST || "127.0.0.1";
const SERVICE_PORT = Number.parseInt(process.env.TRACE_ELECTRON_PORT || "18080", 10);
const STARTUP_TIMEOUT_MS = 120000;
const HEALTH_RETRY_MS = 1000;
const LOG_TAIL_LIMIT = 8;
const FRONTEND_LAUNCH_NONCE = `${Date.now()}`;
const APP_USER_MODEL_ID = "com.traceaml.desktop";

let splashWindow = null;
let welcomeWindow = null;
let mainWindow = null;
let backendProcess = null;
let isQuitting = false;

let splashState = {
  stage: "Preparing desktop shell",
  detail: "Initializing TRACE-AML desktop runtime.",
  progress: 6,
  failed: false,
  ready: false,
  logs: [],
};

function repoRoot() {
  return path.resolve(__dirname, "..");
}

function backendRoot() {
  return app.isPackaged ? path.join(process.resourcesPath, "backend") : repoRoot();
}

function desktopIconPath() {
  const packagedResourceIcon = path.join(process.resourcesPath, "app.ico");
  const packagedIcon = path.join(process.resourcesPath, "app.asar.unpacked", "(1).ico");
  const packagedInlineIcon = path.join(process.resourcesPath, "app.asar", "(1).ico");
  const devIcon = path.join(repoRoot(), "(1).ico");

  if (app.isPackaged) {
    if (fs.existsSync(packagedResourceIcon)) {
      return packagedResourceIcon;
    }
    if (fs.existsSync(packagedIcon)) {
      return packagedIcon;
    }
    if (fs.existsSync(packagedInlineIcon)) {
      return packagedInlineIcon;
    }
  }

  return fs.existsSync(devIcon) ? devIcon : undefined;
}

function serviceBaseUrl() {
  return buildServiceBaseUrl({ host: SERVICE_HOST, port: SERVICE_PORT });
}

function frontendUrl() {
  const baseUrl = buildFrontendUrl({ host: SERVICE_HOST, port: SERVICE_PORT });
  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}desktop_launch=${encodeURIComponent(FRONTEND_LAUNCH_NONCE)}`;
}

function userDataRoot() {
  return path.join(app.getPath("userData"), "TRACE-AML");
}

function backendLogPath() {
  return path.join(userDataRoot(), "logs", "backend.log");
}

function bundledSeedRoot() {
  return app.isPackaged ? path.join(process.resourcesPath, "seed-data") : path.join(repoRoot(), "data");
}

function resolvePythonCommand(root) {
  const candidates = [
    path.join(root, ".venv311", "Scripts", "python.exe"),
    path.join(root, ".venv", "Scripts", "python.exe"),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return "python";
}

function readDotEnv(root) {
  const envPath = path.join(root, ".env");
  if (!fs.existsSync(envPath)) {
    return {};
  }
  return parseDotEnvText(fs.readFileSync(envPath, "utf-8"));
}

function buildBackendProcessEnv(root, { includeSourcePath = true } = {}) {
  const mergedBaseEnv = {
    ...process.env,
    ...readDotEnv(root),
  };
  const env = buildBackendEnv({
    baseEnv: mergedBaseEnv,
    dataRoot: userDataRoot(),
    appMode: "electron",
  });
  env.TRACE_LOG_FILE = backendLogPath();
  if (includeSourcePath) {
    const sourceRoot = path.join(root, "src");
    env.PYTHONPATH = env.PYTHONPATH
      ? `${sourceRoot}${path.delimiter}${env.PYTHONPATH}`
      : sourceRoot;
  }
  return env;
}

function ensureUserDataSeeded() {
  const result = ensureSeedData({
    dataRoot: userDataRoot(),
    seedRoot: bundledSeedRoot(),
    log: rememberLog,
  });

  if (result.reason === "seed-missing") {
    rememberLog(`[Electron] Seed data missing at ${bundledSeedRoot()}`);
  } else if (result.reason === "already-seeded" || result.reason === "data-present") {
    rememberLog(`[Electron] Data root ready at ${userDataRoot()}`);
  }
}

function rememberLog(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed) {
    return;
  }
  splashState.logs = [...splashState.logs, trimmed].slice(-LOG_TAIL_LIMIT);
  publishSplashState();
}

function publishSplashState(patch = {}) {
  splashState = {
    ...splashState,
    ...patch,
  };

  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.send("trace:splash-status", splashState);
  }
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 1120,
    height: 720,
    frame: false,
    resizable: false,
    minimizable: false,
    maximizable: false,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: "#09111a",
    icon: desktopIconPath(),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  splashWindow.once("ready-to-show", () => splashWindow.show());
  splashWindow.on("closed", () => {
    splashWindow = null;
  });
  splashWindow.loadFile(path.join(__dirname, "splash.html"));
}

function createWelcomeWindow() {
  if (welcomeWindow && !welcomeWindow.isDestroyed()) {
    welcomeWindow.focus();
    return;
  }

  welcomeWindow = new BrowserWindow({
    width: 1180,
    height: 780,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: "#081018",
    icon: desktopIconPath(),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  welcomeWindow.once("ready-to-show", () => welcomeWindow.show());
  welcomeWindow.on("closed", () => {
    welcomeWindow = null;
    if (!mainWindow && !isQuitting) {
      app.quit();
    }
  });
  welcomeWindow.loadFile(path.join(__dirname, "welcome.html"));
}

async function clearRendererCache(targetSession = session.defaultSession) {
  if (!targetSession) {
    return;
  }
  try {
    await targetSession.clearCache();
    await targetSession.clearStorageData({ storages: ["cachestorage"] });
  } catch (error) {
    rememberLog(
      `[Electron] Renderer cache clear failed: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

async function createMainWindow() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.focus();
    return mainWindow;
  }

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: "#131313",
    icon: desktopIconPath(),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  await clearRendererCache(mainWindow.webContents.session);
  await mainWindow.loadURL(frontendUrl());
  return mainWindow;
}

function startBackendProcess() {
  if (backendProcess) {
    return;
  }

  ensureUserDataSeeded();
  const root = backendRoot();
  const bundledBackendPath = app.isPackaged ? resolveBundledBackendExecutable(root) : "";
  if (app.isPackaged && !bundledBackendPath) {
    publishSplashState({
      stage: "Startup failed",
      detail: "Bundled backend executable is missing from the packaged app.",
      failed: true,
      ready: false,
      progress: 100,
    });
    return;
  }
  const launchRoot = bundledBackendPath ? path.dirname(bundledBackendPath) : root;
  const env = buildBackendProcessEnv(launchRoot, {
    includeSourcePath: !bundledBackendPath,
  });
  const launchSpec = buildBackendLaunchSpec({
    host: SERVICE_HOST,
    port: SERVICE_PORT,
    projectRoot: root,
    bundledBackendPath,
    pythonCommand: resolvePythonCommand(root),
    configPath: app.isPackaged ? "_internal/config/config.desktop.yaml" : "config/config.demo.yaml",
  });

  publishSplashState({
    stage: "Launching local service",
    detail: `${path.basename(launchSpec.command)} ${launchSpec.args.join(" ")}`,
    progress: 22,
  });

  backendProcess = spawn(launchSpec.command, launchSpec.args, {
    cwd: launchSpec.cwd,
    env,
    windowsHide: true,
  });

  backendProcess.stdout.on("data", (chunk) => {
    String(chunk)
      .split(/\r?\n/)
      .forEach(rememberLog);
  });

  backendProcess.stderr.on("data", (chunk) => {
    String(chunk)
      .split(/\r?\n/)
      .forEach(rememberLog);
  });

  backendProcess.on("exit", (code, signal) => {
    backendProcess = null;
    if (isQuitting) {
      return;
    }

    publishSplashState({
      stage: "Startup failed",
      detail: `Backend exited unexpectedly (${code ?? "unknown"}${signal ? ` / ${signal}` : ""}).`,
      failed: true,
      ready: false,
      progress: 100,
    });
  });
}

function resolveBundledBackendExecutable(root) {
  const candidate = path.join(root, "backend-dist", "trace-aml-backend.exe");
  return fs.existsSync(candidate) ? candidate : "";
}

async function waitForServiceReady() {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;

  publishSplashState({
    stage: "Warming operator workspace",
    detail: `Waiting for ${serviceBaseUrl()}/health`,
    progress: 44,
  });

  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${serviceBaseUrl()}/health`, {
        headers: { Accept: "application/json" },
      });

      if (response.ok) {
        publishSplashState({
          stage: "Workspace ready",
          detail: "FastAPI bridge responded successfully. Desktop session is ready.",
          progress: 100,
          failed: false,
          ready: true,
        });
        return;
      }
    } catch (_error) {
      // Expected while the backend is still starting.
    }

    await sleep(HEALTH_RETRY_MS);
    publishSplashState({
      progress: Math.min(splashState.progress + 4, 92),
      detail: `Connecting to ${serviceBaseUrl()} ...`,
    });
  }

  throw new Error(`Timed out waiting for ${serviceBaseUrl()}/health`);
}

function welcomePayload() {
  return {
    ...buildWelcomeModel({
      username: os.userInfo().username,
      hostname: os.hostname(),
      platform: process.platform,
      appName: "TRACE-AML",
      version: app.getVersion(),
    }),
    serviceUrl: frontendUrl(),
    dataRoot: userDataRoot(),
  };
}

function shutdownBackend() {
  if (!backendProcess) {
    return;
  }

  backendProcess.kill();
  backendProcess = null;
}

async function bootstrapDesktop() {
  createSplashWindow();
  publishSplashState({
    stage: "Preparing desktop shell",
    detail: "Loading Electron shell and desktop assets.",
    progress: 10,
  });
  startBackendProcess();
  await waitForServiceReady();
  await sleep(500);
  await createMainWindow();
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.close();
  }
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

ipcMain.on("trace:renderer-ready", () => {
  publishSplashState();
});

ipcMain.handle("trace:get-splash-state", () => splashState);
ipcMain.handle("trace:get-welcome-model", () => welcomePayload());
ipcMain.handle("trace:launch-workspace", async () => {
  await createMainWindow();
  if (welcomeWindow && !welcomeWindow.isDestroyed()) {
    welcomeWindow.close();
  }
  return { ok: true, url: frontendUrl() };
});
ipcMain.handle("trace:quit-app", () => {
  app.quit();
  return { ok: true };
});

app.whenReady().then(async () => {
  app.setAppUserModelId(APP_USER_MODEL_ID);
  await bootstrapDesktop();
}).catch((error) => {
  publishSplashState({
    stage: "Startup failed",
    detail: error instanceof Error ? error.message : String(error),
    failed: true,
    ready: false,
    progress: 100,
  });
});

app.on("before-quit", () => {
  isQuitting = true;
  shutdownBackend();
});

app.on("window-all-closed", () => {
  app.quit();
});

app.on("activate", async () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    await bootstrapDesktop();
  }
});
