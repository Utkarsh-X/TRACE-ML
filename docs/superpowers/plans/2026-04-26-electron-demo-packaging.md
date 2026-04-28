# TRACE-AML Electron Demo Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing TRACE-AML FastAPI UI/service in a Windows-friendly Electron desktop app with a splash/loading flow, a simple welcome screen, and packaging commands while leaving the Python recognition core untouched.

**Architecture:** Electron will act as a thin shell. The Electron main process will launch the existing `trace-aml service run` backend, wait for the local health endpoint to report ready, and then load the already-built `/ui/...` frontend served by FastAPI. The splash window and welcome screen live in the Electron layer so the existing browser UI can remain the system of record for core app behavior.

**Tech Stack:** Electron, electron-builder, Node.js, existing FastAPI/Typer service, static HTML/CSS/JS frontend, pytest

---

### Task 1: Plan The Electron Surface

**Files:**
- Create: `electron/`
- Create: `docs/superpowers/plans/2026-04-26-electron-demo-packaging.md`
- Modify: `README.md`

- [ ] Confirm the existing backend entrypoint stays `trace-aml service run --host 127.0.0.1 --port <port>`
- [ ] Keep all recognition/storage logic in Python; add only shell/bootstrap code in Electron
- [ ] Define the minimum windows:
  - Splash / loading
  - Welcome screen
  - Main app window pointed at `http://127.0.0.1:<port>/ui/live_ops/index.html`

### Task 2: Add Electron Bootstrap Tests First

**Files:**
- Create: `tests/test_electron_bootstrap.py`

- [ ] Add tests for pure-Python bootstrap helpers that can be verified without launching Electron:
  - service URL generation
  - Electron runtime config payload generation
  - environment shaping for packaged mode
- [ ] Run the targeted pytest file and confirm the new tests fail before implementation

### Task 3: Implement Shared Electron Bootstrap Helpers

**Files:**
- Create: `src/trace_aml/electron/bootstrap.py`
- Modify: `src/trace_aml/__init__.py`

- [ ] Add small helper functions for:
  - normalizing the service host/port
  - choosing the frontend landing route
  - building a welcome payload from OS/user/runtime data
  - preparing environment variables such as `TRACE_DATA_ROOT`
- [ ] Re-run the targeted pytest file until it passes

### Task 4: Scaffold The Electron App

**Files:**
- Create: `electron/package.json`
- Create: `electron/main.js`
- Create: `electron/preload.js`
- Create: `electron/splash.html`
- Create: `electron/welcome.html`
- Create: `electron/renderer/welcome.js`
- Create: `electron/renderer/welcome.css`

- [ ] Add an Electron main process that:
  - starts the Python backend in a child process
  - polls `/health`
  - shows the splash window during startup
  - opens the welcome window when ready
  - opens the main window after the welcome CTA
- [ ] Configure `contextIsolation: true` and `nodeIntegration: false`
- [ ] Expose only a minimal preload API for welcome metadata and app-launch actions

### Task 5: Package And Document The Demo Flow

**Files:**
- Modify: `README.md`
- Create: `electron/.gitignore`
- Create: `scripts/run_electron_demo.ps1`

- [ ] Add `npm` scripts for local Electron run and Windows packaging
- [ ] Document local prerequisites and the handoff steps for sharing a build with friends
- [ ] Verify the Python tests and Electron packaging entrypoints with fresh command output before calling the work complete
