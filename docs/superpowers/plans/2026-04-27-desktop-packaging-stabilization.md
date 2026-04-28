# TRACE-AML Desktop Packaging Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Windows Electron build that starts reliably without a separate terminal backend, keeps encrypted data readable, preserves embedding persistence, and is materially smaller than the current 1.89 GB package.

**Architecture:** Stop shipping copied development virtualenvs inside Electron. Build the Python backend as its own distributable runtime artifact, package that artifact into Electron, and keep Electron responsible only for splash/welcome/window orchestration plus health checks. Preserve Python-from-venv launch only for local development. Use a dedicated desktop backend entrypoint and a shareable desktop config so future rebuilds stay predictable.

**Tech Stack:** Electron, electron-builder, FastAPI, TRACE-AML Python service, `.venv311`, PyInstaller, PowerShell, Node.js

---

## File Structure Map

**Existing files to modify**
- `D:\github FORK\TRACE-ML\electron\main.js`
  Packaged-mode backend resolution, process launch, and startup diagnostics.
- `D:\github FORK\TRACE-ML\electron\runtime.js`
  Launch-spec abstraction for dev mode vs packaged backend executable mode.
- `D:\github FORK\TRACE-ML\electron\package.json`
  Remove bundled virtualenvs from `extraResources`; package compiled backend output only.
- `D:\github FORK\TRACE-ML\src\trace_aml\core\config.py`
  Ensure logs, exports, and portable runtime storage stay under `TRACE_DATA_ROOT`.
- `D:\github FORK\TRACE-ML\src\trace_aml\core\logger.py`
  Keep Windows-safe logger fallback and direct packaged logs to an app-owned path.
- `D:\github FORK\TRACE-ML\README.md`
  Replace the current demo-shell guidance with the stable desktop build workflow.

**New files to create**
- `D:\github FORK\TRACE-ML\src\trace_aml\desktop_backend.py`
  Dedicated packaged-backend entrypoint that preserves `.env`, `TRACE_DATA_ROOT`, and host/port arguments.
- `D:\github FORK\TRACE-ML\config\config.desktop.yaml`
  Shareable desktop config that disables external notification channels by default.
- `D:\github FORK\TRACE-ML\scripts\build_backend.ps1`
  One command to build the packaged Python backend from `.venv311`.
- `D:\github FORK\TRACE-ML\packaging\trace_aml_backend.spec`
  PyInstaller spec defining entrypoint, data files, hidden imports, and output folder.
- `D:\github FORK\TRACE-ML\tests\test_electron_runtime.py`
  Focused tests for packaged launch-spec selection and backend artifact discovery.
- `D:\github FORK\TRACE-ML\docs\desktop-build.md`
  Operator-facing build and smoke-test guide.

**Build outputs to standardize**
- `D:\github FORK\TRACE-ML\build\backend\trace_aml_backend\`
  Backend runtime folder produced by PyInstaller.
- `D:\github FORK\TRACE-ML\electron\dist\win-unpacked\`
  Electron unpacked app.
- `D:\github FORK\TRACE-ML\electron\dist\TRACE-AML-Setup-<version>-x64.exe`
  Windows installer.

## Root Cause Summary

- The current Electron package is shipping `D:\github FORK\TRACE-ML\.venv` and `D:\github FORK\TRACE-ML\.venv311` as raw resources.
- `.venv` alone is roughly 3333 MB of library payload; `.venv311` is roughly 1086 MB.
- The resulting installer and portable executable are about 1.89 GB because the app is behaving like a zipped dev environment.
- The bundled backend itself can start when launched directly from `electron\dist\win-unpacked\resources\backend\.venv311`, so the correct long-term fix is not “keep patching the copied venv.” The correct fix is “stop shipping copied venvs.”
- The packaged app currently lacks strong startup diagnostics, so failures surface as “Connection Lost” instead of actionable backend startup evidence.
- Portable storage is only partially finished today: `TRACE_DATA_ROOT` exists, but some log/export defaults still point to `data/...` directly.
- The compiled backend entrypoint must preserve `.env` loading so `TRACE_VAULT_KEY`, encrypted SMTP settings, and vector/vault paths remain readable in packaged mode.

## Stability Requirements

- **Vault compatibility:** packaged mode must keep `TRACE_VAULT_KEY` available so encrypted portraits, enrollment blobs, and evidence remain readable.
- **Embedding persistence:** LanceDB vectors and the in-memory embedding gallery cache must continue to load from the portable `TRACE_DATA_ROOT`.
- **Portable writes:** logs, exports, screenshots, portraits, vectors, and vault indexes must land under the user-specific data root, not inside the install directory.
- **Safe sharing defaults:** the shareable desktop config should disable outbound email and WhatsApp actions unless explicitly re-enabled.
- **Future rebuild ergonomics:** one script should rebuild the backend, one command should rebuild Electron, and the packaged launch path should not depend on a copied local development environment.

## Task 1: Freeze the Current Debug Path and Measure the Baseline

**Files:**
- Modify: `D:\github FORK\TRACE-ML\README.md`
- Create: `D:\github FORK\TRACE-ML\docs\desktop-build.md`

- [ ] **Step 1: Document the current known-good backend command**

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\trace-aml.exe --config config/config.demo.yaml service run --host 127.0.0.1 --port 18080
```

- [ ] **Step 2: Record the current oversized package evidence**

```powershell
Get-ChildItem -Path .venv,.venv311 -Force |
  Select-Object FullName,@{Name='SizeMB';Expression={[math]::Round((Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum / 1MB,2)}} |
  Format-Table -AutoSize
```

Expected: `.venv` is multiple gigabytes, `.venv311` is about one gigabyte, proving the packaging design is the size problem.

- [ ] **Step 3: Write the docs entry telling developers not to ship copied virtualenvs**

```markdown
## Desktop Packaging Rule

Do not add `.venv` or `.venv311` to Electron `extraResources`.
Desktop builds must package a compiled backend artifact, not a development virtualenv.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/desktop-build.md
git commit -m "docs: define stable desktop packaging baseline"
```

## Task 2: Split Backend Launch Into Dev Mode and Packaged Mode

**Files:**
- Modify: `D:\github FORK\TRACE-ML\electron\runtime.js`
- Modify: `D:\github FORK\TRACE-ML\electron\main.js`
- Test: `D:\github FORK\TRACE-ML\tests\test_electron_runtime.py`

- [ ] **Step 1: Write the failing runtime selection test**

```python
from pathlib import Path

from trace_aml.desktop_runtime import resolve_backend_launch


def test_packaged_launch_prefers_compiled_backend(tmp_path: Path) -> None:
    backend_dir = tmp_path / "backend-dist"
    backend_dir.mkdir()
    backend_exe = backend_dir / "trace-aml-backend.exe"
    backend_exe.write_text("stub", encoding="utf-8")

    spec = resolve_backend_launch(
        packaged=True,
        backend_root=tmp_path,
        host="127.0.0.1",
        port=18080,
    )

    assert spec.command == str(backend_exe)
    assert spec.args == ["--config", "config/config.desktop.yaml", "service", "run", "--host", "127.0.0.1", "--port", "18080"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\python.exe -m pytest tests/test_electron_runtime.py -q
```

Expected: FAIL because the packaged launch path still resolves to Python.

- [ ] **Step 3: Add a dedicated backend launch resolver**

```javascript
function buildBackendLaunchSpec({
  host = DEFAULT_HOST,
  port = DEFAULT_PORT,
  bundledBackendPath = "",
  projectRoot = process.cwd(),
  pythonCommand = "python",
  configPath = "config/config.desktop.yaml",
} = {}) {
  if (bundledBackendPath) {
    return {
      command: bundledBackendPath,
      args: ["--config", configPath, "service", "run", "--host", String(host), "--port", String(port)],
      cwd: projectRoot || path.dirname(bundledBackendPath),
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
```

- [ ] **Step 4: Update `electron/main.js` so packaged mode resolves `backend-dist\\trace-aml-backend.exe`**

```javascript
function resolveBundledBackendExecutable(root) {
  const candidate = path.join(root, "backend-dist", "trace-aml-backend.exe");
  return fs.existsSync(candidate) ? candidate : "";
}

const launchSpec = buildBackendLaunchSpec({
  host: SERVICE_HOST,
  port: SERVICE_PORT,
  projectRoot: root,
  bundledBackendPath: app.isPackaged ? resolveBundledBackendExecutable(root) : "",
  pythonCommand: resolvePythonCommand(root),
  configPath: app.isPackaged ? "config/config.desktop.yaml" : "config/config.demo.yaml",
});
```

- [ ] **Step 5: Re-run the runtime test**

Run:

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\python.exe -m pytest tests/test_electron_runtime.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add electron/main.js electron/runtime.js tests/test_electron_runtime.py
git commit -m "feat: separate dev and packaged backend launch modes"
```

## Task 3: Make Portable Storage Deterministic

**Files:**
- Modify: `D:\github FORK\TRACE-ML\src\trace_aml\core\config.py`
- Modify: `D:\github FORK\TRACE-ML\tests\test_config.py`

- [ ] **Step 1: Write failing config tests for portable defaults**

```python
def test_logging_default_uses_trace_data_root(monkeypatch):
    monkeypatch.setenv("TRACE_DATA_ROOT", "portable_data")
    from trace_aml.core.config import LoggingSettings
    assert LoggingSettings().file_path == "portable_data/logs/trace_aml.log"
```

- [ ] **Step 2: Run the config tests to verify the gap**

Run:

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\python.exe -m pytest tests/test_config.py -q
```

Expected: FAIL because logging and export defaults still point to `data/...`.

- [ ] **Step 3: Update portable defaults**

```python
class PdfReportSettings(BaseModel):
    output_dir: str = f"{_DATA_ROOT}/exports"


class LoggingSettings(BaseModel):
    file_path: str = f"{_DATA_ROOT}/logs/trace_aml.log"
```

- [ ] **Step 4: Re-run the config tests**

Run:

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\python.exe -m pytest tests/test_config.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trace_aml/core/config.py tests/test_config.py
git commit -m "fix: make desktop runtime paths portable"
```

## Task 4: Build the Backend as a Standalone Artifact

**Files:**
- Create: `D:\github FORK\TRACE-ML\src\trace_aml\desktop_backend.py`
- Create: `D:\github FORK\TRACE-ML\config\config.desktop.yaml`
- Create: `D:\github FORK\TRACE-ML\packaging\trace_aml_backend.spec`
- Create: `D:\github FORK\TRACE-ML\scripts\build_backend.ps1`

- [ ] **Step 1: Add the dedicated desktop backend entrypoint**

```python
from trace_aml.cli import app


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Add the desktop-share config**

```yaml
app:
  environment: desktop

notifications:
  email:
    enabled: false
  whatsapp:
    enabled: false
```

- [ ] **Step 3: Add the PyInstaller build script**

```powershell
param(
  [string]$PythonExe = ".\\.venv311\\Scripts\\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "build\\backend"
$spec = Join-Path $root "packaging\\trace_aml_backend.spec"

Remove-Item -LiteralPath $dist -Recurse -Force -ErrorAction SilentlyContinue
& $PythonExe -m PyInstaller --noconfirm $spec
```

- [ ] **Step 4: Add the backend spec**

```python
a = Analysis(
    ["src/trace_aml/desktop_backend.py"],
    pathex=[],
    datas=[
        ("config", "config"),
        ("models", "models"),
        ("src/frontend", "src/frontend"),
        (".env", "."),
    ],
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "fastapi",
    ],
)
```

- [ ] **Step 5: Install PyInstaller into the working build environment**

Run:

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\python.exe -m pip install pyinstaller
```

Expected: `Successfully installed pyinstaller ...`

- [ ] **Step 6: Build the backend**

Run:

```powershell
cd "D:\github FORK\TRACE-ML"
.\scripts\build_backend.ps1
```

Expected: a folder exists at `build\backend\trace_aml_backend\` containing `trace-aml-backend.exe`

- [ ] **Step 7: Smoke-test the compiled backend alone**

Run:

```powershell
cd "D:\github FORK\TRACE-ML\build\backend\trace_aml_backend"
.\trace-aml-backend.exe --config config/config.desktop.yaml service run --host 127.0.0.1 --port 18082
```

In a second terminal:

```powershell
Invoke-WebRequest http://127.0.0.1:18082/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:18082/api/v1/live/snapshot -UseBasicParsing
```

Expected: both return `200`

- [ ] **Step 8: Verify vault/index/vector portability**

Run:

```powershell
Get-ChildItem -Path "$env:APPDATA\\TRACE-AML" -Recurse
```

Expected: vectors, vault blobs, indexes, exports, and logs appear under the portable user-data tree rather than the install folder.

- [ ] **Step 9: Commit**

```bash
git add src/trace_aml/desktop_backend.py config/config.desktop.yaml packaging/trace_aml_backend.spec scripts/build_backend.ps1
git commit -m "build: compile trace-aml backend for desktop packaging"
```

## Task 5: Add Packaged Startup Diagnostics

**Files:**
- Modify: `D:\github FORK\TRACE-ML\electron\main.js`
- Modify: `D:\github FORK\TRACE-ML\src\trace_aml\core\logger.py`

- [ ] **Step 1: Write logs to a deterministic packaged location**

```javascript
function packagedLogPath() {
  return path.join(app.getPath("userData"), "logs", "backend.log");
}
```

- [ ] **Step 2: Inject log path into backend env**

```javascript
env.TRACE_LOG_FILE = packagedLogPath();
```

- [ ] **Step 3: Respect `TRACE_LOG_FILE` in Python logger configuration**

```python
log_path = os.getenv("TRACE_LOG_FILE")
if log_path:
    sink_path = log_path
else:
    sink_path = default_log_path
```

- [ ] **Step 4: Surface backend exit code and last log lines in the splash screen**

```javascript
backendProcess.on("exit", (code, signal) => {
  publishSplashState({
    stage: "Startup failed",
    detail: `Backend exited (${code ?? "unknown"}${signal ? ` / ${signal}` : ""})`,
    failed: true,
    progress: 100,
  });
});
```

- [ ] **Step 5: Verify the failure path intentionally**

Run:

```powershell
$env:TRACE_ELECTRON_PORT="19999"
cd "D:\github FORK\TRACE-ML\electron"
npm start
```

Expected: splash shows actionable startup failure details instead of dropping directly to a silent “Connection Lost” experience.

- [ ] **Step 6: Commit**

```bash
git add electron/main.js src/trace_aml/core/logger.py
git commit -m "feat: add packaged backend startup diagnostics"
```

## Task 6: Rewire Electron Packaging to Ship Only the Compiled Backend

**Files:**
- Modify: `D:\github FORK\TRACE-ML\electron\package.json`

- [ ] **Step 1: Remove `.venv` and `.venv311` from `extraResources`**

Delete:

```json
{
  "from": "../.venv",
  "to": "backend/.venv"
},
{
  "from": "../.venv311",
  "to": "backend/.venv311"
}
```

- [ ] **Step 2: Add the compiled backend output instead**

```json
{
  "from": "../build/backend/trace_aml_backend",
  "to": "backend/backend-dist"
}
```

- [ ] **Step 3: Keep only required data/config assets**

Expected final `extraResources` set:
- `config`
- `models`
- `.env` only if still required for demo mode
- `build/backend/trace_aml_backend`

- [ ] **Step 4: Rebuild the Electron app**

Run:

```powershell
cd "D:\github FORK\TRACE-ML\electron"
npm run dist
```

Expected: installer size drops substantially because raw virtualenvs are no longer included.

- [ ] **Step 5: Record before/after artifact size**

Run:

```powershell
Get-ChildItem -Path dist | Select-Object Name,@{Name='SizeMB';Expression={[math]::Round($_.Length / 1MB, 2)}} | Format-Table -AutoSize
```

Expected: package size is materially smaller than 1896 MB.

- [ ] **Step 6: Commit**

```bash
git add electron/package.json
git commit -m "build: package compiled backend instead of virtualenvs"
```

## Task 7: End-to-End Verification Before Sharing

**Files:**
- Modify: `D:\github FORK\TRACE-ML\docs\desktop-build.md`

- [ ] **Step 1: Verify unpacked app self-starts**

Run:

```powershell
taskkill /f /im TRACE-AML.exe
taskkill /f /im python.exe
cd "D:\github FORK\TRACE-ML\electron\dist\win-unpacked"
.\TRACE-AML.exe
```

Expected: splash -> welcome -> main workspace, with no manual backend terminal.

- [ ] **Step 2: Verify installer build self-starts**

Run installer:

```powershell
cd "D:\github FORK\TRACE-ML\electron\dist"
.\TRACE-AML-Setup-3.0.0-x64.exe
```

Expected: installed app launches and reaches the workspace without the offline overlay.

- [ ] **Step 3: Verify critical endpoints after launch**

Run:

```powershell
Invoke-WebRequest http://127.0.0.1:18080/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:18080/api/v1/live/snapshot -UseBasicParsing
```

Expected: both return `200` while the desktop app is open.

- [ ] **Step 4: Verify the frontend is no longer falling back to `8080`**

Run:

```powershell
Get-Content "D:\github FORK\TRACE-ML\src\frontend\shared\trace_client.js" |
  Select-String "same origin > localhost:8080"
```

Expected: same-origin logic exists and packaged UI binds to the backend it launched.

- [ ] **Step 5: Update the distribution instructions**

```markdown
## Shareable Windows Build

Primary artifact: `electron/dist/TRACE-AML-Setup-<version>-x64.exe`
Fallback artifact: `electron/dist/win-unpacked/TRACE-AML.exe`
Do not share old portable builds created from bundled virtualenv packages.
```

- [ ] **Step 6: Commit**

```bash
git add docs/desktop-build.md
git commit -m "docs: finalize stable desktop build workflow"
```

## Expected Outcome

When this plan is complete:
- Electron packaged mode no longer depends on shipping a development virtualenv.
- Backend startup is deterministic and debuggable.
- Package size drops substantially.
- Encrypted vault data and embeddings remain readable in packaged mode.
- The app can be shared as a normal Windows desktop build instead of a giant dev-environment bundle.

## Execution Order Recommendation

1. Task 3
2. Task 2
3. Task 4
4. Task 5
5. Task 6
6. Task 7

Task 1 can be completed in parallel, but the current unblockers are portable storage correctness and the compiled backend path.
