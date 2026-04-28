# TRACE-AML v3

CLI-first face recognition demo stack for final-year-project presentation.

## What This Version Delivers
- Fresh package-based runtime (`trace_aml/`) with `trace-aml` command surface.
- InsightFace `buffalo_sc` recognition pipeline (SCRFD + ArcFace embeddings).
- LanceDB for person/embedding/detection storage.
- DuckDB analytics helpers for history queries, summary, and CSV export.
- Threaded capture/inference live session tuned for built-in laptop webcam (`device 0`).
- Quality core hardening: per-image scoring, enrollment lifecycle states, and strict active-only matching.
- Temporal decisioning: track-aware smoothing and `accept/review/reject` decisions.
- Rich-styled tactical CLI live view with event feed and health strip for demo-ready presentation.

## Quick Setup
Recommended Python version: **3.11** (best compatibility for InsightFace on Windows).

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Verify Environment
```powershell
trace-aml doctor
```

## Basic Workflow
1. Register a person:
```powershell
trace-aml person add --name "John Doe" --category criminal --images-dir "C:\faces\john"
```
or capture from webcam:
```powershell
trace-aml person add --name "John Doe" --category criminal --capture-count 10
```
or append more captures to an existing person (recommended for robustness):
```powershell
trace-aml person capture --person-id PRC004 --capture-count 20 --capture-mode manual
```

2. Build embeddings:
```powershell
trace-aml train rebuild
```

3. Run live recognition:
```powershell
trace-aml recognize live
```

4. Query history:
```powershell
trace-aml history query --limit 20
```

5. Show summary:
```powershell
trace-aml report summary
```

6. Audit enrollment quality/lifecycle:
```powershell
trace-aml person audit --apply
trace-aml report quality
```

7. Export CSV:
```powershell
trace-aml export csv
```

8. Run Service Layer (for UI integration):
```powershell
trace-aml service run --host 127.0.0.1 --port 8080
```
Key endpoints:
- `GET /api/v1/live/snapshot`
- `GET /api/v1/entities`
- `GET /api/v1/incidents`
- `GET /api/v1/timeline`
- `GET /api/v1/events/stream` (SSE)

## Demo Script (Examiner Flow)
1. `trace-aml doctor`
2. `trace-aml person list`
3. `trace-aml train rebuild`
4. `trace-aml recognize live` (show live webcam recognition and overlay)
5. `trace-aml history query --limit 10`
6. `trace-aml report summary`
7. `trace-aml export csv`

## Config
Default config is `config.yaml`.

You can pass a custom config:
```powershell
trace-aml --config .\config.yaml doctor
```
Profile presets:
```powershell
trace-aml --config .\config.demo.yaml recognize live
trace-aml --config .\config.strict.yaml recognize live
```

Important MVP constraint:
- `camera.device_index` must stay `0` in this cycle.

## Testing
```powershell
pytest
```

## Notes
- Legacy scripts remain in repository as historical reference, but v3 runtime is `trace-aml`.
- Liveness is scaffolded and disabled by default (hooks and metrics are in place for later enforcement).
- Records that do not pass quality/lifecycle gates are blocked from active recognition by default.
- GUI integration (Electron or other framework) can call these same core services later.

## Electron Desktop Demo
An Electron shell now lives in `electron/package.json`. The stable packaging path is:
- build a standalone Python backend artifact
- package that backend artifact into Electron
- launch the desktop app against `config/config.desktop.yaml`

This avoids shipping a full development virtualenv inside the app bundle.

### Local Run
```powershell
.\scripts\run_electron_demo.ps1
```

Manual equivalent:
```powershell
cd .\electron
npm install
npm test
npm start
```

Notes:
- The Electron shell defaults to `127.0.0.1:18080` so it does not collide with a browser-mode service already running on `127.0.0.1:8080`.
- Desktop data is redirected into the OS user-data directory through `TRACE_DATA_ROOT`.
- The Electron launcher forwards values from `.env` into the child Python process so the vault key remains available.
- The shareable desktop profile is `config/config.desktop.yaml`, which disables outbound email/WhatsApp channels by default.

### Windows Packaging
```powershell
.\.venv311\Scripts\python.exe -m pip install pytest pyinstaller
cd .\electron
npm install
npm run dist
```

Expected outputs land in `electron/dist/`:
- NSIS installer
- Portable Windows executable

Packaging now targets a compiled backend artifact rather than bundling `.venv` directly.
See `docs/desktop-build.md` for the full rebuild and smoke-test workflow.
