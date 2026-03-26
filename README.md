# TRACE-ML v3

CLI-first face recognition demo stack for final-year-project presentation.

## What This Version Delivers
- Fresh package-based runtime (`trace_ml/`) with `trace-ml` command surface.
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
trace-ml doctor
```

## Basic Workflow
1. Register a person:
```powershell
trace-ml person add --name "John Doe" --category criminal --images-dir "C:\faces\john"
```
or capture from webcam:
```powershell
trace-ml person add --name "John Doe" --category criminal --capture-count 10
```
or append more captures to an existing person (recommended for robustness):
```powershell
trace-ml person capture --person-id PRC004 --capture-count 20 --capture-mode manual
```

2. Build embeddings:
```powershell
trace-ml train rebuild
```

3. Run live recognition:
```powershell
trace-ml recognize live
```

4. Query history:
```powershell
trace-ml history query --limit 20
```

5. Show summary:
```powershell
trace-ml report summary
```

6. Audit enrollment quality/lifecycle:
```powershell
trace-ml person audit --apply
trace-ml report quality
```

7. Export CSV:
```powershell
trace-ml export csv
```

## Demo Script (Examiner Flow)
1. `trace-ml doctor`
2. `trace-ml person list`
3. `trace-ml train rebuild`
4. `trace-ml recognize live` (show live webcam recognition and overlay)
5. `trace-ml history query --limit 10`
6. `trace-ml report summary`
7. `trace-ml export csv`

## Config
Default config is `config.yaml`.

You can pass a custom config:
```powershell
trace-ml --config .\config.yaml doctor
```
Profile presets:
```powershell
trace-ml --config .\config.demo.yaml recognize live
trace-ml --config .\config.strict.yaml recognize live
```

Important MVP constraint:
- `camera.device_index` must stay `0` in this cycle.

## Testing
```powershell
pytest
```

## Notes
- Legacy scripts remain in repository as historical reference, but v3 runtime is `trace-ml`.
- Liveness is scaffolded and disabled by default (hooks and metrics are in place for later enforcement).
- Records that do not pass quality/lifecycle gates are blocked from active recognition by default.
- GUI integration (Electron or other framework) can call these same core services later.
