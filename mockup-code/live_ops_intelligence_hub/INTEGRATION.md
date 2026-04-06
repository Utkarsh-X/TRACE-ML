# TRACE-AML UI integration

All mockup pages under `mockup-code/` share `mockup-code/shared/trace_api.js`. **Serve the `mockup-code` folder** so `../shared/` resolves:

```powershell
npx http-server .\mockup-code -p 8081 -c-1 --proxy http://127.0.0.1:8080?
```

## Backend (API + optional in-process recognition)

Single terminal — service only (snapshot/timeline update when something else writes to the store; SSE only for service-process events):

```powershell
py -3 -m trace_aml --config config.demo.yaml service run --host 127.0.0.1 --port 8080
```

**End-to-end live pipeline (webcam → detection → same SSE as the UI):** add `--live`. Do **not** run `recognize live` in another terminal (camera conflict).

```powershell
py -3 -m trace_aml --config config.demo.yaml service run --host 127.0.0.1 --port 8080 --live
```

Additional API used by the UI:

- `GET /api/v1/live/overlay` — normalized detection boxes (when `--live` is running)
- `GET /api/v1/entities`, `GET /api/v1/entities/{id}/profile`
- `GET /api/v1/incidents`, `GET /api/v1/incidents/{id}`
- `PATCH /api/v1/incidents/{id}/severity` — body `{ "severity": "low"|"medium"|"high" }`
- `POST /api/v1/incidents/{id}/close`

## Pages (from `mockup-code` root)

| Page | URL |
|------|-----|
| Live Ops | `http://127.0.0.1:8081/live_ops_intelligence_hub/code.html` |
| Entities | `http://127.0.0.1:8081/entity_explorer_clean_canvas/code.html` |
| Incidents | `http://127.0.0.1:8081/incidents_forensic_detail/code.html` |
| History | `http://127.0.0.1:8081/global_history_timeline_forensic_canvas/code.html` |

Optional API override on any page: `?api=http://127.0.0.1:8080` (CORS is enabled on the service for dev).
