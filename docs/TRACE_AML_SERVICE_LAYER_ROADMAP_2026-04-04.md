# TRACE-AML Service Layer Roadmap

Date: 2026-04-04

## Purpose

This roadmap defines the execution plan for the Service Layer that bridges the existing intelligence backend to the UI, while preserving the stable core pipeline:

```text
Detection -> Event -> Alert -> Incident -> Action
```

## Scope

In scope:

- HTTP query surfaces for read models
- Incident operator controls (severity/close) through service endpoints
- Global and entity timeline access for UI
- Real-time stream transport preparation (SSE over existing publisher)
- CLI command to run service

Out of scope:

- GUI implementation
- New recognition model behavior
- Threshold retuning
- Full auth/multi-tenant concerns

## Service Principles

- Read-model first: endpoints return UI-shaped objects, not raw tables
- Keep pipeline untouched
- Keep CLI backward compatible
- Prefer deterministic, explainable responses
- Avoid overengineering before GUI integration

## Phase Plan

## Phase S1: Service Skeleton

Deliverables:

- `trace_aml/service/app.py` app factory
- health and root endpoints
- lazy dependency handling so CLI remains safe without web deps

Status: Completed

## Phase S2: Read-Model API Contract

Deliverables:

- Live snapshot endpoint
- Entity endpoints (list/detail/profile/timeline/incidents)
- Incident endpoints (list/detail)
- Alerts and actions endpoints
- Global timeline endpoint with filters

Status: Completed

## Phase S3: Operator Control Endpoints

Deliverables:

- `PATCH /incidents/{id}/severity`
- `POST /incidents/{id}/close`
- consistent 404 behavior for missing entities/incidents

Status: Completed

## Phase S4: Real-Time Bridge

Deliverables:

- SSE endpoint backed by in-memory publisher
- optional backfill and keepalive behavior

Status: Completed (transport-ready, single-process runtime)

## Phase S5: Runtime Integration

Deliverables:

- shared in-process publisher in CLI runtime
- `recognize live` publishes runtime events through same interface
- `service run` command to start API

Status: Completed

## Phase S6: Data Shape Hardening

Deliverables:

- timeline stable ordering guarantees
- timeline filtering support (`kinds`, `entity_id`, `incident_id`)
- short snapshot TTL cache for efficient polling

Status: Completed

## Phase S7: Validation

Deliverables:

- service API tests
- existing CLI/test suite regression pass

Status: Completed in code; final local run required per environment.

## Next Milestones (Follow-On Plan)

1. Frontend integration contract:
- freeze request/response schemas used by mockup pages
- map each page widget to endpoint(s)

2. Session-mode service strategy:
- decide whether GUI and recognition run same process or split process
- if split, upgrade streaming path from in-memory to broker/websocket relay

3. Security baseline:
- add API key or token guard for write operations (`set-severity`, `close`)
- add request audit logging

4. GUI wiring phase:
- Live Ops polling + stream hookup
- Incident detail + action timeline rendering
- Entity profile and history timeline rendering

## Definition Of Done For Service Layer

Service layer is considered complete for GUI start when:

- all read-model endpoints are live and stable
- incident control endpoints work end-to-end
- timeline views support filtering + stable ordering
- SSE endpoint emits runtime events in expected shape
- existing CLI behavior remains unchanged
