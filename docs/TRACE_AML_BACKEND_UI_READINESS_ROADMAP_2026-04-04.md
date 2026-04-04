# TRACE-AML Backend UI Readiness Roadmap

Date: 2026-04-04

## Purpose

This roadmap defines the implementation path for evolving TRACE-ML from a pipeline-and-CLI backend into a queryable, explainable, UI-ready intelligence backend.

This roadmap does **not** cover API wiring or frontend implementation. It focuses on backend capabilities that must exist before the service layer can be built cleanly.

## Current Baseline

TRACE-ML already has a working operational chain:

```text
Detection -> Event -> Alert -> Incident -> Action
```

This chain is stable and must remain intact.

### What Must Not Change

- recognition model behavior
- temporal recognition logic
- current CLI behavior
- existing incident/action semantics

## Target State

Transform the backend from:

```text
write-oriented pipeline + CLI surfaces
```

into:

```text
queryable, explainable, UI-ready intelligence core
```

## Implementation Principles

- Preserve the current runtime pipeline
- Add read models, not duplicate business logic
- Prefer aggregation over rewrites
- Keep location support minimal and honest
- Prepare real-time interfaces without committing to API transport yet
- Keep existing CLI commands working unchanged

## Phase 1: Domain Model Upgrades

### Goal

Extend the core models so the backend can express richer UI-facing intelligence.

### Changes

- add minimal event location support
- add incident summary
- add action execution context
- add UI/read-model objects:
  - `TimelineItem`
  - `EntitySummary`
  - `EntityProfile`
  - `IncidentDetail`
  - `SystemHealthSnapshot`
  - `LiveOpsSnapshot`

### Outcome

The backend can describe intelligence state, not just store raw records.

## Phase 2: Storage Compatibility And Explainability Persistence

### Goal

Persist the new explainability and location fields without breaking the current store.

### Changes

- extend event persistence for location
- extend incident persistence for summary
- extend action persistence for context
- keep schema migration safe for existing data

### Outcome

Existing repositories continue to work while new intelligence-facing fields become available.

## Phase 3: Read Model Layer

### Goal

Add a backend query layer specifically shaped for UI consumption.

### Required Functions

- `get_live_ops_snapshot()`
- `get_incident_detail(incident_id)`
- `get_entity_profile(entity_id)`
- `get_global_timeline(start, end)`
- `get_recent_alerts(limit)`
- `get_entity(entity_id)`
- `get_entity_timeline(entity_id)`
- `get_entity_incidents(entity_id)`

### Outcome

The backend becomes queryable without exposing raw tables or requiring CLI parsing.

## Phase 4: Timeline Aggregation

### Goal

Unify operational history into one ordered, structured timeline model.

### Aggregation Scope

- event
- alert
- incident
- action

### Outcome

The future UI can render incident timelines, entity timelines, and global history from one backend abstraction.

## Phase 5: Live Snapshot And System Health

### Goal

Prepare the backend to support Live Ops views.

### Deliverables

- active entities view
- active incidents view
- recent alerts view
- system health snapshot
- optional runtime enrichment if a live session is attached

### Outcome

The service layer can later expose a real dashboard snapshot without inventing new backend logic.

## Phase 6: Real-Time Preparation Hooks

### Goal

Prepare the runtime to emit events into a future streaming layer.

### Deliverables

- event stream publisher interface
- in-memory publisher implementation
- subscription hook
- optional session integration

### Outcome

The backend becomes stream-ready without building FastAPI, SSE, or WebSockets yet.

## Phase 7: Validation

### Goal

Ensure the new UI-ready backend does not regress existing behavior.

### Required Validation

- model compatibility
- schema migration compatibility
- timeline aggregation tests
- entity profile tests
- incident detail tests
- live ops snapshot tests
- existing suite remains green

## Execution Order

1. Extend core models
2. Extend store schema and migrations
3. Add read-model service
4. Add timeline aggregation
5. Add live snapshot support
6. Add stream preparation interface
7. Add tests and run full suite

## What Is Explicitly Deferred

- HTTP API layer
- WebSocket/SSE transport
- frontend wiring
- true geospatial intelligence
- full replay engine
- major CLI redesign

## Definition Of Done

The backend is considered UI-ready for the next stage when:

- it exposes stable read-model functions for UI consumption
- incident, entity, alert, and action data are explainable
- unified timelines are available
- live operational snapshot exists
- minimal location support exists in the event model
- runtime streaming hooks are prepared
- existing CLI remains intact
- test suite passes
