# TRACE-AML Frontend-Backend Alignment Report

Date: 2026-04-03

## Purpose

This report evaluates the current TRACE-AML system as a whole and answers one core question:

> Is the existing frontend ready to connect to the existing backend, and if not, what must be fixed before the service layer is built?

The goal is not to suggest random improvements. The goal is to align the current static frontend concepts with the already-built backend intelligence pipeline so the next stage can become a coherent, real-time system.

## Executive Summary

TRACE-AML already has a strong backend intelligence core. The runtime pipeline is no longer a simple recognition loop; it is an operational chain:

```text
detection -> event -> alert -> incident -> action
```

That backend foundation is real, persistent, testable, and operational through the CLI. However, the frontend is not yet a directly connectable product surface. The frontend currently exists as two separate prototypes:

- `MOCK.html`: a multi-screen design compendium containing several full HTML documents in one file
- `mockup.html`: a single-page static dashboard with inline interactions and hardcoded demo state

This creates the central architectural mismatch:

```text
Backend = coherent intelligence runtime
Frontend = static visual concepts with inconsistent information architecture
```

The backend is ready for a service layer. The frontend is not yet ready to be wired directly to that service layer without one round of structural consolidation first.

The most important conclusion is this:

> TRACE-AML should not jump straight into "build API + connect HTML."

It should first do three things:

1. Freeze one canonical frontend information architecture
2. Define backend read models and service contracts
3. Add the missing aggregation/query layer that the UI actually needs

Only after that should the service layer be implemented.

## Scope Of Review

### Frontend Files Reviewed

- `D:\github FORK\TRACE-ML\MOCK.html`
- `D:\github FORK\TRACE-ML\mockup.html`

### Backend Files Reviewed

- `D:\github FORK\TRACE-ML\trace_ml\core\models.py`
- `D:\github FORK\TRACE-ML\trace_ml\core\config.py`
- `D:\github FORK\TRACE-ML\trace_ml\pipeline\session.py`
- `D:\github FORK\TRACE-ML\trace_ml\pipeline\entity_resolver.py`
- `D:\github FORK\TRACE-ML\trace_ml\pipeline\rules_engine.py`
- `D:\github FORK\TRACE-ML\trace_ml\pipeline\incident_manager.py`
- `D:\github FORK\TRACE-ML\trace_ml\pipeline\policy_engine.py`
- `D:\github FORK\TRACE-ML\trace_ml\pipeline\action_engine.py`
- `D:\github FORK\TRACE-ML\trace_ml\store\vector_store.py`
- `D:\github FORK\TRACE-ML\trace_ml\store\analytics.py`
- `D:\github FORK\TRACE-ML\trace_ml\cli.py`

## Current System Snapshot

### What Already Exists In The Backend

TRACE-AML already implements:

- face detection and recognition
- temporal smoothing and decision stabilization
- entity resolution for known and unknown subjects
- event creation for every resolved detection
- rules-based alert generation
- incident grouping and persistence
- action policy evaluation and execution
- CLI access to history, events, alerts, incidents, and actions
- persistent storage for all major operational records

This is a meaningful intelligence runtime, not an MVP toy loop.

### What Already Exists In The Frontend

The frontend already expresses a clear visual ambition:

- tactical operator view
- incident-centric workflow
- entity profile exploration
- history and timeline analysis
- settings and system controls
- map/geospatial-style situational awareness
- logs, counters, and investigation context

Visually, the system direction is strong. Architecturally, it is still mock-driven.

## Frontend Analysis

## 1. Frontend State: Two Different Prototypes

The first important finding is that the frontend is not one application yet.

### Prototype A: `mockup.html`

This file is a one-page static dashboard with tabs and inline JavaScript. It includes:

- `DASHBOARD`
- `REGISTRY`
- `ANALYTICS`
- `SYSTEM CONFIG`
- `GEOSPATIAL NET`

It uses direct DOM manipulation functions such as:

- `switchTab(...)`
- `showRegistryDetail(...)`
- `trackEntity(...)`

This file behaves like an interaction prototype, not a production frontend architecture.

### Prototype B: `MOCK.html`

This file contains multiple complete HTML documents concatenated together. It is not a single app file; it is effectively a collection of separate page comps:

- `TRACE-AML | Live Ops Dashboard`
- `Global History Timeline | TRACE-AML`
- `TRACE-AML | Incidents & Investigation`
- `TRACE-AML | Entity Explorer`
- `TRACE-AML | Database Explorer`

This makes `MOCK.html` useful as a design reference, but not directly usable as a frontend implementation artifact.

### Conclusion

The frontend must first be normalized into one canonical information architecture before any real integration work begins.

## 2. Frontend Page And Component Inventory

### A. Live Ops / Dashboard

Observed concepts across both HTML prototypes:

- live surveillance feed
- bounding box and match overlay
- liveness and inference HUD
- detection log / alert feed
- active incidents panel
- quick entity view
- session stats
- health indicators
- event stream

#### Data Required

- live camera/session status
- current detections
- current tracks
- current temporal decision state
- current focus subject
- recent live events
- recent alerts
- open incidents
- performance metrics
- last action/activity

#### Interactions Implied

- inspect live subject
- inspect current incident
- filter log stream
- switch active source
- observe health in real time

### B. Registry / Entities

Observed concepts:

- registry table
- row click opens side profile
- known and unknown identities
- entity profile
- last seen
- source location
- match confidence
- exemplar images

#### Data Required

- unified entity list
- known person metadata
- unknown subject metadata
- lifecycle/status
- last seen information
- recent detection history
- evidence images or screenshots
- linked incidents/alerts

#### Interactions Implied

- select entity
- inspect profile
- search/filter entities
- delete or manage record
- review supporting evidence

### C. Incidents And Investigation

Observed concepts:

- incident list
- incident timeline
- alert narrative
- evidence/context panel
- map/pathing section
- actions log
- operator workflow

#### Data Required

- incident list with status, severity, timestamps, counts
- incident detail
- ordered merged timeline
- related alerts
- related actions
- related events/detections
- supporting screenshots/evidence
- entity summary for the incident subject

#### Interactions Implied

- select incident
- close incident
- change severity
- inspect alert/action chronology
- move between list and detail contexts

### D. History

Observed concepts:

- global investigative timeline
- synchronized map + timeline + logs
- event blocks
- history navigation

#### Data Required

- chronological stream of events
- grouped timeline items
- cross-entity or per-entity filters
- source attribution
- optional location context
- screenshot/evidence support

#### Interactions Implied

- filter by time window
- focus one entity
- replay or inspect history transitions
- correlate logs and map/timeline

### E. Settings / System Config

Observed concepts:

- video source selection
- threshold sliders
- anti-spoofing toggle
- logging toggles
- rule/policy-like controls
- save/restart actions

#### Data Required

- current runtime configuration snapshot
- editable safe configuration subset
- validation rules
- health and runtime state

#### Interactions Implied

- view settings
- change settings
- save configuration
- restart pipeline or apply changes

### F. Geospatial / Map / Asset Tracking

Observed concepts:

- global map or globe
- target pinning
- active tracking pool
- location strings
- entity positioning / pathing

#### Data Required

- geo coordinates or structured location model
- source-to-location mapping
- tracked entity position history
- current positions or last known positions

#### Interactions Implied

- select tracked target
- pan or inspect location
- link entity selection to map focus

### G. Database Explorer

Observed concepts:

- database table browsing
- storage metrics
- query-like UI
- health indicators

#### Data Required

- table summaries
- storage counts
- ingestion metrics
- health state
- potentially safe query access

#### Interactions Implied

- inspect system storage
- review counts and health
- navigate records

## 3. Frontend Interaction Expectations

Across the prototypes, the frontend clearly expects the system to support:

- list views
- detail views
- selection state
- filters
- time windows
- severity updates
- incident close operations
- synchronized panels
- live feed updates
- explainability context

This means the backend cannot be surfaced as raw CLI commands or raw tables. The frontend needs a proper query and mutation surface.

## Backend Analysis

## 1. Existing Backend Architecture

The central runtime logic in `trace_ml/pipeline/session.py` already performs the following:

```text
recognition
-> entity resolution
-> detection persistence
-> event creation
-> alert generation
-> incident update/create
-> policy evaluation
-> action execution
```

This is a strong architectural foundation and should remain the operational core.

## 2. Existing Data Models

### Detection Layer

`DetectionEvent` already captures:

- `detection_id`
- `timestamp_utc`
- `source`
- `person_id`
- `name`
- `category`
- `confidence`
- `similarity`
- `smoothed_confidence`
- `bbox`
- `track_id`
- `decision_state`
- `decision_reason`
- `quality_flags`
- `liveness_provider`
- `liveness_score`
- `screenshot_path`
- `metadata`

This is good evidence-level data for UI use.

### Entity Layer

`EntityRecord` already captures:

- `entity_id`
- `type` (`known` or `unknown`)
- `status`
- `source_person_id`
- `created_at`
- `last_seen_at`

This is the right core identity model, but it is too thin for a frontend-facing explorer on its own.

### Event Layer

`EventRecord` already captures:

- `event_id`
- `entity_id`
- `timestamp_utc`
- `confidence`
- `decision`
- `track_id`
- `is_unknown`
- `detection_id`
- `source`

This is a valid event backbone.

### Alert Layer

`AlertRecord` already captures:

- `alert_id`
- `entity_id`
- `type`
- `severity`
- `reason`
- `timestamp_utc`
- `first_seen_at`
- `last_seen_at`
- `event_count`

This is sufficient for a first operator alert stream.

### Incident Layer

`IncidentRecord` already captures:

- `incident_id`
- `entity_id`
- `status`
- `start_time`
- `last_seen_time`
- `alert_ids`
- `alert_count`
- `severity`
- `last_action_at`

This is enough for backend incident control, but not enough for a rich investigation UI.

### Action Layer

`ActionRecord` already captures:

- `action_id`
- `incident_id`
- `action_type`
- `trigger`
- `status`
- `reason`
- `timestamp_utc`

This is a clean audit model.

## 3. Existing Storage Capabilities

The vector store already persists and queries:

- persons
- person lifecycle state
- image quality
- embeddings
- detections
- detection decisions
- entities
- events
- alerts
- unknown profiles
- incidents
- actions

This is a strong storage base.

Important strengths:

- strict incident lookup now works correctly
- incident severity and action timestamps persist correctly
- action audit is stored
- event and alert queries already exist

## 4. Existing Analytics Capabilities

The analytics layer currently supports:

- detection history queries
- summary reports
- low-quality enrollment report
- threshold impact summary
- CSV export

This is useful, but it is still detection-centric rather than investigation-centric.

## 5. Existing Interaction Layer

The CLI already supports meaningful operator surfaces:

- `recognize live`
- `history query`
- `report summary`
- `events tail`
- `alerts tail`
- `incident list`
- `incident show`
- `incident close`
- `incident set-severity`
- `action list`

This means the backend logic is present. What is missing is the browser-facing service boundary.

## Gap Analysis

## A. Data Gaps

### 1. No Unified Entity Read Model

The backend stores entities and persons separately, but the UI needs a single rich object for entity exploration.

The frontend expects something closer to:

- identity summary
- known/unknown type
- linked person data if known
- last seen
- incident count
- alert count
- recent events
- recent detections
- screenshots or exemplar imagery

This does not currently exist as one query surface.

### 2. No Incident Detail Read Model

The backend can show incident metadata and alert timeline, but the UI expects a richer incident object with:

- incident summary
- linked entity summary
- alert timeline
- action timeline
- relevant detections
- screenshots
- evidence narrative

Right now, incident detail is still assembled very narrowly for CLI output.

### 3. No Merged History Timeline

The history page concept implies a unified time-sorted stream across:

- detections
- events
- alerts
- incidents
- actions

The backend does not yet provide this as a reusable timeline query.

### 4. No Real Geospatial Model

The current UI implies spatial intelligence. The backend mostly has:

- `source`
- person `city`
- person `country`

That is not enough for:

- maps
- pins
- last known positions
- tracked paths
- geospatial filtering

### 5. No System Health Snapshot Object

The live runtime knows about:

- FPS
- active tracks
- latency
- queue depth
- decision counts
- recent confidence trend

But this information is only drawn into the OpenCV overlay. It is not published as a queryable live snapshot.

### 6. No Database Explorer Read Surface

The `Database Explorer` UI implies:

- counts by table
- storage health
- possibly row previews
- ingestion metrics

The backend has the raw data but no presentation-friendly system metrics API.

## B. Behavior Gaps

### 1. No Browser-Facing Query Layer

The backend currently exposes behavior through CLI commands and internal store methods. The frontend needs:

- service endpoints
- read contracts
- mutation contracts

### 2. No Real-Time UI Stream

There is no:

- FastAPI app
- SSE stream
- WebSocket service

The frontend cannot yet subscribe to live events or live incidents.

### 3. No Runtime Settings Mutation Layer

The settings UI implies:

- read configuration
- update configuration
- apply/restart controls

The backend currently reads YAML config, but does not expose a safe mutation surface for a browser UI.

### 4. No Cross-Panel Synchronization Contract

The frontend expects behaviors like:

- clicking an incident updates the timeline and detail panel
- clicking an entity updates map/timeline/profile
- live feed changes update alerts and incidents

Those interactions require stable DTOs and identifiers, not raw table access.

## C. Structural Gaps

### 1. The Frontend Information Architecture Is Not Frozen

`mockup.html` and `MOCK.html` represent different structures:

- `mockup.html` uses `Dashboard / Registry / Analytics / System Config / Geospatial Net`
- `MOCK.html` uses `Live Ops / Incidents / Entities / History / Settings / Database Explorer`

The service layer cannot be designed cleanly until one UI architecture is chosen as canonical.

### 2. `MOCK.html` Is Not Implementation-Ready

It contains multiple separate HTML documents in one file. That makes it a design pack, not an app shell.

### 3. Some Mock UI Content Is Domain-Misaligned

The `Incidents` and `Database Explorer` sections in `MOCK.html` contain financial intelligence language such as:

- exchanges
- sanctions
- wallets
- DEX swaps
- KYC

TRACE-AML is currently a person-recognition and surveillance intelligence system. Those financial investigation motifs are aesthetically strong, but semantically inconsistent with the backend and current product scope.

This matters because a UI that tells a different story than the backend will confuse both implementation and demonstration.

### 4. Backend Is Table-Centric, UI Needs Read-Model-Centric

The backend store is shaped around persistence tables.

The frontend will need objects shaped around views:

- `LiveOpsSnapshot`
- `IncidentListItem`
- `IncidentDetail`
- `EntityDetail`
- `HistoryTimelineItem`
- `SettingsSnapshot`
- `SystemHealthSnapshot`

Without those, the service layer will become a thin wrapper over unrelated storage methods and quickly become messy.

## Recommended System Improvements

## 1. Freeze One Canonical Frontend Architecture

Recommended canonical structure:

- `Live Ops`
- `Incidents`
- `Entities`
- `History`
- `Settings`

Optional later:

- `Database Explorer`
- `Geospatial`

Why this structure:

- it matches operational workflows
- it matches the backend intelligence pipeline better
- it separates real operator surfaces from diagnostic/admin surfaces

Recommendation:

- use `MOCK.html` as the reference for page taxonomy and visual direction
- use `mockup.html` as a reference for domain-consistent registry and operational semantics
- do not wire both independently

## 2. Add A Query / Read-Model Layer Before The Service Layer

The next backend addition should not be a UI framework. It should be a read layer.

Recommended backend read services:

- `get_live_ops_snapshot()`
- `get_open_incidents(limit, filters)`
- `get_incident_detail(incident_id)`
- `set_incident_severity(incident_id, severity)`
- `close_incident(incident_id)`
- `get_entities(filters)`
- `get_entity_detail(entity_id)`
- `get_history_timeline(filters)`
- `get_actions_for_incident(incident_id)`
- `get_settings_snapshot()`
- `get_system_health_snapshot()`
- `get_storage_metrics()`

These should return frontend-shaped DTOs, not raw storage rows.

## 3. Add Missing Data In A Minimal Way

The goal here is not to overbuild.

### Must Add Soon

- `camera_id`
- `site_id`
- `zone_id`

These can support location-like UI semantics without forcing true GIS immediately.

### Optional If Map Remains Core

- latitude / longitude
- source geolocation
- per-event location payload

If true geospatial data is not real yet, the map should be reframed as:

- site topology
- camera zones
- source network map

That will be more honest and far easier to support.

## 4. Publish Live Runtime State

The runtime in `session.py` already knows valuable live data:

- FPS
- active tracks
- queue depth
- latency
- current focus
- recent confidence trend
- live event feed

Recommendation:

- introduce a lightweight in-memory runtime state publisher
- let the service layer read from that state for `Live Ops`

This avoids scraping UI overlay strings or coupling the service layer to OpenCV rendering.

## 5. Create Investigation-Centric Timeline Queries

The most important UI-facing aggregation to add is a merged timeline.

Recommended timeline item structure:

```text
HistoryTimelineItem
- timestamp
- kind: detection | alert | action | incident_state
- entity_id
- incident_id
- summary
- severity
- source
- screenshot_path
- metadata
```

This will unlock:

- history page
- incident detail timeline
- entity activity view
- replay readiness later

## 6. Real-Time Strategy Recommendation

### Recommended

- HTTP for normal queries and mutations
- SSE for live event streams

### Why SSE First

- most frontend live behavior is server-to-client
- simpler than full WebSocket infrastructure
- sufficient for event feed, alert feed, action feed, and live dashboard counters
- easier to debug and demonstrate

### Where Polling Is Fine

- entities page
- history page
- settings page
- storage metrics

### Where Streaming Is Preferred

- live ops event feed
- alert stream
- incident updates
- action audit feed

## 7. State Ownership Model

### Backend Owns

- authoritative records
- operational truth
- incidents
- severity values
- actions
- timeline generation
- system health
- live runtime state
- configuration truth

### Frontend Owns

- selected page
- selected incident/entity
- filters
- sort order
- time windows
- panel visibility
- viewport and UI-only display state

This division will keep the service layer clean and avoid state drift.

## UI Integration Readiness Assessment

### Is The Frontend Ready To Connect Directly?

No.

### Why Not?

- there are two competing frontend structures
- one file is a design compendium, not an app shell
- some visual concepts expect data the backend does not yet model
- the frontend is currently mock-data driven rather than contract-driven

### What Needs To Happen First?

1. Choose canonical page architecture
2. Break the frontend into real page/component boundaries
3. Define read contracts for each page
4. Remove or defer unsupported features such as fake global geospatial intelligence

## Prioritized Execution Plan

## Phase A: Must Be Done Before Service Layer

### 1. Freeze Canonical UI Architecture

Decision:

- choose one structure
- mark the other mock as reference-only

### 2. Define UI Data Contracts

Create explicit DTOs for:

- `LiveOpsSnapshot`
- `IncidentListItem`
- `IncidentDetail`
- `EntityListItem`
- `EntityDetail`
- `HistoryTimelineItem`
- `SettingsSnapshot`
- `SystemHealthSnapshot`

### 3. Add Missing Aggregation Functions

Implement the read layer that assembles those DTOs from storage/runtime.

### 4. Decide Map Scope Honestly

Either:

- implement real location fields

or:

- downgrade map to site/camera/zone topology for now

## Phase B: Build With Service Layer

### 5. Add FastAPI Service

Recommended initial routes:

- `GET /live/snapshot`
- `GET /stream/events`
- `GET /stream/alerts`
- `GET /stream/incidents`
- `GET /incidents`
- `GET /incidents/{id}`
- `POST /incidents/{id}/severity`
- `POST /incidents/{id}/close`
- `GET /entities`
- `GET /entities/{id}`
- `GET /history`
- `GET /actions`
- `GET /settings`
- `GET /system/health`

### 6. Wire Pages In Order

Recommended connection order:

1. `Live Ops`
2. `Incidents`
3. `Entities`
4. `History`
5. `Settings`

This order follows the operational value chain and gives visible progress quickly.

## Phase C: Defer

These should not block the service layer:

- advanced geospatial intelligence
- relationship graph / knowledge graph
- database explorer as a primary screen
- multi-camera orchestration
- true replay engine
- real outbound action integrations
- case management layer

## Proposed View Contracts

## 1. LiveOpsSnapshot

```text
LiveOpsSnapshot
- session_status
- source
- fps
- active_tracks
- latency_ms
- frame_queue_depth
- result_queue_depth
- decision_counters
- current_focus
- confidence_trend
- recent_events[]
- open_incidents[]
```

## 2. IncidentDetail

```text
IncidentDetail
- incident_id
- entity_id
- entity_summary
- status
- severity
- start_time
- last_seen_time
- last_action_at
- alert_count
- timeline[]
- recent_actions[]
- recent_detections[]
```

## 3. EntityDetail

```text
EntityDetail
- entity_id
- type
- status
- linked_person
- created_at
- last_seen_at
- incident_count
- recent_alerts[]
- recent_events[]
- recent_detections[]
- screenshots[]
```

## 4. HistoryTimelineItem

```text
HistoryTimelineItem
- timestamp
- kind
- entity_id
- incident_id
- title
- summary
- severity
- source
- screenshot_path
- metadata
```

## Risks To Avoid

### 1. Do Not Use CLI Output As An Integration Layer

The frontend should not parse terminal output. It should consume service DTOs.

### 2. Do Not Build The Service Layer On Raw Table Calls Alone

The service layer should not just mirror storage methods one-to-one. That creates a thin but unstable API.

### 3. Do Not Keep Fake Geospatial As A Core Dependency

If location data is not real yet, the map should not drive the architecture.

### 4. Do Not Mix Frontend Taxonomies

Pick one page model and stick to it.

### 5. Do Not Overbuild Real-Time Transport

SSE is enough initially. A full event bus or WebSocket-heavy architecture is unnecessary right now.

## Final Assessment

### What Is Strong Today

- backend intelligence pipeline
- persistence model
- incident and action lifecycle
- CLI operator workflows
- live tactical runtime

### What Is Weak Today

- frontend structural coherence
- service-readiness of frontend artifacts
- backend read-model/query layer
- location/geospatial truth model
- browser-facing real-time interface

### Go / No-Go Decision

#### Backend Readiness For Service Layer

Yes, with a small read-model layer added first.

#### Frontend Readiness For Direct Wiring

No.

#### Recommended Immediate Next Step

Do not start with browser wiring. Start with:

1. canonical frontend architecture freeze
2. read-model contract design
3. aggregation/query layer implementation

Then build the service layer on top of that.

## Final Recommendation

TRACE-AML should now evolve from:

```text
CLI intelligence system + static visual concepts
```

into:

```text
queryable intelligence backend + real-time operator frontend
```

The backend is already close to that target.

The missing bridge is not more ML work. The missing bridge is:

- service contracts
- read models
- live state publishing
- one canonical frontend architecture

That is the correct next engineering move.
