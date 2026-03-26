# TRACE-AML Architecture Audit Report
Date: 2026-03-25  
Project: Tracking and Recognition of Criminal Entities using Advanced Machine Learning (TRACE-AML)  
Audit Type: Technical architecture and implementation status review

## 1. Executive Summary
TRACE-AML has progressed beyond MVP and now operates as a robust CLI-first intelligence pipeline with quality gating, temporal decisioning, and profile-based runtime behavior (`demo` and `strict`). The current system is operationally credible for a final-year project demo and establishes a strong foundation for an incident- and case-centric surveillance platform.

The architecture is now modular, test-backed, configurable, and future-compatible with GUI/API expansion. Recognition behavior shows stable `accept/review/reject` outcomes under non-ideal conditions, with review fallback preventing overconfident misclassification.

## 2. Scope and Mission Alignment
Primary mission currently implemented:
1. Register and manage identities for criminal-entity tracking workflows.
2. Perform live webcam recognition with uncertainty-aware decision states.
3. Persist detections and decisions with auditable metadata.
4. Support quality-aware enrollment and operational reporting.

Current runtime target:
1. Single built-in webcam (`device_index=0`).
2. CLI-driven operator workflow.
3. Local persistence and local analytics.

## 3. Current System Architecture
Core package root: `trace_ml/`

### 3.1 Interface Layer
Primary interface is Typer CLI in `trace_ml/cli.py`.

Command domains currently available:
1. `doctor`
2. `person` (`add`, `list`, `update`, `capture`, `audit`, `delete`)
3. `train rebuild`
4. `recognize live`
5. `history query`
6. `report summary`, `report quality`
7. `export csv`

### 3.2 Configuration and Runtime Contracts
Settings models in `trace_ml/core/config.py`.

Key contract sections:
1. `camera`, `recognition`, `quality`, `temporal`
2. `pipeline`, `store`, `logging`, `liveness`
3. Profile preset configs: `config.demo.yaml`, `config.strict.yaml`

Notable behavior:
1. Profile switching is config-only, no code branch changes.
2. Recognition thresholds and temporal strategy are runtime-tunable.

### 3.3 Domain Model Layer
Defined in `trace_ml/core/models.py`.

Implemented domain objects include:
1. Person lifecycle states: `draft`, `ready`, `active`, `blocked`
2. Decision states: `accept`, `review`, `reject`
3. Enriched detection event schema with decision, track, quality, and liveness metadata.

### 3.4 Recognition and Intelligence Layer
Recognizer implementation: `trace_ml/recognizers/arcface.py`.

Active features:
1. InsightFace `buffalo_sc` backend (SCRFD detector + ArcFace embeddings).
2. Active-only preference in identity retrieval.
3. Robust matching mode with multi-embedding person aggregation.
4. Preprocess fallback for low-light detection recovery.
5. Quality-adaptive threshold relaxation at inference time.

### 3.5 Enrollment Quality and Lifecycle Governance
Quality scoring: `trace_ml/quality/scoring.py`  
Lifecycle gating: `trace_ml/quality/gating.py`

Implemented controls:
1. Per-image quality metrics (face ratio, sharpness, brightness, pose proxy).
2. Fail-fast gating for unusable enrollment inputs.
3. Person audit command to detect and optionally correct lifecycle states.

### 3.6 Temporal Decision Engine
Temporal logic: `trace_ml/pipeline/temporal.py`

Implemented controls:
1. Window voting + confidence smoothing.
2. Track assignment and stale-track purge.
3. Improved track reuse scoring (distance + IoU + recency).
4. Dynamic threshold application from recognizer metadata.
5. Hard reject overrides for inactive/liveness-fail conditions.

### 3.7 Runtime Pipeline
1. Camera capture worker: `trace_ml/pipeline/capture.py`
2. Inference worker: `trace_ml/pipeline/inference.py`
3. Session orchestrator and tactical HUD: `trace_ml/pipeline/session.py`

Operational pattern:
1. Threaded queue pipeline for near-real-time processing.
2. Event feed, counters, latency strip, and track-aware overlay.

### 3.8 Storage and Analytics
Vector store: `trace_ml/store/vector_store.py` (LanceDB)  
Analytics store: `trace_ml/store/analytics.py` (DuckDB over snapshots)

Operational tables include:
1. `persons`, `person_states`
2. `person_embeddings`, `image_quality`
3. `detections`, `detection_decisions`

Analytics outputs include:
1. Filtered history queries
2. Session summary with decision distribution
3. Quality-focused enrollment report
4. CSV export

### 3.9 Liveness Status
Liveness is scaffolded in `trace_ml/liveness/base.py`.

Current status:
1. Stub and pass-through interfaces are present.
2. Full anti-spoof inference is intentionally deferred.

## 4. Test and Quality Status
Test suite location: `tests/`  
Current automated result: `17 passed` (local run).

Coverage focus currently includes:
1. Config defaults and validation
2. Vector store CRUD and active filtering behavior
3. Analytics outputs
4. Temporal state transitions
5. CLI smoke tests
6. Robust recognizer behavior for active preference and threshold relaxation

## 5. Observed Runtime Outcomes (Recent)
Recent command outputs indicate:
1. Stable identification of active subject `PRC004`.
2. High accept frequency with confidence commonly around 60-85+.
3. Meaningful review fallback under difficult frames.
4. Unknown noise is controlled and does not dominate recognized identity stream.

Interpretation:
1. Core recognition is no longer brittle to moderate environmental variation.
2. Decision policy is behaving safely (uncertain frames flow to review).

## 6. Architecture Strengths
1. Clear modular separation across core, pipeline, recognition, quality, and storage.
2. Strong CLI operations for practical demo and operator workflows.
3. Robustness controls now exist at enrollment, inference, and temporal stages.
4. Rich metadata creates strong auditability and explainability.
5. Config-driven strategy enables rapid profile switching.
6. Solid base for API/GUI integration without rewriting core logic.

## 7. Current Gaps and Risks
1. No case management model yet (detections are not grouped into case entities).
2. No alert rules engine yet (event severity is implicit, not policy-driven).
3. Unknown-person tracking lacks explicit clustering identities (`UNK-*`).
4. Liveness remains scaffold-only.
5. Governance controls (RBAC, immutable audit chain, PII policy) are not yet implemented.
6. Single-camera scope remains a hard constraint.

## 8. Readiness Assessment
Current maturity (project context):
1. Demo readiness: High
2. Architectural quality: Strong for final-year scope
3. Production readiness: Moderate, pending governance and operational intelligence layers

## 9. Recommended Immediate Direction
Before GUI, prioritize CLI intelligence extensions:
1. Case and incident workflow layer
2. Rules-based alerting and prioritization
3. Unknown clustering and recurring-entity intelligence
4. Evidence package export and chain-of-custody metadata

This path keeps complexity controlled while making TRACE-AML feel like a serious surveillance intelligence platform.

## 10. Repository Coverage Checklist
Status legend:
1. `Complete` means implemented and validated in current repo state.
2. `Partial` means implemented base exists but not full production depth.
3. `Planned` means intentionally deferred.

Checklist:
1. CLI command surface and operator flow: `Complete`
2. Typed configuration and profile strategy: `Complete`
3. Enrollment quality gate and lifecycle governance: `Complete`
4. Robust recognition and temporal stabilization: `Complete`
5. Storage + analytics + CSV export: `Complete`
6. Tactical HUD and live operator feedback: `Complete`
7. Anti-spoofing model enforcement: `Partial` (interface/scaffold only)
8. Case/incident ontology: `Planned`
9. Alert rules and escalation engine: `Planned`
10. Unknown clustering intelligence: `Planned`
11. Governance controls (RBAC, immutable audit chain): `Planned`
12. API/WebSocket service boundary for GUI: `Planned`

## 11. Explicit Non-Goals in Current Phase
1. Multi-camera ingest orchestration.
2. Distributed deployment and cloud-scale serving.
3. Full legal/compliance control plane.
4. Autonomous enforcement actions without human review.
5. End-to-end anti-spoof model hardening.

## 12. Audit Verdict
Architectural verdict:
1. Current repository is sufficiently complete for a strong final-year CLI-first intelligence demo.
2. Architecture quality is above typical college-project baseline due to lifecycle governance, temporal decisioning, and auditable metadata.
3. Next improvement frontier is operational intelligence (case/incident/rules), not core model replacement.

Readiness verdict:
1. `Green` for closing architecture hardening phase.
2. `Green` for moving into feature-intelligence phase before GUI.
