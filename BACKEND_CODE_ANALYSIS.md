# TRACE-ML Backend Code Analysis

**Date:** April 13, 2026  
**Version:** v3 MVP  
**Scope:** Complete technical analysis of backend infrastructure, pipelines, and processing engines

---

## Table of Contents

1. [Core Infrastructure](#core-infrastructure)
2. [Storage Layer](#storage-layer)
3. [Recognition Pipeline](#recognition-pipeline)
4. [Intelligence Pipeline](#intelligence-pipeline)
5. [Recognizers](#recognizers)
6. [Quality System](#quality-system)
7. [System Integration & Flow](#system-integration--flow)

---

## Core Infrastructure

### Configuration System (`src/trace_aml/core/config.py`)

**Purpose:** Hierarchical YAML-based settings management with Pydantic validation.

**Key Classes:**
- `Settings`: Root configuration object with 11 subsystems
- Nested configs: `AppSettings`, `CameraSettings`, `RecognitionSettings`, `PipelineSettings`, `QualitySettings`, `TemporalSettings`, `RulesSettings`, `ActionsSettings`, `StoreSettings`, `LoggingSettings`, `LivenessSettings`

**Configuration Hierarchy:**

| Config Category | Key Parameters |
|---|---|
| **Camera** | device_index=0 (only), width=1280, height=720, fps=30 |
| **Recognition** | model_name="buffalo_sc", provider="CPUExecutionProvider", thresholds (accept=0.72, review=0.58, similarity=0.45) |
| **Quality** | min_sharpness=55.0, min_brightness=45.0-220.0, min_quality_score=0.38, min_detector_score=0.55 |
| **Temporal** | decision_window=6, smoothing_alpha=0.6, track_ttl=1.8s, max_track_distance=120px, min_commit_confidence=45% |
| **Rules** | reappearance/unknown (10s window, 3 events), instability (std_threshold=0.15), cooldown=15s |
| **Actions** | Severity→action mapping (low/medium/high), cooldown=20s |
| **Pipeline** | frame_queue_size=2, result_queue_size=2, ghost_entity_cleanup enabled |

**Input/Output:**
- **Input:** YAML file (config_path or TRACE_AML_CONFIG env), environment overrides with `TRACE_AML_*` prefix
- **Output:** Immutable `Settings` object with runtime_config_path metadata

**Constraints:**
- Device index restricted to 0 (single built-in webcam for v3 MVP)
- Environment variables use nested delimiter `__` (e.g., `TRACE_AML_RECOGNITION__ACCEPT_THRESHOLD`)

---

### Domain Models (`src/trace_aml/core/models.py`)

**Purpose:** Type-safe data contracts for the entire system.

**Core Enums:**

| Enum | Values | Usage |
|---|---|---|
| `PersonCategory` | criminal, missing, employee, vip | Person classification |
| `PersonLifecycleStatus` | draft→ready→active→blocked | Enrollment progression |
| `DecisionState` | accept, review, reject | Face confidence binning |
| `EntityType` | known, unknown | Known person vs. unknown cluster |
| `EntityStatus` | active, inactive | Entity availability |
| `AlertSeverity` | low, medium, high | Incident priority |
| `AlertType` | REAPPEARANCE, UNKNOWN_RECURRENCE, INSTABILITY | Rule types |
| `IncidentStatus` | open, closed | Incident lifecycle |
| `ActionType` | log, email, alarm | Executable actions |

**Key Records:**

1. **PersonRecord** - Core identity
   - Attributes: person_id, name, category, severity, lifecycle_state, enrollment_score, valid_embeddings
   - Lifecycle: draft (0 embeddings) → ready (≥2 embeddings, adequate quality) → active (≥6 embeddings, ≥6 images, high quality) → blocked

2. **EmbeddingRecord** - Face encodings
   - 512-D vectors normalized to unit length
   - Quality tracking: quality_score, quality_flags
   - Validation: rejects empty embeddings

3. **DetectionEvent** - Raw face capture
   - Contains: bbox, confidence, similarity, smoothed_confidence, decision_state, screenshot_path
   - Links to person_id (if matched) or creates UNK entity

4. **EntityRecord** - Face identity cluster
   - Known: person_id matches PersonRecord
   - Unknown: generated UNK### ID, aggregates unidentified faces

5. **EventRecord** - Normalized detection event
   - Filters DetectionEvent through quality gates
   - Feeds rules engine and incident tracking

6. **AlertRecord** - Rule violation
   - Created by RulesEngine when entity behavior violates thresholds
   - Contains: alert_type, severity, event_count

7. **IncidentRecord** - Grouped alerts
   - Represents active security event: one incident per entity
   - Tracks: alert_ids, alert_count, severity, last_action_at
   - Lifecycle: open → closed

8. **ActionRecord** - Executed response
   - Triggers: on_create (incident birth), on_update (escalation)
   - Types: log, email, alarm
   - Status: success, failed

---

### Health Checks (`src/trace_aml/core/health.py`)

**Purpose:** Startup validation pipeline.

**Checks Performed:**
1. **Dependencies:** numpy, lancedb, duckdb, onnxruntime, rich, typer, insightface
2. **Paths:** Create/verify data directories (vectors, screenshots, exports, logs)
3. **Camera:** Enumerate available devices, test cv2.VideoCapture

**Output:** List[HealthCheck] with (name, status, detail)

**Usage:** CLI `doctor` command validates environment before running

---

### ID Generation (`src/trace_aml/core/ids.py`)

**Strategies:**

| Entity Type | ID Format | Example |
|---|---|---|
| Person (Criminal) | PRC### | PRC001 |
| Person (Non-Criminal) | PRM### | PRM042 |
| Detection | DET-TIMESTAMP-UUID | DET-20260413T121530Z-a1b2c3d4 |
| Event | EVT-TIMESTAMP-UUID | EVT-20260413T121530Z-e5f6g7h8 |
| Alert | ALT-TIMESTAMP-UUID | ALT-20260413T121530Z-i9j0k1l2 |
| Incident | INC-TIMESTAMP-UUID | INC-20260413T121530Z-m3n4o5p6 |
| Action | ACT-TIMESTAMP-UUID | ACT-20260413T121530Z-q7r8s9t0 |
| Unknown Entity | UNK### | UNK001, UNK002, ... |
| Embedding | EMB-{person_id}-UUID | EMB-PRC001-a1b2c3d4e5f6 |

**Assumptions:**
- Existing IDs are read to compute next sequential number
- UUID suffix (8 hex chars) ensures timestamp collision safety
- Unknown entities form separate sequence space

---

### Event Streaming (`src/trace_aml/core/streaming.py`)

**Purpose:** In-memory pub/sub for real-time event propagation.

**Protocols:**

**EventStreamPublisher (Protocol):**
```python
publish(topic: str, payload: dict) → None
subscribe(listener: Callable[[StreamEvent], None]) → Callable[[], None]  # unsubscribe
recent(limit: int = 100) → list[StreamEvent]
latest(topic: str | None = None) → StreamEvent | None
subscriber_count() → int
last_published_at() → str
```

**Implementations:**

1. **NullEventStreamPublisher** - No-op, used when streaming disabled
2. **InMemoryEventStreamPublisher** - Thread-safe circular buffer
   - **max_events:** 512 (ring buffer, oldest dropped)
   - **Locking:** RLock protects listeners dict, events deque, and metadata
   - **Async Delivery:** Synchronous but catches exceptions per listener
   - **Topics:** Detection, event, alert, incident, action, session.state

**Data Structure:**
```python
@dataclass(slots=True)
class StreamEvent:
    topic: str
    payload: dict[str, Any]
    timestamp_utc: str  # ISO format
```

**Constraints:**
- Listeners called synchronously during publish
- Exception in one listener doesn't affect others
- No persistence (in-memory only)

---

### Logging (`src/trace_aml/core/logger.py`)

**Framework:** Loguru

**Configuration:**
- **Console:** stderr, colored format with timestamp, level, message
- **File:** `data/logs/trace_aml.log` with rotation (10 MB), retention (14 days), async write
- **Level:** Configurable via `settings.logging.level` (default INFO)

**Output Example:**
```
<green>2026-04-13 12:15:30</green> | <level>INFO</level> | Ghost entity purge complete: removed 3 entities.
```

---

## Storage Layer

### Vector Store (`src/trace_aml/store/vector_store.py`)

**Purpose:** LanceDB-backed persistence with 12 tables, embedding search, and CRUD operations.

**Database Connection:**
- **Backend:** LanceDB (Apache Arrow + LANCE format)
- **Path:** `data/vectors` (configured via `settings.store.vectors_dir`)
- **Schema:** 12 tables with PyArrow schemas, auto-migration on field additions

**12 Tables Schema:**

| Table | Key Fields | Purpose | Embedding? |
|---|---|---|---|
| **persons** | person_id (PK), name, category, severity, created_at | Master person records | No |
| **person_states** | person_id (PK), lifecycle_state, enrollment_score, valid_embeddings | Lifecycle tracking | No |
| **person_embeddings** | embedding_id (PK), person_id (FK), embedding[512] | Face vectors for active persons | **Yes** (LANCE) |
| **image_quality** | quality_id (PK), person_id (FK), passed, quality_score | Enrollment assessment | No |
| **detections** | detection_id (PK), timestamp_utc, person_id, confidence, embedding[512] | All captures + vectors | **Yes** (LANCE) |
| **detection_decisions** | detection_id (PK), decision_state, smoothed_confidence, top_candidates (JSON) | Decision metadata | No |
| **entities** | entity_id (PK), type (known/unknown), source_person_id, last_seen_at | Identity clusters | No |
| **events** | event_id (PK), entity_id (FK), timestamp_utc, confidence | Normalized detections | No |
| **unknown_profiles** | entity_id (PK), embedding[512], sample_count | Unknown aggregation | **Yes** (LANCE) |
| **alerts** | alert_id (PK), entity_id (FK), type, severity, event_count | Rule violations | No |
| **incidents** | incident_id (PK), entity_id (FK), status, severity, alert_ids (JSON) | Grouped alerts | No |
| **actions** | action_id (PK), incident_id (FK), action_type, status, context (JSON) | Executed responses | No |

**CRUD Operations:**

**Person Management:**
```python
add_or_update_person(person: PersonRecord) → None
get_person(person_id: str) → dict | None
list_persons() → list[dict]
active_person_ids() → set[str]  # lifecycle_state == "active"
delete_person(person_id: str, delete_detections: bool = True) → None
```

**Embedding Search:**
```python
search_embeddings(embedding: list[float], top_k: int) → list[dict]
  Returns: [{person_id, similarity (cosine), quality_flags, ...}]
  Metric: Cosine distance → similarity = 1 - distance

search_embeddings_for_person_ids(embedding: list, person_ids: set, top_k) → list[dict]
  Filters search to active persons only
  Normalizes vectors manually (LANCE cosine metric buggy)
```

**Detection & Decision:**
```python
add_detection(event: DetectionEvent, embedding: list[float]) → None
add_detection_decision(detection_id, track_id, decision_state, smoothed_confidence, ...) → None
list_detections(limit: int) → list[dict]
  Merges detection + detection_decisions on detection_id
  Parses JSON fields (quality_flags, metadata, bbox)
```

**Entity Lifecycle:**
```python
ensure_entity(entity_id, entity_type, status, source_person_id, last_seen_at) → dict
  Upsert semantics: creates or updates last_seen_at

resolve_or_create_unknown_entity(embedding, similarity_threshold) → str
  Finds best unknown profile matching embedding
  If similarity ≥ threshold: merge vectors (average), increment sample_count
  Else: create new UNK### entity
  Returns: entity_id
```

**Event & Alert Tracking:**
```python
add_event(event: EventRecord) → None
get_events(entity_id: str, window_sec: int) → list[dict]
  Returns events since cutoff time, sorted ascending

add_alert(alert: AlertRecord) → None
list_alerts(limit, entity_id, severity) → list[dict]
```

**Incident Management:**
```python
create_incident(incident: IncidentRecord) → None
  Atomic: delete old + add new (with deduplication safety)

update_incident(incident: IncidentRecord) → None
  Calls create_incident

get_active_incident(entity_id: str) → dict | None
  Returns open incident for entity, or None

close_incident(incident_id: str) → bool
set_incident_severity(incident_id: str, severity: str) → bool
set_incident_last_action(incident_id: str, timestamp_utc: str) → bool

deduplicate_incidents() → int  # Cleanup duplicates from non-atomic operations
purge_ghost_entities(min_events: int = 2) → int  # Cleanup warmup artifacts
```

**Data Contracts:**

**Type Conversions (String ↔ JSON):**
- `quality_flags`: list → JSON string, parsed on read
- `bbox`: tuple → JSON string, parsed on read
- `metadata`: dict → JSON string, parsed on read
- `alert_ids`: list → JSON string, parsed on read
- `location`: EventLocation → JSON string, parsed on read
- `context`: dict → JSON string, parsed on read

**Embedding Validation:**
```python
EMBEDDING_DIM = 512  # Fixed dimension
Vectors padded/trimmed to 512-D, stored as list[float32]
```

**Query Fallback:**
- Primary: LanceDB `.query().where()` API (supports SQL subset)
- Fallback: In-memory filtering with regex parser for simple `field = 'value'` clauses
- Reason: LanceDB version compatibility issues

**Scaling Limits:**
- `list_detections()`, `list_embeddings()`: Hard limit 100K-200K per query
- `search_embeddings_for_person_ids()`: Manual dot-product search, O(embeddings) time

---

### Analytics Store (`src/trace_aml/store/analytics.py`)

**Purpose:** DuckDB-based SQL analytics over detection snapshots.

**Methods:**

**history(query: HistoryQuery) → list[dict]**
- Filters detections by: start_ts, end_ts, person_id, category, decision_state, min_confidence
- Orders by timestamp_utc DESC
- Returns up to query.limit rows

**summary() → SummaryReport**
```python
SummaryReport:
  total_detections: int
  unique_persons: int
  avg_confidence: float
  decision_distribution: {decision_state → count}
  blocked_persons: int  # lifecycle_state == "blocked"
  low_quality_persons: int  # enrollment_score < min_quality_score
  top_persons: [{name, category, hits, avg_conf}]  # Top 5 by frequency
```

**low_quality_enrollments(limit: int) → list[dict]**
- Returns persons below quality threshold or in draft/blocked state
- Sorted by: blocked status (asc), enrollment_score (asc)

**threshold_impact() → dict**
- Simulates accept/review/reject bins at current thresholds
- Returns: {accept_band, review_band, reject_band} counts

**export_csv(query: HistoryQuery, output_path: str) → Path**
- Writes CSV with fieldnames: detection_id, timestamp_utc, source, person_id, name, ...

**Data Type Handling:**
- Normalizes quality_flags: list → comma-separated string
- Normalizes confidence/similarity: handles 0-1 and 0-100 ranges

---

## Recognition Pipeline

### Frame Capture (`src/trace_aml/pipeline/capture.py`)

**Purpose:** Threaded webcam frame acquisition with shared buffer for external consumers.

**Class: CameraCapture**

**Architecture:**
```
cv2.VideoCapture(device_index)
         ↓
    Daemon Thread
         ↓
  Latest-frame buffer (shared lock)
         ↓
   Frame queue (for inference)
```

**Public API:**

```python
start() → None
  Opens camera, spawns daemon thread
  Raises: CameraError if device unavailable

stop() → None
  Sets stop_event, joins thread (timeout 1s), releases camera

is_active() → bool
get_latest_frame() → FramePacket | None
  Class methods for external consumers (MJPEG endpoint)
```

**Internal Threading:**

**FramePacket:**
```python
@dataclass
class FramePacket:
    frame: np.ndarray  # BGR, shape (H, W, 3)
    captured_at: float  # time.time()
    frame_index: int  # Monotonic counter
```

**_run() Loop:**
1. Read frame from cv2.VideoCapture (blocking up to 1/fps second)
2. Create FramePacket with captured_at = time.time()
3. Update shared _shared_latest (thread-safe, class-level)
4. Try to enqueue into result_queue (size 2): if full, drop oldest frame
5. Continue until _stop_event set

**Queue Management:**
- `put_nowait()`: Non-blocking enqueue
- If queue full: `get_nowait()` removes oldest, then re-adds new
- Rationale: Prioritize fresh frames, drop stale ones

**Shared State (Thread-Safe):**
```python
_shared_lock = threading.Lock() (RLock)
_shared_latest: FramePacket | None
_shared_active: bool
```

**Constraints:**
- Device index hardcoded to 0 (via settings.camera.device_index validation)
- Resolution: 1280×720, FPS: 30 (configurable, applied via cv2 CAP_PROP)
- Frame format: BGR (OpenCV standard)

---

### Inference Worker (`src/trace_aml/pipeline/inference.py`)

**Purpose:** Asynchronous face recognition from frame queue.

**Class: InferenceWorker**

**Architecture:**
```
Frame Queue → Worker Thread → Recognizer.match() → Result Queue
                                    ↓
                          StoreVector (embeddings search)
```

**Public API:**

```python
start() → None
  Spawns daemon thread

stop() → None
  Sets stop_event, joins thread (timeout 1s)
```

**InferencePacket:**
```python
@dataclass
class InferencePacket:
    frame: np.ndarray
    frame_index: int
    captured_at: float  # From FramePacket
    processed_at: float  # time.time() during inference
    matches: list[RecognitionMatch]  # One per detected face
    embeddings: list[list[float]]  # Corresponding 512-D vectors
    liveness: list[LivenessResult]  # Liveness scores
```

**_run() Loop:**
1. Wait for FramePacket from frame_queue (timeout 0.2s)
2. Call recognizer.match(frame, store)
   - Returns: list[tuple(RecognitionMatch, embedding, LivenessResult)]
3. Unpack into InferencePacket
4. Enqueue result (size 2): same drop-oldest logic as frame capture
5. Continue on timeout

**Key Responsibility:**
- Bridges frame capture → face recognition
- Handles GPU/CPU inference latency

**Thread Safety:**
- No shared state beyond queues
- Daemon thread, safe to abandon on exit

---

### Temporal Decision Engine (`src/trace_aml/pipeline/temporal.py`)

**Purpose:** Stable face identity tracking via voting and confidence smoothing.

**Class: TemporalDecisionEngine**

**State Management:**

**TrackState:**
```python
@dataclass
class TrackState:
    track_id: str  # T0001, T0002, ...
    center: tuple[float, float]  # (x + w/2, y + h/2)
    bbox: tuple[int, int, int, int]  # Current frame position
    last_seen: float  # time.time()
    smoothed_confidence: float  # Exponential moving average (0-1)
    confidences: deque[float]  # Rolling window of similarities
    identities: deque[str]  # Rolling window of person_ids / "unknown"
```

**Config Parameters:**
- `decision_window: int = 6` — Number of frames to consider for voting
- `smoothing_alpha: float = 0.6` — EMA weight for current frame
- `track_ttl_seconds: float = 1.8` — Time before track is discarded
- `max_track_distance_px: int = 120` — Max centroid movement to reuse track
- `min_track_iou: float = 0.08` — Min overlap to reuse track
- `track_reuse_min_score: float = 0.35` — Min composite score to match old track
- `min_accept_votes: int = 2` — Frames showing same person ID for ACCEPT
- `min_commit_confidence: float = 45.0%` — Smoothed confidence gate before DB write

**Algorithm: _assign_track_id(bbox, now)**

1. **Purge stale tracks:** Remove if (now - last_seen) > track_ttl_seconds
2. **For each existing track:**
   - Compute: distance (centroid), IoU (bounding box)
   - If distance > max_distance AND IoU < min_iou: skip
   - Score = 0.55×distance_score + 0.35×IoU + 0.10×recency_score
3. **Reuse best track if:**
   - best_score ≥ track_reuse_min_score
   - Update center, bbox, last_seen
4. **Else:** Create new track T{_next_track:04d}, increment counter

**Algorithm: evaluate(match: RecognitionMatch, now_ts) → TemporalDecision**

1. **Assign/reuse track** using _assign_track_id()
2. **Extract identity:**
   - Priority: match.person_id > top candidate from match.candidate_scores > "unknown"
3. **Accumulate:**
   - Append to state.identities, state.confidences (capped at decision_window)
4. **Smooth confidence:**
   - For first frame: smoothed_conf = match.confidence
   - Else: smoothed_conf = alpha × confidence + (1 - alpha) × prior
5. **Vote on identity:**
   - Counter over non-"unknown" identities
   - top_id = most frequent, votes = count
6. **Determine decision:**
   - **REJECT** if any hard block flag (liveness_fail, person_not_active, person_state_not_active)
   - **ACCEPT** if: top_id ∧ smoothed_conf ≥ accept_thr × 100 ∧ votes ≥ min_accept_votes
   - **REVIEW** if: top_id ∧ smoothed_conf ≥ review_thr × 100
   - Else: **REJECT**
7. **Liveness override:** Force REJECT if liveness_fail in quality_flags

**Return: TemporalDecision**
```python
@dataclass
class TemporalDecision:
    track_id: str
    decision_state: DecisionState
    decision_reason: str  # Explains decision rationale
    smoothed_confidence: float  # 0-1, rounded to 3 decimals
    resolved_person_id: str | None  # top_id or None
    vote_count: int
```

**Quality Flags → Soft Blocks:**
- `person_not_active`: Person's lifecycle ≠ active
- `person_state_not_active`: Computed by ArcFace, redundant with above
- `liveness_fail`: Hard block (REJECT regardless of scores)
- `low_face_quality`: Soft indicator, doesn't block

**Constraints:**
- Track assignment O(n_tracks) per frame, n_tracks ≈ O(1) with TTL
- No persistence: tracks lost on session exit
- UNK entity creation happens later (entity_resolver), not here

---

### Entity Resolver (`src/trace_aml/pipeline/entity_resolver.py`)

**Purpose:** Cluster faces into known person or unknown entity, create event records.

**Class: EntityResolver**

**State:**
```python
_track_entity_map: dict[str, str]  # track_id → entity_id (lifetime mapping)
```

**Algorithm: resolve(match: RecognitionMatch, embedding: list) → EntityResolution**

**Layer 3: Track Ownership Cache**
1. If match.track_id in _track_entity_map:
   - Reuse existing entity_id
   - Update entity.last_seen_at (no new record)
   - Return EntityResolution with cached entity_id
   - **Rationale:** Ensure one entity per track lifetime

2. **If match.person_id (known person):**
   - Call store.ensure_entity(entity_id=person_id, type=KNOWN, source_person_id=person_id)
   - Register track → person_id mapping
   - Return EntityResolution(entity_id=person_id, type=KNOWN, is_unknown=False)

3. **If unknown (match.person_id is None):**
   - Call store.resolve_or_create_unknown_entity(embedding, threshold=unknown_reuse_threshold)
   - Returns: UNK### or existing UNK match
   - Register track → unknown_entity_id mapping
   - Return EntityResolution(entity_id=UNK###, type=UNKNOWN, is_unknown=True)

**Return: EntityResolution**
```python
@dataclass
class EntityResolution:
    entity_id: str
    entity_type: EntityType  # known or unknown
    is_unknown: bool
```

**Method: create_event_record(resolution, match, detection_id, source) → EventRecord**
- Wraps resolution + match into EventRecord
- Sets: entity_id, decision, track_id, is_unknown, detection_id
- Returns ready-to-persist EventRecord

**Constraints:**
- One entity per track_id (cache prevents duplicates)
- Unknown clustering threshold: settings.recognition.unknown_reuse_threshold (default 0.55)

---

### Training & Collection (`src/trace_aml/pipeline/train.py` & `collect.py`)

**train.py: rebuild_embeddings(settings, store, recognizer) → TrainStats**

**Purpose:** Batch process person images, extract embeddings, assess enrollment quality.

**Algorithm:**
1. **For each person:**
   - Check person_images/{person_id} directory exists
   - Extract all .jpg/.jpeg/.png/.bmp files (sorted)
2. **For each image:**
   - cv2.imread() → frame_bgr
   - Call recognizer.primary_face_from_image(frame)
     - Detects largest face, returns FaceCandidate or None
   - Call build_assessment(settings, person_id, img_path, frame, bbox)
     - Computes quality_score (sharpness, brightness, pose, face_ratio)
   - Store QualityAssessment in vector_store.image_quality
   - If passed: create EmbeddingRecord, append to records list
3. **Replace person embeddings:**
   - store.replace_person_embeddings(person_id, records)
   - Clears old, bulk-inserts new
4. **Decide lifecycle:**
   - Call decide_person_lifecycle(total_images, valid_images, embeddings_count, avg_quality)
   - Returns: PersonLifecycleStatus + enrollment_score
5. **Update person state:**
   - store.set_person_state(person_id, lifecycle, enrollment_score, valid_embeddings, ...)

**Return: TrainStats**
```python
@dataclass
class TrainStats:
    persons_total: int
    persons_processed: int
    embeddings_created: int
    skipped_images: int
    active_persons: int
    ready_persons: int
    blocked_persons: int
```

---

**collect.py: capture_from_webcam() & import_from_directory()**

**capture_from_webcam(settings, person_id, count=10, auto=True, interval_seconds=0.35)**
- Opens cv2.VideoCapture
- Display loop with counter "Captured X/10"
- Auto mode: every 0.35s
- Manual mode: spacebar or auto
- Quit: 'q' key
- Saves to data/person_images/{person_id}/capture_{n:03d}.jpg

**import_from_directory(settings, person_id, source_dir)**
- Scans source_dir for images
- Copies to data/person_images/{person_id}/import_{n:03d}.jpg

---

### Session & Recognition Loop (`src/trace_aml/pipeline/session.py`)

**Purpose:** Main recognition orchestration, temporal decisions, alert generation, action execution.

**Class: RecognitionSession**

**State:**
```python
# Core components
settings: Settings
store: VectorStore
recognizer: ArcFaceRecognizer
stream_publisher: EventStreamPublisher

# Subsystems
entity_resolver: EntityResolver
rules_engine: RulesEngine
incident_manager: IncidentManager
policy_engine: PolicyEngine
action_engine: ActionEngine
temporal: TemporalDecisionEngine

# Control flags
_camera_enabled: bool
_recognition_enabled: bool
_camera_lock: threading.Lock()
_capture: CameraCapture | None
_inference: InferenceWorker | None
_frame_queue: queue.Queue
_result_queue: queue.Queue

# Tracking
_committed_tracks: set[str]  # Tracks past commitment gate
```

**Camera Control API:**

```python
enable_camera() → dict[str, Any]
  Creates fresh queues, CameraCapture, InferenceWorker
  Spawns threads
  Returns: {status, message, ...}

disable_camera() → dict[str, Any]
  Stops threads, releases resources

is_camera_enabled() → bool
get_camera_status() → dict

enable_recognition() → dict[str, Any]
  Requires camera_enabled
  Starts inference thread

disable_recognition() → dict[str, Any]
  Stops inference, keeps camera running

get_recognition_status() → dict
```

**Main Event Processing: _save_detection()**

**Layer 5: Minimum Unknown Surface Threshold**
- If decision == REJECT ∧ no person_id ∧ smoothed_conf < min_unknown_surface_threshold:
  - Drop face (too ambiguous, likely noise)
  - Prevents warmup-phase ghost entity creation

**Persistence Control:**
- Obeys settings.recognition.persist_unknown (default False)
- Obeys settings.recognition.persist_review (default True)

**Detection Capture:**
- Create DetectionEvent with screenshot_path
- Save frame to data/screenshots/{detection_id}.jpg
- Write to store.detections

**Decision Tracking:**
- Log frame-level decision to add_detection_decision()
- Publish "detection" event to stream

**Entity Resolution:**
- Call entity_resolver.resolve(match, embedding)
- Creates/reuses entity_id (known person or UNK###)

**Event Pipeline:**
1. Create EventRecord via entity_resolver.create_event_record()
2. Persist to store.events
3. Call rules_engine.process_event(event) → list[AlertRecord]

**Rules Engine Processing:**
- Check reappearance rule (3+ events in 10s window)
- Check unknown_recurrence rule (unknown entity, 3+ events in 10s)
- Check instability rule (std deviation of confidences > 0.15)
- Cooldown per rule type: 15 seconds

**Alert Handling:**
- Create or update incident via incident_manager.handle_alert()
  - If active incident exists for entity: append alert_id, update last_seen_time
  - Else: create new incident with alert_id
- Persist incident to store

**Policy & Action:**
- Evaluate policy via policy_engine.evaluate(incident, trigger)
  - Reads incident.severity → actions config → resolved ActionTypes
  - Trigger: on_create (incident born) or on_update (escalation)
- Execute actions via action_engine.execute(incident, actions, trigger)
  - Checks cooldown: delta ≥ settings.actions.cooldown_sec
  - Runs action._run(): log/email/alarm (currently print stubs)
  - Persists ActionRecord
  - Updates incident.last_action_at

**Event Publishing:**
- pubsub topics: detection, event, alert, incident, action, session.state
- Payload: JSON-serializable dicts

**HUD Overlay (_overlay_panel, _annotate):**
- Draw FPS, active tracks, latency, decision counters
- Per-face: bbox color (ACCEPT=red, REVIEW=orange, REJECT=green)
- Event feed: 7-item queue of latest actions
- Confidence trend: 6-item deque of recent smoothed_confidences

**Constraints:**
- Single camera (device 0)
- Queues size 2: prioritize freshness, drop stale frames
- Daemon threads: safe to exit without explicit cleanup

---

## Intelligence Pipeline

### Rules Engine (`src/trace_aml/pipeline/rules_engine.py`)

**Purpose:** Deterministic alert generation based on entity behavior.

**Class: RulesEngine**

**State:**
```python
config: Settings
store: VectorStore
cache: dict[(entity_id, rule_type), float]  # Cooldown timestamps
```

**Three Deterministic Rules:**

#### 1. Reappearance Rule
```python
_check_reappearance(event: EventRecord) → list[AlertRecord]
  window = config.rules.reappearance.window_sec (default 10)
  min_events = config.rules.reappearance.min_events (default 3)
  
  Get events = store.get_events(entity_id, window_sec=10)
  If len(events) ≥ 3 ∧ cooldown_check(entity_id, REAPPEARANCE):
    Create AlertRecord(type=REAPPEARANCE, severity=calculated)
```

**Trigger:** Entity re-detected 3+ times within 10-second window

**Severity:** Mapped as:
- count ≥ 5: MEDIUM
- count < 5: LOW

#### 2. Unknown Recurrence Rule
```python
_check_unknown_recurrence(event: EventRecord) → list[AlertRecord]
  If event.is_unknown:
    window = config.rules.unknown.window_sec (default 10)
    min_events = config.rules.unknown.min_events (default 3)
    Get events = store.get_events(entity_id, window_sec=10)
    If len(events) ≥ 3 ∧ cooldown_check(entity_id, UNKNOWN_RECURRENCE):
      Create AlertRecord(type=UNKNOWN_RECURRENCE, severity=HIGH)
```

**Trigger:** Unknown entity (UNK###) re-detected 3+ times within 10s

**Severity:** Always HIGH (elevated priority for unknowns)

#### 3. Instability Rule
```python
_check_instability(event: EventRecord) → list[AlertRecord]
  window = config.rules.instability.window_sec (default 10)
  std_threshold = config.rules.instability.std_threshold (default 0.15)
  
  Get events = store.get_events(entity_id, window_sec=10)
  confidences = [float(e.confidence) for e in events]  # Normalize to 0-1
  If len(confidences) ≥ 3:
    std = np.std(confidences)
    If std > 0.15 ∧ cooldown_check(entity_id, INSTABILITY):
      Create AlertRecord(type=INSTABILITY, severity=MEDIUM)
```

**Trigger:** High variance in confidence scores (jittery identity?)

**Severity:** MEDIUM (suggests enrollment or detection issues)

**Cooldown Mechanism:**
```python
_cooldown(entity_id: str, rule_type: AlertType) → bool
  key = (entity_id, rule_type)
  now = time.time()
  last = cache.get(key, 0)
  if now - last < cooldown_sec (default 15):
    return False  # Suppress duplicate alerts
  cache[key] = now
  return True
```

**Rationale:** Prevent alert spam for same entity/rule within 15-second window

**AlertRecord Generation:**
```python
_build_alert(event, alert_type, count) → AlertRecord
  Return AlertRecord(
    alert_id=new_alert_id(),
    entity_id=event.entity_id,
    type=alert_type,
    severity=_map_severity(event, alert_type, count),
    reason=f"{alert_type} detected with {count} events",
    timestamp_utc=event.timestamp_utc,
    first_seen_at=event.timestamp_utc,
    last_seen_at=event.timestamp_utc,
    event_count=count,
  )
```

**Constraints:**
- Purely logic-based: no ML, deterministic
- Stateless between calls (cache via time, not persistence)
- Cache cleared on session exit (in-memory)

---

### Incident Manager (`src/trace_aml/pipeline/incident_manager.py`)

**Purpose:** Group alerts into incidents, one per entity.

**Class: IncidentManager**

**Algorithm: handle_alert(alert: AlertRecord) → (IncidentRecord, trigger_label)**

1. **Check for active incident:**
   ```python
   active = store.get_active_incident(alert.entity_id)
   ```

2. **If active incident exists:**
   - Append alert.alert_id to alert_ids list (if not already present)
   - Update last_seen_time = alert.timestamp_utc
   - Preserve start_time, status=OPEN
   - Increment alert_count
   - Persist via store.update_incident()
   - Return (updated_incident, "on_update")

3. **Else: Create new incident**
   ```python
   created = IncidentRecord(
     incident_id=new_incident_id(),
     entity_id=alert.entity_id,
     status=OPEN,
     start_time=alert.timestamp_utc,
     last_seen_time=alert.timestamp_utc,
     alert_ids=[alert.alert_id],
     alert_count=1,
     severity=alert.severity,
     summary=_build_summary(alert),
   )
   ```
   - Persist via store.create_incident()
   - Return (created_incident, "on_create")

**Trigger Labels:**
- `"on_create"`: Brand new incident
- `"on_update"`: Existing incident updated

**Summary Generation:**
```python
_build_summary(alert) → str
  Return f"{alert.type}: {alert.reason}"
  Example: "UNKNOWN_RECURRENCE: Unknown_Recurrence detected with 3 events"
```

**Severity Inheritance:**
- New incident: severity = alert.severity (initially LOW/MEDIUM/HIGH)
- Updated incident: severity preserved (not escalated by this module)

**Constraints:**
- One incident per entity_id (ensure_entity pattern)
- Incidents persist until manually closed
- Alert history: full alert_ids list accumulated

---

### Policy Engine (`src/trace_aml/pipeline/policy_engine.py`)

**Purpose:** Map incident severity + trigger → prescribed action set.

**Class: PolicyEngine**

**Data Structure: ActionPolicyBySeverity**
```python
@dataclass
class ActionPolicyBySeverity:
  low: list[str] = []
  medium: list[str] = ["log"]  # Default
  high: list[str] = ["log", "email", "alarm"]  # Default
```

**Config Example:**
```yaml
actions:
  enabled: true
  on_create:
    low: []
    medium: ["log"]
    high: ["log", "email", "alarm"]
  on_update:
    low: []
    medium: ["log"]
    high: ["log"]
  cooldown_sec: 20
```

**Algorithm: evaluate(incident: IncidentRecord, trigger: ActionTrigger) → list[ActionType]**

1. **Check enabled:**
   - If not self.config.actions.enabled: return []

2. **Determine trigger:**
   - trigger_value = "on_create" or "on_update"

3. **Route to config:**
   ```python
   if trigger_value == "on_create":
     configured = self.config.actions.on_create.{severity}
   else:
     configured = self.config.actions.on_update.{severity}
   ```
   - severity = incident.severity (low/medium/high)

4. **Resolve action types:**
   - For each string in configured:
     - Try ActionType(value)
     - Catch ValueError: skip invalid
   - Return resolved list

**Example Flows:**
- New HIGH incident, on_create: [log, email, alarm]
- Updated MEDIUM incident, on_update: [log]
- New LOW incident, on_create: []

**Constraints:**
- Action list must be explicitly configured (no defaults beyond class)
- Non-existent actions silently skipped

---

### Action Engine (`src/trace_aml/pipeline/action_engine.py`)

**Purpose:** Execute prescribed actions with audit logging.

**Class: ActionEngine**

**Algorithm: execute(incident, actions: list[ActionType], trigger) → list[ActionRecord]**

1. **If no actions:** return []

2. **Check cooldown:**
   - Parse incident.last_action_at (ISO string)
   - If last action < cooldown_sec ago: return []
   - Rationale: Prevent action spam per incident

3. **For each action in actions:**
   - Call _run(action_type, incident, trigger)
   - Create ActionRecord with status (success/failed)
   - Persist to store.insert_action()
   - Append to emitted list

4. **If any actions emitted:**
   - Update incident.last_action_at = utc_now_iso()

5. **Return emitted list**

**Method: _run(action_type, incident, trigger) → (bool, str)**

**Current Stubs (for v3 MVP):**
```python
if action_type == ActionType.log:
  print(f"[ACTION] log incident {incident_id}")
  return True, "logged"

if action_type == ActionType.email:
  print(f"[ACTION] email sent for {incident_id}")
  return True, "email_sent"

if action_type == ActionType.alarm:
  print(f"[ACTION] alarm triggered for {incident_id}")
  return True, "alarm_triggered"
```

**Real Implementation (Future):**
- Log: Structured logging to syslog or log aggregator
- Email: SMTP integration, templated messages
- Alarm: Webhook, sound, or hardware trigger

**ActionRecord Creation:**
```python
ActionRecord(
  action_id=new_action_id(),
  incident_id=incident.incident_id,
  action_type=action_type,
  trigger=trigger,
  status=success|failed,
  reason=explanation,
  context=_context_for(incident, trigger, action_type, reason),
)
```

**Context Dict:**
```python
{
  "incident_id": ...,
  "entity_id": ...,
  "incident_status": ...,
  "incident_severity": ...,
  "incident_summary": ...,
  "trigger": "on_create" | "on_update",
  "action_type": "log" | "email" | "alarm",
  "explanation": "logged" | "email_sent" | ...
}
```

**Cooldown Mechanism:**
```python
_cooldown_allows(incident) → bool
  last = parse_iso(incident.last_action_at)
  if last is None: return True
  delta = (now - last).total_seconds()
  return delta ≥ cooldown_sec (default 20)
```

**Constraints:**
- Per-incident cooldown (not per-action-type)
- Atomic: all actions execute if cooldown passes
- No retry: failures are logged but not escalated

---

## Recognizers

### ArcFace Recognizer (`src/trace_aml/recognizers/arcface.py`)

**Purpose:** InsightFace wrapper for face detection and embedding extraction.

**Class: ArcFaceRecognizer**

**Initialization:**
```python
def __init__(self, settings: Settings) → None
  Load buffalo_sc model on first _ensure_app() call
  Set liveness checker (default PassThroughLiveness)
```

**Model:** buffalo_sc
- Detector: SCRFD (face bounding boxes)
- Recognizer: ArcFace (512-D embeddings, cosine similarity)
- Provider: CPUExecutionProvider (configurable)
- Det size: (640, 640) from settings

**Method: detect_faces(frame_bgr) → list[FaceCandidate]**

1. **Detect:** `self._app.get(frame_bgr)` → list of face objects
2. **Fallback (if no faces ∧ enable_preprocess_fallback):**
   - Apply CLAHE (contrast-limited adaptive histogram equalization)
   - Apply gamma correction if brightness < 85
   - Re-detect on enhanced frame
3. **Extract:**
   - For each face:
     - bbox = (x1, y1, x2, y2) → (x, y, w, h)
     - embedding = normalize to 512-D unit vector
     - detector_score = confidence from SCRFD
   - Return list[FaceCandidate]

**Face Quality Scoring: _face_quality(frame, bbox, detector_score)**

```python
Inputs: frame_bgr (H×W×3), bbox (x,y,w,h), detector_score

Components:
  face_ratio = (w×h) / (frame_w×frame_h)
  brightness = cv2.cvtColor(frame, BGR2GRAY).mean()
  
Subscores:
  ratio_score = clip(face_ratio / 0.10, 0-1)
  brightness_score = clip(1 - |brightness - 120| / 120, 0-1)
  det_score = clip(detector_score, 0-1)

Final:
  face_quality = 0.45×det_score + 0.30×ratio_score + 0.25×brightness_score
```

**Output:** 0-1 float

**Threshold Relaxation (Robust Matching):**

If `settings.recognition.robust_matching` enabled:
```python
face_quality_score = _face_quality(...)
relax = (1 - clip(face_quality, 0-1)) × threshold_relaxation
dynamic_accept = max(0.50, accept_threshold - relax)
dynamic_review = max(0.35, review_threshold - 1.1×relax)
```

**Logic:** Low-quality detections lower the bar to accept/review

**Quality Flags (applied before temporal):**
- `liveness_fail`: Liveness score < threshold ∧ strict_reject enabled
- `low_face_quality`: face_quality < low_quality_threshold (0.40)
- `person_not_active`: person_id not in active_person_ids()
- `person_state_not_active`: person.lifecycle_state ≠ "active"

**Main Method: match(frame_bgr, vector_store) → list[tuple[RecognitionMatch, embedding, LivenessResult]]**

**Layer 1: Runtime Face Quality Gate**
```
candidates = detect_faces(frame) [all detected faces]
candidates = [c for c in candidates if c.detector_score ≥ min_detector_score (0.55)]
```
**Rationale:** Drop uncertain detections before embedding (prevents ghost entities)

**For each candidate:**

1. **Liveness check:**
   - Crop face region from frame
   - liveness = self._liveness.evaluate(crop)

2. **Embedding search (two-tier):**
   - Tier 1: Search active persons only
     - top_k = max(active_gallery_search_k=96, top_k=5)
   - If no results: Tier 2: Search all persons
     - top_k = max(top_k × 8=40, 32)

3. **Person aggregation:**
   - Call _aggregate_person_scores(best_rows, active_ids)
   - Groups embeddings by person_id
   - Ranks by: 0.70×best_sim + 0.30×mean_top_sim + support_bonus
   - Returns list sorted by robust_similarity descending

4. **Best match selection:**
   - best = candidate_scores[0] if exists else None
   - similarity = best.similarity or 0.0
   - confidence = round(similarity × 100, 2)
   - person_id = best.person_id or ""

5. **Quality flags:**
   - Liveness_fail: strict rejection if enabled
   - low_face_quality: soft indicator
   - person_not_active / person_state_not_active: hard blocks

6. **Is_match decision (pre-temporal):**
   - true if: person_id ∧ person_id in active_ids ∧ no hard_block_flags ∧ similarity ≥ dyn_review_thr
   - Decision still pending (temporal will finalize)

7. **Create RecognitionMatch:**
   ```python
   RecognitionMatch(
     person_id=person_id or None,
     name=person.name if person else "Unknown",
     category=person.category or "unknown",
     similarity=similarity,
     confidence=confidence,
     bbox=candidate.bbox,
     is_match=is_match,  # Pre-temporal; temporal will update
     decision_state=REJECT,  # Will be updated by temporal
     decision_reason="temporal_pending",
     smoothed_confidence=confidence,  # Will be smoothed by temporal
     quality_flags=quality_flags,
     candidate_scores=candidate_scores,
     metadata={...},  # detector_score, face_quality, dynamic_thresholds, etc.
   )
   ```

8. **Return tuple:**
   - (RecognitionMatch, embedding, LivenessResult)

**Constraints:**
- Embeddings only 512-D (validation on creation)
- No model switching at runtime (single buffalo_sc)
- Liveness always computed (even if disabled) for metadata
- Search limited to 100K embeddings per query

---

## Quality System

### Scoring (`src/trace_aml/quality/scoring.py`)

**Purpose:** Image quality metrics for enrollment validation.

**Function: score_face_image(settings, frame_bgr, bbox) → QualityComponents**

**Inputs:**
- frame_bgr: BGR image (H×W×3)
- bbox: (x, y, w, h) bounding box from detector

**Output: QualityComponents**
```python
@dataclass
class QualityComponents:
    sharpness: float
    face_ratio: float
    brightness: float
    pose_score: float
    quality_score: float  # Weighted average (0-1)
    reasons: list[str]  # Failure reasons
    passed: bool  # All checks passed
```

**Metric 1: Sharpness**
```python
gray = cv2.cvtColor(frame, BGR2GRAY)
crop = extract_face_region(frame, bbox)
gray_crop = cv2.cvtColor(crop, BGR2GRAY)
sharpness = cv2.Laplacian(gray_crop, CV_64F).var()
```
**Threshold:** min_sharpness = 55.0
**Scoring:** normalize(sharpness, 55, 137.5) clamped to 0-1

**Metric 2: Face Ratio**
```python
face_area = w × h
frame_area = frame_w × frame_h
face_ratio = face_area / frame_area
```
**Threshold:** min_face_ratio = 0.03 (3% of frame)
**Scoring:** normalize(face_ratio, 0.03, 0.075) clamped to 0-1

**Metric 3: Brightness**
```python
gray_crop = cv2.cvtColor(crop, BGR2GRAY)
brightness = gray_crop.mean()
```
**Thresholds:** min=45, max=220
**Scoring:** 1 - clamp(|brightness - 120| / 120, 0-1)

**Metric 4: Pose Score (Centering + Aspect)**
```python
face_center = ((x1+x2)/2, (y1+y2)/2)
frame_center = (w/2, h/2)
center_distance = (|dx|/frame_center_x + |dy|/frame_center_y) / 1.6
centering_score = 1 - clamp(center_distance, 0-1)

aspect = w / h
aspect_score = 1 - clamp(|aspect - 0.8|, 0-1)

pose_score = 0.65×centering + 0.35×aspect
```
**Threshold:** min_pose_score = 0.28

**Metric 5: Overall Quality Score**
```python
quality_score = 0.30×face_score + 0.30×sharpness_score 
              + 0.20×brightness_score + 0.20×pose_score
```

**Constraints:**
- All checks must pass for passed=True
- Reasons accumulated if any fail
- Edge case: no bbox → passed=False, all zeros

**Function: build_assessment(settings, person_id, source_path, frame, bbox) → QualityAssessment**
- Wraps score_face_image() result
- Creates QualityAssessment domain model
- Ready to persist

---

### Gating (`src/trace_aml/quality/gating.py`)

**Purpose:** Lifecycle state machine based on enrollment progress.

**Function: decide_person_lifecycle(settings, total_images, valid_images, embeddings_count, avg_quality) → LifecycleDecision**

**State Transitions:**

```
┌─────────────────────────────────────────┐
│  total_images = 0                       │
│  → DRAFT (reason: no_images_uploaded)   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  embeddings_count = 0 ∨ valid_images = 0│
│  → BLOCKED (reason: no_valid_embeddings)│
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  embeddings < min_embeddings_ready (2)  │
│  → DRAFT (reason: insufficient_...)     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  avg_quality < min_quality_score (0.38) │
│  → BLOCKED (reason: average_quality_...) │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  valid_images < min_valid_images (6)    │
│  ∨ embeddings < min_embeddings_active(6)│
│  → READY (reason: meets_minimum_...)    │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  All gates passed                       │
│  → ACTIVE (reason: quality_gates_passed)│
└─────────────────────────────────────────┘
```

**Return: LifecycleDecision**
```python
@dataclass
class LifecycleDecision:
    state: PersonLifecycleStatus
    reason: str
    enrollment_score: float  # avg_quality from input
```

**Thresholds:**
- min_embeddings_ready: 2
- min_embeddings_active: 6
- min_valid_images: 6
- min_quality_score: 0.38

**Constraints:**
- Unidirectional progression implied (not enforced)
- blocked state sticky until manually updated
- enrollment_score always set to avg_quality (for reporting)

---

## System Integration & Flow

### End-to-End Recognition Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CAPTURE PHASE                                                │
│    CameraCapture._run()                                         │
│    → cv2.imread(device 0, 1280×720) per frame                  │
│    → FramePacket(frame, captured_at, frame_index)              │
│    → Queue to result_queue (size 2)                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. INFERENCE PHASE                                              │
│    InferenceWorker._run()                                       │
│    → recognizer.detect_faces(frame)                            │
│      - SCRFD detector (buffalo_sc)                             │
│      - Layer 1: filter by min_detector_score (0.55)            │
│    → For each candidate:                                        │
│      - score_face_quality()                                     │
│      - robust_thresholds (dynamic accept/review)               │
│      - search_embeddings() for person_id                       │
│      - _aggregate_person_scores() (groupby + ranking)          │
│      - Create RecognitionMatch + quality_flags                 │
│    → InferencePacket(frame, matches, embeddings, liveness)    │
│    → Queue to result_queue (size 2)                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TEMPORAL DECISION PHASE                                      │
│    TemporalDecisionEngine.evaluate()                            │
│    → _assign_track_id(bbox)                                    │
│      - Match to existing track (IoU, distance, recency)        │
│      - Or create new track T####                              │
│    → Accumulate: confidences, identities (deque, size=6)       │
│    → Smooth confidence: EMA(alpha=0.6)                         │
│    → Vote on identity: Counter, most_common                    │
│    → Decision:                                                 │
│      - REJECT if any hard block                                │
│      - ACCEPT if votes ≥ 2 AND smoothed ≥ accept_thr          │
│      - REVIEW if votes ≥ 1 AND smoothed ≥ review_thr          │
│    → Return: TemporalDecision(track_id, decision_state, ...)  │
│                                                                 │
│    Layer 4: Minimum Commit Confidence Gate                     │
│    → Track must reach 45% smoothed confidence before DB write  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. ENTITY RESOLUTION PHASE                                      │
│    EntityResolver.resolve()                                     │
│    → Layer 3: Track ownership cache                            │
│      - If track_id already owns entity: reuse, update          │
│    → If person_id (known):                                     │
│      - ensure_entity(entity_id=person_id, type=KNOWN)          │
│    → Else (unknown):                                           │
│      - resolve_or_create_unknown_entity(embedding, 0.55)       │
│      - Returns UNK### (new or reused)                          │
│    → Register track → entity mapping                           │
│    → Return: EntityResolution(entity_id, type)                │
│                                                                 │
│    Layer 5: Minimum Unknown Surface Threshold                  │
│    → If REJECT ∧ no person_id ∧ smoothed < 35%: DROP          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. DETECTION RECORDING PHASE                                    │
│    RecognitionSession._save_detection()                         │
│    → Decide persistence: observe persist_unknown/persist_review │
│    → Save screenshot to data/screenshots/{detection_id}.jpg    │
│    → Persist DetectionEvent to store.detections                │
│    → [Cooldown check: log_cooldown_seconds (5s)]               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. EVENT CREATION PHASE                                         │
│    EntityResolver.create_event_record()                         │
│    → Wrap resolution + match into EventRecord                  │
│    → Persist to store.events                                   │
│    → Publish "event" topic                                     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. RULES ENGINE PHASE                                           │
│    RulesEngine.process_event()                                  │
│    → Check reappearance (3+ events in 10s)                     │
│    → Check unknown_recurrence (UNK, 3+ events in 10s)          │
│    → Check instability (std > 0.15)                            │
│    → [Cooldown: 15s per rule/entity]                           │
│    → Return list[AlertRecord]                                  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. INCIDENT GROUPING PHASE                                      │
│    IncidentManager.handle_alert()                               │
│    → If active incident exists:                                │
│      - Append alert_id, update last_seen_time                 │
│      - Trigger: "on_update"                                    │
│    → Else:                                                     │
│      - Create new incident with alert_id                       │
│      - Trigger: "on_create"                                    │
│    → Return (IncidentRecord, trigger_label)                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. POLICY EVALUATION PHASE                                      │
│    PolicyEngine.evaluate(incident, trigger)                    │
│    → Route: config.actions.{on_create|on_update}.{severity}   │
│    → Map action strings → ActionType enums                     │
│    → Return list[ActionType]                                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10. ACTION EXECUTION PHASE                                      │
│    ActionEngine.execute(incident, actions, trigger)            │
│    → Check cooldown (20s per incident)                         │
│    → For each action:                                          │
│      - _run(action_type, incident, trigger)                    │
│      - Create ActionRecord(status=success|failed)              │
│      - Persist to store.actions                                │
│    → Update incident.last_action_at                            │
│    → Return list[ActionRecord]                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5-Layer Quality Gates

| Layer | Component | Check | Outcome |
|---|---|---|---|
| **1** | ArcFace | min_detector_score ≥ 0.55 | Drop if too uncertain |
| **2** | Enrollment | Image quality metrics (sharpness, brightness, pose) | Mark invalid, skip from training |
| **3** | Entity Resolver | Track ownership cache | Prevent duplicate entities per track |
| **4** | Temporal | min_commit_confidence ≥ 45% | Hold back DB write until confident |
| **5** | Session | min_unknown_surface_threshold ≥ 35% | Drop ambiguous unknowns (noise) |

### Cooldown Mechanisms

| Component | Cooldown | Rationale |
|---|---|---|
| **Log Cooldown** | 5 seconds per (person_id, decision_state) | Avoid repeated screenshots of same person |
| **Rules Cooldown** | 15 seconds per (entity_id, rule_type) | Prevent alert spam |
| **Action Cooldown** | 20 seconds per incident | Prevent action cascade |

### Threading Model

| Component | Thread Type | Lifecycle | Safety |
|---|---|---|---|
| **CameraCapture** | Daemon thread | start() → _run() → stop() | _shared_lock (RLock) for latest frame |
| **InferenceWorker** | Daemon thread | start() → _run() → stop() | Queue thread-safe (built-in lock) |
| **Session main** | Main thread | enable_camera() → process loop | _camera_lock (Lock) guards state |
| **Streaming** | Event listeners (sync) | subscribe() → listener(event) | _lock (RLock) for listeners dict |

**Key Property:** Queues are thread-safe; no manual locking needed for frame/result queues.

---

## Key Assumptions & Design Constraints

1. **Single Source:** One built-in webcam (device index 0 enforced)
2. **Real-Time:** 30 FPS target, frame drops prioritized over latency
3. **Deterministic Rules:** No ML thresholds; all rule logic hard-coded
4. **Stateless Pipeline:** Per-session state (tracks, unknowns) lost on restart
5. **Embedding Search:** O(n_embeddings) lookup; no vector indices at v3
6. **One Incident per Entity:** Prevents incident explosion
7. **Cooldown Semantics:** Simple time-based (in-memory cache, loses state on restart)
8. **Action Stubs:** log/email/alarm currently print-only; no actual integrations

---

## Future Optimization Opportunities

1. **Vector Indexing:** Add FAISS or Annoy for sub-linear embedding search
2. **Persistence:** Replace cooldown cache with database TTL
3. **Async Actions:** Queue actions for async execution instead of blocking
4. **HTTP API:** Expose REST endpoints for live monitoring
5. **Config Hot-Reload:** Watch config file for changes without restart
6. **Distributed Incidents:** Multi-camera incident correlation
7. **Liveness Integration:** Real liveness checkers (FIDO, challenge-response)
8. **Dynamic Thresholds:** Learn optimal per-person thresholds from data

---

**End of Backend Code Analysis Document**
