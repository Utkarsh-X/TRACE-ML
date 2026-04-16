# TRACE-ML Codebase Architecture Map

**Last Updated:** April 16, 2026  
**Scope:** Detection storage, recognition pipeline, person/entity management, profile photos, frontend display

---

## 1. DETECTION STORAGE & SCHEMA

### Location
- **Primary:** [src/trace_aml/store/vector_store.py](src/trace_aml/store/vector_store.py) (lines 150-250)
- **Models:** [src/trace_aml/core/models.py](src/trace_aml/core/models.py) (lines 159-177)
- **Analytics Query:** [src/trace_aml/store/analytics.py](src/trace_aml/store/analytics.py)

### Detections Table Schema (LANCE format)
Located in [vector_store.py](src/trace_aml/store/vector_store.py#L150-L177)

```python
DETECTIONS_TABLE: detections.lance
├── detection_id (string, PK)
├── timestamp_utc (string)
├── source (string) = "webcam:0"
├── person_id (string, FK → persons)
├── name (string)
├── category (string)
├── confidence (float32) = raw inference score
├── similarity (float32) = embedding match similarity
├── smoothed_confidence (float32) = temporal-smoothed confidence
├── bbox (string) = JSON tuple [x, y, w, h]
├── track_id (string) = temporal tracking ID
├── decision_state (string) ∈ {accept, review, reject}
├── decision_reason (string)
├── quality_flags (string) = JSON array
├── liveness_provider (string)
├── liveness_score (float32)
├── screenshot_path (string)
├── metadata (string) = JSON object
└── embedding (list[float32], 512-dim) = face vector
```

### Detection Decisions Table Schema
Separate audit trail table at [vector_store.py](src/trace_aml/store/vector_store.py#L180-L198)

```python
DETECTION_DECISIONS_TABLE: detection_decisions.lance
├── detection_id (string, PK)
├── track_id (string)
├── decision_state (string)
├── decision_reason (string)
├── smoothed_confidence (float32)
├── quality_flags (string) = JSON
├── top_candidates (string) = JSON [{name, similarity, person_id}, ...]
├── liveness_provider (string)
├── liveness_score (float32)
└── created_at (string)
```

### DetectionEvent Model
[src/trace_aml/core/models.py](src/trace_aml/core/models.py#L159-L177)

```python
class DetectionEvent(BaseModel):
    detection_id: str
    timestamp_utc: str
    source: str = "webcam:0"
    person_id: str | None = None
    name: str = "Unknown"
    category: str = "unknown"
    confidence: float = 0.0       # Raw inference score
    similarity: float = 0.0        # Embedding similarity
    smoothed_confidence: float = 0.0  # Temporal smoothing
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    track_id: str = ""
    decision_state: DecisionState  # enum: accept|review|reject
    decision_reason: str
    quality_flags: list[str]       # [blur, dark, pose, ...]
    liveness_provider: str = "none"
    liveness_score: float = 0.0
    screenshot_path: str
    metadata: dict[str, Any]
```

### Key Data Flow
1. **Detection** → Raw face detected, bboxes extracted
2. **Embedding** → 512-dim face vector computed
3. **Recognition** → Vector similarity search against person gallery
4. **Temporal** → Multi-frame confidence smoothing & decision stabilization
5. **Persistence** → Both detection and decision_decisions records created
6. **Screenshot** → JPEG saved to `data/screenshots/{detection_id}.jpg`

---

## 2. RECOGNITION PIPELINE

### Entry Points

#### a) Live Recognition Session
[src/trace_aml/pipeline/session.py](src/trace_aml/pipeline/session.py) - Main orchestrator

```python
class RecognitionSession:
    # Core components
    recognizer: ArcFaceRecognizer
    temporal: TemporalDecisionEngine
    entity_resolver: EntityResolver
    rules_engine: RulesEngine
    incident_manager: IncidentManager
    
    # Camera & inference control
    enable_camera() → starts CameraCapture + InferenceWorker
    enable_recognition() → starts inference processing
    disable_camera() / disable_recognition()
```

#### b) Inference Worker Thread
[src/trace_aml/pipeline/inference.py](src/trace_aml/pipeline/inference.py)

```python
class InferenceWorker(Thread):
    # Per-frame processing
    1. Frame capture from camera queue
    2. Face detection (SCRFD detector, CPU-only)
    3. Face embedding (ArcFace, 512-dim vectors)
    4. Quality assessment (sharpness, brightness, pose, face_ratio)
    5. Gallery search (top-k matching against active persons)
    6. Output: InferencePacket with RecognitionMatch objects
```

### Pipeline Flow Diagram
```
Camera Frame (video input)
    ↓
[CameraCapture] → Frame Queue
    ↓
[InferenceWorker] → 
    ├─ Face Detection (SCRFD)
    ├─ Embedding Extraction (ArcFace)
    ├─ Quality Assessment
    └─ Gallery Search (topk=5 persons)
    ↓
[RecognitionMatch] + [EmbeddingVector]
    ↓
Result Queue → [RecognitionSession._annotate()]
    ↓
[_apply_temporal_decision()]
    ├─ Track ID assignment
    ├─ Confidence smoothing
    └─ Decision state (accept|review|reject)
    ↓
[_save_detection()] (commit gate check)
    ├─ DetectionEvent → store.detections
    ├─ Screenshot → data/screenshots/{id}.jpg
    ├─ DecisionRecord → store.detection_decisions
    └─ EntityResolver → create/link entity
    ↓
[RulesEngine.process_event()] → Alerts
    ↓
[IncidentManager] → Incidents
    ↓
[PolicyEngine + ActionEngine] → Notifications
```

### Key Functions in session.py

#### Recognition Entry Point
[session.py](src/trace_aml/pipeline/session.py#L301-L420) - `_save_detection()`

```python
def _save_detection(self, frame, match: RecognitionMatch, 
                    embedding: list[float], liveness: LivenessResult):
    # Step 1: Check minimum unknown surface threshold
    if decision == DecisionState.reject and not match.person_id:
        if match.smoothed_confidence < min_unknown_surface_threshold:
            return  # Drop noisy faces
    
    # Step 2: Check persistence flags
    if decision == DecisionState.reject and not settings.recognition.persist_unknown:
        persist_detection = False
    
    # Step 3: Create DetectionEvent + save to store
    event = DetectionEvent(
        detection_id=new_detection_id(),
        timestamp_utc=utc_now_iso(),
        confidence=match.confidence,
        similarity=match.similarity,
        smoothed_confidence=match.smoothed_confidence,
        bbox=match.bbox,
        decision_state=decision,
        quality_flags=match.quality_flags,
        liveness_provider=liveness.provider,
        screenshot_path=str(screenshot_path),
    )
    
    # Step 4: Store detection + embedding
    self.store.add_detection(event, embedding=embedding)
    
    # Step 5: Store decision record
    self.store.add_detection_decision(
        detection_id=detection_id,
        decision_state=decision.value,
        smoothed_confidence=match.smoothed_confidence,
        top_candidates=match.candidate_scores,
    )
    
    # Step 6: Resolve/create entity
    resolution = self.entity_resolver.resolve(match, embedding)
    
    # Step 7: Create event record
    core_event = self.entity_resolver.create_event_record(resolution, match)
    self.store.add_event(core_event)
    
    # Step 8: Rules engine → alerts
    alerts = self.rules_engine.process_event(core_event)
    for alert in alerts:
        self.store.add_alert(alert)
        incident = self.incident_manager.handle_alert(alert)
```

#### Temporal Decision Application
[session.py](src/trace_aml/pipeline/session.py#L397-L418) - `_apply_temporal_decision()`

```python
def _apply_temporal_decision(self, match: RecognitionMatch, now: float):
    # Get temporal engine result
    temporal = self.temporal.evaluate(match, now_ts=now)
    
    # Update match with temporal decision
    match.track_id = temporal.track_id
    match.smoothed_confidence = temporal.smoothed_confidence
    match.decision_state = temporal.decision_state
    match.decision_reason = temporal.decision_reason
    
    # If accepted/review + person exists → link person
    if temporal.decision_state in {DecisionState.accept, DecisionState.review}:
        person = self.store.get_person(temporal.resolved_person_id)
        match.person_id = person["person_id"]
        match.name = person["name"]
        match.category = person["category"]
    else:
        # Rejected → mark as unknown
        match.person_id = None
        match.name = "Unknown"
        match.category = "unknown"
```

### Inference Worker Location
[src/trace_aml/pipeline/inference.py](src/trace_aml/pipeline/inference.py)

Key responsibilities:
- **Face Detection:** SCRFD detector (confidence > 0.5)
- **Embedding:** ArcFace 512-dimensional vectors (L2 normalized)
- **Quality Scoring:** Sharpness, brightness, face ratio, pose
- **Gallery Search:** Vector similarity against active persons (topk=5)
- **Output:** `RecognitionMatch` with `candidate_scores` list

### Temporal Decision Engine
[src/trace_aml/pipeline/temporal.py](src/trace_aml/pipeline/temporal.py)

Features:
- Multi-frame smoothing (running average of confidences)
- Track ID assignment for continuity
- Decision state voting (accept if stable across frames)
- Minimum confidence + vote thresholds for commitment
- Prevents warmup-phase ghost entities

---

## 3. PERSON & ENTITY RECORD MANAGEMENT

### PersonRecord Model
[src/trace_aml/core/models.py](src/trace_aml/core/models.py#L86-L104)

```python
class PersonRecord(BaseModel):
    person_id: str                         # PRC001, PRM002, etc.
    name: str
    category: PersonCategory               # criminal|missing|employee|vip
    severity: str = ""
    dob: str = ""
    gender: str = ""
    last_seen_city: str = ""
    last_seen_country: str = ""
    notes: str = ""
    profile_photo_path: str = ""           # Best recognition image
    profile_photo_confidence: float = 0.0  # Confidence of profile photo
    lifecycle_state: PersonLifecycleStatus # draft→ready→active→blocked
    lifecycle_reason: str
    enrollment_score: float = 0.0          # Quality of embeddings (0-100)
    valid_embeddings: int = 0              # Count of usable vectors
    created_at: str
    updated_at: str
```

### Persons Table Schema
[src/trace_aml/store/vector_store.py](src/trace_aml/store/vector_store.py#L100-L130)

```python
PERSONS_TABLE: persons.lance
├── person_id (string, PK)
├── name (string)
├── category (string)
├── severity (string)
├── dob (string)
├── gender (string)
├── last_seen_city (string)
├── last_seen_country (string)
├── notes (string)
├── profile_photo_path (string)           ← Best match screenshot path
├── profile_photo_confidence (float32)    ← Detection confidence
├── lifecycle_state (string)
├── lifecycle_reason (string)
├── enrollment_score (float32)
├── valid_embeddings (int32)
├── created_at (string)
└── updated_at (string)
```

### Person Lifecycle States
```
draft ─→ ready ─→ active ─→ blocked
  ↓
awaiting_training / awaiting_images
↓
ready = sufficient embeddings, enrollment_score > threshold
↓
active = ready for recognition matching
↓
blocked = excluded from matching
```

### EntityRecord Model
[src/trace_aml/core/models.py](src/trace_aml/core/models.py#L180-L187)

```python
class EntityRecord(BaseModel):
    entity_id: str                    # PRC001 or UNK0001
    type: EntityType                  # known|unknown
    status: EntityStatus = EntityStatus.active
    source_person_id: str | None      # For 'known' type
    created_at: str
    last_seen_at: str
```

### Entities Table
```python
ENTITIES_TABLE: entities.lance
├── entity_id (string, PK)
├── type (string)  # known|unknown
├── status (string)
├── source_person_id (string, FK → persons)
├── created_at (string)
└── last_seen_at (string)
```

### Person/Entity Resolution
[src/trace_aml/core/entity_resolver.py](src/trace_aml/core/entity_resolver.py)

```python
EntityResolver.resolve(match: RecognitionMatch, embedding: list) 
    ├─ If match.person_id exists → link to known person
    │  └─ create_entity(entity_id=person_id, type='known')
    └─ Else (unknown) → resolve_or_create_unknown_entity()
       ├─ Search unknown_profiles table for similar embeddings
       ├─ If similarity > threshold → reuse existing UNK0001
       └─ Else → create new UNK#### entity
```

### Person API Endpoints
[src/trace_aml/service/person_api.py](src/trace_aml/service/person_api.py)

```python
POST   /api/v1/persons              # Create person
GET    /api/v1/persons              # List all persons
GET    /api/v1/persons/{id}         # Get person detail
PATCH  /api/v1/persons/{id}         # Update metadata
DELETE /api/v1/persons/{id}         # Delete person
POST   /api/v1/persons/{id}/images  # Upload training images
POST   /api/v1/train/rebuild        # Rebuild embeddings
```

---

## 4. PROFILE PHOTO IMPLEMENTATION

### Profile Photo Fields in PersonRecord
[src/trace_aml/core/models.py](src/trace_aml/core/models.py#L96-L97)

```python
profile_photo_path: str = ""              # Relative path to screenshot
profile_photo_confidence: float = 0.0    # Confidence of detection
```

### Auto-Update Trigger
[src/trace_aml/pipeline/session.py](src/trace_aml/pipeline/session.py#L340-L360)

**Current Implementation:**
- Not yet fully integrated in recognition loop
- Placeholder in PersonRecord schema

**Planned Enhancement** (per [ENHANCEMENT_PLAN.md](ENHANCEMENT_PLAN.md#L13-L38)):

```python
PROFILE_PHOTO_CONFIG = {
    "min_confidence_threshold": 0.60,      # Only save if confidence >= 60%
    "auto_update": True,                   # Auto-update to better match
    "max_profile_filesize": 500_000,       # 500KB max
}

# When detection.smoothed_confidence > PROFILE_THRESHOLD:
# 1. Compare with existing profile photo confidence
# 2. If new detection > existing → swap profile
# 3. Update persons table profile_photo_path + confidence
```

### Storage Location
```
data/
├── screenshots/
│   ├── DET_UUID_1.jpg        ← Detection screenshot (raw)
│   ├── DET_UUID_2.jpg
│   └── ...
└── person_images/
    ├── PRC001/
    │   ├── training_1.jpg
    │   ├── training_2.jpg
    │   └── profile_photo.jpg  ← Best recognition match (optional)
    └── PRM002/
        └── ...
```

### Profile Photo Endpoint (Planned)
[src/trace_aml/service/person_api.py](src/trace_aml/service/person_api.py) (pending)

```python
POST /api/v1/persons/{id}/profile-photo
    Input: screenshot_path, confidence
    Action: Update persons.profile_photo_path + confidence
    
GET /api/v1/persons/{id}/profile-photo
    Output: {path, confidence, updated_at}
```

---

## 5. ENTITIES PAGE STRUCTURE & DISPLAY

### Frontend Location
[src/frontend/entities/](src/frontend/entities/)

- **HTML:** [index.html](src/frontend/entities/index.html)
- **Controller:** [entities.js](src/frontend/entities/entities.js)

### Two-View Pattern

#### a) Overview View (Grid of All Entities)
[entities.js](src/frontend/entities/entities.js#L13-L112)

```javascript
loadEntityList()
    ├─ TraceClient.entities({limit: 300})
    └─ renderGrid(list)
        ├─ Entity card per row:
        │  ├─ Type badge (known/unknown)
        │  ├─ Avatar icon (person / person_off)
        │  ├─ Name + entity_id (truncated)
        │  ├─ Last seen timestamp
        │  └─ Open incidents count
        └─ Filter by type (known|unknown) + search

applyFilters()
    └─ Client-side text + type filter
```

**Entity Card Fields:**
- Entity ID (PRC001, UNK0001, etc.)
- Name (from linked person or "Unknown")
- Type badge (KNOWN / UNKNOWN)
- Last seen timestamp (formatted)
- Open incident count (with visual indicator)
- Clickable → loads detail view

#### b) Detail View (Full Entity Profile)
[entities.js](src/frontend/entities/entities.js#L127-L212)

```javascript
loadEntityProfile(entityId)
    ├─ TraceClient.entityProfile(entityId)
    └─ Renders:
        ├─ Header section
        │  ├─ Entity name + ID
        │  ├─ Status (active/inactive)
        │  ├─ Category / severity
        │  └─ Last seen clock
        ├─ Photo + Stats grid
        │  ├─ Profile photo (if exists)
        │  ├─ Appearance count
        │  ├─ Open cases count
        │  ├─ Detection summary
        │  └─ Confidence bar
        ├─ Timeline section
        │  └─ Events/alerts/incidents (reverse chronological)
        └─ Incidents section
           └─ Open cases linked to entity
```

### Stat Bar (Compact Overview)
[index.html](src/frontend/entities/index.html#L22-L44)

```html
<div class="stat-bar">
  <div class="stat-bar__item">
    <span class="stat-bar__label">Total Entities</span>
    <span class="stat-bar__value">142</span>
  </div>
  <div class="stat-bar__item">
    <span class="stat-bar__label">Known</span>
    <span class="stat-bar__value">23</span>
  </div>
  <div class="stat-bar__item">
    <span class="stat-bar__label">Unknown</span>
    <span class="stat-bar__value">119</span>
  </div>
  <div class="stat-bar__item">
    <span class="stat-bar__label">Alerts Today</span>
    <span class="stat-bar__value">7</span>
  </div>
</div>
```

### Entity Card Layout
[index.html](src/frontend/entities/index.html#L51-L108)

```html
<div class="entity-card">
  <!-- Accent border per type -->
  <div class="entity-card::before" style="top-border-color: #ff6b35 | #4da6ff"></div>
  
  <!-- Header row -->
  <div class="flex items-center justify-between">
    <span class="entity-card__badge--known|unknown">KNOWN / UNKNOWN</span>
    <span class="ec-arrow">→</span>
  </div>
  
  <!-- Avatar + Name block -->
  <div class="flex items-center gap-3">
    <div class="entity-card__avatar">
      <person | person_off icon>
    </div>
    <div>
      <div class="entity-name">John Doe</div>
      <div class="entity-id">PRC001</div>
    </div>
  </div>
  
  <!-- Footer separator -->
  <div class="entity-card__footer">
    <span class="timestamp">Last seen: 2026-04-16 14:32 UTC</span>
    <span class="incident-badge" v-if="open_incidents > 0">
      3 OPEN
    </span>
  </div>
</div>
```

### Entity Profile View
[index.html](src/frontend/entities/index.html#L274-L370)

```html
<!-- Header -->
<div class="flex justify-between items-start">
  <div>
    <h1 id="entity-display-name">John Doe</h1>
    <div class="flex gap-6">
      <span id="entity-status">ACTIVE</span>
      <span id="entity-type">KNOWN</span>
      <span id="entity-severity">2 OPEN</span>
    </div>
  </div>
  <span id="entity-clock">Last: Apr 16, 14:32 UTC</span>
</div>

<!-- Photo + Stats Grid (2 cols) -->
<div class="flex flex-col lg:flex-row gap-8">
  <!-- Left: Photo placeholder -->
  <div class="w-full lg:w-1/3">
    <div id="entity-photo" class="bg-surface-low aspect-video flex items-center justify-center">
      [Profile photo area]
    </div>
  </div>
  
  <!-- Right: Stats -->
  <div class="flex-1 grid grid-cols-2 gap-4">
    <div class="stat-box">
      <span class="stat-label">Appearances</span>
      <span class="stat-value" id="stat-appearances">42</span>
    </div>
    <div class="stat-box">
      <span class="stat-label">Cases</span>
      <span class="stat-value" id="stat-incidents">2</span>
    </div>
  </div>
</div>

<!-- Timeline section -->
<div id="entity-timeline-root">
  <!-- Timeline items (reverse chronological) -->
</div>

<!-- Incidents section -->
<div id="entity-incidents-root">
  <!-- Linked cases -->
</div>
```

### Timeline Rendering
[entities.js](src/frontend/entities/entities.js#L189-L212)

```javascript
renderTimeline(timeline)
    └─ For each item (reverse chronological, max 25):
        ├─ Badge: INCIDENT | ALERT | EVENT
        ├─ Timestamp (formatted)
        ├─ Summary text
        └─ Confidence indicator (if detection)
```

### API Endpoints Used by Entities Page
[src/trace_aml/service/app.py](src/trace_aml/service/app.py#L323-L400)

```python
GET /api/v1/entities                   # List all (limit=300)
GET /api/v1/entities/{id}              # Entity summary
GET /api/v1/entities/{id}/profile      # Full profile (linked_person, timeline, etc.)
GET /api/v1/entities/{id}/timeline     # Timeline events
GET /api/v1/entities/{id}/incidents    # Linked cases
GET /api/v1/entities/{id}/suggestions  # Merge suggestions (similar unknown profiles)
```

---

## 6. DETECTION SCORE FLOW THROUGH SYSTEM

### Score Journey: Inference → Storage → Analytics

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INFERENCE LAYER (ArcFace embedding match)               │
├─────────────────────────────────────────────────────────────┤
│ Input: Face embedding (512-d vector)                        │
│ Process: L2 cosine distance against gallery                 │
│ Output: match.similarity (0.0 - 1.0)                        │
│         match.confidence (from SCRFD detector, 0-1)         │
│         candidate_scores = [                                │
│           {name, person_id, similarity: 0.75},              │
│           {name, person_id, similarity: 0.68},              │
│           ...                                               │
│         ]                                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TEMPORAL SMOOTHING (Multi-frame stabilization)          │
├─────────────────────────────────────────────────────────────┤
│ Input: match.confidence (per-frame)                         │
│ Process: Running average across track.last_n_frames         │
│ Output: match.smoothed_confidence                           │
│         decision_state = accept|review|reject               │
│ Logic: Smoothing eliminates jitter from single frames       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. DETECTION PERSISTENCE (DetectionEvent)                  │
├─────────────────────────────────────────────────────────────┤
│ Store fields:                                               │
│   - confidence: float32 (from inference, unchanged)         │
│   - similarity: float32 (embedding match)                   │
│   - smoothed_confidence: float32 (temporal-smoothed)        │
│   - decision_state: string (accept|review|reject)          │
│   - quality_flags: [blur, dark, pose, ...]                 │
│   - top_candidates: JSON with candidate_scores             │
│   - bbox: JSON tuple                                        │
│   - screenshot_path: JPEG file reference                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. DUAL TABLE STORAGE                                       │
├─────────────────────────────────────────────────────────────┤
│ detections.lance:                                           │
│   - Contains full detection with embedding vector           │
│   - Indexed for vector similarity search                    │
│   - Row: {detection_id, confidence, smoothed_confidence,    │
│           decision_state, embedding[512], ...}             │
│                                                             │
│ detection_decisions.lance:                                 │
│   - Audit trail for decision logic                          │
│   - Row: {detection_id, decision_state,                     │
│           smoothed_confidence, top_candidates, ...}        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ANALYTICS LAYER (DuckDB SQL queries)                     │
├─────────────────────────────────────────────────────────────┤
│ register_detections() → Arrow table                         │
│ Fields normalized:                                          │
│   SELECT confidence, smoothed_confidence, decision_state    │
│   FROM detections_view                                      │
│   WHERE entity_id = ? AND timestamp > ?                     │
│                                                             │
│ Aggregations:                                               │
│   - AVG(smoothed_confidence) per person                     │
│   - COUNT(*) by decision_state                              │
│   - Distribution by threshold (60%, 70%, 80%)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. FRONTEND DISPLAY                                         │
├─────────────────────────────────────────────────────────────┤
│ History page:                                               │
│   - confidence column (raw inference)                       │
│   - smoothed column (temporal-adjusted)                     │
│   - decision_state badge (color-coded)                      │
│                                                             │
│ Entities detail:                                            │
│   - Confidence bar (max smoothed_confidence)                │
│   - Timeline showing recent detections                      │
│   - Detection count by decision state                       │
│                                                             │
│ CSV export:                                                 │
│   - "confidence", "smoothed_confidence"                     │
│   - "decision_state", "decision_reason"                     │
│   - "top_candidates" (JSON)                                │
└─────────────────────────────────────────────────────────────┘
```

### Key Score Fields Across Tables

| Layer | Field | Type | Purpose | Range |
|-------|-------|------|---------|-------|
| **Inference** | confidence | float32 | Raw detector score | 0.0-1.0 |
| | similarity | float32 | Embedding match | 0.0-1.0 |
| **Temporal** | smoothed_confidence | float32 | Multi-frame average | 0.0-1.0 |
| **Decision** | decision_state | string | Accept/Review/Reject | enum |
| **Quality** | quality_score | float32 | Image assessment | 0.0-1.0 |
| | sharpness | float32 | Laplacian variance | 0.0-100 |
| | brightness | float32 | Grayscale mean | 0-255 |
| | face_ratio | float32 | Face % of frame | 0.0-1.0 |

---

## 7. KEY FILE LOCATIONS SUMMARY

### Core Data Models
| File | Purpose |
|------|---------|
| [src/trace_aml/core/models.py](src/trace_aml/core/models.py) | All domain models (PersonRecord, DetectionEvent, EntityRecord, etc.) |
| [src/trace_aml/core/errors.py](src/trace_aml/core/errors.py) | Custom exception types |
| [src/trace_aml/core/ids.py](src/trace_aml/core/ids.py) | ID generation (person_id, detection_id, entity_id) |

### Storage Layer
| File | Purpose |
|------|---------|
| [src/trace_aml/store/vector_store.py](src/trace_aml/store/vector_store.py) | LanceDB persistence (12 tables, embedding indexing) |
| [src/trace_aml/store/embedding_cache.py](src/trace_aml/store/embedding_cache.py) | In-memory gallery cache for fast search |
| [src/trace_aml/store/analytics.py](src/trace_aml/store/analytics.py) | DuckDB SQL analytics over detections |

### Recognition Pipeline
| File | Purpose |
|------|---------|
| [src/trace_aml/pipeline/session.py](src/trace_aml/pipeline/session.py) | Main orchestration loop, detection persistence |
| [src/trace_aml/pipeline/inference.py](src/trace_aml/pipeline/inference.py) | Face detection + embedding extraction thread |
| [src/trace_aml/pipeline/temporal.py](src/trace_aml/pipeline/temporal.py) | Multi-frame confidence smoothing |
| [src/trace_aml/pipeline/collect.py](src/trace_aml/pipeline/collect.py) | Image acquisition (webcam, files) |

### Entity & Rules Management
| File | Purpose |
|------|---------|
| [src/trace_aml/core/entity_resolver.py](src/trace_aml/core/entity_resolver.py) | Person ↔ Unknown entity linking |
| [src/trace_aml/core/rules_engine.py](src/trace_aml/core/rules_engine.py) | Alert generation (reappearance, instability) |
| [src/trace_aml/core/incident_manager.py](src/trace_aml/core/incident_manager.py) | Incident grouping + deduplication |

### Service Layer
| File | Purpose |
|------|---------|
| [src/trace_aml/service/app.py](src/trace_aml/service/app.py) | FastAPI server + all endpoints |
| [src/trace_aml/service/person_api.py](src/trace_aml/service/person_api.py) | Person CRUD + training endpoints |
| [src/trace_aml/service/analytics_api.py](src/trace_aml/service/analytics_api.py) | History query endpoints |

### Frontend
| File | Purpose |
|------|---------|
| [src/frontend/entities/index.html](src/frontend/entities/index.html) | Entity registry UI (grid + detail view) |
| [src/frontend/entities/entities.js](src/frontend/entities/entities.js) | Entity controller (load, filter, display) |
| [src/frontend/shared/trace_client.js](src/frontend/shared/trace_client.js) | API client library |
| [src/frontend/enrollment/index.html](src/frontend/enrollment/index.html) | Person enrollment form |

---

## 8. DETECTION PERSISTENCE LOGIC

### When Detections Are Saved
[session.py](src/trace_aml/pipeline/session.py#L310-L320)

```python
# Condition 1: Must pass minimum unknown surface threshold
if decision == DecisionState.reject and not match.person_id:
    if match.smoothed_confidence < min_unknown_surface_threshold:
        return  # Drop noisy faces (default: 0.15)

# Condition 2: Check persistence flags
persist_detection = True
if decision == DecisionState.reject and not settings.recognition.persist_unknown:
    persist_detection = False  # Drop unknown rejects (default: False)
if decision == DecisionState.review and not self.settings.recognition.persist_review:
    persist_detection = False  # Drop reviews (default: True)

# Condition 3: Cooldown check (avoid duplicate-saving same person)
person_key = f"pid:{person_id}:{decision}:{track_id}"
if not self._should_log(person_key):  # Default cooldown: 1.0 seconds
    persist_detection = False
```

### Storage Sequence
1. **DetectionEvent** created with full data
2. **Screenshot** saved: `data/screenshots/{detection_id}.jpg`
3. **store.add_detection()** → detections.lance
4. **store.add_detection_decision()** → detection_decisions.lance
5. **Event published** → subscribers (WebSocket, logging)

---

## 9. EMBEDDING GALLERY SEARCH

### In-Memory Cache Strategy
[embedding_cache.py](src/trace_aml/store/embedding_cache.py)

```python
class EmbeddingGalleryCache:
    # Loaded once at startup from LanceDB
    # Updated incrementally when person embeddings change
    # Fast cosine similarity search (O(n) linear scan, optimized)
    
    def search(embedding: list[float], topk: int = 5) -> list[dict]:
        # Returns top-k matches with scores
        # Normalizes input vector
        # Computes L2 distances
        # Returns [{person_id, name, similarity, embedding_id}, ...]
```

### Person Search Filter
[vector_store.py](src/trace_aml/store/vector_store.py#L800-L850)

```python
def search_embeddings_for_person_ids(
    embedding: list,
    person_ids: set,
    top_k: int = 5
) -> list[dict]:
    # Filter search to specific person IDs only
    # Used during incremental training to verify improvements
    
    # Returns only active persons (lifecycle_state == 'active')
    # Normalizes vectors (LANCE cosine metric has bugs)
```

---

## Key Takeaways

✅ **Detections** → Dual table design (raw detection + decision audit trail)  
✅ **Recognition** → Multi-stage pipeline (inference → temporal → entity resolution → rules)  
✅ **Persons** → Lifecycle-managed records (draft → ready → active → blocked)  
✅ **Profile Photos** → Schema in place, auto-capture TBD (ENHANCEMENT_PLAN.md)  
✅ **Entities Display** → Two-view pattern (overview grid + detail profile)  
✅ **Score Flow** → Inference → Temporal Smoothing → Storage → Analytics → Frontend

