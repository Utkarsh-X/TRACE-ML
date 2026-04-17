# TRACE-ML Database Schema and API Analysis

**Last Updated:** 2026-04-17  
**Focus:** Entity/Person fields in database vs. API response vs. UI display capabilities

---

## Executive Summary

TRACE-AML uses **LanceDB** (Arrow-backed vector store) for all persistence, organized into 13 tables. The database contains significantly more metadata than is currently exposed via REST APIs or displayed in the UI. This document identifies stored fields that could enhance the database explorer, entity profiles, and incident tracking without requiring schema changes.

---

## 1. Core Storage Layer: LanceDB Schema

### Connection & Configuration
- **Backend:** LanceDB (managed via `VectorStore` in `src/trace_aml/store/vector_store.py`)
- **Persistence:** Arrow Lance files in `data/vectors/` directory
- **Migration:** Automatic schema migrations with default values for new columns
- **Query API:** Direct SQL queries via DuckDB; Search/filter via LanceDB vector API

---

## 2. Complete Database Table Definitions

### 2.1 PERSONS TABLE
**Primary identity store for known enrolled persons.**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
person_id              | string    | PRIMARY KEY (PRC### or custom ID)
name                   | string    | Full name / alias
category               | string    | Enum: criminal, missing, employee, vip
severity               | string    | Operator-assigned risk level
dob                    | string    | Date of birth (ISO format)
gender                 | string    | M, F, Other, Unknown
last_seen_city         | string    | Last known location city
last_seen_country      | string    | Last known location country
notes                  | string    | Free-form operator notes
profile_photo_path     | string    | Path to best face crop (256×256 JPEG)
profile_photo_confidence | float32 | Confidence of best enrollment image
created_at             | string    | UTC ISO timestamp, never changes
updated_at             | string    | UTC ISO timestamp, updates on any edit
```

**Optional stored elsewhere:** lifecycle_state, enrollment_score, valid_embeddings (in PERSON_STATES table)

---

### 2.2 PERSON_STATES TABLE
**Enrollment progress and quality metrics for each person.**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
person_id              | string    | FK → persons.person_id
lifecycle_state        | string    | draft, ready, active, blocked
lifecycle_reason       | string    | Why in current state ("new enrollment", "ready_for_deployment", etc.)
enrollment_score       | float32   | Aggregate quality metric (0-1)
valid_embeddings       | int32     | Count of usable face embeddings
valid_images           | int32     | Count of quality-approved images
total_images           | int32     | Total images on disk (may include rejected)
updated_at             | string    | UTC ISO timestamp
```

**Derived at runtime:** Not visible in API but computed when listing persons

---

### 2.3 EMBEDDINGS TABLE
**Face embeddings extracted from training images.**

```
Column Name              | Type              | Purpose
──────────────────────────────────────────────────────────
embedding_id           | string            | Unique ID per embedding
person_id              | string            | FK → persons.person_id
source_path            | string            | Path to source training image
created_at             | string            | Timestamp of extraction
quality_score          | float32           | Image quality (0-1)
quality_flags          | string            | JSON list: ["blurry", "poor_pose", "low_light"]
embedding              | list[512 float32] | Face vector (512-dim, ArcFace)
```

**No API access:** Embeddings are for backend vector search only; never serialized to API responses

---

### 2.4 IMAGE_QUALITY TABLE
**Quality assessment results for every training image.**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
quality_id             | string    | Unique ID per assessment
person_id              | string    | FK → persons.person_id
source_path            | string    | Path to source image file
passed                 | bool      | True if meets quality threshold
quality_score          | float32   | Overall score (0-1)
sharpness              | float32   | Blur metric (0-1)
face_ratio             | float32   | Face size in frame (0-1)
brightness             | float32   | Exposure level (0-1)
pose_score             | float32   | Head rotation (0-1; 1=frontal)
reasons                | string    | JSON list of failure reasons ["blurry", "dark", ...]
created_at             | string    | Assessment timestamp
```

**Not exposed in API:** Only used internally for enrollment heuristics

---

### 2.5 DETECTIONS TABLE
**Every face detection event during live recognition.**

```
Column Name              | Type              | Purpose
──────────────────────────────────────────────────────────
detection_id           | string            | PRIMARY KEY (unique per event)
timestamp_utc          | string            | When detected
source                 | string            | e.g., "webcam:0", "rtsp://…"
person_id              | string            | Matched person_id (NULL if unknown)
name                   | string            | Matched name (copied at detection time)
category               | string            | Matched category (criminal, missing, etc.)
confidence             | float32           | Raw recognition score (0-1)
similarity             | float32           | Cosine similarity to best person embedding
smoothed_confidence    | float32           | Temporal smoothing over track
bbox                   | string            | JSON [x1, y1, x2, y2]
track_id               | string            | FK → Multi-frame face track
decision_state         | string            | accept, review, reject (confidence binning)
decision_reason        | string            | Why decision made
quality_flags          | string            | JSON list of concerns at detection time
liveness_provider      | string            | "none", "minifasnet", or other
liveness_score         | float32           | Liveness confidence (0-1) if checked
screenshot_path        | string            | Path to cropped face JPEG (256×256)
metadata               | string            | JSON object for extensibility
embedding              | list[512 float32] | Face vector at detection time
```

**API exposure:** Detections are returned in entity profiles/timeline; embedding never exposed

---

### 2.6 DETECTION_DECISIONS TABLE
**Normalized decision-making context (denormalized from DETECTIONS for analytics).**

```
Column Name              | Type              | Purpose
──────────────────────────────────────────────────────────
detection_id           | string            | FK → detections.detection_id
track_id               | string            | Multi-frame tracking ID
decision_state         | string            | accept, review, reject
decision_reason        | string            | Rule or threshold reason
smoothed_confidence    | float32           | Temporal filtering result
quality_flags          | string            | JSON list
top_candidates         | string            | JSON with alternate matches
liveness_provider      | string            | Liveness checker used
liveness_score         | float32           | Liveness score
created_at             | string            | Timestamp
```

**Purpose:** Analytical queries on decision patterns without scanning full DETECTIONS table

---

### 2.7 ENTITIES TABLE
**Abstract identity clustering (known persons and unknown clusters).**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
entity_id              | string    | PRIMARY KEY (PRC### for known, UNK### for unknown)
type                   | string    | "known" (← person_id) or "unknown"
status                 | string    | active, inactive
source_person_id       | string    | If known: → persons.person_id
created_at             | string    | Entity creation timestamp
last_seen_at           | string    | Most recent detection
```

**Critical:** Entities are the UI-facing identity abstraction; persons are enrollment records

---

### 2.8 EVENTS TABLE
**Operational event log (detections grouped by entity and decision).**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
event_id               | string    | PRIMARY KEY (unique per event record)
entity_id              | string    | FK → entities.entity_id
timestamp_utc          | string    | When event occurred
confidence             | float32   | Recognition confidence
decision               | string    | accept, review, reject
track_id               | string    | Multi-frame track ID
is_unknown             | bool      | True if entity_id is UNK###
detection_id           | string    | FK → detections.detection_id
source                 | string    | Source identifier (camera, etc.)
location               | string    | JSON {lat, lon, source}
```

**API exposure:** Serialized in entity timeline; location field often empty or "{}

---

### 2.9 UNKNOWN_PROFILES TABLE
**Centroid embeddings for unknown entity clusters.**

```
Column Name              | Type              | Purpose
──────────────────────────────────────────────────────────
entity_id              | string            | PK (UNK###)
embedding              | list[512 float32] | Average of cluster members
sample_count           | int32             | Number of detections in cluster
created_at             | string            | Cluster creation
last_seen_at           | string            | Most recent addition
```

**No API exposure:** Used only for unknown → unknown matching; embedding never serialized

---

### 2.10 ALERTS TABLE
**Rule-triggered notifications (e.g., person reappeared, instability detected).**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
alert_id               | string    | PRIMARY KEY
entity_id              | string    | FK → entities.entity_id
type                   | string    | REAPPEARANCE, UNKNOWN_RECURRENCE, INSTABILITY
severity               | string    | low, medium, high
reason                 | string    | Human-readable trigger description
timestamp_utc          | string    | When alert triggered
first_seen_at          | string    | First time threshold crossed
last_seen_at           | string    | Most recent occurrence
event_count            | int32     | Number of triggering events
```

**API exposure:** Full record returned in entity profile and incident detail

---

### 2.11 INCIDENTS TABLE
**Aggregated alert sequences (grouping related alerts).**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
incident_id            | string    | PRIMARY KEY
entity_id              | string    | FK → entities.entity_id
status                 | string    | open, closed
start_time             | string    | First alert timestamp
last_seen_time         | string    | Most recent alert timestamp
alert_ids              | string    | JSON array of alert_id strings
alert_count            | int32     | Count of alerts
severity               | string    | low, medium, high
summary                | string    | Human or auto-generated summary
last_action_at         | string    | When policy action last executed
```

**API exposure:** Full record in incidents list and detail endpoints

---

### 2.12 ACTIONS TABLE
**Policy engine executions (log, email, alarm, etc.).**

```
Column Name              | Type      | Purpose
─────────────────────────────────────────────────────────
action_id              | string    | PRIMARY KEY
incident_id            | string    | FK → incidents.incident_id
action_type            | string    | log, email, alarm
trigger                | string    | on_create, on_update
status                 | string    | success, failed
reason                 | string    | Why action triggered
context                | string    | JSON metadata (email recipient, alarm config, etc.)
timestamp_utc          | string    | Execution time
```

**API exposure:** Full record in incident detail and timeline

---

---

## 3. Current API Response Schemas

### 3.1 GET /api/v1/entities — Entity List
**Response: `list[EntitySummary]`**

```python
class EntitySummary(BaseModel):
    entity_id: str                      # PRC### or UNK###
    type: EntityType                    # "known" | "unknown"
    status: EntityStatus                # "active" | "inactive"
    name: str                           # From linked person or "Unknown"
    category: str                       # From linked person or "unknown"
    person_id: str                      # Empty if unknown entity
    created_at: str                     # Entity creation timestamp
    last_seen_at: str                   # Most recent detection
    open_incident_count: int            # Live count of open incidents
    recent_alert_count: int             # Live count of recent alerts
```

**Fields returned:** 10  
**Fields available in DB but NOT returned:**  
- Person's `severity`, `dob`, `gender`, `notes`  
- Person's enrollment stats: `lifecycle_state`, `lifecycle_reason`, `enrollment_score`, `valid_embeddings`, `valid_images`  
- Location info: `last_seen_city`, `last_seen_country`

---

### 3.2 GET /api/v1/entities/{entity_id} — Single Entity Detail
**Response: `EntitySummary` (same as above)**

---

### 3.3 GET /api/v1/entities/{entity_id}/profile — Full Entity Profile
**Response: `EntityProfile`**

```python
class EntityProfile(BaseModel):
    entity: EntitySummary                    # (see above)
    linked_person: dict[str, Any] | None    # Full PersonRecord if known
    incidents: list[IncidentSummary]        # Open/recent incidents
    recent_alerts: list[AlertRecord]        # Recent alerts (limit 20)
    recent_detections: list[dict]           # Detection events (limit 20)
    timeline: list[TimelineItem]            # Mixed events/alerts/incidents/actions
    screenshot_paths: list[str]             # Paths to captured face crops
    stats: dict[str, Any]                   # {timeline_items: N, incident_count: N, ...}
```

**PersonRecord fields returned (if known entity):**
```python
class PersonRecord(BaseModel):
    person_id: str
    name: str
    category: PersonCategory                # "criminal" | "missing" | "employee" | "vip"
    severity: str                           # Free-form (e.g., "HIGH", "MEDIUM", "")
    dob: str                                # ISO date or ""
    gender: str                             # "M" | "F" | "" | "Other" | "Unknown"
    last_seen_city: str
    last_seen_country: str
    notes: str
    profile_photo_path: str
    profile_photo_confidence: float         # 0.0-1.0
    lifecycle_state: PersonLifecycleStatus  # "draft" | "ready" | "active" | "blocked"
    lifecycle_reason: str
    enrollment_score: float
    valid_embeddings: int
    created_at: str
    updated_at: str
```

**NOT returned in profile:**  
- `valid_images`, `total_images` (available in store but not serialized)

---

### 3.4 GET /api/v1/entities/{entity_id}/timeline — Entity Timeline
**Response: `list[TimelineItem]`**

```python
class TimelineItem(BaseModel):
    item_id: str                        # Event/Alert/Incident/Action ID
    kind: TimelineItemKind              # "event" | "alert" | "incident" | "action"
    timestamp_utc: str
    entity_id: str
    incident_id: str                    # If applicable
    severity: str                       # If alert/incident
    title: str                          # "EVENT ACCEPT", "ALERT REAPPEARANCE", etc.
    summary: str                        # Human-readable description
    source: str                         # "webcam:0", "rules_engine", etc.
    screenshot_path: str                # Path for events
    location: EventLocation             # {lat, lon, source}
    metadata: dict[str, Any]            # Kind-specific extras
```

**Detection timeline items include:**
- screenshot_path
- bbox (from detection, rarely serialized in API response)
- metadata: {track_id, is_unknown, detection_id}

**NOT included in timeline:**
- confidence scores (except in summary text)
- decision_reason details
- quality_flags
- liveness_score
- Full detection coordinates or face crop path

---

### 3.5 GET /api/v1/entities/{entity_id}/incidents — Entity Incidents
**Response: `list[IncidentSummary]`**

```python
class IncidentSummary(BaseModel):
    incident_id: str
    entity_id: str
    status: IncidentStatus               # "open" | "closed"
    severity: AlertSeverity              # "low" | "medium" | "high"
    summary: str
    start_time: str
    last_seen_time: str
    alert_count: int
    last_action_at: str
```

---

### 3.6 GET /api/v1/health — System Health Snapshot
**Response: `SystemHealthSnapshot`**

```python
class SystemHealthSnapshot(BaseModel):
    status: str                         # "ok"
    active_entity_count: int            # Total entities
    open_incident_count: int            # Open incidents
    recent_alert_count: int             # Recent alerts (within 1 hour?)
    total_detection_count: int          # All-time detections
    latest_event_at: str                # Most recent event
    latest_alert_at: str                # Most recent alert
    publisher_subscribers: int          # SSE listener count
    last_published_at: str              # Last event stream update
    runtime: dict[str, Any]             # System metrics
```

---

---

## 4. Fields Stored in Database but NOT Currently Exposed by API

### 4.1 Person Metadata (Available via `/api/v1/entities/{id}/profile` → linked_person)

| Field | Type | DB Location | Currently in API? | Use Case |
|-------|------|-------------|-------------------|----------|
| `severity` | string | PERSONS | ✓ (in linked_person only) | Risk level indicator |
| `dob` | string | PERSONS | ✓ (in linked_person) | Age calculation, identity verification |
| `gender` | string | PERSONS | ✓ (in linked_person) | Demographic context |
| `last_seen_city` | string | PERSONS | ✗ | Geographic tracking |
| `last_seen_country` | string | PERSONS | ✗ | Geographic tracking |
| `notes` | string | PERSONS | ✓ (in linked_person) | Operator context |
| `profile_photo_confidence` | float | PERSONS | ✓ (in linked_person) | Quality indicator for best photo |
| `lifecycle_state` | string | PERSON_STATES | ✓ (in linked_person via lifecycle_state) | Enrollment status |
| `lifecycle_reason` | string | PERSON_STATES | ✓ (in linked_person) | Why in current state |
| `enrollment_score` | float | PERSON_STATES | ✓ (in linked_person) | Quality aggregate for person |
| `valid_embeddings` | int | PERSON_STATES | ✓ (in linked_person) | Biometric confidence |
| `valid_images` | int | PERSON_STATES | ✗ | Training data quality |
| `total_images` | int | PERSON_STATES | ✗ | Training data volume |

---

### 4.2 Detection Quality & Context (Available in DETECTIONS, Not in API)

| Field | DB Table | Currently Serialized? | Use Case |
|-------|----------|----------------------|----------|
| `bbox` | DETECTIONS | ✗ | Face bounding box for cropping |
| `similarity` | DETECTIONS | ✗ | Embedding distance to matched person |
| `quality_flags` | DETECTIONS | ✗ | Detected issues: [blurry, low_light, poor_pose] |
| `liveness_score` | DETECTIONS | ✗ | Liveness check confidence (0-1) |
| `liveness_provider` | DETECTIONS | ✗ | Which liveness checker was used |
| `decision_reason` | DETECTIONS | ✗ | Why confidence→decision_state conversion happened |
| `metadata` | DETECTIONS | ✗ | Custom JSON metadata |
| `embedding` | DETECTIONS | ✗ (never) | Face vector (512-dim, kept internal) |

---

### 4.3 Image Quality Assessment (Available in IMAGE_QUALITY, Not in API)

| Field | Type | Use Case |
|-------|------|----------|
| `sharpness` | float | Blur metric for focus quality |
| `face_ratio` | float | Face size in frame |
| `brightness` | float | Exposure level |
| `pose_score` | float | Head rotation (1=frontal, 0=profile) |
| `reasons` | string (JSON) | Detailed failure reasons ["blurry", "dark", "profile"] |

**None of these are exposed in API responses.** Enrollment quality data is only used internally to score a person and decide readiness.

---

### 4.4 Event Location Data (Available in EVENTS, Often Empty)

| Field | Type | Currently Serialized? |
|-------|------|----------------------|
| `location.lat` | float | ✓ (in TimelineItem) but rarely populated |
| `location.lon` | float | ✓ (in TimelineItem) but rarely populated |
| `location.source` | string | ✓ (in TimelineItem) |

**Storage note:** Location is stored as JSON string `"{\"lat\": null, \"lon\": null, \"source\": \"...\"}"` in EVENTS.location field.

---

---

## 5. API Endpoints Summary

### Entities & Profiles

| Endpoint | Method | Returns | Field Count | Notes |
|----------|--------|---------|-------------|-------|
| `/api/v1/entities` | GET | `list[EntitySummary]` | 10 | Query params: limit, type_filter, status |
| `/api/v1/entities/{id}` | GET | `EntitySummary` | 10 | Single entity detail |
| `/api/v1/entities/{id}/profile` | GET | `EntityProfile` | 40+ | Includes linked_person if known (adds 14+ fields) |
| `/api/v1/entities/{id}/timeline` | GET | `list[TimelineItem]` | 11 | Query params: start, end, limit |
| `/api/v1/entities/{id}/incidents` | GET | `list[IncidentSummary]` | 8 | Related incidents only |
| `/api/v1/entities/{id}/portrait` | GET | Binary JPEG | — | Highest-quality face crop (256×256) |

### Incidents

| Endpoint | Method | Returns | Notes |
|----------|--------|---------|-------|
| `/api/v1/incidents` | GET | `list[IncidentSummary]` | Query: limit, skip, status, entity_id |
| `/api/v1/incidents/{id}` | GET | `IncidentDetail` | Full incident with related events/alerts/actions |
| `/api/v1/incidents/{id}/severity` | PATCH | `IncidentDetail` | Update severity (low/medium/high) |
| `/api/v1/incidents/{id}/close` | POST | `IncidentDetail` | Close incident (change status→closed) |

### System

| Endpoint | Method | Returns | Notes |
|----------|--------|---------|-------|
| `/health` | GET | `SystemHealthSnapshot` | System metrics |
| `/api/v1/live/snapshot` | GET | `LiveOpsSnapshot` | Active entities, incidents, alerts |
| `/api/v1/live/overlay` | GET | JSON | Current detection boxes (from live recognition) |

---

---

## 6. Frontend Data Usage

### Database Explorer (`src/frontend/database/index.html`)

**Calls:** `TraceClient.entities({ limit })`  
**Renders per entity:**
- entity_id
- name
- category
- status
- type
- created_at (not displayed, but available)
- last_seen_at (not displayed, but available)
- open_incident_count (displayed as "incidents" count)
- recent_alert_count (displayed as "alerts" count)

**NOT displayed:**
- person_id (available in EntitySummary)
- severity (only available via linked_person in profile endpoint)
- gender, dob (only in linked_person)
- location (last_seen_city/country in PERSONS table)
- enrollment stats (lifecycle_state, valid_embeddings)

---

### Entity Profile (`src/frontend/entities/index.html`)

**Calls:** `TraceClient.entity(id)` + `TraceClient.entityProfile(id)`

**Displays from EntitySummary:**
- entity_id, type, status, name, category

**Displays from linked_person (if known):**
- Full PersonRecord: name, category, severity, dob, gender, location, notes, lifecycle_state, enrollment_score

**Displays from timeline/incidents:**
- Recent detections (with screenshots)
- Recent alerts
- Incident list
- Combined timeline (events + alerts + incidents + actions)

**NOT displayed but available:**
- valid_images, total_images (PERSON_STATES)
- Face bounding boxes (DETECTIONS.bbox)
- Liveness scores
- Quality flags per detection

---

### Live Operations (`src/frontend/live_ops/index.html`)

**Calls:** `TraceClient.liveSnapshot()`

**Displays:**
- Active entities (limit 3)
- Active incidents (limit 3)
- Recent alerts (limit 3)
- System health (total counts, latest timestamps)

---

---

## 7. Recommendations: Fields That Could Enhance UI Display

### 7.1 Database Explorer Enhancements

**Additional columns to consider:**

| Field | Source | Impact | Difficulty |
|-------|--------|--------|------------|
| Severity | PERSONS (via linked_person) | Quick risk assessment in table | ✓ Easy (fetch profile) |
| Detection count | Computed from DETECTIONS | Activity indicator | ✓ Easy (in recent_alerts via API) |
| Last detection timestamp | EVENTS.timestamp_utc | Recency sorting/grouping | ✓ Easy (already in EntitySummary) |
| Location (city/country) | PERSONS.last_seen_city/country | Geographic context | ⚠️ Moderate (API change needed) |
| Lifecycle state | PERSON_STATES.lifecycle_state | Enrollment readiness | ⚠️ Moderate (API change needed) |

**Changes required:**
1. **Easy:** Fetch entity profiles for each entity in list (increases latency; cache recommended)
2. **Moderate:** Add `severity`, `last_seen_city`, `last_seen_country`, `lifecycle_state` to EntitySummary in API
3. **Best practice:** Add optional fields to EntitySummary response at minimal cost to bandwidth

---

### 7.2 Entity Profile Enhancements

**Currently displayed:** Person metadata, incidents, alerts, timeline, screenshots

**Could be added (no DB changes needed):**

| Field | Source | Display Idea |
|-------|--------|--------------|
| Total detection count | COUNT(DETECTIONS.person_id) | Statistics panel |
| Detection trend (24h, 7d) | Time-windowed COUNT(DETECTIONS) | Activity sparkline |
| Quality flags frequency | COUNT per DETECTIONS.quality_flags | Quality concerns indicator |
| Liveness check stats | Stats on DETECTIONS.liveness_score | Biometric health indicator |
| Face bounding boxes | DETECTIONS.bbox | Visual detection regions on timeline |
| Training image stats | PERSON_STATES.valid_images/total_images | Enrollment coverage info |
| Geographic distribution | GROUP BY location from DETECTIONS | Heat map of sightings |
| Top candidate matches | DETECTION_DECISIONS.top_candidates | Alternative person suggestions |

---

### 7.3 Timeline Display Enhancements

**Currently shown:** Title, summary, timestamp, screenshot (for detections)

**Could be added:**

| Field | Source |
|-------|--------|
| Confidence score | DETECTIONS.confidence (in detection events) |
| Decision state | DETECTIONS.decision_state (accept/review/reject) |
| Quality flags | DETECTIONS.quality_flags (e.g., "blurry", "low_light") |
| Liveness status | DETECTIONS.liveness_score |
| Face bounding box | DETECTIONS.bbox (render box on screenshot) |
| Similarity score | DETECTIONS.similarity (vs. best embedding) |
| Location | EVENTS.location (lat/lon if available) |

---

### 7.4 Incident Detail Enhancements

**Currently shown:** Incident summary, alerts, actions, recent detections, timeline

**Could be added:**

| Field | Source |
|-------|--------|
| Incident duration | Compute: last_seen_time - start_time |
| Confidence trend | Aggregate DETECTIONS.confidence over incident period |
| Geographic extent | DETECTIONS grouped by source/location |
| Quality concerns | Aggregated DETECTIONS.quality_flags |
| Liveness evidence | Aggregate DETECTIONS.liveness_score |

---

---

## 8. Implementation Notes for UI Teams

### 8.1 Low-Hanging Fruit (Minimal Backend Changes)

1. **Display severity in entity list**
   - Endpoint change: Optional — can fetch via existing profile endpoint with caching
   - UI: Add `severity` column to database explorer table
   - Data source: `linked_person.severity` if known entity

2. **Show last_seen timestamp in list**
   - No change needed — `last_seen_at` already in EntitySummary
   - UI: Format and display in table

3. **Add incident & alert counts to table rows**
   - No change needed — `open_incident_count` and `recent_alert_count` already in EntitySummary
   - UI: Display as badge/count in table

---

### 8.2 Medium-Effort Enhancements (API Extension)

1. **Extend EntitySummary with location fields**
   ```python
   class EntitySummary(BaseModel):
       # ... existing fields ...
       last_seen_city: str = ""          # NEW
       last_seen_country: str = ""       # NEW
       severity: str = ""                # NEW (from linked_person)
       lifecycle_state: str = ""         # NEW (from linked_person)
   ```
   - Backend impact: Read from PERSONS via _entity_person() join
   - API endpoints: `/api/v1/entities`, `/api/v1/entities/{id}`

2. **Add detection stats to entity profile**
   - Compute: `total_detections`, `avg_confidence`, `detection_trend`
   - Backend: SQL query on DETECTIONS table grouped by entity

3. **Include face bounding boxes in timeline**
   - Field: `DETECTIONS.bbox` (stored as JSON string)
   - Serialize: Parse bbox in TimelineItem.metadata for event items
   - UI: Render box overlay on screenshot image

---

### 8.3 Data Availability by Entity Type

**Known Entities (type=known):**
- All person metadata available
- All enrollment statistics available
- Quality flags per training image available

**Unknown Entities (type=unknown):**
- Name/category/severity: NOT available (marked "Unknown")
- Detection data: Available (quality flags, liveness, etc.)
- Enrollment stats: NOT applicable (no training phase)
- Location: Available from detections

---

---

## 9. Database Query Examples (DuckDB)

### 9.1 High-Activity Entities (Last 24 Hours)

```sql
SELECT 
    e.entity_id,
    e.type,
    p.name,
    COUNT(d.detection_id) as detection_count,
    AVG(d.confidence) as avg_confidence,
    MAX(d.timestamp_utc) as last_detection
FROM entities e
LEFT JOIN persons p ON e.source_person_id = p.person_id
LEFT JOIN detections d ON 
    (e.type = 'known' AND d.person_id = p.person_id) OR
    (e.type = 'unknown' AND e.entity_id IN (
        SELECT entity_id FROM events 
        WHERE is_unknown = TRUE AND detection_id = d.detection_id
    ))
WHERE d.timestamp_utc > datetime('now', '-1 day')
GROUP BY e.entity_id
ORDER BY detection_count DESC
LIMIT 20
```

### 9.2 Quality Issues in Training Dataset

```sql
SELECT 
    p.person_id,
    p.name,
    COUNT(iq.quality_id) as total_images,
    SUM(CASE WHEN iq.passed THEN 1 ELSE 0 END) as passed_count,
    ROUND(AVG(iq.quality_score) * 100, 1) as avg_quality_pct,
    GROUP_CONCAT(DISTINCT iq.reasons) as failure_reasons
FROM persons p
LEFT JOIN image_quality iq ON p.person_id = iq.person_id
GROUP BY p.person_id
HAVING passed_count < total_images
ORDER BY avg_quality_pct ASC
```

### 9.3 Geographic Distribution of Detections

```sql
SELECT 
    d.source,
    COUNT(*) as detection_count,
    COUNT(DISTINCT e.entity_id) as unique_entities,
    MAX(d.timestamp_utc) as latest_detection
FROM detections d
LEFT JOIN events e ON d.detection_id = e.detection_id
GROUP BY d.source
ORDER BY detection_count DESC
```

---

---

## 10. Summary Table: Field Availability Matrix

| Field | Stored? | API Endpoint | In UI? | Notes |
|-------|---------|--------------|--------|-------|
| **Entity Identity** |
| entity_id | ✓ ENTITIES | ✓ All | ✓ | PRC/UNK ID |
| type | ✓ ENTITIES | ✓ All | ✓ | known/unknown |
| status | ✓ ENTITIES | ✓ All | ✓ | active/inactive |
| **Person Identity** |
| person_id | ✓ PERSONS | ✓ Profile | ✓ | Only for known |
| name | ✓ PERSONS | ✓ All | ✓ | |
| category | ✓ PERSONS | ✓ All | ✓ | criminal, missing, etc. |
| **Person Context** |
| severity | ✓ PERSONS | ⚠️ Profile only | ⚠️ Not in list | |
| dob | ✓ PERSONS | ✓ Profile | ✓ Profile | |
| gender | ✓ PERSONS | ✓ Profile | ✓ Profile | |
| notes | ✓ PERSONS | ✓ Profile | ✓ Profile | |
| **Person Location** |
| last_seen_city | ✓ PERSONS | ✗ | ✗ | Available in DB |
| last_seen_country | ✓ PERSONS | ✗ | ✗ | Available in DB |
| **Person Enrollment** |
| lifecycle_state | ✓ PERSON_STATES | ✓ Profile | ✓ Profile | draft/ready/active/blocked |
| lifecycle_reason | ✓ PERSON_STATES | ✓ Profile | ✓ Profile | |
| enrollment_score | ✓ PERSON_STATES | ✓ Profile | ✓ Profile | Quality aggregate |
| valid_embeddings | ✓ PERSON_STATES | ✓ Profile | ✓ Profile | Biometric count |
| valid_images | ✓ PERSON_STATES | ✗ | ✗ | Available in DB |
| total_images | ✓ PERSON_STATES | ✗ | ✗ | Available in DB |
| **Detection Quality** |
| quality_flags | ✓ DETECTIONS | ✗ | ✗ | Available in DB |
| liveness_score | ✓ DETECTIONS | ✗ | ✗ | Available in DB |
| liveness_provider | ✓ DETECTIONS | ✗ | ✗ | Available in DB |
| **Detection Context** |
| bbox | ✓ DETECTIONS | ✗ | ✗ | Available in DB |
| similarity | ✓ DETECTIONS | ✗ | ✗ | Available in DB |
| decision_reason | ✓ DETECTIONS | ✗ | ✗ | Available in DB |
| **Detection Assets** |
| screenshot_path | ✓ DETECTIONS | ✓ Profile | ✓ | Face crop JPEG |
| **Temporal** |
| created_at | ✓ ENTITIES | ✓ All | ✗ | Entity creation |
| last_seen_at | ✓ ENTITIES | ✓ All | ⚠️ Sortable only | Entity last detection |
| timestamp_utc | ✓ EVENT/DETECTION | ✓ Timeline | ✓ | Individual event time |
| **Operational** |
| incident_count | Computed | ✓ All | ✓ | Open incidents |
| alert_count | Computed | ✓ All | ✓ | Recent alerts |

---

**End of Document**
