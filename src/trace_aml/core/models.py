"""Typed domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PersonCategory(StrEnum):
    criminal = "criminal"
    missing = "missing"
    employee = "employee"
    vip = "vip"


class PersonLifecycleStatus(StrEnum):
    draft = "draft"
    ready = "ready"
    active = "active"
    blocked = "blocked"


class DecisionState(StrEnum):
    accept = "accept"
    review = "review"
    reject = "reject"


class EntityType(StrEnum):
    known = "known"
    unknown = "unknown"


class EntityStatus(StrEnum):
    active = "active"
    inactive = "inactive"


class AlertSeverity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class AlertType(StrEnum):
    reappearance = "REAPPEARANCE"
    unknown_recurrence = "UNKNOWN_RECURRENCE"
    instability = "INSTABILITY"


class IncidentStatus(StrEnum):
    open = "open"
    closed = "closed"


class ActionType(StrEnum):
    log        = "log"
    email      = "email"
    whatsapp   = "whatsapp"    # Primary messaging channel (Evolution API)
    pdf_report = "pdf_report"  # PDF/HTML incident report generation
    alarm      = "alarm"       # Deprecated — kept as backward-compat alias for whatsapp


class ActionTrigger(StrEnum):
    on_create = "on_create"
    on_update = "on_update"


class ActionStatus(StrEnum):
    success = "success"
    failed = "failed"


class TimelineItemKind(StrEnum):
    event = "event"
    alert = "alert"
    incident = "incident"
    action = "action"


class PersonRecord(BaseModel):
    person_id: str
    name: str
    category: PersonCategory
    severity: str = ""
    dob: str = ""
    gender: str = ""
    last_seen_city: str = ""
    last_seen_country: str = ""
    notes: str = ""
    profile_photo_path: str = ""  # Path to best recognition image
    profile_photo_confidence: float = 0.0  # Confidence of best profile photo
    lifecycle_state: PersonLifecycleStatus = PersonLifecycleStatus.draft
    lifecycle_reason: str = "new enrollment"
    enrollment_score: float = 0.0
    valid_embeddings: int = 0
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class EmbeddingRecord(BaseModel):
    embedding_id: str
    person_id: str
    source_path: str
    embedding: list[float]
    quality_score: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("embedding")
    @classmethod
    def validate_dim(cls, value: list[float]) -> list[float]:
        if len(value) == 0:
            raise ValueError("Embedding must not be empty")
        return value


class FaceCandidate(BaseModel):
    bbox: tuple[int, int, int, int]
    embedding: list[float]
    detector_score: float = 0.0
    # Optional: 5 facial keypoints [[x,y], ...] from InsightFace detector.
    # Used by the composite quality gate for pose-yaw estimation.
    kps: list[list[float]] | None = None
    # Geometric yaw estimate computed from kps (degrees, 0=frontal, 90=profile).
    pose_yaw: float = 0.0


class QualityAssessment(BaseModel):
    person_id: str
    source_path: str
    passed: bool
    quality_score: float
    sharpness: float
    face_ratio: float
    brightness: float
    pose_score: float
    reasons: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class RecognitionMatch(BaseModel):
    person_id: str | None = None
    name: str = "Unknown"
    category: str = "unknown"
    similarity: float = 0.0
    confidence: float = 0.0
    bbox: tuple[int, int, int, int]
    is_match: bool = False
    decision_state: DecisionState = DecisionState.reject
    decision_reason: str = "no decision"
    track_id: str = ""
    smoothed_confidence: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionEvent(BaseModel):
    detection_id: str
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    source: str = "webcam:0"
    person_id: str | None = None
    name: str = "Unknown"
    category: str = "unknown"
    confidence: float = 0.0
    similarity: float = 0.0
    smoothed_confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    track_id: str = ""
    decision_state: DecisionState = DecisionState.reject
    decision_reason: str = "no decision"
    quality_flags: list[str] = Field(default_factory=list)
    liveness_provider: str = "none"
    liveness_score: float = 0.0
    screenshot_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityRecord(BaseModel):
    entity_id: str
    type: EntityType
    status: EntityStatus = EntityStatus.active
    source_person_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    last_seen_at: str = Field(default_factory=utc_now_iso)


class EventLocation(BaseModel):
    lat: float | None = None
    lon: float | None = None
    source: str = ""


class EventRecord(BaseModel):
    event_id: str
    entity_id: str
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    confidence: float = 0.0
    decision: DecisionState = DecisionState.reject
    track_id: str = ""
    is_unknown: bool = True
    detection_id: str = ""
    source: str = "webcam:0"
    location: EventLocation = Field(default_factory=EventLocation)


class AlertRecord(BaseModel):
    alert_id: str
    entity_id: str
    type: AlertType
    severity: AlertSeverity
    reason: str
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    first_seen_at: str = Field(default_factory=utc_now_iso)
    last_seen_at: str = Field(default_factory=utc_now_iso)
    event_count: int = 1
    acknowledged: bool = False
    acknowledged_at: str = ""


class IncidentRecord(BaseModel):
    incident_id: str
    entity_id: str
    status: IncidentStatus = IncidentStatus.open
    start_time: str = Field(default_factory=utc_now_iso)
    last_seen_time: str = Field(default_factory=utc_now_iso)
    alert_ids: list[str] = Field(default_factory=list)
    alert_count: int = 0
    severity: AlertSeverity = AlertSeverity.low
    summary: str = ""
    last_action_at: str = ""


class ActionRecord(BaseModel):
    action_id: str
    incident_id: str
    action_type: ActionType
    trigger: ActionTrigger
    status: ActionStatus
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp_utc: str = Field(default_factory=utc_now_iso)


class HistoryQuery(BaseModel):
    start_ts: str | None = None
    end_ts: str | None = None
    person_id: str | None = None
    category: str | None = None
    decision_state: str | None = None
    min_confidence: float = 0.0
    limit: int = 100


class SummaryReport(BaseModel):
    total_detections: int
    unique_persons: int
    avg_confidence: float
    decision_distribution: dict[str, int] = Field(default_factory=dict)
    blocked_persons: int = 0
    low_quality_persons: int = 0
    top_persons: list[dict[str, Any]]


class EntitySummary(BaseModel):
    entity_id: str
    type: EntityType
    status: EntityStatus = EntityStatus.active
    name: str = "Unknown"
    category: str = "unknown"
    person_id: str = ""
    created_at: str = ""
    last_seen_at: str = ""
    open_incident_count: int = 0
    recent_alert_count: int = 0


class IncidentSummary(BaseModel):
    incident_id: str
    entity_id: str
    status: IncidentStatus = IncidentStatus.open
    severity: AlertSeverity = AlertSeverity.low
    summary: str = ""
    start_time: str = ""
    last_seen_time: str = ""
    alert_count: int = 0
    last_action_at: str = ""


class TimelineItem(BaseModel):
    item_id: str
    kind: TimelineItemKind
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    entity_id: str = ""
    incident_id: str = ""
    severity: str = ""
    title: str
    summary: str = ""
    source: str = ""
    screenshot_path: str = ""
    location: EventLocation = Field(default_factory=EventLocation)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityProfile(BaseModel):
    entity: EntitySummary
    linked_person: dict[str, Any] | None = None
    incidents: list[IncidentSummary] = Field(default_factory=list)
    recent_alerts: list[AlertRecord] = Field(default_factory=list)
    recent_detections: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)
    screenshot_paths: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class IncidentDetail(BaseModel):
    incident: IncidentSummary
    entity: EntitySummary
    alerts: list[AlertRecord] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    recent_detections: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)


class SystemHealthSnapshot(BaseModel):
    status: str = "ok"
    active_entity_count: int = 0
    open_incident_count: int = 0
    recent_alert_count: int = 0
    total_detection_count: int = 0
    latest_event_at: str = ""
    latest_alert_at: str = ""
    publisher_subscribers: int = 0
    last_published_at: str = ""
    runtime: dict[str, Any] = Field(default_factory=dict)


class LiveOpsSnapshot(BaseModel):
    active_entities: list[EntitySummary] = Field(default_factory=list)
    active_incidents: list[IncidentSummary] = Field(default_factory=list)
    recent_alerts: list[AlertRecord] = Field(default_factory=list)
    system_health: SystemHealthSnapshot = Field(default_factory=SystemHealthSnapshot)
