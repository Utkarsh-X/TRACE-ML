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


class PersonLifecycleStatus(StrEnum):
    draft = "draft"
    ready = "ready"
    active = "active"
    blocked = "blocked"


class DecisionState(StrEnum):
    accept = "accept"
    review = "review"
    reject = "reject"


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
