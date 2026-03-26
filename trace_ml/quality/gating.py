"""Lifecycle gating for enrollment quality."""

from __future__ import annotations

from dataclasses import dataclass

from trace_ml.core.config import Settings
from trace_ml.core.models import PersonLifecycleStatus


@dataclass
class LifecycleDecision:
    state: PersonLifecycleStatus
    reason: str
    enrollment_score: float


def decide_person_lifecycle(
    settings: Settings,
    total_images: int,
    valid_images: int,
    embeddings_count: int,
    avg_quality: float,
) -> LifecycleDecision:
    quality_cfg = settings.quality
    if total_images == 0:
        return LifecycleDecision(
            state=PersonLifecycleStatus.draft,
            reason="no_images_uploaded",
            enrollment_score=0.0,
        )

    if embeddings_count == 0 or valid_images == 0:
        return LifecycleDecision(
            state=PersonLifecycleStatus.blocked,
            reason="no_valid_embeddings",
            enrollment_score=avg_quality,
        )

    if embeddings_count < quality_cfg.min_embeddings_ready:
        return LifecycleDecision(
            state=PersonLifecycleStatus.draft,
            reason="insufficient_embeddings_for_ready",
            enrollment_score=avg_quality,
        )

    if avg_quality < quality_cfg.min_quality_score:
        return LifecycleDecision(
            state=PersonLifecycleStatus.blocked,
            reason="average_quality_below_threshold",
            enrollment_score=avg_quality,
        )

    if (
        valid_images < quality_cfg.min_valid_images
        or embeddings_count < quality_cfg.min_embeddings_active
    ):
        return LifecycleDecision(
            state=PersonLifecycleStatus.ready,
            reason="meets_minimum_but_not_active_gate",
            enrollment_score=avg_quality,
        )

    return LifecycleDecision(
        state=PersonLifecycleStatus.active,
        reason="quality_gates_passed",
        enrollment_score=avg_quality,
    )
