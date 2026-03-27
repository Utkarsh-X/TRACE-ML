"""Entity resolution and event creation core for TRACE-AML."""

from __future__ import annotations

from dataclasses import dataclass

from trace_ml.core.config import Settings
from trace_ml.core.ids import new_event_id
from trace_ml.core.models import (
    EntityStatus,
    EntityType,
    EventRecord,
    RecognitionMatch,
)
from trace_ml.store.vector_store import VectorStore


@dataclass
class EntityResolution:
    entity_id: str
    entity_type: EntityType
    is_unknown: bool


class EntityResolver:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings = settings
        self.store = store

    def resolve(self, match: RecognitionMatch, embedding: list[float]) -> EntityResolution:
        if match.person_id:
            self.store.ensure_entity(
                entity_id=match.person_id,
                entity_type=EntityType.known,
                status=EntityStatus.active,
                source_person_id=match.person_id,
            )
            return EntityResolution(
                entity_id=match.person_id,
                entity_type=EntityType.known,
                is_unknown=False,
            )

        unknown_entity_id = self.store.resolve_or_create_unknown_entity(
            embedding=embedding,
            similarity_threshold=self.settings.recognition.unknown_reuse_threshold,
        )
        return EntityResolution(
            entity_id=unknown_entity_id,
            entity_type=EntityType.unknown,
            is_unknown=True,
        )

    @staticmethod
    def create_event_record(
        resolution: EntityResolution,
        match: RecognitionMatch,
        detection_id: str = "",
        source: str = "webcam:0",
    ) -> EventRecord:
        return EventRecord(
            event_id=new_event_id(),
            entity_id=resolution.entity_id,
            confidence=float(match.confidence),
            decision=match.decision_state,
            track_id=match.track_id,
            is_unknown=resolution.is_unknown,
            detection_id=detection_id,
            source=source,
        )
