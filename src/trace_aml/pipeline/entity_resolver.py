"""Entity resolution and event creation core for TRACE-AML."""

from __future__ import annotations

from dataclasses import dataclass

from trace_aml.core.config import Settings
from trace_aml.core.ids import new_event_id
from trace_aml.core.models import (
    EntityStatus,
    EntityType,
    EventLocation,
    EventRecord,
    RecognitionMatch,
)
from trace_aml.store.vector_store import VectorStore


@dataclass
class EntityResolution:
    entity_id: str
    entity_type: EntityType
    is_unknown: bool


class EntityResolver:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings = settings
        self.store = store
        # In-memory map: track_id → entity_id.
        # Guarantees at most ONE entity is ever created per track across its lifetime.
        self._track_entity_map: dict[str, str] = {}

    def resolve(self, match: RecognitionMatch, embedding: list[float]) -> EntityResolution:
        # ── Layer 3: Track ownership cache ────────────────────────────────────
        # If this track already owns an entity from a previous frame, reuse it.
        # Never allow a second entity creation for the same track_id.
        if match.track_id and match.track_id in self._track_entity_map:
            existing_id = self._track_entity_map[match.track_id]
            # Update entity last_seen without creating a new record.
            entity_type = EntityType.known if match.person_id else EntityType.unknown
            self.store.ensure_entity(
                entity_id=existing_id,
                entity_type=entity_type,
                status=EntityStatus.active,
                source_person_id=match.person_id,
            )
            return EntityResolution(
                entity_id=existing_id,
                entity_type=entity_type,
                is_unknown=not bool(match.person_id),
            )
        # ─────────────────────────────────────────────────────────────────────

        if match.person_id:
            self.store.ensure_entity(
                entity_id=match.person_id,
                entity_type=EntityType.known,
                status=EntityStatus.active,
                source_person_id=match.person_id,
            )
            resolution = EntityResolution(
                entity_id=match.person_id,
                entity_type=EntityType.known,
                is_unknown=False,
            )
        else:
            unknown_entity_id = self.store.resolve_or_create_unknown_entity(
                embedding=embedding,
                similarity_threshold=self.settings.recognition.unknown_reuse_threshold,
            )
            resolution = EntityResolution(
                entity_id=unknown_entity_id,
                entity_type=EntityType.unknown,
                is_unknown=True,
            )

        # Register this track → entity mapping for all future frames.
        if match.track_id:
            self._track_entity_map[match.track_id] = resolution.entity_id
        return resolution


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
            location=EventLocation(source=source),
        )
