"""LanceDB-backed storage for persons, quality metadata, embeddings and detections."""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from loguru import logger

from trace_aml.core.config import Settings
from trace_aml.core.errors import StorageError
from trace_aml.core.ids import next_unknown_entity_id
from trace_aml.core.models import (
    ActionRecord,
    AlertRecord,
    DetectionEvent,
    EmbeddingRecord,
    EntityRecord,
    EntityStatus,
    EntityType,
    EventRecord,
    IncidentRecord,
    IncidentStatus,
    PersonLifecycleStatus,
    PersonRecord,
    QualityAssessment,
)

PERSONS_TABLE = "persons"
PERSON_STATES_TABLE = "person_states"
EMBEDDINGS_TABLE = "person_embeddings"
IMAGE_QUALITY_TABLE = "image_quality"
DETECTIONS_TABLE = "detections"
DETECTION_DECISIONS_TABLE = "detection_decisions"
ENTITIES_TABLE = "entities"
EVENTS_TABLE = "events"
UNKNOWN_PROFILES_TABLE = "unknown_profiles"
ALERTS_TABLE = "alerts"
INCIDENTS_TABLE = "incidents"
ACTIONS_TABLE = "actions"
EMBEDDING_DIM = 512


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStore:
    """Primary persistence layer using LanceDB."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vectors_dir = Path(settings.store.vectors_dir)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.db = lancedb.connect(str(self.vectors_dir))
        except Exception as exc:
            raise StorageError(f"Failed to connect LanceDB: {exc}") from exc
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self.persons = self._open_or_create(
            PERSONS_TABLE,
            pa.schema(
                [
                    pa.field("person_id", pa.string()),
                    pa.field("name", pa.string()),
                    pa.field("category", pa.string()),
                    pa.field("severity", pa.string()),
                    pa.field("dob", pa.string()),
                    pa.field("gender", pa.string()),
                    pa.field("last_seen_city", pa.string()),
                    pa.field("last_seen_country", pa.string()),
                    pa.field("notes", pa.string()),
                    pa.field("profile_photo_path", pa.string()),
                    pa.field("profile_photo_confidence", pa.float32()),
                    pa.field("created_at", pa.string()),
                    pa.field("updated_at", pa.string()),
                ]
            ),
        )

        self.person_states = self._open_or_create(
            PERSON_STATES_TABLE,
            pa.schema(
                [
                    pa.field("person_id", pa.string()),
                    pa.field("lifecycle_state", pa.string()),
                    pa.field("lifecycle_reason", pa.string()),
                    pa.field("enrollment_score", pa.float32()),
                    pa.field("valid_embeddings", pa.int32()),
                    pa.field("valid_images", pa.int32()),
                    pa.field("total_images", pa.int32()),
                    pa.field("updated_at", pa.string()),
                ]
            ),
        )

        self.embeddings = self._open_or_create(
            EMBEDDINGS_TABLE,
            pa.schema(
                [
                    pa.field("embedding_id", pa.string()),
                    pa.field("person_id", pa.string()),
                    pa.field("source_path", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("quality_score", pa.float32()),
                    pa.field("quality_flags", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
                ]
            ),
        )

        self.image_quality = self._open_or_create(
            IMAGE_QUALITY_TABLE,
            pa.schema(
                [
                    pa.field("quality_id", pa.string()),
                    pa.field("person_id", pa.string()),
                    pa.field("source_path", pa.string()),
                    pa.field("passed", pa.bool_()),
                    pa.field("quality_score", pa.float32()),
                    pa.field("sharpness", pa.float32()),
                    pa.field("face_ratio", pa.float32()),
                    pa.field("brightness", pa.float32()),
                    pa.field("pose_score", pa.float32()),
                    pa.field("reasons", pa.string()),
                    pa.field("created_at", pa.string()),
                ]
            ),
        )

        self.detections = self._open_or_create(
            DETECTIONS_TABLE,
            pa.schema(
                [
                    pa.field("detection_id", pa.string()),
                    pa.field("timestamp_utc", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("person_id", pa.string()),
                    pa.field("name", pa.string()),
                    pa.field("category", pa.string()),
                    pa.field("confidence", pa.float32()),
                    pa.field("similarity", pa.float32()),
                    pa.field("smoothed_confidence", pa.float32()),
                    pa.field("bbox", pa.string()),
                    pa.field("track_id", pa.string()),
                    pa.field("decision_state", pa.string()),
                    pa.field("decision_reason", pa.string()),
                    pa.field("quality_flags", pa.string()),
                    pa.field("liveness_provider", pa.string()),
                    pa.field("liveness_score", pa.float32()),
                    pa.field("screenshot_path", pa.string()),
                    pa.field("metadata", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
                ]
            ),
        )

        self.detection_decisions = self._open_or_create(
            DETECTION_DECISIONS_TABLE,
            pa.schema(
                [
                    pa.field("detection_id", pa.string()),
                    pa.field("track_id", pa.string()),
                    pa.field("decision_state", pa.string()),
                    pa.field("decision_reason", pa.string()),
                    pa.field("smoothed_confidence", pa.float32()),
                    pa.field("quality_flags", pa.string()),
                    pa.field("top_candidates", pa.string()),
                    pa.field("liveness_provider", pa.string()),
                    pa.field("liveness_score", pa.float32()),
                    pa.field("created_at", pa.string()),
                ]
            ),
        )

        self.entities = self._open_or_create(
            ENTITIES_TABLE,
            pa.schema(
                [
                    pa.field("entity_id", pa.string()),
                    pa.field("type", pa.string()),
                    pa.field("status", pa.string()),
                    pa.field("source_person_id", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("last_seen_at", pa.string()),
                ]
            ),
        )

        self.events = self._open_or_create(
            EVENTS_TABLE,
            pa.schema(
                [
                    pa.field("event_id", pa.string()),
                    pa.field("entity_id", pa.string()),
                    pa.field("timestamp_utc", pa.string()),
                    pa.field("confidence", pa.float32()),
                    pa.field("decision", pa.string()),
                    pa.field("track_id", pa.string()),
                    pa.field("is_unknown", pa.bool_()),
                    pa.field("detection_id", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("location", pa.string()),
                ]
            ),
            migration_defaults={
                "location": "{}",
            },
        )

        self.alerts = self._open_or_create(
            ALERTS_TABLE,
            pa.schema(
                [
                    pa.field("alert_id", pa.string()),
                    pa.field("entity_id", pa.string()),
                    pa.field("type", pa.string()),
                    pa.field("severity", pa.string()),
                    pa.field("reason", pa.string()),
                    pa.field("timestamp_utc", pa.string()),
                    pa.field("first_seen_at", pa.string()),
                    pa.field("last_seen_at", pa.string()),
                    pa.field("event_count", pa.int32()),
                ]
            ),
        )

        self.unknown_profiles = self._open_or_create(
            UNKNOWN_PROFILES_TABLE,
            pa.schema(
                [
                    pa.field("entity_id", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
                    pa.field("sample_count", pa.int32()),
                    pa.field("created_at", pa.string()),
                    pa.field("last_seen_at", pa.string()),
                ]
            ),
        )

        self.incidents = self._open_or_create(
            INCIDENTS_TABLE,
            pa.schema(
                [
                    pa.field("incident_id", pa.string()),
                    pa.field("entity_id", pa.string()),
                    pa.field("status", pa.string()),
                    pa.field("start_time", pa.string()),
                    pa.field("last_seen_time", pa.string()),
                    pa.field("alert_ids", pa.string()),
                    pa.field("alert_count", pa.int32()),
                    pa.field("severity", pa.string()),
                    pa.field("summary", pa.string()),
                    pa.field("last_action_at", pa.string()),
                ]
            ),
            migration_defaults={
                "severity": "low",
                "summary": "",
                "last_action_at": "",
            },
        )

        self.actions = self._open_or_create(
            ACTIONS_TABLE,
            pa.schema(
                [
                    pa.field("action_id", pa.string()),
                    pa.field("incident_id", pa.string()),
                    pa.field("action_type", pa.string()),
                    pa.field("trigger", pa.string()),
                    pa.field("status", pa.string()),
                    pa.field("reason", pa.string()),
                    pa.field("context", pa.string()),
                    pa.field("timestamp_utc", pa.string()),
                ]
            ),
            migration_defaults={
                "context": "{}",
            },
        )

    @staticmethod
    def _default_for_arrow_type(field_type: pa.DataType) -> Any:
        if pa.types.is_string(field_type):
            return ""
        if pa.types.is_boolean(field_type):
            return False
        if pa.types.is_integer(field_type):
            return 0
        if pa.types.is_floating(field_type):
            return 0.0
        if pa.types.is_list(field_type):
            return []
        return None

    def _migrate_table_if_needed(
        self,
        name: str,
        table,
        schema: pa.Schema,
        migration_defaults: dict[str, Any],
    ):
        existing_cols = self._table_columns(table)
        target_cols = set(schema.names)
        missing = [field.name for field in schema if field.name not in existing_cols]
        if not missing:
            return table

        old_rows = table.to_arrow().to_pylist()
        migrated_rows: list[dict[str, Any]] = []
        for row in old_rows:
            payload: dict[str, Any] = {}
            for field in schema:
                if field.name in row:
                    payload[field.name] = row[field.name]
                elif field.name in migration_defaults:
                    payload[field.name] = migration_defaults[field.name]
                else:
                    payload[field.name] = self._default_for_arrow_type(field.type)
            # Ignore any unknown/legacy columns.
            payload = {k: v for k, v in payload.items() if k in target_cols}
            migrated_rows.append(payload)

        self.db.drop_table(name)
        self.db.create_table(name, data=migrated_rows, schema=schema)
        return self.db.open_table(name)

    def _open_or_create(
        self,
        name: str,
        schema: pa.Schema,
        migration_defaults: dict[str, Any] | None = None,
    ):
        list_tables = getattr(self.db, "list_tables", None)
        if callable(list_tables):
            raw = list_tables()
            raw_names = getattr(raw, "tables", raw)
            names = set()
            for entry in raw_names:
                if isinstance(entry, str):
                    names.add(entry)
                elif isinstance(entry, (tuple, list)) and entry:
                    names.add(str(entry[0]))
                else:
                    names.add(str(entry))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                names = set(self.db.table_names())
        if name in names:
            table = self.db.open_table(name)
            if migration_defaults:
                table = self._migrate_table_if_needed(
                    name=name,
                    table=table,
                    schema=schema,
                    migration_defaults=migration_defaults,
                )
            return table
        return self.db.create_table(name, data=[], schema=schema)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "''")

    def _query_rows(self, table, where: str | None = None, limit: int = 10_000) -> list[dict[str, Any]]:
        try:
            query = table.query()
            if where:
                query = query.where(where)
            query = query.limit(limit)
            return query.to_list()
        except Exception:
            rows = table.to_arrow().to_pylist()
            if where:
                # Fallback parser for simple equality filters like: field = 'value'
                match = re.fullmatch(r"\s*([a-zA-Z0-9_]+)\s*=\s*'([^']*)'\s*", where)
                if match:
                    field, value = match.group(1), match.group(2)
                    rows = [row for row in rows if str(row.get(field, "")) == value]
            return rows[:limit]

    def _table_columns(self, table) -> set[str]:
        schema = table.schema
        if hasattr(schema, "names"):
            return set(schema.names)
        return set()

    def _filtered_row(self, table, row: dict[str, Any]) -> dict[str, Any]:
        columns = self._table_columns(table)
        return {k: v for k, v in row.items() if k in columns}

    @staticmethod
    def _parse_iso_ts(value: str) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_json_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = []
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        return []

    @staticmethod
    def _parse_json_object(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    def add_or_update_person(self, person: PersonRecord) -> None:
        self.delete_person(person.person_id, delete_detections=False)
        self.persons.add([self._filtered_row(self.persons, person.model_dump())])
        self.set_person_state(
            person_id=person.person_id,
            lifecycle_state=person.lifecycle_state,
            lifecycle_reason=person.lifecycle_reason,
            enrollment_score=person.enrollment_score,
            valid_embeddings=person.valid_embeddings,
            valid_images=0,
            total_images=0,
        )

    def set_person_state(
        self,
        person_id: str,
        lifecycle_state: PersonLifecycleStatus | str,
        lifecycle_reason: str,
        enrollment_score: float,
        valid_embeddings: int,
        valid_images: int,
        total_images: int,
    ) -> None:
        escaped = self._escape(person_id)
        self.person_states.delete(f"person_id = '{escaped}'")
        payload = {
            "person_id": person_id,
            "lifecycle_state": str(lifecycle_state),
            "lifecycle_reason": lifecycle_reason,
            "enrollment_score": float(enrollment_score),
            "valid_embeddings": int(valid_embeddings),
            "valid_images": int(valid_images),
            "total_images": int(total_images),
            "updated_at": utc_now_iso(),
        }
        self.person_states.add([self._filtered_row(self.person_states, payload)])

    def get_person_state(self, person_id: str) -> dict[str, Any] | None:
        escaped = self._escape(person_id)
        rows = self._query_rows(self.person_states, where=f"person_id = '{escaped}'", limit=1)
        if not rows:
            return None
        row = rows[0]
        row.setdefault("lifecycle_state", PersonLifecycleStatus.draft.value)
        row.setdefault("lifecycle_reason", "state_not_initialized")
        row.setdefault("enrollment_score", 0.0)
        row.setdefault("valid_embeddings", 0)
        row.setdefault("valid_images", 0)
        row.setdefault("total_images", 0)
        return row

    def list_person_states(self) -> list[dict[str, Any]]:
        return self._query_rows(self.person_states, limit=100_000)

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        escaped = self._escape(person_id)
        rows = self._query_rows(self.persons, where=f"person_id = '{escaped}'", limit=1)
        if not rows:
            return None
        person = rows[0]
        state = self.get_person_state(person_id) or {}
        person["lifecycle_state"] = state.get("lifecycle_state", PersonLifecycleStatus.draft.value)
        person["lifecycle_reason"] = state.get("lifecycle_reason", "state_not_initialized")
        person["enrollment_score"] = float(state.get("enrollment_score", 0.0))
        person["valid_embeddings"] = int(state.get("valid_embeddings", 0))
        person["valid_images"] = int(state.get("valid_images", 0))
        person["total_images"] = int(state.get("total_images", 0))
        return person

    def list_persons(self) -> list[dict[str, Any]]:
        persons = self._query_rows(self.persons, limit=100_000)
        state_map = {row["person_id"]: row for row in self.list_person_states()}
        for person in persons:
            state = state_map.get(person.get("person_id", ""), {})
            person["lifecycle_state"] = state.get("lifecycle_state", PersonLifecycleStatus.draft.value)
            person["lifecycle_reason"] = state.get("lifecycle_reason", "state_not_initialized")
            person["enrollment_score"] = float(state.get("enrollment_score", 0.0))
            person["valid_embeddings"] = int(state.get("valid_embeddings", 0))
            person["valid_images"] = int(state.get("valid_images", 0))
            person["total_images"] = int(state.get("total_images", 0))
        return persons

    def active_person_ids(self) -> set[str]:
        states = self.list_person_states()
        if not states:
            return set()
        return {
            row.get("person_id", "")
            for row in states
            if str(row.get("lifecycle_state", "")) == PersonLifecycleStatus.active.value
        }

    def delete_person(self, person_id: str, delete_detections: bool = True) -> None:
        escaped = self._escape(person_id)
        try:
            related_incidents = [
                row
                for row in self.list_incidents(limit=100_000)
                if str(row.get("entity_id", "")) == person_id
            ]
            self.persons.delete(f"person_id = '{escaped}'")
            self.person_states.delete(f"person_id = '{escaped}'")
            self.embeddings.delete(f"person_id = '{escaped}'")
            self.image_quality.delete(f"person_id = '{escaped}'")
            self.entities.delete(f"entity_id = '{escaped}'")
            self.events.delete(f"entity_id = '{escaped}'")
            self.alerts.delete(f"entity_id = '{escaped}'")
            self.incidents.delete(f"entity_id = '{escaped}'")
            for incident in related_incidents:
                incident_id = self._escape(str(incident.get("incident_id", "")))
                self.actions.delete(f"incident_id = '{incident_id}'")
            if delete_detections:
                self.detections.delete(f"person_id = '{escaped}'")
        except Exception as exc:
            raise StorageError(f"Delete failed for {person_id}: {exc}") from exc

    def clear_image_quality(self, person_id: str) -> None:
        escaped = self._escape(person_id)
        self.image_quality.delete(f"person_id = '{escaped}'")

    def add_image_quality(self, assessment: QualityAssessment) -> None:
        payload = assessment.model_dump()
        payload["quality_id"] = f"{assessment.person_id}:{Path(assessment.source_path).name}:{assessment.created_at}"
        payload["reasons"] = json.dumps(assessment.reasons)
        payload = self._filtered_row(self.image_quality, payload)
        self.image_quality.add([payload])

    def list_image_quality(self, person_id: str) -> list[dict[str, Any]]:
        escaped = self._escape(person_id)
        rows = self._query_rows(self.image_quality, where=f"person_id = '{escaped}'", limit=100_000)
        for row in rows:
            raw_reasons = row.get("reasons", "[]")
            try:
                row["reasons"] = json.loads(raw_reasons) if isinstance(raw_reasons, str) else raw_reasons
            except Exception:
                row["reasons"] = []
        return rows

    def replace_person_embeddings(self, person_id: str, records: list[EmbeddingRecord]) -> None:
        escaped = self._escape(person_id)
        self.embeddings.delete(f"person_id = '{escaped}'")
        if not records:
            logger.warning("No embeddings provided for person {}", person_id)
            return

        payload = []
        for rec in records:
            row = rec.model_dump()
            row["embedding"] = [float(v) for v in row["embedding"]]
            row["quality_flags"] = json.dumps(row.get("quality_flags", []))
            payload.append(self._filtered_row(self.embeddings, row))
        self.embeddings.add(payload)

    def count_embeddings(self, person_id: str | None = None) -> int:
        if person_id:
            escaped = self._escape(person_id)
            return len(self._query_rows(self.embeddings, where=f"person_id = '{escaped}'", limit=100_000))
        return len(self._query_rows(self.embeddings, limit=100_000))

    def search_embeddings(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if len(embedding) != EMBEDDING_DIM:
            raise StorageError(f"Expected embedding dim {EMBEDDING_DIM}, got {len(embedding)}")
        rows = self.embeddings.search(embedding).metric("cosine").limit(top_k).to_list()
        results: list[dict[str, Any]] = []
        for row in rows:
            distance = float(row.get("_distance", 1.0))
            similarity = max(0.0, 1.0 - distance)
            raw_flags = row.get("quality_flags", "[]")
            try:
                flags = json.loads(raw_flags) if isinstance(raw_flags, str) else raw_flags
            except Exception:
                flags = []
            row["quality_flags"] = flags
            row["similarity"] = similarity
            results.append(row)
        return results

    def search_embeddings_for_person_ids(
        self,
        embedding: list[float],
        person_ids: set[str] | list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if len(embedding) != EMBEDDING_DIM:
            raise StorageError(f"Expected embedding dim {EMBEDDING_DIM}, got {len(embedding)}")
        ids = {str(pid) for pid in person_ids if str(pid)}
        if not ids:
            return []

        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(query) + 1e-9)
        rows = self._query_rows(self.embeddings, limit=200_000)

        results: list[dict[str, Any]] = []
        for row in rows:
            pid = str(row.get("person_id", ""))
            if pid not in ids:
                continue
            vector = np.asarray(row.get("embedding", []), dtype=np.float32).reshape(-1)
            if vector.shape[0] != EMBEDDING_DIM:
                continue
            similarity = float(np.dot(query, vector) / (q_norm * float(np.linalg.norm(vector) + 1e-9)))
            similarity = max(0.0, min(1.0, similarity))

            raw_flags = row.get("quality_flags", "[]")
            try:
                flags = json.loads(raw_flags) if isinstance(raw_flags, str) else raw_flags
            except Exception:
                flags = []

            out = dict(row)
            out["quality_flags"] = flags
            out["similarity"] = similarity
            results.append(out)

        results.sort(key=lambda item: float(item.get("similarity", 0.0)), reverse=True)
        return results[:top_k]

    def add_detection(self, event: DetectionEvent, embedding: list[float] | None) -> None:
        emb = embedding if embedding and len(embedding) == EMBEDDING_DIM else [0.0] * EMBEDDING_DIM
        payload = event.model_dump()
        payload["bbox"] = json.dumps(payload["bbox"])
        payload["metadata"] = json.dumps(payload["metadata"])
        payload["quality_flags"] = json.dumps(payload.get("quality_flags", []))
        payload["confidence"] = float(payload["confidence"])
        payload["similarity"] = float(payload["similarity"])
        payload["smoothed_confidence"] = float(payload.get("smoothed_confidence", payload["confidence"]))
        payload["person_id"] = payload["person_id"] or ""
        payload["embedding"] = [float(v) for v in emb]
        self.detections.add([self._filtered_row(self.detections, payload)])

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        escaped = self._escape(entity_id)
        rows = self._query_rows(self.entities, where=f"entity_id = '{escaped}'", limit=1)
        return rows[0] if rows else None

    def list_entities(self, limit: int = 10_000, type_filter: str | None = None) -> list[dict[str, Any]]:
        rows = self._query_rows(self.entities, limit=limit)
        if type_filter:
            rows = [row for row in rows if str(row.get("type", "")) == type_filter]
        return rows

    def ensure_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        status: EntityStatus = EntityStatus.active,
        source_person_id: str | None = None,
        last_seen_at: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        existing = self.get_entity(entity_id)
        payload = EntityRecord(
            entity_id=entity_id,
            type=entity_type,
            status=status,
            source_person_id=source_person_id,
            created_at=existing.get("created_at", now) if existing else now,
            last_seen_at=last_seen_at or now,
        ).model_dump()
        escaped = self._escape(entity_id)
        self.entities.delete(f"entity_id = '{escaped}'")
        self.entities.add([self._filtered_row(self.entities, payload)])
        return payload

    def add_event(self, event: EventRecord) -> None:
        payload = event.model_dump()
        payload["confidence"] = float(payload["confidence"])
        payload["location"] = json.dumps(payload.get("location", {}))
        self.events.add([self._filtered_row(self.events, payload)])

    def get_events(self, entity_id: str, window_sec: int) -> list[dict[str, Any]]:
        rows = self.list_events(limit=20_000, entity_id=entity_id)
        cutoff = datetime.now(timezone.utc).timestamp() - float(window_sec)
        kept: list[dict[str, Any]] = []
        for row in rows:
            ts = self._parse_iso_ts(str(row.get("timestamp_utc", "")))
            if ts is None:
                continue
            if ts.timestamp() >= cutoff:
                kept.append(row)
        kept.sort(key=lambda r: str(r.get("timestamp_utc", "")))
        return kept

    def list_events(self, limit: int = 100, entity_id: str | None = None) -> list[dict[str, Any]]:
        if entity_id:
            escaped = self._escape(entity_id)
            rows = self._query_rows(self.events, where=f"entity_id = '{escaped}'", limit=max(10_000, limit))
        else:
            rows = self._query_rows(self.events, limit=max(10_000, limit))
        for row in rows:
            row["location"] = self._parse_json_object(row.get("location", "{}"))
        rows.sort(key=lambda r: str(r.get("timestamp_utc", "")), reverse=True)
        return rows[:limit]

    def add_alert(self, alert: AlertRecord) -> None:
        payload = alert.model_dump()
        payload["type"] = str(payload["type"])
        payload["severity"] = str(payload["severity"])
        payload["event_count"] = int(payload.get("event_count", 1))
        self.alerts.add([self._filtered_row(self.alerts, payload)])

    def list_alerts(
        self,
        limit: int = 100,
        entity_id: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._query_rows(self.alerts, limit=max(limit, 10_000))
        if entity_id:
            rows = [row for row in rows if str(row.get("entity_id", "")) == entity_id]
        if severity:
            rows = [row for row in rows if str(row.get("severity", "")).lower() == severity.lower()]
        rows.sort(key=lambda r: str(r.get("timestamp_utc", "")), reverse=True)
        return rows[:limit]

    def create_incident(self, incident: IncidentRecord) -> None:
        payload = incident.model_dump()
        payload["status"] = str(payload["status"])
        payload["severity"] = str(payload["severity"])
        payload["alert_ids"] = json.dumps(payload.get("alert_ids", []))
        payload["alert_count"] = int(payload.get("alert_count", len(incident.alert_ids)))
        payload["summary"] = str(payload.get("summary", ""))
        payload["last_action_at"] = str(payload.get("last_action_at", ""))
        row = self._filtered_row(self.incidents, payload)
        escaped = self._escape(incident.incident_id)
        
        # Delete any existing incidents with this ID
        # Lance delete is synchronous but we verify the result
        try:
            self.incidents.delete(f"incident_id = '{escaped}'")
        except Exception:
            pass  # Safe to continue if delete fails (no matching record)
        
        # Add the new incident record
        self.incidents.add([row])
        
        # Verify we don't have duplicates (safety check)
        # This prevents the delete+add race condition from causing duplicates
        existing = self._query_rows(self.incidents, where=f"incident_id = '{escaped}'", limit=100)
        if len(existing) > 1:
            # Duplicate detected - remove all but the most recent
            for dup in existing[1:]:
                try:
                    self.incidents.delete(f"incident_id = '{escaped}'")
                    # Re-add only the intended record
                    self.incidents.add([row])
                    break
                except Exception:
                    pass  # Continue on error

    def update_incident(self, incident: IncidentRecord) -> None:
        self.create_incident(incident)

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        escaped = self._escape(incident_id)
        rows = self._query_rows(self.incidents, where=f"incident_id = '{escaped}'", limit=1)
        if not rows:
            return None
        row = rows[0]
        row["alert_ids"] = self._parse_json_list(row.get("alert_ids", "[]"))
        row["alert_count"] = int(row.get("alert_count", len(row["alert_ids"])))
        row["status"] = str(row.get("status", IncidentStatus.open.value))
        row["severity"] = str(row.get("severity", "low"))
        row["summary"] = str(row.get("summary", ""))
        row["last_action_at"] = str(row.get("last_action_at", ""))
        return row

    def list_incidents(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        rows = self._query_rows(self.incidents, limit=max(limit, 10_000))
        if status:
            rows = [row for row in rows if str(row.get("status", "")).lower() == status.lower()]
        for row in rows:
            row["alert_ids"] = self._parse_json_list(row.get("alert_ids", "[]"))
            row["alert_count"] = int(row.get("alert_count", len(row["alert_ids"])))
            row["status"] = str(row.get("status", IncidentStatus.open.value))
            row["severity"] = str(row.get("severity", "low"))
            row["summary"] = str(row.get("summary", ""))
            row["last_action_at"] = str(row.get("last_action_at", ""))
        rows.sort(key=lambda r: str(r.get("last_seen_time", "")), reverse=True)
        return rows[:limit]

    def get_active_incident(self, entity_id: str) -> dict[str, Any] | None:
        incidents = self.list_incidents(limit=10_000, status=IncidentStatus.open.value)
        for row in incidents:
            if str(row.get("entity_id", "")) == entity_id:
                return row
        return None

    def close_incident(self, incident_id: str) -> bool:
        incident = self.get_incident(incident_id)
        if not incident:
            return False
        if str(incident.get("status", "")) == IncidentStatus.closed.value:
            return True
        model = IncidentRecord(
            incident_id=str(incident.get("incident_id", "")),
            entity_id=str(incident.get("entity_id", "")),
            status=IncidentStatus.closed,
            start_time=str(incident.get("start_time", utc_now_iso())),
            last_seen_time=str(incident.get("last_seen_time", utc_now_iso())),
            alert_ids=self._parse_json_list(incident.get("alert_ids", [])),
            alert_count=int(incident.get("alert_count", 0)),
            severity=str(incident.get("severity", "low")),
            summary=str(incident.get("summary", "")),
            last_action_at=str(incident.get("last_action_at", "")),
        )
        self.update_incident(model)
        return True

    def set_incident_severity(self, incident_id: str, severity: str) -> bool:
        incident = self.get_incident(incident_id)
        if not incident:
            return False
        model = IncidentRecord(
            incident_id=str(incident.get("incident_id", "")),
            entity_id=str(incident.get("entity_id", "")),
            status=str(incident.get("status", IncidentStatus.open.value)),
            start_time=str(incident.get("start_time", utc_now_iso())),
            last_seen_time=str(incident.get("last_seen_time", utc_now_iso())),
            alert_ids=self._parse_json_list(incident.get("alert_ids", [])),
            alert_count=int(incident.get("alert_count", 0)),
            severity=severity,
            summary=str(incident.get("summary", "")),
            last_action_at=str(incident.get("last_action_at", "")),
        )
        self.update_incident(model)
        return True

    def set_incident_last_action(self, incident_id: str, timestamp_utc: str) -> bool:
        incident = self.get_incident(incident_id)
        if not incident:
            return False
        model = IncidentRecord(
            incident_id=str(incident.get("incident_id", "")),
            entity_id=str(incident.get("entity_id", "")),
            status=str(incident.get("status", IncidentStatus.open.value)),
            start_time=str(incident.get("start_time", utc_now_iso())),
            last_seen_time=str(incident.get("last_seen_time", utc_now_iso())),
            alert_ids=self._parse_json_list(incident.get("alert_ids", [])),
            alert_count=int(incident.get("alert_count", 0)),
            severity=str(incident.get("severity", "low")),
            summary=str(incident.get("summary", "")),
            last_action_at=timestamp_utc,
        )
        self.update_incident(model)
        return True

    def insert_action(self, action: ActionRecord) -> None:
        payload = action.model_dump()
        payload["action_type"] = str(payload["action_type"])
        payload["trigger"] = str(payload["trigger"])
        payload["status"] = str(payload["status"])
        payload["context"] = json.dumps(payload.get("context", {}))
        self.actions.add([self._filtered_row(self.actions, payload)])

    def get_actions(self, incident_id: str, limit: int = 500) -> list[dict[str, Any]]:
        return self.list_actions(limit=limit, incident_id=incident_id)

    def list_actions(self, limit: int = 500, incident_id: str | None = None) -> list[dict[str, Any]]:
        if incident_id:
            escaped = self._escape(incident_id)
            rows = self._query_rows(self.actions, where=f"incident_id = '{escaped}'", limit=max(limit, 10_000))
        else:
            rows = self._query_rows(self.actions, limit=max(limit, 10_000))
        for row in rows:
            row["context"] = self._parse_json_object(row.get("context", "{}"))
        rows.sort(key=lambda r: str(r.get("timestamp_utc", "")), reverse=True)
        return rows[:limit]

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector) + 1e-9)
        return (vector / norm).astype(np.float32)

    def resolve_or_create_unknown_entity(self, embedding: list[float], similarity_threshold: float) -> str:
        query = self._normalize_vector(np.asarray(embedding, dtype=np.float32).reshape(-1))
        profiles = self._query_rows(self.unknown_profiles, limit=100_000)
        best_entity = ""
        best_similarity = -1.0
        best_profile: dict[str, Any] | None = None

        for row in profiles:
            vector = np.asarray(row.get("embedding", []), dtype=np.float32).reshape(-1)
            if vector.shape[0] != EMBEDDING_DIM:
                continue
            vector = self._normalize_vector(vector)
            similarity = float(np.dot(query, vector))
            if similarity > best_similarity:
                best_similarity = similarity
                best_entity = str(row.get("entity_id", ""))
                best_profile = row

        now = utc_now_iso()
        if best_entity and best_similarity >= similarity_threshold and best_profile is not None:
            count = int(best_profile.get("sample_count", 1))
            prev = self._normalize_vector(np.asarray(best_profile.get("embedding", []), dtype=np.float32).reshape(-1))
            merged = self._normalize_vector(((prev * count) + query) / max(1, count + 1))
            escaped = self._escape(best_entity)
            self.unknown_profiles.delete(f"entity_id = '{escaped}'")
            self.unknown_profiles.add(
                [
                    self._filtered_row(
                        self.unknown_profiles,
                        {
                            "entity_id": best_entity,
                            "embedding": [float(v) for v in merged.tolist()],
                            "sample_count": count + 1,
                            "created_at": best_profile.get("created_at", now),
                            "last_seen_at": now,
                        },
                    )
                ]
            )
            self.ensure_entity(
                entity_id=best_entity,
                entity_type=EntityType.unknown,
                status=EntityStatus.active,
                source_person_id=None,
                last_seen_at=now,
            )
            return best_entity

        existing_unknowns = [row.get("entity_id", "") for row in self.list_entities(type_filter=EntityType.unknown.value)]
        new_entity_id = next_unknown_entity_id(existing_unknowns)
        self.unknown_profiles.add(
            [
                self._filtered_row(
                    self.unknown_profiles,
                    {
                        "entity_id": new_entity_id,
                        "embedding": [float(v) for v in query.tolist()],
                        "sample_count": 1,
                        "created_at": now,
                        "last_seen_at": now,
                    },
                )
            ]
        )
        self.ensure_entity(
            entity_id=new_entity_id,
            entity_type=EntityType.unknown,
            status=EntityStatus.active,
            source_person_id=None,
            last_seen_at=now,
        )
        return new_entity_id

    def add_detection_decision(
        self,
        detection_id: str,
        track_id: str,
        decision_state: str,
        decision_reason: str,
        smoothed_confidence: float,
        quality_flags: list[str],
        top_candidates: list[dict[str, Any]],
        liveness_provider: str,
        liveness_score: float,
    ) -> None:
        escaped = self._escape(detection_id)
        self.detection_decisions.delete(f"detection_id = '{escaped}'")
        payload = {
            "detection_id": detection_id,
            "track_id": track_id,
            "decision_state": decision_state,
            "decision_reason": decision_reason,
            "smoothed_confidence": float(smoothed_confidence),
            "quality_flags": json.dumps(quality_flags),
            "top_candidates": json.dumps(top_candidates),
            "liveness_provider": liveness_provider,
            "liveness_score": float(liveness_score),
            "created_at": utc_now_iso(),
        }
        self.detection_decisions.add([self._filtered_row(self.detection_decisions, payload)])

    def list_detections(self, limit: int = 10_000) -> list[dict[str, Any]]:
        detections = self._query_rows(self.detections, limit=limit)
        decisions = self._query_rows(self.detection_decisions, limit=limit * 2)
        decision_map = {row.get("detection_id", ""): row for row in decisions}
        for row in detections:
            det_id = row.get("detection_id", "")
            decision = decision_map.get(det_id, {})

            row["decision_state"] = row.get("decision_state") or decision.get("decision_state", "reject")
            row["decision_reason"] = row.get("decision_reason") or decision.get("decision_reason", "")
            row["track_id"] = row.get("track_id") or decision.get("track_id", "")
            row["smoothed_confidence"] = float(
                row.get("smoothed_confidence", decision.get("smoothed_confidence", row.get("confidence", 0.0)))
            )
            raw_flags = row.get("quality_flags") or decision.get("quality_flags", "[]")
            try:
                row["quality_flags"] = json.loads(raw_flags) if isinstance(raw_flags, str) else raw_flags
            except Exception:
                row["quality_flags"] = []
            row["liveness_provider"] = row.get("liveness_provider") or decision.get("liveness_provider", "none")
            row["liveness_score"] = float(row.get("liveness_score", decision.get("liveness_score", 0.0)))
            row["bbox"] = self._parse_json_list(row.get("bbox", "[]"))
            row["metadata"] = self._parse_json_object(row.get("metadata", "{}"))
        return detections

    def delete_unknown_entity(self, entity_id: str) -> None:
        """Remove a single unknown entity and all its associated data.

        Used by purge_ghost_entities to clean up warmup-phase artifacts.
        Does NOT remove known-person detection records.
        """
        escaped = self._escape(entity_id)
        try:
            self.entities.delete(f"entity_id = '{escaped}'")
            self.unknown_profiles.delete(f"entity_id = '{escaped}'")
            self.events.delete(f"entity_id = '{escaped}'")
            self.alerts.delete(f"entity_id = '{escaped}'")
            # Remove associated incidents and their actions.
            related_incidents = [
                row for row in self.list_incidents(limit=100_000)
                if str(row.get("entity_id", "")) == entity_id
            ]
            self.incidents.delete(f"entity_id = '{escaped}'")
            for incident in related_incidents:
                inc_id = self._escape(str(incident.get("incident_id", "")))
                if inc_id:
                    self.actions.delete(f"incident_id = '{inc_id}'")
        except Exception as exc:
            logger.warning("Failed to delete unknown entity {}: {}", entity_id, exc)

    def purge_ghost_entities(self, min_events: int = 2) -> int:
        """Remove UNK entities that have fewer than `min_events` detection events.

        These are warmup-phase ghost entities created before the temporal
        commitment gate was in place. Returns the count of purged entities.
        """
        unknowns = self.list_entities(type_filter=EntityType.unknown.value)
        purged = 0
        for entity in unknowns:
            eid = str(entity.get("entity_id", ""))
            if not eid:
                continue
            events = self.list_events(entity_id=eid, limit=100)
            if len(events) < min_events:
                self.delete_unknown_entity(eid)
                purged += 1
                logger.info("Purged ghost entity {} ({} events < {})", eid, len(events), min_events)
        if purged:
            logger.info("Ghost entity purge complete: removed {} entities.", purged)
        return purged

    def deduplicate_incidents(self) -> int:
        """Remove duplicate incidents caused by non-atomic delete+add operations.

        This method detects and removes duplicate incident_id records, keeping
        only the most recently updated one. Returns the count of deduplicated records removed.
        
        The duplication issue occurs when multiple incidents share the same incident_id,
        which can happen due to race conditions in the create_incident() method.
        """
        all_incidents = self._query_rows(self.incidents, limit=100_000)
        incident_groups: dict[str, list[dict]] = {}
        
        # Group incidents by incident_id
        for row in all_incidents:
            iid = str(row.get("incident_id", ""))
            if iid:
                if iid not in incident_groups:
                    incident_groups[iid] = []
                incident_groups[iid].append(row)
        
        deduped_count = 0
        for incident_id, duplicates in incident_groups.items():
            if len(duplicates) > 1:
                logger.info("Found {} duplicate incident records for {}", len(duplicates), incident_id)
                # Delete ALL duplicates and keep only one (most recently updated)
                escaped = self._escape(incident_id)
                self.incidents.delete(f"incident_id = '{escaped}'")
                
                # Re-add only the most recent one
                # Sort by last_seen_time, descending
                latest = max(
                    duplicates,
                    key=lambda r: str(r.get("last_seen_time", ""))
                )
                self.incidents.add([latest])
                deduped_count += len(duplicates) - 1
                logger.info("Deduped incident {} - removed {} duplicate records", incident_id, len(duplicates) - 1)
        
        if deduped_count > 0:
            logger.warning("Deduplication complete: removed {} duplicate incident records", deduped_count)
        return deduped_count

