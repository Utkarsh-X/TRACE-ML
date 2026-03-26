"""LanceDB-backed storage for persons, quality metadata, embeddings and detections."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from loguru import logger

from trace_ml.core.config import Settings
from trace_ml.core.errors import StorageError
from trace_ml.core.models import (
    DetectionEvent,
    EmbeddingRecord,
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

    def _open_or_create(self, name: str, schema: pa.Schema):
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
            return self.db.open_table(name)
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
            if where and "person_id = '" in where:
                value = where.split("person_id = '", 1)[1].rsplit("'", 1)[0]
                rows = [row for row in rows if row.get("person_id") == value]
            return rows[:limit]

    def _table_columns(self, table) -> set[str]:
        schema = table.schema
        if hasattr(schema, "names"):
            return set(schema.names)
        return set()

    def _filtered_row(self, table, row: dict[str, Any]) -> dict[str, Any]:
        columns = self._table_columns(table)
        return {k: v for k, v in row.items() if k in columns}

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
            self.persons.delete(f"person_id = '{escaped}'")
            self.person_states.delete(f"person_id = '{escaped}'")
            self.embeddings.delete(f"person_id = '{escaped}'")
            self.image_quality.delete(f"person_id = '{escaped}'")
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
            row["bbox"] = row.get("bbox", "[]")
        return detections
